import os
import json
import base64
from io import BytesIO
import time
import numpy as np

# Beets imports
from beets.plugins import BeetsPlugin
from beets.ui import Subcommand, decargs, UserError
from beets.util import displayable_path, syspath
from beets.library import Item

# Faiss import
import faiss

def print_obj(item):
    print(item)

# VGGish Plugin Utility Imports
# NOTE: These utilities should ideally be in a shared utility file, but are kept
# here for the self-contained plugin structure.
def string_to_tensor(s: str) -> np.ndarray:
    """Deserializes a Base64 string back into a NumPy array."""
    if not s: return None
    try:
        data = base64.b64decode(s)
        with BytesIO(data) as bio:
            # allow_pickle=True is needed for np.load to handle the saved format
            return np.load(bio, allow_pickle=True)['data']
    except Exception as e:
        print(f"Error deserializing VGGish data: {e}")
        return None
        
# --- 1. Vectorization and Metric Logic ---

# VGGish embedding dimension
D = 128 

VECTORIZE_SCHEMES = {
    'mean': lambda tensor: np.mean(tensor, axis=0),
}

METRIC_SCHEMES = {
    'cosine': {
        'index_factory': lambda D: faiss.IndexFlatIP(D),
        'normalize': True, # Requires L2 normalization before insertion
        'pre_normalize': True, # Need to normalize query vector for IP metric
    },
    'l2': {
        'index_factory': lambda D: faiss.IndexFlatL2(D),
        'normalize': False, # L2 distance works on unnormalized vectors
        'pre_normalize': False,
    }
}

def get_faiss_index_id_field(embedding_name: str, vectorize_scheme: str, metric_name: str) -> str:
    """Creates a unique field name for the Faiss index ID, encoding the scheme."""
    return f'faiss_{embedding_name}_{vectorize_scheme}_{metric_name}'

# --- 2. Faiss Core Manager (Re-introducing the Map for Speed) ---

class FaissManager:
    """Handles loading, updating, and saving the Faiss index and ID mapping."""
    
    def __init__(self, index_path, metric_config):
        self.index_path = index_path
        self.metric_config = metric_config
        # item_id_map: Maps Faiss Vector ID (integer) -> Beets Item ID (integer)
        self.item_id_map = {} 
        self._index = None

    def load_index(self):
        """Loads the Faiss index from disk or initializes a new one."""
        if self._index is not None:
            return self._index
            
        map_path = self.index_path + '.map' # Store map adjacent to index
        
        # Load Faiss Index
        if os.path.exists(self.index_path):
            self._index = faiss.read_index(self.index_path)
            if self._index.d != D:
                 raise UserError(f"Index dimension mismatch. Expected {D}, got {self._index.d}.")
            
            # Load ID Map (O(1) lookup guarantee)
            if os.path.exists(map_path):
                with open(map_path, 'r') as f:
                    # JSON keys are strings, so convert them back to ints
                    self.item_id_map = {int(k): v for k, v in json.load(f).items()}
            else:
                # If we lose the map, the index is useless
                raise UserError(f"Found Faiss index at {self.index_path} but missing critical ID map at {map_path}.")
            
            print(f"Loaded Faiss index with {self._index.ntotal} vectors.")
        else:
            # Initialize new index
            self._index = self.metric_config['index_factory'](D)
            print("Initialized new Faiss index.")

        return self._index

    def add_vector(self, vector: np.ndarray, beets_item_id: int):
        """Adds a single vector to the index. Returns the index ID assigned by Faiss."""
        index = self.load_index()
        
        # 1. Prepare Vector
        vector = vector.reshape(1, -1).astype('float32')
        
        # 2. Apply Normalization if required by the metric (e.g., for 'cosine')
        if self.metric_config['normalize']:
            faiss.normalize_L2(vector)
            
        # 3. Add to Faiss and get the assigned ID
        faiss_id = index.ntotal
        index.add(vector)
        
        # 4. Update the map for O(1) lookup speed
        self.item_id_map[faiss_id] = beets_item_id
        
        return faiss_id

    def save_index(self):
        """Saves the Faiss index and the ID map to disk."""
        if self._index is None:
            return
            
        # 1. Save Faiss Index (binary file)
        faiss.write_index(self._index, self.index_path)
        
        # 2. Save ID Map (JSON file)
        map_path = self.index_path + '.map'
        with open(map_path, 'w') as f:
            # item_id_map keys are integers, must be saved as strings in JSON
            json.dump(self.item_id_map, f, indent=4)
        
        print(f"Saved Faiss index ({self._index.ntotal} vectors) and ID map to {self.index_path}.")

# --- 3. Beets Plugin Class ---

class FaissPlugin(BeetsPlugin):
    """
    Beets plugin for converting raw embeddings to Faiss vectors, building a 
    Faiss index, and providing the Virtual Radio DJ command.
    """
    
    def __init__(self, name='faiss'):
        super().__init__(name)
        
        # Shared Configuration
        self.config.add({
            'embedding': 'vggish',
            'vectorize': 'mean',
            'metric': 'cosine',
            'index': 'faiss.db',
            'auto': False,
            # vrdj specific config
            'default_limit': 10,
        })
        
        self.faiss_manager = None
        self._index_field_name = None
        
        if self.config['auto'].get(bool):
            self.register_listener('item_imported', self._index_vectors)
            self.register_listener('album_imported', self._index_vectors)
            #self._log.info("Automatic Faiss indexing enabled.")

    def _setup_manager(self):
        """Initializes configuration and the FaissManager."""
        
        embedding_name = self.config['embedding'].get(str)
        vectorize_scheme = self.config['vectorize'].get(str)
        metric_name = self.config['metric'].get(str)
        index_path = self.config['index'].get(str)
        
        # 1. Validation checks
        if vectorize_scheme not in VECTORIZE_SCHEMES:
            raise UserError(f"Unknown vectorize scheme: {vectorize_scheme}")
        if metric_name not in METRIC_SCHEMES:
            raise UserError(f"Unknown metric: {metric_name}")
            
        # 2. Determine the unique attribute field name
        self._index_field_name = get_faiss_index_id_field(
            embedding_name, vectorize_scheme, metric_name
        )
        
        # 3. Initialize the manager
        metric_config = METRIC_SCHEMES[metric_name]
        self.faiss_manager = FaissManager(index_path, metric_config)
        self.faiss_manager.load_index()
        
        return embedding_name, vectorize_scheme, metric_name, metric_config

    def _process_item_to_faiss(self, item, force=False):
        """
        Converts a raw embedding into a Faiss vector and indexes it.
        Returns True if item was indexed/updated, False otherwise.
        """
        if not self._index_field_name:
            self._setup_manager()
            
        faiss_id_key = self._index_field_name
        
        # 1. Check if Faiss ID already exists
        if not force and item.get(faiss_id_key) is not None and item.get(faiss_id_key) != "":
            self._log.debug(f"Skipping {displayable_path(item.path)}: Faiss ID already exists.")
            return False

        # 2. Get the raw embedding string
        embedding_name = self.config['embedding'].get(str)
        embedding_string = item.get(embedding_name)
        
        if not embedding_string:
            self._log.debug(f"Skipping {displayable_path(item.path)}: No raw '{embedding_name}' embedding found.")
            return False

        # 3. Deserialize raw tensor
        raw_tensor = string_to_tensor(embedding_string)
        if raw_tensor is None:
            self._log.error(f"Failed to deserialize embedding for {displayable_path(item.path)}.")
            return False
            
        # 4. Vectorize (convert N x 128 to 1 x D)
        vectorize_scheme = self.config['vectorize'].get(str)
        vectorize_func = VECTORIZE_SCHEMES[vectorize_scheme]
        
        try:
            vector = vectorize_func(raw_tensor)
            if vector.shape != (D,):
                raise ValueError(f"Vectorization resulted in {vector.shape}. Expected ({D},).")
        except Exception as e:
            self._log.error(f"Vectorization failed for {displayable_path(item.path)}: {e}")
            return False

        # 5. Add vector to Faiss index, update map, and get ID
        new_faiss_id = self.faiss_manager.add_vector(vector, item.id)
        
        # 6. Store Faiss ID on the item (stored as a string in flexible field)
        item[faiss_id_key] = str(new_faiss_id) # Store as string in flexible field
        item.store()
        
        self._log.info(f"Indexed {displayable_path(item.path)} as Faiss ID {new_faiss_id}.")
        return True


    def _index_item_vectors(self, lib, item):
        return self._index_album_vectors(lib, [item])
    def _index_album_vectors(self, lib, album):
        return self._index_album_vectors(lib, album.items())

    def _index_vectors(self, lib, item=None, album=None):
        """Listener hook for 'database_change' event (Auto-indexing)."""
        items = list()
        if album is not None:
            items = list(album.items())
        if item is not None:
            items.append(item)
        

        self._setup_manager()
        
        indexed_count = 0
        with lib.transaction():
            for item in items:
                if self._process_item_to_faiss(item):
                    indexed_count += 1
                    
        if indexed_count > 0:
            self.faiss_manager.save_index()

    # --- Utility: Vector Retrieval and Aggregation ---

    def _aggregate_vectors(self, items, embedding_name, vectorize_scheme):
        """Retrieves and averages vectors for a list of items."""
        vectorize_func = VECTORIZE_SCHEMES[vectorize_scheme]
        all_vectors = []
        
        for item in items:
            embedding_string = item.get(embedding_name)
            if not embedding_string:
                self._log.warning(f"Could not retrieve embedding for seed item: {item.title}. Skipping.")
                continue

            # Deserialize and Vectorize (Steps B & C)
            raw_tensor = string_to_tensor(embedding_string)
            if raw_tensor is not None:
                try:
                    vector = vectorize_func(raw_tensor)
                    all_vectors.append(vector)
                except Exception as e:
                    self._log.error(f"Vectorization failed for seed item {item.title}: {e}")
        
        if not all_vectors:
            raise UserError("No usable embedded vectors found in the seed items for aggregation.")

        # Aggregate (Step D): Compute the mean across all vectors
        aggregated_vector = np.mean(np.array(all_vectors), axis=0)
        
        return aggregated_vector.reshape(1, -1).astype('float32')
        
    # --- Subcommands ---

    def commands(self):
        """Defines the 'beet faiss' and 'beet vrdj' subcommands."""
        
        faiss_command = Subcommand(
            'faiss',
            parser=None,
            help='calculate vectors and build Faiss index',
        )
        faiss_command.func = self._faiss_command_func
        
        vrdj_command = Subcommand(
            'vrdj',
            help='generate a similarity-based playlist from a query',
        )
        vrdj_command.parser.add_option(
            '-n', '--number', dest='limit', type=int,
            default=self.config['default_limit'].get(int),
            help='number of similar tracks to produce (excluding seeds)',
        )
        
        vrdj_command.func = self._vrdj_command_func
        
        return [faiss_command, vrdj_command]

    def _faiss_command_func(self, lib, opts, args):
        """
        Handler for the 'beet faiss [query]' command.
        """
        self._setup_manager()
        
        query = decargs(args)
        
        # Get items matching the query
        raw_field = self.config['embedding'].get(str)
        
        # Use '::' for regex matching in flexible fields
        query_with_filter = [f'{raw_field}::.+', f'-{self._index_field_name}::.+'] + query
        
        items_to_index = lib.items(query_with_filter)
        
        self._log.info(f"Starting Faiss indexing for {len(items_to_index)} items matching query: {query}")
        
        indexed_count = 0
        with lib.transaction():
            for item in items_to_index:
                if self._process_item_to_faiss(item, force=False):
                    indexed_count += 1

        if indexed_count > 0:
            self.faiss_manager.save_index()
            
        self._log.info(f"Finished Faiss indexing. {indexed_count} new vectors added.")
        
    def _vrdj_command_func(self, lib, opts, args):
        """
        Handler for the 'beet vrdj [query]' command. Generates a similar playlist.
        """
        embedding_name, vectorize_scheme, metric_name, metric_config = self._setup_manager()
        limit = opts.limit
        
        # 1. Retrieve Seed Items and Auto-index if necessary
        seed_items = lib.items(decargs(args))
        if not seed_items:
            self._log.warning("Query returned no seed items. Aborting VDJ.")
            return

        self._log.info(f"Found {len(seed_items)} seed items. Checking index status...")
        
        indexed_count = 0
        with lib.transaction():
            for item in seed_items:
                if self._process_item_to_faiss(item, force=False):
                    indexed_count += 1
        
        if indexed_count > 0:
            self._log.info(f"Indexed {indexed_count} new seed items. Saving index.")
            self.faiss_manager.save_index()
            # Reloading index to ensure it is up-to-date, though FaissManager.add_vector handles live updates.
            # This line is primarily for robustness if the index was freshly created.
            self.faiss_manager.load_index() 

        # 2. Aggregate Vectors
        query_vector = self._aggregate_vectors(seed_items, embedding_name, vectorize_scheme)

        # 3. Pre-normalize query vector if required by the metric (e.g., for cosine/IP)
        if metric_config['pre_normalize']:
            faiss.normalize_L2(query_vector)

        # 4. Search Faiss (Step E)
        index = self.faiss_manager._index
        # We search for limit + len(seed_items) to filter out the seed items themselves later.
        K = limit + len(seed_items) 
        
        # Ensure K is not larger than the total number of items indexed
        if K > index.ntotal:
            K = index.ntotal

        self._log.info(f"Searching Faiss index for top {K} similar tracks...")
        D_scores, I_indices = index.search(query_vector, K)
        
        # Faiss results are returned in NumPy arrays
        D_scores = D_scores[0]
        I_indices = I_indices[0]

        # 5. Filter out Seed Items and Map to Beets Items
        faiss_item_map = self.faiss_manager.item_id_map
        seed_item_ids = {item.id for item in seed_items}
        playlist_items = []

        for faiss_score, faiss_id in zip(D_scores, I_indices):
            beets_id = faiss_item_map.get(faiss_id)
            
            # Skip if ID is invalid or if the item is one of the seeds
            if beets_id is None or beets_id in seed_item_ids:
                continue


            item = lib.get_item(beets_id)
            if item:
                self._log.info(f'{faiss_id} {beets_id} {faiss_score} {item}')
                playlist_items.append(item)
                if len(playlist_items) >= limit:
                    break
        
        if not playlist_items:
            self._log.info("No new, similar tracks found in the index.")
            return

        # 6. Output Method 1: Filter Output (Standard beets listing)
        self._log.info(f"\n--- Virtual DJ: {len(playlist_items)} Similar Tracks ---")
        for item in playlist_items:
            # print_obj respects global formatting flags like -f
            print_obj(item)

        # 7. Output Method 2: M3U Playlist File
        playlist_filename = f'vrdj_{time.strftime("%Y%m%d_%H%M%S")}.m3u'
        playlist_path = os.path.join(os.getcwd(), playlist_filename)
        
        try:
            with open(syspath(playlist_path), 'w', encoding='utf-8') as f:
                # M3U Header
                f.write('#EXTM3U\n')
                for item in playlist_items:
                    # Extended M3U Format: #EXTINF:<length>,<artist> - <title>
                    # Use -1 for unknown/default length
                    extinf_line = f'#EXTINF:-1,{item.artist} - {item.title}\n'
                    f.write(extinf_line)
                    # Absolute path to the file (MPD compatibility)
                    f.write(displayable_path(item.path) + '\n')
            
            self._log.info(f"\n--- Playlist saved to {playlist_filename} ---")
        except Exception as e:
            self._log.error(f"Failed to write M3U playlist file: {e}")
