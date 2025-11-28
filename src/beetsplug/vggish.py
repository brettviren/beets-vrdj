'''
Produce vggish embeddings of beets items.
'''
import os
import json
import base64
from io import BytesIO
import time
import numpy as np

# Beets imports
from beets.plugins import BeetsPlugin
#from beets.dbcommon import add_field
#from beets.library import write_item
from beets.ui import Subcommand, decargs, UserError
from beets.util import displayable_path

# Torch/VGGish imports
import torch
from torchvggish import vggish_input, vggish

# --- 1. Tensor Serialization Utilities ---

def tensor_to_string(tensor: np.ndarray) -> str:
    """
    Serializes a NumPy array (from a VGGish tensor) to a Base64 string
    using binary compression (np.savez_compressed).
    This minimizes size for storage in the SQLite database.
    """
    if tensor is None:
        return ""
    
    with BytesIO() as bio:
        # Use savez_compressed for efficient binary storage
        np.savez_compressed(bio, data=tensor)
        bio.seek(0)
        return base64.b64encode(bio.read()).decode('utf-8')

def string_to_tensor(s: str) -> np.ndarray:
    """
    Deserializes a Base64 string back into a NumPy array.
    """
    if not s:
        return None
    
    try:
        data = base64.b64decode(s)
        with BytesIO(data) as bio:
            # Load from the compressed binary format
            return np.load(bio)['data']
    except Exception as e:
        print(f"Error deserializing VGGish data: {e}")
        return None

# --- 2. VGGish Core Functionality ---

class VGGishCore:
    """Handles VGGish model loading and embedding calculation."""
    
    def __init__(self):
        self._model = None
        self._device = torch.device('cpu') # Always prefer CPU for background tasks
        
    def set_device(self, device):
        self._device = torch.device(device)
        # normally, not the best thing to do but torchvggish does not seem to
        # completely move to a device when using .to().
        torch.set_default_device(device)
        

    def get_model(self):
        """Loads or returns the cached VGGish model."""
        if self._model is None:
            print(f"Loading VGGish model (using {self._device})...")
            try:
                # Instantiate the model from torchvggish and map to CPU
                model = vggish()
                model.eval()
                model = model.to(self._device) # note, does not actually work for gpu
                self._model = model
            except Exception as e:
                raise UserError(f"Failed to load VGGish model: {e}")
                
        return self._model

    def get_vggish_sequential_embeddings(self, item_path: str) -> np.ndarray | None:
        """
        Computes ALL VGGish time-slice embeddings for an audio file.
        Returns the (Time Slices, 128) matrix.
        """
        model = self.get_model()
        
        try:
            with torch.no_grad():
                # torchvggish handles loading, resampling, and segmenting
                audio_tensor = vggish_input.wavfile_to_examples(item_path)
                audio_tensor = audio_tensor.to(self._device)
                embeddings = model.forward(audio_tensor)
                
            return embeddings.cpu().numpy().astype('float32')
        except Exception as e:
            # Catch errors like audio decoding failures
            print(f"Error processing {item_path}: {e}")
            return None

# --- 3. Beets Plugin Class ---

class VGGishPlugin(BeetsPlugin):
    """
    Beets plugin for calculating and storing VGGish audio embeddings.
    """
    # The name of the field storing the serialized VGGish tensor
    EMBEDDING_FIELD = 'vggish'
    
    def __init__(self, name='vggish'):
        super().__init__(name)
        
        # Define the new field in the database. 
        # By default, beets items have all fields as strings, which is what we need.
        # add_field(self.EMBEDDING_FIELD, write=True)
        
        # Configuration setup (used to check for 'auto' mode)
        self.config.add({
            'auto': False,
            'force': False,
            'device': 'cpu',
        })
        
        # Initialize the VGGish processor (lazily loads the model)
        self.vggish_core = VGGishCore()
        
        # Register the import hook if 'auto' is enabled
        if self.config['auto'].get(bool):
            self.register_listener('item_imported', self.process_embedding)
            self.register_listener('album_imported', self.process_embedding)
            # self._log.info(f"Automatic VGGish embedding generation enabled.")

    # --- Event Hook (Automatic Import) ---


    def process_embedding(lib, item=None, album=None):
        if item is not None:
            self.process_embedding(lib, item)
        if album is not None:
            for item in album.items():
                self.process_embedding(lib, item)

    def process_item_embedding(self, item):

        """Calculates and stores the VGGish embedding for a single item."""
        
        if not self.config['force'].get(bool) and item.get(self.EMBEDDING_FIELD):
            self._log.debug(f"Skipping {displayable_path(item.path)}: already indexed.")
            return

        device = self.config['device'].get(str)
        if device:
            self.vggish_core.set_device(device)

        self._log.info(f"Generating VGGish embedding for {displayable_path(item.path)}")
        start_time = time.time()
        
        # 1. Calculate the sequential embeddings
        embeddings_tensor = self.vggish_core.get_vggish_sequential_embeddings(
            displayable_path(item.path) # Use beets util path
        )
        
        if embeddings_tensor is not None:
            # 2. Serialize the tensor to a string
            embedding_string = tensor_to_string(embeddings_tensor)
            
            # 3. Store the string attribute on the item
            item[self.EMBEDDING_FIELD] = embedding_string
            
            # 4. Save the item back to the database
            #write_item(item)
            item.store()
            
            self._log.info(f"Successfully stored {embeddings_tensor.shape[0]} slices in {time.time() - start_time:.2f}s.")
        else:
            self._log.error(f"Failed to generate embedding for {displayable_path(item.path)}.")

    # --- Subcommand (Manual Processing) ---
    
    def commands(self):
        """Defines the 'beet embedding' subcommand."""
        
        vggish_command = Subcommand(
            'vggish',
            parser=None,
            help=f'calculate and store VGGish embedding',
        )
        vggish_command.func = self._vggish_command_func
        
        return [vggish_command]

    def _vggish_command_func(self, lib, opts, args):
        """
        Handler for the 'beet embedding [query]' command.
        """
        # Parse arguments into a query
        query = decargs(args)
        
        # Get items matching the query
        items = lib.items(query)
        
        self._log.info(f"Starting VGGish embedding for {len(items)} items matching query: {query}")
        
        # Process each item
        with lib.transaction():
            for item in items:
                self.process_item_embedding(item)

        self._log.info("Finished VGGish embedding calculation.")
