The beets-vrdj package provides two plugins, currently named after the technology they use:

-   **`vggish`:** calculate and store VGGish embeddings for songs
-   **`faiss`:** process embeddings and store and query them in an FAISS vector db

Besides commands matching these plugin names the `faiss` plug provides the command

-   **vrdj:** virtual radio DJ takes a beets query, aggregates the vectors from its items to form a faiss query and then produces a list of songs that are most "similar".
