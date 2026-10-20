
# =============================================================================
# CELL: Build an embedding service
# -----------------------------------------------------------------------------
# We wrap the sentence-transformers model in a class so the model is loaded
# ONCE (it takes a few seconds) and reused forever. The class exposes two
# methods: encode(list_of_texts) and encode_one(single_text).
# =============================================================================

class EmbeddingService:
    """Wraps a sentence-transformers model with simple encode methods."""
    def __init__(self, model_name: str = None):
        # Use the config model name unless the caller overrides it.
        self.model_name = model_name or config.EMBEDDING_MODEL

        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is not installed. Run the pip cell at the top."
            )
        # Create a cache folder so the model is downloaded once and reused.
        cache_dir = os.path.join(config.DATA_DIR, "model_cache")
        os.makedirs(cache_dir, exist_ok=True)
        # Load the model. This is the slow part (a few seconds) - it reads the
        # neural network weights from disk. After this, encoding is fast.
        # device="cpu" because we run on a laptop without a GPU.
        self.model = SentenceTransformer(self.model_name, cache_folder=cache_dir, device="cpu")
        # Record how many numbers each vector has (384 for our model).
        self._dimension = self.model.get_sentence_embedding_dimension()
    @property
    def dimension(self) -> int:
        """Return the number of dimensions per vector (e.g. 384)."""
        return self._dimension
    def encode(self, texts: list) -> np.ndarray:
        """Turn a list of texts into a 2D numpy array of embedding vectors.

        Returns an array of shape (len(texts), dimension). Each row is the
        vector for the corresponding input text, L2-normalized (length 1) so
        we can use dot-product as cosine similarity.
        """
        vectors = self.model.encode(
            texts,
            show_progress_bar=False,   # keep output clean
            normalize_embeddings=True, # L2-normalize for fast cosine similarity
            convert_to_numpy=True,     # return a numpy array
        )
        return vectors  # shape: (num_texts, dimension)

    def encode_one(self, text: str) -> np.ndarray:
        """Encode a single text (convenience wrapper).

        Returns a 1D vector of shape (dimension,).
        """
        return self.encode([text])[0]
# --- Create the one shared embedding service (singleton). ---
# The model is expensive to load, so we do it once at import time and share it.
print("Loading embedding model (first run downloads ~90MB)...")
try:
    embedding_service = EmbeddingService()
    print(f"  Model loaded: {embedding_service.model_name}")
    print(f"  Vector dimension: {embedding_service.dimension}")
    # --- Quick demo: embed two sentences and measure their similarity. ---
    v1 = embedding_service.encode_one("The satellite provides internet access.")
    v2 = embedding_service.encode_one("Broadband is delivered via orbiting spacecraft.")
    v3 = embedding_service.encode_one("The stock market crashed yesterday.")
    # Cosine similarity (vectors are normalized, so dot product = cosine).
    def cosine(a, b):
        return float(np.dot(a, b))
    print(f"")
    print(f"Cosine similarity demo:")
    print(f"  'satellite internet' vs 'broadband spacecraft': {cosine(v1, v2):.3f}  (related -> high)")
    print(f"  'satellite internet' vs 'stock market crash':    {cosine(v1, v3):.3f}  (unrelated -> low)")
except ImportError as e:
    print(f"  SKIPPED: {e}")
    embedding_service = None