
import numpy as np


class EmbeddingModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Returns an (N, dim) float32 array of L2-normalized embeddings.
        Normalizing up front means cosine similarity later is just a
        dot product -- cheaper and simpler in the vector store.
        """
        if isinstance(texts, str):
            texts = [texts]
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.astype("float32")

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]
