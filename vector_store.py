
import json
import numpy as np
from pathlib import Path
from dataclasses import asdict

from chunking import Chunk


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self._vectors = np.zeros((0, dim), dtype="float32")
        self._chunks: list[Chunk] = []

    def __len__(self):
        return len(self._chunks)

    def add(self, vectors: np.ndarray, chunks: list[Chunk]):
        if vectors.shape[0] != len(chunks):
            raise ValueError("Number of vectors and chunks must match")
        if vectors.shape[1] != self.dim:
            raise ValueError(
                f"Vector dim {vectors.shape[1]} does not match store dim {self.dim}"
            )
        self._vectors = np.vstack([self._vectors, vectors.astype("float32")])
        self._chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, top_k: int = 4):
        """
        Returns a list of (score, Chunk) tuples, highest similarity first.
        score is cosine similarity in [-1, 1] (in practice mostly [0, 1]
        for text embeddings).
        """
        if len(self._chunks) == 0:
            return []

        top_k = min(top_k, len(self._chunks))

        scores = self._vectors @ query_vector


        top_idx = np.argpartition(-scores, top_k - 1)[:top_k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        return [(float(scores[i]), self._chunks[i]) for i in top_idx]

    # -- persistence -------------------------------------------------
    def save(self, path: str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "vectors.npy", self._vectors)
        with open(path / "chunks.json", "w") as f:
            json.dump(
                {"dim": self.dim, "chunks": [asdict(c) for c in self._chunks]},
                f,
            )

    @classmethod
    def load(cls, path: str) -> "VectorStore":
        path = Path(path)
        vectors = np.load(path / "vectors.npy")
        with open(path / "chunks.json") as f:
            data = json.load(f)
        store = cls(dim=data["dim"])
        chunks = [Chunk(**c) for c in data["chunks"]]
        store.add(vectors, chunks)
        return store


if __name__ == "__main__":
    # Sanity-check with synthetic vectors (no embedding model needed).
    rng = np.random.default_rng(0)
    dim = 8
    store = VectorStore(dim=dim)

    raw = rng.normal(size=(5, dim)).astype("float32")
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)  # normalize like real embeddings
    chunks = [Chunk(text=f"chunk {i}", source="test.txt", chunk_id=i) for i in range(5)]
    store.add(raw, chunks)

    query = raw[2] + rng.normal(scale=0.01, size=dim).astype("float32")
    query /= np.linalg.norm(query)

    results = store.search(query, top_k=3)
    print("Top matches for a query close to chunk 2:")
    for score, chunk in results:
        print(f"  score={score:.4f}  {chunk.text}")
    assert results[0][1].chunk_id == 2, "Nearest neighbor search sanity check failed"
    print("Sanity check passed.")
