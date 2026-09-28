

from pathlib import Path

from chunking import chunk_document, Chunk
from embeddings import EmbeddingModel
from vector_store import VectorStore
from generator import LocalGenerator
from image_understanding import ImageUnderstander


class RAGPipeline:
    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        generator_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        chunking_method: str = "table_aware",
        lazy_generator: bool = True,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunking_method = chunking_method

        self.embedder = EmbeddingModel(embedding_model_name)
        self.store = VectorStore(dim=self.embedder.dim)

        self._generator_model_name = generator_model_name
        self._generator = None if lazy_generator else LocalGenerator(generator_model_name)
        self._image_understander = None  # lazy -- only loaded if an image is ever ingested
        self.image_summaries: dict[str, str] = {}

    @property
    def generator(self) -> LocalGenerator:
        # The generator model is the heaviest thing to load, so we defer
        # loading it until the first query unless the caller opts out.
        if self._generator is None:
            self._generator = LocalGenerator(self._generator_model_name)
        return self._generator

    @property
    def image_understander(self) -> ImageUnderstander:
        if self._image_understander is None:
            self._image_understander = ImageUnderstander()
        return self._image_understander

    
    def ingest_texts(self, texts_by_source: dict[str, str]):
        """texts_by_source: {"source_name": "raw document text", ...}"""
        all_chunks = []
        for source, text in texts_by_source.items():
            chunks = chunk_document(
                text,
                source=source,
                method=self.chunking_method,
                chunk_size=self.chunk_size,
                overlap=self.chunk_overlap,
            )
            all_chunks.extend(chunks)

        if not all_chunks:
            return 0

        vectors = self.embedder.encode([c.text for c in all_chunks])
        self.store.add(vectors, all_chunks)
        return len(all_chunks)

    def ingest_images(self, image_paths_by_source: dict[str, str]):
        """image_paths_by_source: {"filename": "/path/to/image.png", ...}
        Each image is described once by the vision model, and that
        description is what gets embedded and searched -- the actual
        image is only re-examined at query time if it turns out to be
        the best match (see query())."""
        chunks = []
        descriptions = []
        for source, path in image_paths_by_source.items():
            description = self.image_understander.describe(path)
            self.image_summaries[source] = description
            chunks.append(
                Chunk(
                    text=description,
                    source=source,
                    chunk_id=0,
                    metadata={"type": "image", "image_path": path},
                )
            )
            descriptions.append(description)

        if not chunks:
            return 0

        vectors = self.embedder.encode(descriptions)
        self.store.add(vectors, chunks)
        return len(chunks)

    def ingest_directory(self, dir_path: str, glob: str = "*.txt"):
        texts = {}
        for path in Path(dir_path).glob(glob):
            texts[path.name] = path.read_text(encoding="utf-8")
        return self.ingest_texts(texts)

    
    def retrieve(self, question: str, top_k: int = 4):
        query_vector = self.embedder.encode_one(question)
        return self.store.search(query_vector, top_k=top_k)

    def query(self, question: str, top_k: int = 4, min_score: float = 0.2) -> dict:
        retrieved = self.retrieve(question, top_k=top_k)

        # Don't trust the model to self-police "I don't know" -- if nothing
        # retrieved clears the similarity bar, refuse before ever calling
        # the generator. min_score is a threshold you should tune per
        # embedding model / corpus; 0.2 is a reasonable starting point for
        # all-MiniLM-L6-v2.
        if not retrieved or retrieved[0][0] < min_score:
            top_score = retrieved[0][0] if retrieved else 0.0
            print(f"  [debug] top retrieval score {top_score:.3f} < min_score {min_score} -> refusing")
            return {
                "question": question,
                "answer": (
                    "That doesn't appear to be covered in the documents I have "
                    "access to, so I can't answer it from this knowledge base."
                ),
                "sources": [],
                "top_score": top_score,
            }

        top_chunk = retrieved[0][1]
        if top_chunk.metadata.get("type") == "image":

            answer = self.image_understander.answer(top_chunk.metadata["image_path"], question)
        else:
            answer = self.generator.generate(question, retrieved)

        return {
            "question": question,
            "answer": answer,
            "sources": [
                {"source": c.source, "chunk_id": c.chunk_id, "score": score, "text": c.text}
                for score, c in retrieved
            ],
        }


    def save(self, path: str):
        self.store.save(path)

    def load(self, path: str):
        self.store = VectorStore.load(path)