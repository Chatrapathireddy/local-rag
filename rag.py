
import hashlib
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain_chroma import Chroma

from chunking import chunk_document

REFUSAL_MESSAGE = (
    "That doesn't appear to be covered in the documents I have "
    "access to, so I can't answer it from this knowledge base."
)

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that answers questions using ONLY the "
            "provided context. If the context does not contain the answer, say "
            "you don't have enough information -- do not make anything up. "
            "Cite which source each fact comes from using the [source] tags given.",
        ),
        (
            "human",
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer the question using only the context above.",
        ),
    ]
)

_ROLE_MAP = {"system": "system", "human": "user", "ai": "assistant"}


def _document_id(source: str, chunk_id: int, doc_type: str) -> str:
    identity = f"{source}\0{doc_type}\0{chunk_id}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


class LocalEmbeddings(Embeddings):
    """Adapts our sentence-transformers wrapper to LangChain's Embeddings
    interface, so any LangChain vector store can use it."""

    def __init__(self, embedding_model):
        self.model = embedding_model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode_one(text).tolist()


class RAGPipeline:
    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        generator_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        chunking_method: str = "table_aware",
        persist_directory: str | Path = Path(__file__).resolve().parent / "chroma_db",
        embedder=None,
        generator=None,
        image_understander=None,
    ):
        # embedder / generator / image_understander can be injected (used
        # for testing); otherwise they are created lazily from model names.
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunking_method = chunking_method

        if embedder is None:
            from embeddings import EmbeddingModel

            embedder = EmbeddingModel(embedding_model_name)
        self.embedder = embedder

        self.vector_store = Chroma(
            collection_name="local_rag",
            embedding_function=LocalEmbeddings(embedder),
            persist_directory=str(persist_directory),
            collection_metadata={"hnsw:space": "cosine"},
        )

        self._generator_model_name = generator_model_name
        self._generator = generator
        self._image_understander = image_understander

        # {filename: description} for every image ingested, so the UI can
        # show what the vision model saw right after an upload.
        self.image_summaries: dict[str, str] = {}

        self._top_k = 3
        self._min_score = 0.2
        self.chain = self._build_chain()

    @property
    def generator(self):
        if self._generator is None:
            from generator import LocalGenerator

            self._generator = LocalGenerator(self._generator_model_name)
        return self._generator

    @property
    def image_understander(self):
        if self._image_understander is None:
            from image_understanding import ImageUnderstander

            self._image_understander = ImageUnderstander()
        return self._image_understander

    # -- ingestion -----------------------------------------------------
    def ingest_texts(self, texts_by_source: dict[str, str]) -> int:
        docs = []
        ids = []
        for source, text in texts_by_source.items():
            chunks = chunk_document(
                text,
                source=source,
                method=self.chunking_method,
                chunk_size=self.chunk_size,
                overlap=self.chunk_overlap,
            )
            for c in chunks:
                docs.append(
                    Document(
                        page_content=c.text,
                        metadata={
                            "source": c.source,
                            "chunk_id": c.chunk_id,
                            "type": c.metadata.get("type", "prose"),
                        },
                    )
                )
                ids.append(_document_id(c.source, c.chunk_id, c.metadata.get("type", "prose")))
        if docs:
            self.vector_store.add_documents(docs, ids=ids)
        return len(docs)

    def ingest_images(self, image_paths_by_source: dict[str, str]) -> int:
        """Each image is described once by the vision model; the description
        is what gets embedded and searched. The image file path is kept in
        metadata so the image can be re-examined at answer time."""
        docs = []
        ids = []
        for source, path in image_paths_by_source.items():
            description = self.image_understander.describe(path)
            self.image_summaries[source] = description
            docs.append(
                Document(
                    page_content=description,
                    metadata={
                        "source": source,
                        "chunk_id": 0,
                        "type": "image",
                        "image_path": path,
                    },
                )
            )
            ids.append(_document_id(source, 0, "image"))
        if docs:
            self.vector_store.add_documents(docs, ids=ids)
        return len(docs)

    # -- the chain -------------------------------------------------------
    def _retrieve(self, question: str) -> dict:
        results = self.vector_store.similarity_search_with_relevance_scores(
            question, k=self._top_k
        )
        return {"question": question, "results": results}  # [(Document, score), ...]

    def _should_refuse(self, state: dict) -> bool:
        results = state["results"]
        top_score = results[0][1] if results else 0.0
        if not results or top_score < self._min_score:
            print(f"  [debug] top retrieval score {top_score:.3f} < min_score {self._min_score} -> refusing")
            return True
        return False

    def _top_is_image(self, state: dict) -> bool:
        return state["results"][0][0].metadata.get("type") == "image"

    def _prompt_inputs(self, state: dict) -> dict:
        blocks = [
            f"[source: {doc.metadata['source']}#{doc.metadata['chunk_id']}]\n{doc.page_content}"
            for doc, _score in state["results"]
        ]
        return {"context": "\n\n".join(blocks), "question": state["question"]}

    def _generate_text(self, prompt_value) -> str:
        messages = [
            {"role": _ROLE_MAP.get(m.type, "user"), "content": m.content}
            for m in prompt_value.to_messages()
        ]
        return self.generator.generate_from_messages(messages)

    def _answer_from_image(self, state: dict) -> str:
        top_doc = state["results"][0][0]
        return self.image_understander.answer(top_doc.metadata["image_path"], state["question"])

    def _build_chain(self):
        # Path 1: nothing relevant was retrieved -> refuse, never call a model.
        refuse = RunnablePassthrough.assign(answer=RunnableLambda(lambda _s: REFUSAL_MESSAGE))

        # Path 2: best match is an image -> ask the vision model directly.
        from_image = RunnablePassthrough.assign(answer=RunnableLambda(self._answer_from_image))

        # Path 3: best match is text -> prompt template -> text model.
        from_text = RunnablePassthrough.assign(
            answer=RunnableLambda(self._prompt_inputs) | PROMPT | RunnableLambda(self._generate_text)
        )

        router = RunnableBranch(
            (self._should_refuse, refuse),
            (self._top_is_image, from_image),
            from_text,
        )
        return RunnableLambda(self._retrieve) | router

    # -- public query ------------------------------------------------------
    def query(self, question: str, top_k: int = 4, min_score: float = 0.2) -> dict:
        self._top_k = top_k
        self._min_score = min_score
        state = self.chain.invoke(question)

        sources = []
        answered = state["answer"] != REFUSAL_MESSAGE
        if answered:
            sources = [
                {
                    "source": doc.metadata["source"],
                    "chunk_id": doc.metadata["chunk_id"],
                    "score": score,
                    "text": doc.page_content,
                }
                for doc, score in state["results"]
            ]
        return {"question": question, "answer": state["answer"], "sources": sources}