

from rag import RAGPipeline

SAMPLE_DOCS = {
    "rag_overview.txt": (
        "Retrieval-Augmented Generation (RAG) is a technique that combines "
        "a retrieval system with a text generation model. Instead of relying "
        "purely on knowledge baked into a model's weights during training, "
        "RAG looks up relevant information from an external knowledge base "
        "at query time and feeds it to the generator as context. This has "
        "two big advantages: it reduces hallucination, because the model "
        "can ground its answer in retrieved text, and it lets the system "
        "answer questions about information the model was never trained on, "
        "such as private company documents or very recent events, without "
        "retraining the model."
    ),
    "vector_search.txt": (
        "Vector search works by converting text into numerical vectors "
        "called embeddings, using an embedding model. Texts with similar "
        "meaning end up with vectors that point in similar directions in "
        "high-dimensional space. To find relevant text for a query, the "
        "query is also embedded, and then compared against every stored "
        "vector using a similarity metric, most commonly cosine similarity. "
        "The chunks with the highest similarity scores are considered the "
        "most relevant and are returned to the generator."
    ),
    "chunking_strategy.txt": (
        "Before text can be embedded, long documents must be split into "
        "smaller chunks, because embedding models work best on short "
        "passages and because retrieval needs to be precise. A common "
        "chunk size is a few hundred to a thousand characters. Chunks are "
        "usually created with some overlap between consecutive chunks so "
        "that a sentence or fact that falls near a chunk boundary is not "
        "cut in half and lost from both chunks."
    ),
}


def main():
    print("Loading embedding model and building pipeline...")
    pipeline = RAGPipeline(
        embedding_model_name="all-MiniLM-L6-v2",
        generator_model_name="HuggingFaceTB/SmolLM2-360M-Instruct" ,
        chunk_size=400,
        chunk_overlap=80,
    )

    print("Ingesting sample documents...")
    n_chunks = pipeline.ingest_texts(SAMPLE_DOCS)
    print(f"Ingested {n_chunks} chunks from {len(SAMPLE_DOCS)} documents.\n")

    questions = [
        "What is RAG and why is it useful?",
        "How does vector search decide which chunks are relevant?",
        "Why do documents get split into chunks before embedding?",
        "What is the capital of France?",  # not in the corpus -> should say it doesn't know
    ]

    for q in questions:
        print(f"Q: {q}")
        result = pipeline.query(q, top_k=3)
        print(f"A: {result['answer']}\n")
        print("Sources used:")
        for s in result["sources"]:
            print(f"  - {s['source']}#{s['chunk_id']} (score={s['score']:.3f})")
        print("-" * 70)


if __name__ == "__main__":
    main()
