# RAG From Scratch (fully local, no API keys)

A minimal but complete Retrieval-Augmented Generation pipeline, built with
nothing but `numpy`, `sentence-transformers`, and `transformers`. No
LangChain, no FAISS/Chroma/Pinecone, no OpenAI/Anthropic API calls. Every
stage is implemented directly so you can see exactly what's happening.

## Architecture

```
raw documents
      |
      v
 [chunking.py]        split into overlapping text chunks
      |
      v
 [embeddings.py]       chunk text -> normalized vectors (sentence-transformers)
      |
      v
 [vector_store.py]     store vectors in a numpy array, cosine-similarity search
      |
      |   <-- query text also goes through embeddings.py -->
      v
   top-k relevant chunks
      |
      v
 [generator.py]        chunks + question -> prompt -> local LLM -> answer
      |
      v
   grounded answer + cited sources
```

`rag.py` (`RAGPipeline`) wires these four stages together into `.ingest_texts()`
and `.query()`.

## Why each piece works the way it does

- **Chunking**: embedding models produce one vector per input and have a
  limited context window, so long documents are split into ~400-800
  character pieces first. Overlap between chunks stops a fact from being
  cut in half at a chunk boundary and lost.
- **Embeddings**: `all-MiniLM-L6-v2` maps text to a 384-dim vector such that
  semantically similar text ends up with similar vectors. Vectors are
  L2-normalized on creation.
- **Vector store**: because vectors are normalized, cosine similarity
  between a query and every stored chunk is just one matrix multiply
  (`vectors @ query`), followed by `argpartition` to grab the top-k. This
  is brute-force search — no approximate-nearest-neighbor index — which
  is exact and plenty fast up to hundreds of thousands of chunks on one
  machine.
- **Generation**: the retrieved chunks are inserted into a prompt that
  explicitly instructs the model to answer only from that context and to
  say when it doesn't know. This is what makes it retrieval-*augmented*
  generation rather than the model just answering from memory.

## Setup

```bash
pip install -r requirements.txt
python demo.py
```

First run downloads two models from Hugging Face:
- `sentence-transformers/all-MiniLM-L6-v2` (~80MB) — embeddings
- `Qwen/Qwen2.5-1.5B-Instruct` (~3GB) — generation

Both are cached afterwards (`~/.cache/huggingface`), so later runs are
offline and fast.

## Usage in your own code

```python
from rag import RAGPipeline

pipeline = RAGPipeline()

pipeline.ingest_texts({
    "handbook.txt": open("handbook.txt").read(),
    "faq.txt": open("faq.txt").read(),
})
# or: pipeline.ingest_directory("my_docs/", glob="*.txt")

result = pipeline.query("What is the vacation policy?", top_k=4)
print(result["answer"])
for s in result["sources"]:
    print(s["source"], s["chunk_id"], s["score"])

# Persist the vector store so you don't have to re-embed next time
pipeline.save("my_index/")
# pipeline.load("my_index/")   # to restore it later
```

## Swapping models

- **Bigger/better embeddings**: try `all-mpnet-base-v2` (higher quality,
  slower) or `bge-small-en-v1.5`.
- **Bigger/better generation**: try `microsoft/Phi-3-mini-4k-instruct` or
  `HuggingFaceTB/SmolLM2-1.7B-Instruct`. Just change `generator_model_name`
  in `RAGPipeline(...)`.
- **GPU**: if a CUDA GPU is available, `generator.py` will use it
  automatically.

## Hardware notes

- Embedding + numpy search: trivial CPU load, works anywhere.
- The 1.5B generator model runs on CPU but is noticeably faster on GPU.
  If you want CPU-only and faster, drop to `SmolLM2-360M-Instruct` (lower
  quality answers, but tiny and fast).

## What "traditional" means here

This is the classic RAG recipe from the original retrieval-augmented
generation papers and early production systems: dense embeddings + cosine
similarity + top-k retrieval + a single generation pass with the retrieved
context stuffed into the prompt. It does *not* include newer additions
like re-ranking, hybrid (BM25 + dense) search, query rewriting, or
multi-hop retrieval — those are natural next steps once this baseline
works, and each of the four files here is a clean place to extend.

## Files

| File | Responsibility |
|---|---|
| `chunking.py` | Split documents into overlapping chunks |
| `embeddings.py` | Wrap sentence-transformers for text -> vector |
| `vector_store.py` | Numpy-only storage + cosine similarity search |
| `generator.py` | Local LLM prompt construction + generation |
| `rag.py` | `RAGPipeline` orchestrating ingest + query |
| `demo.py` | Runnable end-to-end example |
