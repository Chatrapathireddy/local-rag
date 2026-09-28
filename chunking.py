
import re
from dataclasses import dataclass, field


@dataclass
class Chunk:
    text: str
    source: str          
    chunk_id: int         
    metadata: dict = field(default_factory=dict)


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_into_sentences(text: str) -> list[str]:
    """Very lightweight sentence splitter (no external NLP deps)."""
    text = text.strip()
    if not text:
        return []
    sentences = _SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text_fixed(
    text: str,
    source: str,
    chunk_size: int = 800,
    overlap: int = 150,
) -> list[Chunk]:
    """
    Slide a fixed-size window over the raw characters of `text`.

    chunk_size: max characters per chunk
    overlap: how many characters of the previous chunk to repeat at the
             start of the next one, so a fact split across a chunk
             boundary still appears whole in at least one chunk.
    """
    text = text.strip()
    if not text:
        return []
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    chunk_id = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        piece = text[start:end].strip()
        if piece:
            chunks.append(Chunk(text=piece, source=source, chunk_id=chunk_id))
            chunk_id += 1
        if end == n:
            break
        start = end - overlap  # step forward, but re-include the overlap
    return chunks


def chunk_text_sentence_aware(
    text: str,
    source: str,
    chunk_size: int = 800,
    overlap_sentences: int = 1,
) -> list[Chunk]:
    """
    Greedily pack whole sentences into chunks up to ~chunk_size characters.
    Carries the last `overlap_sentences` sentences of a chunk into the
    start of the next chunk, for continuity.
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    chunks = []
    current: list[str] = []
    current_len = 0
    chunk_id = 0

    def flush():
        nonlocal current, current_len, chunk_id
        if current:
            chunks.append(
                Chunk(text=" ".join(current), source=source, chunk_id=chunk_id)
            )
            chunk_id += 1

    for sentence in sentences:
        sentence_len = len(sentence) + 1
        if current and current_len + sentence_len > chunk_size:
            flush()
            # carry overlap
            current = current[-overlap_sentences:] if overlap_sentences else []
            current_len = sum(len(s) + 1 for s in current)
        current.append(sentence)
        current_len += sentence_len

    flush()
    return chunks


def chunk_document(
    text: str,
    source: str,
    method: str = "sentence",
    chunk_size: int = 800,
    overlap: int = 150,
) -> list[Chunk]:
    """Convenience dispatcher used by the pipeline."""
    if method == "sentence":
        overlap_sentences = max(1, overlap // 200)  # rough heuristic
        return chunk_text_sentence_aware(
            text, source, chunk_size=chunk_size, overlap_sentences=overlap_sentences
        )
    elif method == "fixed":
        return chunk_text_fixed(text, source, chunk_size=chunk_size, overlap=overlap)
    elif method == "table_aware":
        return chunk_document_with_tables(text, source, chunk_size=chunk_size, overlap=overlap)
    else:
        raise ValueError(f"Unknown chunking method: {method}")




_ALIGNED_COLUMNS_RE = re.compile(r"\s{2,}")


def _is_table_line(line: str) -> bool:
    """A line counts as 'table-like' if it looks like a markdown pipe row
    (e.g. '| Name | Score |') or has 3+ columns separated by 2+ spaces
    (a common way plain-text tables are aligned)."""
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.count("|") >= 2:
        return True
    columns = _ALIGNED_COLUMNS_RE.split(stripped)
    return len(columns) >= 3


def _split_into_blocks(text: str) -> list[tuple[str, str]]:
    """Splits text into ('table', text) / ('prose', text) blocks, merging
    consecutive lines of the same type together."""
    lines = text.split("\n")
    raw_blocks: list[tuple[str, str]] = []
    current_type = None
    current_lines: list[str] = []

    for line in lines:
        block_type = "table" if _is_table_line(line) else "prose"
        if current_type is None:
            current_type = block_type
            current_lines = [line]
        elif block_type == current_type:
            current_lines.append(line)
        else:
            raw_blocks.append((current_type, "\n".join(current_lines)))
            current_type = block_type
            current_lines = [line]
    if current_lines:
        raw_blocks.append((current_type, "\n".join(current_lines)))

 
    cleaned: list[tuple[str, str]] = []
    for btype, btext in raw_blocks:
        if btype == "table" and btext.count("\n") < 1:
            btype = "prose"
        if cleaned and cleaned[-1][0] == btype:
            cleaned[-1] = (btype, cleaned[-1][1] + "\n" + btext)
        else:
            cleaned.append((btype, btext))
    return cleaned


def chunk_document_with_tables(
    text: str,
    source: str,
    chunk_size: int = 800,
    overlap: int = 150,
) -> list[Chunk]:
    blocks = _split_into_blocks(text)
    all_chunks: list[Chunk] = []
    chunk_id = 0

    for btype, btext in blocks:
        btext = btext.strip("\n")
        if not btext.strip():
            continue

        if btype == "table":
            # Kept whole, no matter how long -- splitting a table loses
            # its headers for every row after the split point.
            all_chunks.append(
                Chunk(text=btext, source=source, chunk_id=chunk_id, metadata={"type": "table"})
            )
            chunk_id += 1
        else:
            overlap_sentences = max(1, overlap // 200)
            sub_chunks = chunk_text_sentence_aware(
                btext, source, chunk_size=chunk_size, overlap_sentences=overlap_sentences
            )
            for c in sub_chunks:
                c.chunk_id = chunk_id
                c.metadata = {"type": "prose"}
                all_chunks.append(c)
                chunk_id += 1

    return all_chunks


if __name__ == "__main__":
    sample = (
        "RAG stands for Retrieval-Augmented Generation. It combines a retriever "
        "with a generator. The retriever finds relevant chunks of text from a "
        "knowledge base. The generator then uses those chunks as context to "
        "answer a question. This avoids hallucination and lets the model use "
        "up-to-date or private information it wasn't trained on."
    )
    for c in chunk_document(sample, source="demo.txt", chunk_size=120, overlap=40):
        print(f"[{c.chunk_id}] ({len(c.text)} chars) {c.text}")