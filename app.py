
import tempfile
from pathlib import Path

import streamlit as st

from history import clear_history, get_history, initialize_history, save_exchange
from rag import RAGPipeline

st.set_page_config(page_title="Local RAG", page_icon="🔎")
initialize_history()


@st.cache_resource(show_spinner="Loading models...")
def get_pipeline():
    return RAGPipeline(
        embedding_model_name="all-MiniLM-L6-v2",
        generator_model_name="HuggingFaceTB/SmolLM2-360M-Instruct",
        chunk_size=400,
        chunk_overlap=80,
    )


if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = set()
if "_tmp_dir" not in st.session_state:
    st.session_state._tmp_dir = tempfile.mkdtemp()
if "image_summaries" not in st.session_state:
    st.session_state.image_summaries = {}

st.title("🔎 Ask my documents")

uploaded = st.file_uploader(
    "Upload .txt documents or images (diagrams, charts, photos)",
    type=["txt", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

if uploaded:
    pipeline = get_pipeline()
    st.session_state.image_summaries.update(pipeline.image_summaries)
    new_files = [f for f in uploaded if f.name not in st.session_state.ingested_files]
    if new_files:
        with st.spinner(f"Indexing {len(new_files)} file(s)..."):
            for f in new_files:
                if f.name.lower().endswith(".txt"):
                    text = f.read().decode("utf-8", errors="ignore")
                    n = pipeline.ingest_texts({f.name: text})
                else:
               
                    img_path = Path(st.session_state._tmp_dir) / f.name
                    img_path.write_bytes(f.read())
                    try:
                        n = pipeline.ingest_images({f.name: str(img_path)})
                    except OSError as exc:
                        st.error(
                            "Could not load the image model. Check your internet "
                            "connection and Hugging Face model access, then try again."
                        )
                        st.exception(exc)
                        st.stop()
                    st.session_state.image_summaries[f.name] = pipeline.image_summaries[f.name]
                st.session_state.ingested_files.add(f.name)
        st.success(f"Indexed {len(new_files)} new file(s).")

if st.session_state.image_summaries:
    st.subheader("Image summaries")
    for source, summary in st.session_state.image_summaries.items():
        st.markdown(f"**{source}**")
        st.write(summary)

if st.session_state.ingested_files:
    st.caption(f"Loaded: {', '.join(st.session_state.ingested_files)}")

    with st.form("question_form"):
        question = st.text_input("Your question:")
        submitted = st.form_submit_button("Ask")
    if submitted and question.strip():
        with st.spinner("Thinking..."):
            result = pipeline.query(question, top_k=3, min_score=0.2)
        save_exchange(question, result["answer"])
        st.write(result["answer"])
else:
    st.info("Upload a .txt file or an image above to get started.")

with st.sidebar:
    st.subheader("Question history")
    history = get_history()
    if history:
        if st.button("Clear history"):
            clear_history()
            st.rerun()
        for entry in history:
            with st.expander(entry["question"]):
                st.caption(entry["created_at"])
                st.write(entry["answer"])
    else:
        st.caption("No questions yet.")