''' 
Vahid Azimi, MD

08/08/2025

pdf_qa_web.py

'''

## Import necessary libraries

import os
import io
import zipfile
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

## Config
ENV_PATH = ".env"            # fixed path to env file
CORPUS_INDEX = "corpus_index"  # folder produced by corpus builder
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MIN_CHARS = 20

st.set_page_config(page_title="🧪 Lab Document Q&A", layout="centered")
st.title("🧪 Lab Document Q&A")
st.caption("For internal lab use: quick retrieval and summarization from validated document sets. Avoid PHI unless policy allows.")

# 1) Load API key (local only; nothing stored server-side)
if os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH)
user_api_key = os.getenv("OPENAI_API_KEY")
if not user_api_key:
    st.error(f"OPENAI_API_KEY not found. Ensure `{ENV_PATH}` exists with your key.")
    st.stop()

    
# Shared embeddings + LLM
embeddings = OpenAIEmbeddings(openai_api_key=user_api_key)
llm = ChatOpenAI(model="gpt-5", temperature=1, openai_api_key=user_api_key)
    
# Sidebar: all data-source UI
st.sidebar.header("Data source")
data_source = st.sidebar.radio(
    "Use:",
    [
        "Base corpus (prebuilt FAISS)",
        "My uploaded files (PDF/DOCX, session-only)",
        "Upload a saved FAISS (.zip) from my computer",
    ],
    index=0,
)

def split_and_clean(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    chunks = [d for d in chunks if d.page_content and d.page_content.strip()]
    chunks = [d for d in chunks if len(d.page_content.strip()) >= MIN_CHARS]
    return chunks

def zip_dir_to_bytes(dir_path: str) -> bytes:
    """Zip a directory into an in-memory bytes object"""
    memfile = io.BytesIO()
    with zipfile.ZipFile(memfile, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(dir_path):
            for f in files:
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, dir_path)
                zf.write(full_path, arcname=rel_path)
    memfile.seek(0)
    return memfile.read()

def load_faiss_from_zip_bytes(zip_bytes: bytes):
    """Extract a FAISS zip to a temp dir and load it (session-only)."""
    tmp_dir = tempfile.mkdtemp(prefix="faiss_")
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        zf.extractall(tmp_dir)
    vs = FAISS.load_local(tmp_dir, embeddings, allow_dangerous_deserialization=True)
    return vs

def load_uploaded_file_to_documents(uploaded_file):
    """
    Handles PDF (via PyMuPDFLoader) and DOCX (via Docx2txtLoader).
    Uses a temp file path per upload. Returns list[Document].
    """
    name = uploaded_file.name
    suffix = os.path.splitext(name)[1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        if suffix == ".pdf":
            loader = PyMuPDFLoader(tmp_path)
        elif suffix == ".docx":
            loader = Docx2txtLoader(tmp_path)
        else:
            st.sidebar.error(f"Unsupported file type: {suffix}. Please upload PDF or DOCX.")
            return []
        docs = loader.load()
        # attach filename metadata consistently
        for d in docs:
            d.metadata = {**d.metadata, "source": name}
        return docs
    except Exception as e:
        st.sidebar.error(f"Failed to read {name}: {e}")
        return []

# 2) Build vectorstore depending on choice
vectorstore = None

# A) Base corpus (prebuilt)
if data_source == "Base corpus (prebuilt FAISS)":
    @st.cache_resource
    def load_base_index():
        return FAISS.load_local(
            CORPUS_INDEX, embeddings, allow_dangerous_deserialization=True
        )
    try:
        vectorstore = load_base_index()
        st.sidebar.success("Loaded base FAISS index.")
    except Exception as e:
        st.sidebar.error(f"Failed to load `{CORPUS_INDEX}`: {e}")
        st.stop()

# B) User uploads (PDF or DOCX) — session-only; offer FAISS download
elif data_source == "My uploaded files (PDF/DOCX, session-only)":
    st.sidebar.subheader("Upload PDF/DOCX file(s)")
    uploaded_files = st.sidebar.file_uploader(
        "Select files", type=["pdf", "docx"], accept_multiple_files=True
    )
    if not uploaded_files:
        st.sidebar.info("Upload at least one PDF or DOCX to proceed.")
        st.stop()

    docs = []
    with st.spinner("Processing uploads..."):
        for f in uploaded_files:
            docs.extend(load_uploaded_file_to_documents(f))

        chunks = split_and_clean(docs)
        if not chunks:
            st.sidebar.error("No extractable text found in the uploaded files.")
            st.stop()

        try:
            vectorstore = FAISS.from_documents(chunks, embeddings)
            st.sidebar.success("Built a temporary FAISS index from uploads.")
        except Exception as e:
            st.sidebar.error(f"Failed to embed uploaded files: {e}")
            st.stop()

    # Offer a download of this session’s FAISS as a zip (local to the user)
    with tempfile.TemporaryDirectory() as tmp_vs_dir:
        vectorstore.save_local(tmp_vs_dir)
        zip_bytes = zip_dir_to_bytes(tmp_vs_dir)
        st.sidebar.download_button(
            label="⬇️ Download this FAISS index (.zip)",
            data=zip_bytes,
            file_name="my_faiss_index.zip",
            mime="application/zip",
        )

# C) User uploads a previously saved FAISS zip (session-only)
else:
    st.sidebar.subheader("Upload your saved FAISS (.zip)")
    zip_upload = st.sidebar.file_uploader("Select a .zip", type="zip")
    if not zip_upload:
        st.sidebar.info("Upload a FAISS .zip to proceed.")
        st.stop()
    try:
        vectorstore = load_faiss_from_zip_bytes(zip_upload.read())
        st.sidebar.success("Loaded your FAISS index from the uploaded zip.")
    except Exception as e:
        st.sidebar.error(f"Failed to load FAISS from zip: {e}")
        st.stop()

# Chat settings & memory
st.sidebar.markdown("**⚠️ PHI caution:** Follow institutional policy when entering identifiers.")
top_k = st.sidebar.slider("Top-k evidence chunks", min_value=2, max_value=10, value=4, step=1)

# Maintain per-session chat history (NOT persisted)
if "messages" not in st.session_state:
    # seed with a system-style instruction (kept out of the model call, but rendered to user for context)
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Ask me about your selected documents and I’ll cite sources. Follow-ups are welcome."}
    ]

if "chat_history_tuples" not in st.session_state:
    # ConversationalRetrievalChain expects list[tuple[str, str]] of (human, ai)
    st.session_state.chat_history_tuples = []

# Build a retriever each run so top_k updates immediately
retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

# Conversational Retrieval chain (returns 'answer' + 'source_documents')
conv_chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-5", temperature=1, openai_api_key=user_api_key),
    retriever=retriever,
    return_source_documents=True,
    # You can tweak this prompt to be more lab-specific:
    combine_docs_chain_kwargs={
        "prompt": None  # use default; swap in a lab-focused prompt if desired
    }
)

# Helpers
def render_sources(docs):
    """Render deduped source chunks with preview + expandable full text."""
    seen = set()
    shown = 0
    for d in docs:
        src = d.metadata.get("source", "unknown")
        pg = d.metadata.get("page")
        key = (src, pg)
        if key in seen:
            continue
        seen.add(key)

        full_text = (d.page_content or "").strip()
        preview = full_text.replace("\n", " ")
        short_preview = preview[:300] + ("…" if len(preview) > 300 else "")

        label = f"**{src}**" + (f", p. {pg}" if pg is not None else "")
        st.markdown(label)
        st.markdown(f"> {short_preview}")
        with st.expander("Show full text"):
            st.write(full_text if full_text else "_(empty)_")
        st.markdown("---")
        shown += 1
    if shown == 0:
        st.markdown("_No sources returned._")

def count_dedup_sources(docs):
    seen = set()
    for d in docs:
        key = (d.metadata.get("source", "unknown"), d.metadata.get("page"))
        seen.add(key)
    return len(seen)

# Render history (answers + their saved sources)
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant" and "sources" in m:
            dedup_n = count_dedup_sources(m["sources"])
            with st.expander(f"📚 Sources ({dedup_n})", expanded=False):
                render_sources(m["sources"])

# Input & turn handling
user_msg = st.chat_input("Type your question or follow-up…")
if user_msg:
    # show user message
    st.session_state.messages.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    # assistant turn
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                result = conv_chain({
                    "question": user_msg,
                    "chat_history": st.session_state.chat_history_tuples
                })
                answer = result["answer"]
                sources = result.get("source_documents", [])

                # render answer
                st.markdown(answer)

                # per-answer sources (also rendered live here)
                dedup_n = count_dedup_sources(sources)
                with st.expander(f"📚 Sources ({dedup_n})", expanded=False):
                    render_sources(sources)

                # update session memory (for follow-ups)
                st.session_state.chat_history_tuples.append((user_msg, answer))

                # persist answer + sources so they remain visible in history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as e:
                err = f"Failed to process your message: {e}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})