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
from langchain.chains import RetrievalQA

## Config
ENV_PATH = ".env"            # fixed path to env file
CORPUS_INDEX = "corpus_index"  # folder produced by corpus builder
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MIN_CHARS = 20

st.set_page_config(page_title="📚 PDF / DOCX Q&A", layout="centered")
st.title("📚 PDF / DOCX Q&A")

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

# 3) Retrieval + QA (uses whichever vectorstore was selected)
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# 4) Main page: Q&A
question = st.text_input("🔎 Ask a question about the selected data source:")
if question:
    with st.spinner("Thinking..."):
        try:
            result = qa_chain(question)
        except Exception as e:
            st.error(f"Failed to process your question: {e}")
        else:
            st.markdown("### 💡 Answer")
            st.write(result["result"])

            with st.expander("📚 Source snippets"):
                for i, doc in enumerate(result["source_documents"]):
                    src = doc.metadata.get("source", "unknown")
                    pg = doc.metadata.get("page")
                    label = f"**Source:** {src}"
                    if pg:
                        label += f" — page {pg}"
                    st.markdown(label)
                    st.write(doc.page_content)
                    st.markdown("---")
else:
    st.caption("Choose your data source in the sidebar, then ask a question here.")