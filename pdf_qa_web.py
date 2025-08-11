''' 
Vahid Azimi, MD

08/08/2025

pdf_qa_web.py

'''

## Import necessary libraries

import os
import tempfile
import streamlit as st
from dotenv import dotenv_values
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from dotenv import load_dotenv, find_dotenv

## Config
ENV_PATH = ".env"            # fixed path to env file
CORPUS_INDEX = "corpus_index"  # folder produced by corpus builder
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MIN_CHARS = 20

# Streamlit page config
st.set_page_config(page_title="📚 PDF Q&A App", layout="centered")
st.title("📚 Ask Questions About Your PDFs")

# Load .env
if os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH)
user_api_key = os.getenv("OPENAI_API_KEY")
if not user_api_key:
    st.error(f"OPENAI_API_KEY not found. Ensure `{ENV_PATH}` exists with your key.")
    st.stop()
else:
    st.success("✅ OpenAI API key loaded from .env")
    
# Shared embeddings + LLM
embeddings = OpenAIEmbeddings(openai_api_key=user_api_key)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=user_api_key)
    
# Sidebar: all data-source UI
st.sidebar.header("Data source")
data_source = st.sidebar.radio(
    "Use:",
    ["Base corpus (prebuilt FAISS)", "My uploaded PDFs (session-only)"],
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

# Build vectorstore depending on choice
vectorstore = None

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

else:
    st.sidebar.subheader("Upload one or more PDFs")
    uploaded_pdfs = st.sidebar.file_uploader(
        "Select PDF(s)", type="pdf", accept_multiple_files=True
    )

    if not uploaded_pdfs:
        st.sidebar.info("Upload at least one PDF to proceed.")
        st.stop()

    docs = []
    with st.spinner("Processing uploaded PDFs..."):
        for f in uploaded_pdfs:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(f.read())
                tmp_path = tmp.name
            try:
                loader = PyMuPDFLoader(tmp_path)
                pages = loader.load()
                # tag each page with source filename
                for p in pages:
                    p.metadata = {**p.metadata, "source": f.name}
                docs.extend(pages)
            except Exception as e:
                st.sidebar.error(f"Failed to read {f.name}: {e}")
                st.stop()

        chunks = split_and_clean(docs)
        if not chunks:
            st.sidebar.error("No extractable text found in the uploaded PDFs.")
            st.stop()

        try:
            vectorstore = FAISS.from_documents(chunks, embeddings)
            st.sidebar.success("Built a temporary FAISS index from uploads.")
        except Exception as e:
            st.sidebar.error(f"Failed to embed uploaded PDFs: {e}")
            st.stop()

# 4) Retrieval + QA
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# 5) Main page: just the Q&A
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
                    st.markdown(f"**Source:** {src}")
                    st.write(doc.page_content)
                    st.markdown("---")
else:
    st.caption("Choose your data source in the sidebar, then ask a question here.")