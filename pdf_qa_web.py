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

# Streamlit page config
st.set_page_config(page_title="📚 PDF Q&A App", layout="centered")
st.title("📚 Ask Questions About Your PDFs")

# --- Automatically load .env from a fixed path ---
default_env_path = "."

# Load .env
if os.path.exists(default_env_path):
    load_dotenv(dotenv_path=default_env_path)
    user_api_key = os.getenv("OPENAI_API_KEY")
    if not user_api_key:
        st.error("⚠️ .env file found, but `OPENAI_API_KEY` is missing.")
        st.stop()
    else:
        st.success("✅ Loaded OpenAI API key from .env")
else:
    st.error(f"❌ .env file not found at: `{default_env_path}`")
    st.stop()
    
## Load Pretrained Corpus Index
st.markdown("### 📚 Step 2: Loading pre-trained PDF corpus...")
CORPUS_INDEX = "corpus_index"

@st.cache_resource
def load_corpus_index(api_key):
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    return FAISS.load_local(CORPUS_INDEX, embeddings, allow_dangerous_deserialization=True)

try:
    corpus_vectorstore = load_corpus_index(user_api_key)
    st.success("✅ Corpus loaded.")
except Exception as e:
    st.error(f"❌ Failed to load prebuilt index: {e}")
    st.stop()
    
## Optional PDF Upload
st.markdown("### 📄 Step 3 (Optional): Upload your own PDF to include in Q&A")
uploaded_pdf = st.file_uploader("Upload a PDF", type="pdf")

extra_vectorstore = None

if uploaded_pdf:
    # ... inside the `if uploaded_pdf:` block
    with st.spinner("🔄 Processing uploaded PDF..."):
        # Save to temp
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_pdf.read())
            tmp_path = tmp_file.name

        # 1) Load with PyMuPDF (handles more PDFs)
        try:
            loader = PyMuPDFLoader(tmp_path)
            pages = loader.load()
        except Exception as e:
            st.error(f"Failed to read PDF: {e}")
            st.stop()

        # 2) Split to chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(pages)

        # 3) Filter out empty/whitespace chunks
        chunks = [d for d in chunks if d.page_content and d.page_content.strip()]

        # Optional: enforce a minimum size to avoid junk fragments
        MIN_CHARS = 20
        chunks = [d for d in chunks if len(d.page_content.strip()) >= MIN_CHARS]

        # 4) Guard: if still empty, explain likely cause (scanned PDF)
        if not chunks:
            st.error(
                "I couldn’t extract any text from that PDF. "
                "It may be scanned/image-only or encrypted. "
                "Try another file or run OCR to extract text."
            )
            st.stop()

        # 5) Build the vectorstore safely
        embedder = OpenAIEmbeddings(openai_api_key=user_api_key)
        try:
            extra_vectorstore = FAISS.from_documents(chunks, embedder)
        except Exception as e:
            st.error(f"Failed to embed uploaded PDF: {e}")
            st.stop()
        
## Merge corpus with user upload (if any)
if extra_vectorstore:
    try:
        corpus_vectorstore.merge_from(extra_vectorstore)
    except Exception as e:
        st.error(f"Failed to merge uploaded PDF into corpus index: {e}")
        st.stop()

retriever = corpus_vectorstore.as_retriever()
    
## Set up LLM and QA Chain
llm = ChatOpenAI(model="gpt-5", temperature=1, openai_api_key=user_api_key)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

##Q&A Interface
st.markdown("### 🧠 Step 4: Ask a Question")
question = st.text_input("Type your question here:")

if question:
    with st.spinner("💬 Thinking..."):
        try:
            result = qa_chain(question)
            st.markdown("### 💡 Answer")
            st.write(result["result"])

            with st.expander("📚 Source Chunks"):
                for i, doc in enumerate(result["source_documents"]):
                    st.markdown(f"**Source:** {doc.metadata.get('source', 'unknown')}")
                    st.write(doc.page_content)
                    st.markdown("---")
        except Exception as e:
            st.error(f"❌ Failed to process your question: {e}")