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
from langchain.document_loaders import PyPDFLoader
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