''' 
Vahid Azimi, MD

08/08/2025

build_corpus_index.py
This script extracts text from PDF files, splits it into chunks, embeds the text using OpenAI embeddings, and saves the resulting vector store to a local directory.
'''

## Import necessary libraries
import os
import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document

CORPUS_DIR = "pdfs"
INDEX_DIR = "corpus_index"

## Function to extract text from PDF
def extract_text(path):
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)