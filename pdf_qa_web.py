''' 
Vahid Azimi, MD

08/08/2025

index_pdfs.py
This script extracts text from PDF files, splits it into chunks, embeds the text using OpenAI embeddings, and saves the resulting vector store to a local directory.
'''

## Import necessary libraries

import os
import streamlit as st
import tempfile
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document