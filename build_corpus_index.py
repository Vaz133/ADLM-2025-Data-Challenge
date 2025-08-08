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

## Function to build the corpus index
# This function processes all PDF files in the specified directory, extracts text, splits it into chunks, and returns a list of Document objects.
def build_corpus_index(path):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = []

    for file in os.listdir(CORPUS_DIR):
        if file.endswith(".pdf"):
            path = os.path.join(CORPUS_DIR, file)
            print(f"Processing: {file}")
            text = extract_text(path)
            chunks = splitter.split_text(text)
            for chunk in chunks:
                documents.append(Document(page_content=chunk, metadata={"source": file}))
    return documents

## Function to embed documents and save the index
def embed_and_save_index(documents):
    embedder = OpenAIEmbeddings()
    print("Embedding corpus...")
    vectorstore = FAISS.from_documents(documents, OpenAIEmbeddings())
    vectorstore.save_local(INDEX_DIR)
    print("✅ Saved index to corpus_index/")

## Main function to build the corpus index
def main():
    path = r'C:\Users\vazimi\Desktop\Sandbox\ADLM-2025-Data-Challenge'
    documents=build_corpus_index(path)
    embed_and_save_index(documents)

if __name__ == "__main__":
    main()
