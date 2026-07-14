"""
ingest.py
---------
This script is responsible for the "ingestion" phase of the RAG pipeline.

What it does, step by step:
1. Loads all documents (.txt and .pdf) from the documents/ folder.
2. Splits each document into small overlapping chunks (so the retriever
   can find precise, relevant pieces of text instead of whole documents).
3. Converts each chunk into a numerical vector (embedding) using a free
   HuggingFace sentence-transformers model.
4. Stores all the vectors inside a FAISS vector database (fully local,
   no paid service required).
5. Saves the FAISS index to disk (vectorstore/) so chatbot.py can load
   it later without repeating this work.

Run this script once before running the chatbot:
    python ingest.py
"""

import os
import sys

# LangChain document loaders - read raw files into LangChain "Document" objects
from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader

# Splits long documents into smaller overlapping chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Free, local sentence-transformers embedding model wrapper
from langchain_huggingface import HuggingFaceEmbeddings

# FAISS vector store wrapper (local, free, no server required)
from langchain_community.vectorstores import FAISS

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
DOCUMENTS_DIR = "documents"          # folder containing source documents
VECTORSTORE_DIR = "vectorstore"      # folder where the FAISS index will be saved
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 500        # number of characters per chunk
CHUNK_OVERLAP = 50      # overlap between chunks so context isn't lost at the edges


def load_documents(documents_dir: str):
    """
    Loads every .txt and .pdf file inside `documents_dir` and returns
    a list of LangChain Document objects.

    Error handling:
    - If the folder does not exist, or contains no supported files,
      a clear error is raised instead of failing silently.
    """
    if not os.path.isdir(documents_dir):
        raise FileNotFoundError(
            f"The documents folder '{documents_dir}' was not found. "
            f"Please create it and add at least one .txt or .pdf file."
        )

    all_docs = []

    # Load all .txt files
    try:
        txt_loader = DirectoryLoader(
            documents_dir,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=False,
        )
        all_docs.extend(txt_loader.load())
    except Exception as e:
        print(f"[WARNING] Could not load .txt files: {e}")

    # Load all .pdf files
    try:
        pdf_files = [
            f for f in os.listdir(documents_dir) if f.lower().endswith(".pdf")
        ]
        for pdf_file in pdf_files:
            pdf_path = os.path.join(documents_dir, pdf_file)
            pdf_loader = PyPDFLoader(pdf_path)
            all_docs.extend(pdf_loader.load())
    except Exception as e:
        print(f"[WARNING] Could not load .pdf files: {e}")

    if len(all_docs) == 0:
        raise ValueError(
            f"No documents were found in '{documents_dir}'. "
            f"Add at least one .txt or .pdf file and try again."
        )

    print(f"[INFO] Loaded {len(all_docs)} document(s) from '{documents_dir}'.")
    return all_docs


def split_documents(documents):
    """
    Splits documents into smaller overlapping chunks.

    Why we do this:
    Language models and retrievers work better with short, focused pieces
    of text rather than very long documents. Overlap ensures that context
    at chunk boundaries is not lost.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"[INFO] Split documents into {len(chunks)} chunks.")
    return chunks


def build_vectorstore(chunks):
    """
    Converts text chunks into embeddings and stores them in a FAISS index.
    """
    print(f"[INFO] Loading embedding model '{EMBEDDING_MODEL_NAME}' (this may take a moment)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    print("[INFO] Creating FAISS vector store from chunks...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def save_vectorstore(vectorstore, path: str):
    """
    Saves the FAISS index to disk so it can be reloaded later without
    re-computing embeddings every time.
    """
    os.makedirs(path, exist_ok=True)
    vectorstore.save_local(path)
    print(f"[INFO] Vector store saved to '{path}'.")


def main():
    """
    Runs the full ingestion pipeline: load -> split -> embed -> save.
    """
    try:
        documents = load_documents(DOCUMENTS_DIR)
        chunks = split_documents(documents)
        vectorstore = build_vectorstore(chunks)
        save_vectorstore(vectorstore, VECTORSTORE_DIR)
        print("[SUCCESS] Ingestion complete! You can now run chatbot.py or app.py.")
    except Exception as e:
        print(f"[ERROR] Ingestion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
