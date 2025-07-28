import os
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def load_and_embed_docs(doc_folder="rag_docs", persist_dir="rag_index"):
    os.makedirs(doc_folder, exist_ok=True)
    os.makedirs(persist_dir, exist_ok=True)
    index_file = os.path.join(persist_dir, "index.faiss")

    embeddings = OpenAIEmbeddings()

    # Reuse index if it exists
    if os.path.exists(index_file):
        return FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)

    # Load PDFs
    documents = []
    for fname in os.listdir(doc_folder):
        if fname.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(doc_folder, fname))
            try:
                documents.extend(loader.load())
            except Exception as e:
                print(f"Error loading {fname}: {e}")

    # Exit early if no docs found
    if not documents:
        print(f"[RAG] No valid PDF documents found in '{doc_folder}'. Skipping RAG.")
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    if not docs:
        print(f"[RAG] Documents found but produced no text chunks. Skipping RAG.")
        return None

    db = FAISS.from_documents(docs, embeddings)
    db.save_local(persist_dir)
    return db


def get_rag_context(query, db, k=4):
    docs = db.similarity_search(query, k=k)
    context = ""
    for i, d in enumerate(docs):
        meta = d.metadata.get("source", "unknown")
        context += f"Source [{i+1}] ({meta}):\n{d.page_content.strip()}\n\n"
    return context
