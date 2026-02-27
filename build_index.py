from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import json
import os

# Define constants
EMBEDDING_MODEL = "sentence-transformers/all-distilroberta-v1"
INDEX_FOLDER = "faiss_index"
JSONL_FILE = "merged.jsonl"

# Load JSONL documents
def load_jsonl_docs(jsonl_path):
    docs = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            doc = Document(
                page_content=data["content"],
                metadata={"filename": data["filename"]}
            )
            docs.append(doc)
    return docs

# Import Document class for creating documents
from langchain_core.documents import Document

# Load and split documents
documents = load_jsonl_docs(JSONL_FILE)

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(documents)

# Embed and save FAISS index
embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
db = FAISS.from_documents(split_docs, embedding)
db.save_local(INDEX_FOLDER)

print(f"✅ FAISS index saved to {INDEX_FOLDER}/")