import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from src.config import DATA_PATH, VECTOR_DB_PATH, EMBEDDING_MODEL

def load_documents():
    """
    Loads all PDF files from the DATA_PATH directory.
    """
    documents = []
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"❌ Data folder not found. Created '{DATA_PATH}'. Please put a PDF there!")
        return []

    # Loop through all files in the data folder
    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(DATA_PATH, file)
            print(f"📄 Loading: {file}...")
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
    
    return documents

def ingest_data():
    # 1. Load Data
    docs = load_documents()
    if not docs:
        print("⚠️ No documents found. Skipping ingestion.")
        return

    # 2. Split Data (Chunking)
    # Why 1000? It's roughly 2-3 paragraphs. Good context size.
    # Why 200 overlap? Ensures we don't cut a sentence in half at the edge of a chunk.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    print(f"🧩 Split {len(docs)} pages into {len(chunks)} small chunks.")

    # 3. Create Embeddings & Store in Vector DB
    print("🔮 Generating embeddings and storing in ChromaDB... (This may take a moment)")
    
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    
    # This creates the DB if it doesn't exist, or adds to it if it does
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_PATH
    )
    
    print(f"✅ Success! Vector DB created at '{VECTOR_DB_PATH}'")

if __name__ == "__main__":
    ingest_data()