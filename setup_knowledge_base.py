import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Document, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from pubmed_fetch import fetch_pubmed_abstracts

# Load environment variables
load_dotenv()

def ingest_data():
    print("Step 1: Fetching data from PubMed...")
    # Fetch abstracts for "Rare Diseases" + "Differential Diagnosis"
    # We can fetch more or specific topics as needed.
    queries = [
        "rare disease differential diagnosis",
        "clinical guidelines rare diseases",
        "uncommon presentation of common diseases"
    ]
    
    all_abstracts = []
    for query in queries:
        print(f"  - Fetching for: {query}")
        abstracts = fetch_pubmed_abstracts(query, max_results=20)
        all_abstracts.extend(abstracts)
    
    print(f"Total abstracts fetched: {len(all_abstracts)}")
    
    # Convert to LlamaIndex Documents
    documents = [Document(text=text) for text in all_abstracts]
    
    print("Step 2: Initializing Embedding Model (HuggingFace)...")
    # Use a high-quality local embedding model
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    print("Step 3: Creating/Loading Vector Store (ChromaDB)...")
    # Create a persistent ChromaDB client
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection("cdss_knowledge_base")
    
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    print("Step 4: Indexing Data...")
    index = VectorStoreIndex.from_documents(
        documents, 
        storage_context=storage_context, 
        embed_model=embed_model
    )
    
    print("Success! Knowledge base created at ./chroma_db")

if __name__ == "__main__":
    ingest_data()
