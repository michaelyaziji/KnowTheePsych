import chromadb
from chromadb.config import Settings
import os
from typing import List

class VectorStore:
    def __init__(self):
        # Initialize ChromaDB with local persistence
        self.client = chromadb.Client(Settings(
            persist_directory="chroma_db",
            anonymized_telemetry=False
        ))
        
        # Create or get the collection
        self.collection = self.client.get_or_create_collection(
            name="psychology_documents",
            metadata={"hnsw:space": "cosine"}
        )
    
    def store_documents(self, documents: List[str]):
        """Store documents in the vector database."""
        # Get all current IDs
        existing = self.collection.get()
        if existing and 'ids' in existing and existing['ids']:
            self.collection.delete(ids=existing['ids'])
        # Add new documents
        ids = [str(i) for i in range(len(documents))]
        self.collection.add(
            documents=documents,
            ids=ids
        )
    
    def get_relevant_chunks(self, query: str = None, n_results: int = 5) -> List[str]:
        """Retrieve relevant document chunks based on a query."""
        if query is None:
            # If no query provided, return all documents
            results = self.collection.get()
            return results['documents']
        
        # Search for relevant chunks
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results['documents'][0]
    
    def clear(self):
        """Clear all documents from the vector store."""
        self.collection.delete(where={"id": {"$ne": None}}) 