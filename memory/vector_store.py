"""
Vector store wrapper for ChromaDB.

This module provides a clean interface for storing and retrieving research
documents using ChromaDB for RAG-based blog generation.
"""

import os
import re
from typing import List, Dict, Optional, Any
import chromadb
from chromadb.config import Settings
import logging

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Wrapper for ChromaDB vector store operations.
    Handles embedding storage and similarity search for RAG.
    """
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initialize ChromaDB client with persistence.
        
        Args:
            persist_directory: Directory to persist the vector database
        """
        self.persist_directory = persist_directory
        
        # Ensure directory exists
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        self.collection = None
        self.collection_name = None
        
        logger.info(f"VectorStore initialized with persist_directory: {persist_directory}")
    
    def create_collection(self, topic: str) -> str:
        """
        Create a new collection for a blog topic.
        
        Args:
            topic: The blog topic
            
        Returns:
            collection_name: Slug version of topic
        """
        # Generate collection name from topic
        collection_name = self._slugify(topic)
        self.collection_name = collection_name
        
        # Delete existing collection if it exists
        try:
            self.client.delete_collection(name=collection_name)
            logger.info(f"Deleted existing collection: {collection_name}")
        except Exception:
            pass  # Collection doesn't exist, which is fine
        
        # Create new collection
        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"topic": topic}
        )
        
        logger.info(f"Created collection: {collection_name}")
        return collection_name
    
    def get_collection(self, collection_name: str):
        """
        Get an existing collection.
        
        Args:
            collection_name: Name of the collection
        """
        self.collection = self.client.get_collection(name=collection_name)
        self.collection_name = collection_name
        logger.info(f"Retrieved collection: {collection_name}")
    
    def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: List[str]
    ):
        """
        Add research documents to collection with embeddings.
        ChromaDB will automatically generate embeddings.
        
        Args:
            documents: List of document texts
            metadatas: List of metadata dicts for each document
            ids: List of unique IDs for each document
        """
        if self.collection is None:
            raise ValueError("No collection created. Call create_collection() first.")
        
        if not (len(documents) == len(metadatas) == len(ids)):
            raise ValueError("documents, metadatas, and ids must have the same length")
        
        self.collection.add(
            documents=documents,
            metadatas=metadatas,  # type: ignore[arg-type]
            ids=ids
        )
        
        logger.info(f"Added {len(documents)} documents to collection: {self.collection_name}")
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 3
    ) -> List[str]:
        """
        Perform similarity search for relevant documents.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of k most relevant document texts
        """
        if self.collection is None:
            raise ValueError("No collection created. Call create_collection() first.")
        
        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )
        
        # Extract documents from results
        documents = results['documents'][0] if results['documents'] else []
        
        logger.info(f"Similarity search for '{query[:50]}...' returned {len(documents)} results")
        return documents
    
    def similarity_search_with_metadata(
        self, 
        query: str, 
        k: int = 3
    ) -> List[Dict]:
        """
        Perform similarity search and return documents with metadata.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of dicts with 'document' and 'metadata' keys
        """
        if self.collection is None:
            raise ValueError("No collection created. Call create_collection() first.")
        
        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )
        
        # Combine documents with metadata
        combined_results = []
        if results['documents']:
            documents = results['documents'][0]
            metadatas = results['metadatas'][0] if results['metadatas'] else [{}] * len(documents)
            
            for doc, meta in zip(documents, metadatas):
                combined_results.append({
                    'document': doc,
                    'metadata': meta
                })
        
        return combined_results
    
    def get_collection_count(self) -> int:
        """
        Get the number of documents in the current collection.
        
        Returns:
            Number of documents
        """
        if self.collection is None:
            return 0
        
        return self.collection.count()
    
    def reset(self):
        """Delete all collections and reset the database."""
        self.client.reset()
        self.collection = None
        self.collection_name = None
        logger.info("VectorStore reset - all collections deleted")
    
    @staticmethod
    def _slugify(text: str) -> str:
        """
        Convert text to URL-friendly slug.
        
        Args:
            text: Text to slugify
            
        Returns:
            Slugified text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters
        text = re.sub(r'[^\w\s-]', '', text)
        
        # Replace spaces and multiple hyphens with single hyphen
        text = re.sub(r'[-\s]+', '_', text)
        
        # Limit length
        text = text[:50]
        
        # Remove leading/trailing hyphens
        text = text.strip('_')
        
        return text if text else "blog_collection"