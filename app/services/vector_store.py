"""Async Chroma vector store service with enhanced security"""

import asyncio
import os
import uuid
from typing import List, Dict, Any, Optional, Tuple
from contextlib import asynccontextmanager

import chromadb
from chromadb.config import Settings
from chromadb.api import AsyncClientAPI, AsyncClient

from app.config import settings
from app.utils.logging_config import get_logger
from app.utils.helpers import ensure_directory_exists, validate_user_id

logger = get_logger(__name__)


class VectorStoreError(Exception):
    """Custom exception for vector store errors"""
    pass


class ChromaVectorStore:
    """Async Chroma vector store service with enhanced security and performance"""
    
    def __init__(self):
        """Initialize the vector store"""
        try:
            # Ensure directory exists
            ensure_directory_exists(settings.chroma_db_path)
            
            # Create async Chroma client
            self.client = chromadb.AsyncClient(
                Settings(
                    chroma_db_impl="duckdb+parquet",
                    persist_directory=settings.chroma_db_path,
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Connection test will be done in startup
            logger.info(f"Chroma vector store initialized at {settings.chroma_db_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise VectorStoreError(f"Vector store initialization failed: {e}")
    
    async def _test_connection(self) -> bool:
        """Test vector store connection"""
        try:
            # Simple test - list collections
            collections = await self.client.list_collections()
            logger.info(f"Vector store connection OK. Collections: {len(collections)}")
            return True
        except Exception as e:
            logger.error(f"Vector store connection test failed: {e}")
            raise VectorStoreError(f"Vector store unavailable: {e}")
    
    def _get_user_collection_name(self, user_id: str, document_id: Optional[str] = None) -> str:
        """
        Generate secure collection name for user
        
        Args:
            user_id: User ID (validated)
            document_id: Document ID (optional)
            
        Returns:
            Collection name
        """
        # Validate user ID format
        if not validate_user_id(user_id):
            raise VectorStoreError("Invalid user ID format")
        
        if document_id:
            # Sanitize document ID
            clean_doc_id = "".join(c for c in document_id if c.isalnum() or c in '-_')[:50]
            return f"user_{user_id}_doc_{clean_doc_id}"
        
        return f"user_{user_id}_general"
    
    async def create_or_get_collection(
        self, 
        user_id: str, 
        document_id: Optional[str] = None,
        reset_if_exists: bool = False
    ) -> Any:
        """
        Create or get collection for user with async operations
        
        Args:
            user_id: User ID
            document_id: Document ID
            reset_if_exists: Reset collection if it exists
            
        Returns:
            Chroma collection
            
        Raises:
            VectorStoreError: On collection errors
        """
        try:
            collection_name = self._get_user_collection_name(user_id, document_id)
            
            # Check if collection exists
            existing_collections = await self.client.list_collections()
            collection_names = [col.name for col in existing_collections]
            
            if collection_name in collection_names:
                if reset_if_exists:
                    logger.info(f"Resetting existing collection: {collection_name}")
                    await self.client.delete_collection(collection_name)
                else:
                    logger.info(f"Getting existing collection: {collection_name}")
                    return await self.client.get_collection(collection_name)
            
            # Create new collection with metadata
            collection = await self.client.create_collection(
                name=collection_name,
                metadata={
                    "user_id": user_id, 
                    "document_id": document_id or "general",
                    "created_at": str(asyncio.get_event_loop().time())
                }
            )
            
            logger.info(f"Created new collection: {collection_name}")
            return collection
            
        except Exception as e:
            logger.error(f"Error creating/getting collection: {e}")
            raise VectorStoreError(f"Collection operation failed: {e}")
    
    async def add_documents(
        self,
        user_id: str,
        document_id: str,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        reset_collection: bool = False
    ) -> List[str]:
        """
        Add documents to vector store asynchronously
        
        Args:
            user_id: User ID
            document_id: Document ID
            texts: List of text chunks
            embeddings: List of embedding vectors
            metadatas: List of metadata for each chunk
            reset_collection: Reset collection before adding
            
        Returns:
            List of document IDs
            
        Raises:
            VectorStoreError: On add operation errors
        """
        if not texts or not embeddings:
            raise VectorStoreError("Texts and embeddings cannot be empty")
        
        if len(texts) != len(embeddings):
            raise VectorStoreError("Number of texts and embeddings must match")
        
        try:
            # Get or create collection
            collection = await self.create_or_get_collection(
                user_id, 
                document_id, 
                reset_if_exists=reset_collection
            )
            
            # Generate secure document IDs
            doc_ids = [f"{document_id}_{i}_{str(uuid.uuid4())[:8]}" for i in range(len(texts))]
            
            # Prepare metadata with security validation
            if metadatas is None:
                metadatas = [{} for _ in range(len(texts))]
            
            # Ensure metadata doesn't contain sensitive information
            clean_metadatas = []
            for i, metadata in enumerate(metadatas):
                clean_meta = {}
                
                # Only allow safe metadata keys
                safe_keys = {
                    'user_id', 'document_id', 'chunk_index', 'total_chunks',
                    'chunk_length', 'start_char', 'filename', 'file_type', 'source'
                }
                
                for key, value in metadata.items():
                    if key in safe_keys and isinstance(value, (str, int, float)):
                        clean_meta[key] = value
                
                # Add required metadata
                clean_meta.update({
                    "user_id": user_id,
                    "document_id": document_id,
                    "chunk_index": i,
                    "text_length": len(texts[i]),
                    "vector_id": doc_ids[i]
                })
                
                clean_metadatas.append(clean_meta)
            
            # Add documents to collection in batches for better performance
            batch_size = 100
            for i in range(0, len(texts), batch_size):
                end_idx = min(i + batch_size, len(texts))
                
                await collection.add(
                    ids=doc_ids[i:end_idx],
                    embeddings=embeddings[i:end_idx],
                    documents=texts[i:end_idx],
                    metadatas=clean_metadatas[i:end_idx]
                )
                
                # Small delay between batches
                if end_idx < len(texts):
                    await asyncio.sleep(0.1)
            
            logger.info(f"Added {len(texts)} documents to collection for user {user_id}")
            return doc_ids
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise VectorStoreError(f"Failed to add documents: {e}")
    
    async def search_similar(
        self,
        user_id: str,
        document_id: str,
        query_embedding: List[float],
        top_k: int = 5,
        min_similarity: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents asynchronously
        
        Args:
            user_id: User ID
            document_id: Document ID
            query_embedding: Query embedding vector
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of similar documents with metadata
            
        Raises:
            VectorStoreError: On search errors
        """
        if not query_embedding:
            raise VectorStoreError("Query embedding cannot be empty")
        
        try:
            collection_name = self._get_user_collection_name(user_id, document_id)
            
            # Check if collection exists
            existing_collections = await self.client.list_collections()
            collection_names = [col.name for col in existing_collections]
            
            if collection_name not in collection_names:
                logger.warning(f"Collection not found: {collection_name}")
                return []
            
            collection = await self.client.get_collection(collection_name)
            
            # Perform search
            results = await collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, 50),  # Limit to 50 max for security
                include=["documents", "metadatas", "distances"]
            )
            
            # Process results
            search_results = []
            
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    distance = results['distances'][0][i]
                    similarity = max(0.0, 1.0 - distance)  # Convert distance to similarity
                    
                    if similarity >= min_similarity:
                        # Validate metadata before returning
                        metadata = results['metadatas'][0][i] or {}
                        
                        # Ensure user owns this document
                        if metadata.get('user_id') != user_id:
                            logger.warning(f"Access denied: user {user_id} tried to access document owned by {metadata.get('user_id')}")
                            continue
                        
                        result_item = {
                            'id': results['ids'][0][i],
                            'document': results['documents'][0][i],
                            'metadata': metadata,
                            'similarity': round(similarity, 4),
                            'distance': round(distance, 4)
                        }
                        search_results.append(result_item)
            
            logger.info(f"Found {len(search_results)} similar documents for user {user_id}")
            return search_results
            
        except Exception as e:
            logger.error(f"Error searching similar documents: {e}")
            raise VectorStoreError(f"Search operation failed: {e}")
    
    async def get_collection_stats(self, user_id: str, document_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get collection statistics asynchronously
        
        Args:
            user_id: User ID
            document_id: Document ID
            
        Returns:
            Collection statistics
        """
        try:
            collection_name = self._get_user_collection_name(user_id, document_id)
            
            existing_collections = await self.client.list_collections()
            collection_names = [col.name for col in existing_collections]
            
            if collection_name not in collection_names:
                return {"exists": False, "count": 0}
            
            collection = await self.client.get_collection(collection_name)
            count = await collection.count()
            
            return {
                "exists": True,
                "name": collection_name,
                "count": count,
                "metadata": collection.metadata
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"exists": False, "count": 0, "error": str(e)}
    
    async def delete_user_collection(self, user_id: str, document_id: Optional[str] = None) -> bool:
        """
        Delete user collection asynchronously
        
        Args:
            user_id: User ID
            document_id: Document ID
            
        Returns:
            True if collection was deleted
        """
        try:
            collection_name = self._get_user_collection_name(user_id, document_id)
            
            existing_collections = await self.client.list_collections()
            collection_names = [col.name for col in existing_collections]
            
            if collection_name in collection_names:
                await self.client.delete_collection(collection_name)
                logger.info(f"Deleted collection: {collection_name}")
                return True
            else:
                logger.warning(f"Collection not found for deletion: {collection_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            return False
    
    async def list_user_collections(self, user_id: str) -> List[str]:
        """
        List all collections for a user asynchronously
        
        Args:
            user_id: User ID
            
        Returns:
            List of collection names
        """
        try:
            all_collections = await self.client.list_collections()
            user_prefix = f"user_{user_id}_"
            
            user_collections = [
                col.name for col in all_collections
                if col.name.startswith(user_prefix)
            ]
            
            return user_collections
            
        except Exception as e:
            logger.error(f"Error listing user collections: {e}")
            return []
    
    async def delete_documents_by_ids(
        self, 
        user_id: str, 
        document_id: str, 
        doc_ids: List[str]
    ) -> bool:
        """
        Delete specific documents by their IDs
        
        Args:
            user_id: User ID
            document_id: Document ID
            doc_ids: List of document IDs to delete
            
        Returns:
            True if deletion was successful
        """
        try:
            collection_name = self._get_user_collection_name(user_id, document_id)
            
            existing_collections = await self.client.list_collections()
            collection_names = [col.name for col in existing_collections]
            
            if collection_name not in collection_names:
                logger.warning(f"Collection not found: {collection_name}")
                return False
            
            collection = await self.client.get_collection(collection_name)
            
            # Verify ownership before deletion
            results = await collection.get(ids=doc_ids, include=["metadatas"])
            
            for metadata in results.get('metadatas', []):
                if metadata and metadata.get('user_id') != user_id:
                    raise VectorStoreError(f"Access denied: user {user_id} cannot delete documents")
            
            # Delete documents
            await collection.delete(ids=doc_ids)
            
            logger.info(f"Deleted {len(doc_ids)} documents for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting documents by IDs: {e}")
            return False
    
    async def update_document_metadata(
        self,
        user_id: str,
        document_id: str,
        doc_id: str,
        new_metadata: Dict[str, Any]
    ) -> bool:
        """
        Update metadata for a specific document
        
        Args:
            user_id: User ID
            document_id: Document ID
            doc_id: Specific document ID
            new_metadata: New metadata to set
            
        Returns:
            True if update was successful
        """
        try:
            collection_name = self._get_user_collection_name(user_id, document_id)
            
            existing_collections = await self.client.list_collections()
            collection_names = [col.name for col in existing_collections]
            
            if collection_name not in collection_names:
                return False
            
            collection = await self.client.get_collection(collection_name)
            
            # Verify ownership
            results = await collection.get(ids=[doc_id], include=["metadatas"])
            if not results.get('metadatas') or not results['metadatas'][0]:
                return False
            
            existing_metadata = results['metadatas'][0]
            if existing_metadata.get('user_id') != user_id:
                raise VectorStoreError(f"Access denied: user {user_id} cannot update document")
            
            # Merge metadata safely
            safe_keys = {
                'chunk_index', 'total_chunks', 'chunk_length', 'start_char',
                'filename', 'file_type', 'source', 'updated_at'
            }
            
            updated_metadata = existing_metadata.copy()
            for key, value in new_metadata.items():
                if key in safe_keys and isinstance(value, (str, int, float)):
                    updated_metadata[key] = value
            
            # Update metadata
            await collection.update(
                ids=[doc_id],
                metadatas=[updated_metadata]
            )
            
            logger.info(f"Updated metadata for document {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating document metadata: {e}")
            return False
    
    async def get_collection_size_stats(self) -> Dict[str, Any]:
        """
        Get overall size statistics for monitoring
        
        Returns:
            Dictionary with size statistics
        """
        try:
            all_collections = await self.client.list_collections()
            
            total_collections = len(all_collections)
            total_documents = 0
            
            # Count documents in each collection
            for collection in all_collections:
                try:
                    col_obj = await self.client.get_collection(collection.name)
                    count = await col_obj.count()
                    total_documents += count
                except Exception as e:
                    logger.warning(f"Error counting documents in {collection.name}: {e}")
            
            return {
                "total_collections": total_collections,
                "total_documents": total_documents,
                "average_docs_per_collection": round(
                    total_documents / total_collections, 2
                ) if total_collections > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting size stats: {e}")
            return {"error": str(e)}
    
    async def cleanup_empty_collections(self) -> int:
        """
        Clean up empty collections for maintenance
        
        Returns:
            Number of collections cleaned up
        """
        try:
            all_collections = await self.client.list_collections()
            cleaned_count = 0
            
            for collection in all_collections:
                try:
                    col_obj = await self.client.get_collection(collection.name)
                    count = await col_obj.count()
                    
                    if count == 0:
                        await self.client.delete_collection(collection.name)
                        cleaned_count += 1
                        logger.info(f"Cleaned up empty collection: {collection.name}")
                        
                except Exception as e:
                    logger.warning(f"Error cleaning collection {collection.name}: {e}")
            
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return 0
    
    async def close(self):
        """Close the vector store client"""
        try:
            # ChromaDB client doesn't have an explicit close method
            # But we can reset if needed
            logger.info("Vector store client closed")
        except Exception as e:
            logger.warning(f"Error closing vector store: {e}")


# Global vector store instance
vector_store = ChromaVectorStore() 