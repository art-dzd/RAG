"""Chroma vector store service with enhanced security"""

import asyncio
import os
import uuid
from typing import List, Dict, Any, Optional, Tuple

import chromadb
from chromadb.config import Settings

from app.config import settings
from app.utils.logging_config import get_logger
from app.utils.helpers import ensure_directory_exists, validate_user_id

logger = get_logger(__name__)


class VectorStoreError(Exception):
    """Custom exception for vector store errors"""
    pass


class ChromaVectorStore:
    """Chroma vector store service with enhanced security and performance"""
    
    def __init__(self):
        """Initialize the vector store"""
        try:
            ensure_directory_exists(settings.chroma_db_path)
            
            # Configure Chroma with security settings
            chroma_settings = Settings(
                persist_directory=settings.chroma_db_path,
                allow_reset=False,
                is_persistent=True,
                anonymized_telemetry=False
            )
            
            # Initialize client
            self._client = chromadb.PersistentClient(
                path=settings.chroma_db_path,
                settings=chroma_settings
            )
            self._collection = None
            
            logger.info("ChromaDB vector store initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise VectorStoreError(f"Vector store initialization failed: {e}")
    
    def get_client(self):
        """Get ChromaDB client"""
        return self._client
    
    async def _test_connection(self) -> bool:
        """Test vector store connection"""
        try:
            # Simple test - list collections
            loop = asyncio.get_event_loop()
            collections = await loop.run_in_executor(None, self._client.list_collections)
            logger.info(f"Vector store connection OK. Collections: {len(collections)}")
            return True
        except Exception as e:
            logger.error(f"Vector store connection test failed: {e}")
            raise VectorStoreError(f"Vector store unavailable: {e}")
    
    async def create_collection(self, collection_name: str, user_id: str) -> bool:
        """Create a new collection for user documents"""
        try:
            validate_user_id(user_id)
            safe_name = f"user_{user_id}_{collection_name}"
            
            loop = asyncio.get_event_loop()
            collection = await loop.run_in_executor(
                None, 
                lambda: self._client.get_or_create_collection(
                    name=safe_name,
                    metadata={"user_id": user_id, "created_at": str(asyncio.get_event_loop().time())}
                )
            )
            
            logger.info(f"Collection {safe_name} created/retrieved for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create collection {collection_name}: {e}")
            raise VectorStoreError(f"Collection creation failed: {e}")
    
    async def add_documents(
        self, 
        collection_name: str, 
        user_id: str,
        texts: List[str], 
        embeddings: List[List[float]], 
        metadatas: Optional[List[Dict[str, Any]]] = None,
        document_ids: Optional[List[str]] = None
    ) -> bool:
        """Add documents to collection with embeddings"""
        try:
            validate_user_id(user_id)
            safe_name = f"user_{user_id}_{collection_name}"
            
            if not texts or not embeddings:
                raise VectorStoreError("Empty texts or embeddings provided")
            
            if len(texts) != len(embeddings):
                raise VectorStoreError("Texts and embeddings length mismatch")
            
            # Generate IDs if not provided
            if document_ids is None:
                document_ids = [str(uuid.uuid4()) for _ in texts]
            
            # Prepare metadata
            if metadatas is None:
                metadatas = [{"user_id": user_id} for _ in texts]
            else:
                for meta in metadatas:
                    meta["user_id"] = user_id
            
            loop = asyncio.get_event_loop()
            collection = await loop.run_in_executor(
                None, 
                lambda: self._client.get_collection(safe_name)
            )
            
            await loop.run_in_executor(
                None,
                lambda: collection.add(
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas,
                    ids=document_ids
                )
            )
            
            logger.info(f"Added {len(texts)} documents to collection {safe_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents to {collection_name}: {e}")
            raise VectorStoreError(f"Document addition failed: {e}")
    
    async def query_documents(
        self, 
        collection_name: str, 
        user_id: str,
        query_embedding: List[float], 
        n_results: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Query documents from collection"""
        try:
            validate_user_id(user_id)
            safe_name = f"user_{user_id}_{collection_name}"
            
            # Add user filter to metadata
            if filter_metadata is None:
                filter_metadata = {"user_id": user_id}
            else:
                filter_metadata["user_id"] = user_id
            
            loop = asyncio.get_event_loop()
            collection = await loop.run_in_executor(
                None, 
                lambda: self._client.get_collection(safe_name)
            )
            
            results = await loop.run_in_executor(
                None,
                lambda: collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(n_results, settings.max_results),
                    where=filter_metadata,
                    include=["documents", "metadatas", "distances"]
                )
            )
            
            logger.info(f"Query returned {len(results.get('documents', [[]])[0])} results")
            return results
            
        except Exception as e:
            logger.error(f"Failed to query collection {collection_name}: {e}")
            raise VectorStoreError(f"Query failed: {e}")
    
    async def delete_collection(self, collection_name: str, user_id: str) -> bool:
        """Delete user collection"""
        try:
            validate_user_id(user_id)
            safe_name = f"user_{user_id}_{collection_name}"
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self._client.delete_collection(safe_name)
            )
            
            logger.info(f"Deleted collection {safe_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete collection {collection_name}: {e}")
            raise VectorStoreError(f"Collection deletion failed: {e}")
    
    async def list_collections(self, user_id: str) -> List[str]:
        """List all collections for user"""
        try:
            validate_user_id(user_id)
            
            loop = asyncio.get_event_loop()
            all_collections = await loop.run_in_executor(None, self._client.list_collections)
            
            prefix = f"user_{user_id}_"
            user_collections = [
                col.name[len(prefix):] 
                for col in all_collections 
                if col.name.startswith(prefix)
            ]
            
            logger.info(f"Found {len(user_collections)} collections for user {user_id}")
            return user_collections
            
        except Exception as e:
            logger.error(f"Failed to list collections for user {user_id}: {e}")
            raise VectorStoreError(f"Collection listing failed: {e}")
    
    async def get_collection_stats(self, collection_name: str, user_id: str) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            validate_user_id(user_id)
            safe_name = f"user_{user_id}_{collection_name}"
            
            loop = asyncio.get_event_loop()
            collection = await loop.run_in_executor(
                None, 
                lambda: self._client.get_collection(safe_name)
            )
            
            count = await loop.run_in_executor(None, collection.count)
            
            return {
                "name": collection_name,
                "document_count": count,
                "user_id": user_id
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats for collection {collection_name}: {e}")
            raise VectorStoreError(f"Stats retrieval failed: {e}")


# Global vector store instance
vector_store = ChromaVectorStore()