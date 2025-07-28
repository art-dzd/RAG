"""Сервис для работы с векторной базой данных Chroma"""

import os
import uuid
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings

from app.config import settings
from app.utils.logging_config import get_logger
from app.utils.helpers import ensure_directory_exists

logger = get_logger(__name__)


class VectorStoreError(Exception):
    """Исключение для ошибок векторного хранилища"""
    pass


class ChromaVectorStore:
    """Сервис для работы с Chroma векторной базой данных"""
    
    def __init__(self):
        """Инициализация векторного хранилища"""
        # Убедиться что директория существует
        ensure_directory_exists(settings.chroma_db_path)
        
        # Создать клиент Chroma
        self.client = chromadb.PersistentClient(
            path=settings.chroma_db_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        logger.info(f"Chroma векторное хранилище инициализировано в {settings.chroma_db_path}")
    
    def get_user_collection_name(self, user_id: str, document_id: Optional[str] = None) -> str:
        """
        Получить имя коллекции для пользователя
        
        Args:
            user_id: ID пользователя
            document_id: ID документа (опционально)
            
        Returns:
            Имя коллекции
        """
        if document_id:
            return f"user_{user_id}_doc_{document_id}"
        return f"user_{user_id}_general"
    
    def create_or_get_collection(
        self, 
        user_id: str, 
        document_id: Optional[str] = None,
        reset_if_exists: bool = False
    ) -> Any:
        """
        Создать или получить коллекцию для пользователя
        
        Args:
            user_id: ID пользователя
            document_id: ID документа
            reset_if_exists: Сбросить коллекцию если существует
            
        Returns:
            Коллекция Chroma
            
        Raises:
            VectorStoreError: При ошибке создания коллекции
        """
        try:
            collection_name = self.get_user_collection_name(user_id, document_id)
            
            # Проверить существование коллекции
            existing_collections = [col.name for col in self.client.list_collections()]
            
            if collection_name in existing_collections:
                if reset_if_exists:
                    logger.info(f"Сброс существующей коллекции {collection_name}")
                    self.client.delete_collection(collection_name)
                else:
                    logger.info(f"Получение существующей коллекции {collection_name}")
                    return self.client.get_collection(collection_name)
            
            # Создать новую коллекцию
            collection = self.client.create_collection(
                name=collection_name,
                metadata={"user_id": user_id, "document_id": document_id or "general"}
            )
            
            logger.info(f"Создана новая коллекция {collection_name}")
            return collection
            
        except Exception as e:
            logger.error(f"Ошибка создания/получения коллекции: {e}")
            raise VectorStoreError(f"Не удалось создать коллекцию: {e}")
    
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
        Добавить документы в векторное хранилище
        
        Args:
            user_id: ID пользователя
            document_id: ID документа
            texts: Список текстов
            embeddings: Список эмбеддингов
            metadatas: Список метаданных для каждого текста
            reset_collection: Сбросить коллекцию перед добавлением
            
        Returns:
            Список ID добавленных документов
            
        Raises:
            VectorStoreError: При ошибке добавления документов
        """
        try:
            if len(texts) != len(embeddings):
                raise VectorStoreError("Количество текстов и эмбеддингов должно совпадать")
            
            # Получить коллекцию
            collection = self.create_or_get_collection(
                user_id, 
                document_id, 
                reset_if_exists=reset_collection
            )
            
            # Сгенерировать ID для документов
            doc_ids = [str(uuid.uuid4()) for _ in range(len(texts))]
            
            # Подготовить метаданные
            if metadatas is None:
                metadatas = [{} for _ in range(len(texts))]
            
            # Добавить базовые метаданные
            for i, metadata in enumerate(metadatas):
                metadata.update({
                    "user_id": user_id,
                    "document_id": document_id,
                    "chunk_index": i,
                    "text_length": len(texts[i])
                })
            
            # Добавить документы в коллекцию
            collection.add(
                ids=doc_ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
            
            logger.info(f"Добавлено {len(texts)} документов в коллекцию для пользователя {user_id}")
            return doc_ids
            
        except Exception as e:
            logger.error(f"Ошибка добавления документов: {e}")
            raise VectorStoreError(f"Не удалось добавить документы: {e}")
    
    async def search_similar(
        self,
        user_id: str,
        document_id: str,
        query_embedding: List[float],
        top_k: int = 5,
        min_similarity: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Поиск похожих документов
        
        Args:
            user_id: ID пользователя
            document_id: ID документа
            query_embedding: Эмбеддинг запроса
            top_k: Количество результатов
            min_similarity: Минимальная схожесть
            
        Returns:
            Список найденных документов с метаданными
            
        Raises:
            VectorStoreError: При ошибке поиска
        """
        try:
            collection_name = self.get_user_collection_name(user_id, document_id)
            
            # Проверить существование коллекции
            existing_collections = [col.name for col in self.client.list_collections()]
            if collection_name not in existing_collections:
                logger.warning(f"Коллекция {collection_name} не найдена")
                return []
            
            collection = self.client.get_collection(collection_name)
            
            # Выполнить поиск
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            # Обработать результаты
            search_results = []
            
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    # Вычислить схожесть из расстояния (cosine distance -> similarity)
                    distance = results['distances'][0][i]
                    similarity = 1 - distance  # Преобразование расстояния в схожесть
                    
                    if similarity >= min_similarity:
                        result_item = {
                            'id': results['ids'][0][i],
                            'document': results['documents'][0][i],
                            'metadata': results['metadatas'][0][i],
                            'similarity': similarity,
                            'distance': distance
                        }
                        search_results.append(result_item)
            
            logger.info(f"Найдено {len(search_results)} релевантных документов для пользователя {user_id}")
            return search_results
            
        except Exception as e:
            logger.error(f"Ошибка поиска: {e}")
            raise VectorStoreError(f"Не удалось выполнить поиск: {e}")
    
    def get_collection_stats(self, user_id: str, document_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Получить статистику коллекции
        
        Args:
            user_id: ID пользователя
            document_id: ID документа
            
        Returns:
            Статистика коллекции
        """
        try:
            collection_name = self.get_user_collection_name(user_id, document_id)
            
            existing_collections = [col.name for col in self.client.list_collections()]
            if collection_name not in existing_collections:
                return {"exists": False, "count": 0}
            
            collection = self.client.get_collection(collection_name)
            count = collection.count()
            
            return {
                "exists": True,
                "name": collection_name,
                "count": count,
                "metadata": collection.metadata
            }
            
        except Exception as e:
            logger.error(f"Ошибка получения статистики: {e}")
            return {"exists": False, "count": 0, "error": str(e)}
    
    def delete_user_collection(self, user_id: str, document_id: Optional[str] = None) -> bool:
        """
        Удалить коллекцию пользователя
        
        Args:
            user_id: ID пользователя
            document_id: ID документа
            
        Returns:
            True если коллекция удалена
        """
        try:
            collection_name = self.get_user_collection_name(user_id, document_id)
            
            existing_collections = [col.name for col in self.client.list_collections()]
            if collection_name in existing_collections:
                self.client.delete_collection(collection_name)
                logger.info(f"Коллекция {collection_name} удалена")
                return True
            else:
                logger.warning(f"Коллекция {collection_name} не найдена для удаления")
                return False
                
        except Exception as e:
            logger.error(f"Ошибка удаления коллекции: {e}")
            return False
    
    def list_user_collections(self, user_id: str) -> List[str]:
        """
        Получить список коллекций пользователя
        
        Args:
            user_id: ID пользователя
            
        Returns:
            Список имён коллекций
        """
        try:
            all_collections = [col.name for col in self.client.list_collections()]
            user_prefix = f"user_{user_id}_"
            
            user_collections = [
                col for col in all_collections 
                if col.startswith(user_prefix)
            ]
            
            return user_collections
            
        except Exception as e:
            logger.error(f"Ошибка получения списка коллекций: {e}")
            return []


# Глобальный экземпляр векторного хранилища
vector_store = ChromaVectorStore() 