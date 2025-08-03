"""Основной RAG сервис"""

import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from app.config import settings
from app.services.openai_service import openai_service, OpenAIServiceError
from app.services.vector_store import vector_store, VectorStoreError
from app.services.file_parser import file_parser_service, FileParserError
from app.utils.logging_config import get_logger, performance_logger
from app.utils.helpers import generate_unique_id

logger = get_logger(__name__)


class RAGServiceError(Exception):
    """Исключение для ошибок RAG сервиса"""
    pass


class DocumentProcessor:
    """Процессор документов для RAG"""
    
    def __init__(self):
        """Инициализация процессора"""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
        
        logger.info(
            f"DocumentProcessor инициализирован: "
            f"chunk_size={settings.chunk_size}, "
            f"chunk_overlap={settings.chunk_overlap}"
        )
    
    def split_text_into_chunks(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Разбить текст на чанки
        
        Args:
            text: Исходный текст
            metadata: Метаданные для чанков
            
        Returns:
            Список документов LangChain
        """
        if not text.strip():
            logger.warning("Попытка разбить пустой текст")
            return []
        
        # Создать документ LangChain
        doc = Document(page_content=text, metadata=metadata or {})
        
        # Разбить на чанки
        chunks = self.text_splitter.split_documents([doc])
        
        # Добавить индексы чанков
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_length": len(chunk.page_content),
                "start_char": text.find(chunk.page_content),
            })
        
        logger.info(f"Текст разбит на {len(chunks)} чанков")
        return chunks
    
    def prepare_chunks_for_indexing(self, chunks: List[Document]) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Подготовить чанки для индексации
        
        Args:
            chunks: Список документов LangChain
            
        Returns:
            Кортеж (тексты, метаданные)
        """
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        
        return texts, metadatas


class RAGService:
    """Основной RAG сервис"""
    
    def __init__(self):
        """Инициализация RAG сервиса"""
        self.document_processor = DocumentProcessor()
        self.openai_service = openai_service
        self.vector_store = vector_store
        self.file_parser = file_parser_service
        
        logger.info("RAG сервис инициализирован")
    
    async def process_document(
        self,
        user_id: str,
        file_path: str,
        document_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Обработать документ: парсинг, разбиение на чанки, создание эмбеддингов, индексация
        
        Args:
            user_id: ID пользователя
            file_path: Путь к файлу
            document_id: ID документа (если None - генерируется автоматически)
            
        Returns:
            Результат обработки
            
        Raises:
            RAGServiceError: При ошибке обработки
        """
        performance_logger.start_timer("process_document")
        
        if document_id is None:
            document_id = generate_unique_id()
        
        try:
            logger.info(f"Начинаю обработку документа {file_path} для пользователя {user_id}")
            
            # 1. Парсинг файла
            logger.info("Шаг 1: Парсинг файла")
            parsed_data = self.file_parser.parse_file(file_path)
            extracted_text = parsed_data['extracted_text']
            
            # 2. Разбиение на чанки
            logger.info("Шаг 2: Разбиение на чанки")
            chunks = self.document_processor.split_text_into_chunks(
                extracted_text,
                metadata={
                    "user_id": user_id,
                    "document_id": document_id,
                    "filename": parsed_data['filename'],
                    "file_type": parsed_data['file_type'],
                    "source": file_path
                }
            )
            
            if not chunks:
                raise RAGServiceError("Не удалось создать чанки из документа")
            
            # 3. Подготовка данных для индексации
            texts, metadatas = self.document_processor.prepare_chunks_for_indexing(chunks)
            
            # 4. Создание эмбеддингов
            logger.info("Шаг 3: Создание эмбеддингов")
            embeddings = await self.openai_service.create_embeddings(texts)
            
            # 5. Сохранение в векторную БД
            logger.info("Шаг 4: Сохранение в векторную БД")
            
            # Создать коллекцию для документа
            collection_name = f"doc_{document_id}"
            await self.vector_store.create_collection(collection_name, user_id)
            
            # Добавить документы в коллекцию
            success = await self.vector_store.add_documents(
                collection_name=collection_name,
                user_id=user_id,
                texts=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                document_ids=[f"{document_id}_{i}" for i in range(len(texts))]
            )
            
            result = {
                "success": True,
                "document_id": document_id,
                "user_id": user_id,
                "file_info": parsed_data,
                "chunks_count": len(chunks),
                "embeddings_count": len(embeddings),
                "processing_time_seconds": round(performance_logger.end_timer("process_document", success=True, chunks_count=len(chunks), embeddings_count=len(embeddings)), 2),
                "processed_at": datetime.utcnow().isoformat()
            }
            
            logger.info(
                f"Документ успешно обработан. "
                f"Создано {len(chunks)} чанков."
            )
            
            return result
            
        except (FileParserError, VectorStoreError, OpenAIServiceError) as e:
            performance_logger.end_timer("process_document", success=False, error=str(e))
            logger.error(f"Ошибка обработки документа: {e}")
            raise RAGServiceError(f"Ошибка обработки документа: {e}")
        except Exception as e:
            performance_logger.end_timer("process_document", success=False, error=str(e))
            logger.error(f"Неожиданная ошибка при обработке документа: {e}")
            raise RAGServiceError(f"Неожиданная ошибка: {e}")
    
    async def query_document(
        self,
        user_id: str,
        document_id: str,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        top_k: int = None,
        min_similarity: float = 0.3
    ) -> Dict[str, Any]:
        """
        Выполнить запрос к документу с использованием RAG
        
        Args:
            user_id: ID пользователя
            document_id: ID документа
            query: Текст запроса
            chat_history: История чата
            top_k: Количество релевантных чанков
            min_similarity: Минимальная схожесть для включения чанка
            
        Returns:
            Ответ с метаданными
            
        Raises:
            RAGServiceError: При ошибке выполнения запроса
        """
        performance_logger.start_timer("query_document")
        
        if top_k is None:
            top_k = settings.top_k_results
        
        try:
            logger.info(f"Выполняю запрос '{query}' к документу {document_id} пользователя {user_id}")
            
            # 1. Создать эмбеддинг для запроса
            query_embedding = await self.openai_service.create_single_embedding(query)
            
            # 2. Поиск релевантных чанков
            collection_name = f"doc_{document_id}"
            search_results = await self.vector_store.query_documents(
                collection_name=collection_name,
                user_id=user_id,
                query_embedding=query_embedding,
                n_results=top_k
            )
            
            # Проверить результаты поиска
            documents = search_results.get('documents', [[]])[0]
            metadatas = search_results.get('metadatas', [[]])[0]
            distances = search_results.get('distances', [[]])[0]
            
            if not documents:
                logger.warning(f"Не найдено релевантных чанков для запроса: {query}")
                return {
                    "success": False,
                    "error": "Не найдено релевантной информации в документе",
                    "query": query,
                    "found_chunks": 0
                }
            
            # 3. Подготовить контекстные документы
            context_documents = documents
            
            # 4. Подготовить сообщения для GPT
            messages = []
            if chat_history:
                messages.extend(chat_history)
            
            # Добавить текущий запрос
            messages.append({"role": "user", "content": query})
            
            # 5. Генерация ответа
            response = await self.openai_service.generate_chat_response(
                messages=messages,
                context_documents=context_documents,
                max_tokens=1000,
                temperature=0.7
            )
            
            # 6. Подготовка результата
            result = {
                "success": True,
                "query": query,
                "answer": response['content'],
                "context_chunks": [
                    {
                        "text": doc,
                        "similarity": 1.0 - dist,  # Конвертируем расстояние в схожесть
                        "metadata": meta
                    }
                    for doc, dist, meta in zip(documents, distances, metadatas)
                ],
                "found_chunks": len(documents),
                "response_metadata": {
                    "model": response['model'],
                    "usage": response['usage'],
                    "response_time_ms": response['response_time_ms'],
                    "total_processing_time_ms": int(performance_logger.end_timer("query_document", success=True, found_chunks=len(documents)) * 1000)
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(
                f"Запрос обработан, найдено {len(documents)} релевантных чанков"
            )
            
            return result
            
        except (VectorStoreError, OpenAIServiceError) as e:
            performance_logger.end_timer("query_document", success=False, error=str(e))
            logger.error(f"Ошибка выполнения запроса: {e}")
            raise RAGServiceError(f"Ошибка выполнения запроса: {e}")
        except Exception as e:
            performance_logger.end_timer("query_document", success=False, error=str(e))
            logger.error(f"Неожиданная ошибка при выполнении запроса: {e}")
            raise RAGServiceError(f"Неожиданная ошибка: {e}")
    
    async def get_document_stats(self, user_id: str, document_id: str) -> Dict[str, Any]:
        """
        Получить статистику документа
        
        Args:
            user_id: ID пользователя
            document_id: ID документа
            
        Returns:
            Статистика документа
        """
        try:
            collection_name = f"doc_{document_id}"
            stats = await self.vector_store.get_collection_stats(collection_name, user_id)
            return {
                "user_id": user_id,
                "document_id": document_id,
                "vector_store_stats": stats,
                "is_indexed": stats.get("document_count", 0) > 0,
                "chunks_count": stats.get("document_count", 0)
            }
        except Exception as e:
            logger.error(f"Ошибка получения статистики документа: {e}")
            return {
                "user_id": user_id,
                "document_id": document_id,
                "error": str(e),
                "is_indexed": False,
                "chunks_count": 0
            }
    
    async def delete_document(self, user_id: str, document_id: str) -> bool:
        """
        Удалить документ из векторной БД
        
        Args:
            user_id: ID пользователя
            document_id: ID документа
            
        Returns:
            True если документ удалён
        """
        try:
            collection_name = f"doc_{document_id}"
            result = await self.vector_store.delete_collection(collection_name, user_id)
            if result:
                logger.info(f"Документ {document_id} пользователя {user_id} удалён")
            return result
        except Exception as e:
            logger.error(f"Ошибка удаления документа: {e}")
            return False
    
    async def list_user_documents(self, user_id: str) -> List[str]:
        """
        Получить список документов пользователя
        
        Args:
            user_id: ID пользователя
            
        Returns:
            Список ID документов
        """
        try:
            collections = await self.vector_store.list_collections(user_id)
            # Извлечь ID документов из имён коллекций
            document_ids = []
            for collection_name in collections:
                if "_doc_" in collection_name:
                    doc_id = collection_name.split("_doc_")[1]
                    document_ids.append(doc_id)
            return document_ids
        except Exception as e:
            logger.error(f"Ошибка получения списка документов: {e}")
            return []


# Глобальный экземпляр RAG сервиса
rag_service = RAGService() 