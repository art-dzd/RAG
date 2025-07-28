"""FastAPI приложение для RAG сервиса"""

import os
import shutil
from typing import List, Optional, Dict, Any
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ValidationError
from sqlalchemy.orm import Session
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.config import settings
from app.database.database import get_db, create_tables
from app.database.models import User, Document, Conversation, Message
from app.services.rag_service import rag_service, RAGServiceError
from app.services.file_parser import FileParserError
from app.utils.logging_config import setup_logging, get_logger
from app.utils.helpers import (
    sanitize_filename, 
    generate_unique_id, 
    ensure_directory_exists,
    get_file_size_mb,
    is_valid_file_extension
)

# Настройка логирования
setup_logging(settings.log_level, settings.log_file)
logger = get_logger(__name__)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)

# Создание FastAPI приложения
app = FastAPI(
    title="RAG Telegram Bot API",
    description="API для RAG-сервиса в Telegram боте",
    version="1.0.0",
    debug=settings.debug,
    lifespan=lifespan
)

# Добавить rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Обработчики ошибок
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Обработчик ошибок валидации"""
    logger.warning(f"Ошибка валидации запроса: {exc}")
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Ошибка валидации данных",
            "errors": exc.errors(),
            "body": exc.body
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Общий обработчик исключений"""
    logger.error(f"Необработанная ошибка: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Внутренняя ошибка сервера",
            "error": str(exc) if settings.debug else "Произошла неожиданная ошибка"
        }
    )

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic модели для API
class UserCreate(BaseModel):
    telegram_id: str
    username: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None


class QueryRequest(BaseModel):
    user_id: str
    document_id: str
    query: str
    chat_history: Optional[List[Dict[str, str]]] = None


class QueryResponse(BaseModel):
    success: bool
    query: str
    answer: Optional[str] = None
    error: Optional[str] = None
    found_chunks: int = 0
    response_metadata: Optional[Dict[str, Any]] = None


class DocumentProcessResponse(BaseModel):
    success: bool
    document_id: str
    user_id: str
    chunks_count: int = 0
    processing_time_seconds: float = 0.0
    error: Optional[str] = None


# События жизненного цикла приложения
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения"""
    # Startup
    logger.info("Запуск FastAPI приложения")
    
    # Создать таблицы базы данных
    create_tables()
    logger.info("Таблицы базы данных созданы")
    
    # Убедиться что необходимые директории существуют
    ensure_directory_exists(settings.chroma_db_path)
    ensure_directory_exists("./data/user_files")
    ensure_directory_exists("./logs")
    
    logger.info("FastAPI приложение успешно запущено")
    
    yield
    
    # Shutdown
    logger.info("Завершение работы FastAPI приложения")


# Эндпоинты API

@app.get("/")
async def root():
    """Корневой эндпоинт"""
    return {
        "message": "RAG Telegram Bot API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/health")
async def health_check():
    """Проверка здоровья сервиса"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "services": {}
    }
    
    # Проверить RAG сервис
    try:
        # Простая проверка доступности сервисов
        from app.services.rag_service import rag_service
        health_status["services"]["rag"] = "healthy"
    except Exception as e:
        health_status["services"]["rag"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # Проверить базу данных
    try:
        db = next(get_db())
        db.execute("SELECT 1")
        health_status["services"]["database"] = "healthy"
        db.close()
    except Exception as e:
        health_status["services"]["database"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    return health_status

@app.get("/metrics")
async def get_metrics(db: Session = Depends(get_db)):
    """Получить метрики сервиса"""
    try:
        # Подсчитать общие метрики
        total_users = db.query(User).count()
        total_documents = db.query(Document).count()
        processed_documents = db.query(Document).filter(Document.is_processed == True).count()
        total_conversations = db.query(Conversation).count()
        total_messages = db.query(Message).count()
        
        # Метрики за последние 24 часа
        from datetime import timedelta
        yesterday = datetime.utcnow() - timedelta(days=1)
        
        new_users_24h = db.query(User).filter(User.created_at >= yesterday).count()
        new_documents_24h = db.query(Document).filter(Document.uploaded_at >= yesterday).count()
        new_messages_24h = db.query(Message).filter(Message.created_at >= yesterday).count()
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "total_metrics": {
                "users": total_users,
                "documents": total_documents,
                "processed_documents": processed_documents,
                "conversations": total_conversations,
                "messages": total_messages,
                "processing_success_rate": round(processed_documents / total_documents * 100, 2) if total_documents > 0 else 0
            },
            "last_24h_metrics": {
                "new_users": new_users_24h,
                "new_documents": new_documents_24h,
                "new_messages": new_messages_24h
            }
        }
        
    except Exception as e:
        logger.error(f"Ошибка получения метрик: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка получения метрик: {e}")


@app.post("/users/", response_model=dict)
async def create_or_get_user(user_data: UserCreate, db: Session = Depends(get_db)):
    """Создать или получить пользователя"""
    try:
        # Проверить существование пользователя
        existing_user = db.query(User).filter(User.telegram_id == user_data.telegram_id).first()
        
        if existing_user:
            # Обновить время последней активности
            existing_user.last_activity = datetime.utcnow()
            db.commit()
            logger.info(f"Пользователь {user_data.telegram_id} найден")
            
            return {
                "id": existing_user.id,
                "telegram_id": existing_user.telegram_id,
                "username": existing_user.username,
                "created_at": existing_user.created_at.isoformat(),
                "is_new": False
            }
        
        # Создать нового пользователя
        new_user = User(
            telegram_id=user_data.telegram_id,
            username=user_data.username,
            first_name=user_data.first_name,
            last_name=user_data.last_name
        )
        
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        logger.info(f"Создан новый пользователь {user_data.telegram_id}")
        
        return {
            "id": new_user.id,
            "telegram_id": new_user.telegram_id,
            "username": new_user.username,
            "created_at": new_user.created_at.isoformat(),
            "is_new": True
        }
        
    except Exception as e:
        logger.error(f"Ошибка создания/получения пользователя: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка базы данных: {e}")


@app.post("/upload/", response_model=DocumentProcessResponse)
@limiter.limit("5/minute")
async def upload_and_process_file(
    request: Request,
    user_id: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Загрузить и обработать файл"""
    try:
        logger.info(f"Получен файл {file.filename} от пользователя {user_id}")
        
        # Проверить пользователя
        user = db.query(User).filter(User.telegram_id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="Пользователь не найден")
        
        # Валидация файла
        if not file.filename:
            raise HTTPException(status_code=400, detail="Имя файла не указано")
        
        if not is_valid_file_extension(file.filename, settings.allowed_file_types):
            raise HTTPException(
                status_code=400, 
                detail=f"Неподдерживаемый тип файла. Разрешены: {', '.join(settings.allowed_file_types)}"
            )
        
        # Создать безопасное имя файла
        safe_filename = sanitize_filename(file.filename)
        document_id = generate_unique_id()
        
        # Создать путь для сохранения
        user_dir = f"./data/user_files/{user_id}"
        ensure_directory_exists(user_dir)
        file_path = os.path.join(user_dir, f"{document_id}_{safe_filename}")
        
        # Сохранить файл
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        file_size_mb = get_file_size_mb(file_path)
        
        # Проверить размер файла
        if file_size_mb > settings.max_file_size_mb:
            os.remove(file_path)  # Удалить файл
            raise HTTPException(
                status_code=413, 
                detail=f"Файл слишком большой: {file_size_mb:.2f}MB > {settings.max_file_size_mb}MB"
            )
        
        logger.info(f"Файл сохранён: {file_path} ({file_size_mb:.2f}MB)")
        
        try:
            # Обработать документ через RAG сервис
            result = await rag_service.process_document(
                user_id=user_id,
                file_path=file_path,
                document_id=document_id
            )
            
            # Сохранить информацию о документе в БД
            document_record = Document(
                user_id=user.id,
                filename=safe_filename,
                original_filename=file.filename,
                file_path=file_path,
                file_type=file.filename.split('.')[-1].lower(),
                file_size_mb=file_size_mb,
                file_hash=result['file_info']['file_hash'],
                is_processed=True,
                total_chunks=result['chunks_count'],
                processed_at=datetime.utcnow()
            )
            
            db.add(document_record)
            db.commit()
            
            logger.info(f"Документ {document_id} успешно обработан для пользователя {user_id}")
            
            return DocumentProcessResponse(
                success=True,
                document_id=document_id,
                user_id=user_id,
                chunks_count=result['chunks_count'],
                processing_time_seconds=result['processing_time_seconds']
            )
            
        except (RAGServiceError, FileParserError) as e:
            # Удалить файл при ошибке обработки
            if os.path.exists(file_path):
                os.remove(file_path)
            
            logger.error(f"Ошибка обработки документа: {e}")
            raise HTTPException(status_code=500, detail=f"Ошибка обработки файла: {e}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Неожиданная ошибка при загрузке файла: {e}")
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {e}")


@app.post("/query/", response_model=QueryResponse)
@limiter.limit("30/minute")
async def query_document(request: Request, query_req: QueryRequest, db: Session = Depends(get_db)):
    """Выполнить запрос к документу"""
    try:
        logger.info(f"Запрос к документу {query_req.document_id}: {query_req.query}")
        
        # Проверить пользователя
        user = db.query(User).filter(User.telegram_id == query_req.user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="Пользователь не найден")
        
        # Проверить документ
        document = db.query(Document).filter(
            Document.user_id == user.id,
            Document.file_path.contains(query_req.document_id)
        ).first()
        
        if not document:
            raise HTTPException(status_code=404, detail="Документ не найден")
        
        if not document.is_processed:
            raise HTTPException(status_code=400, detail="Документ ещё не обработан")
        
        # Выполнить запрос через RAG сервис
        result = await rag_service.query_document(
            user_id=query_req.user_id,
            document_id=query_req.document_id,
            query=query_req.query,
            chat_history=query_req.chat_history
        )
        
        # Сохранить сообщение в БД (найти или создать беседу)
        conversation = db.query(Conversation).filter(
            Conversation.user_id == user.id,
            Conversation.document_id == document.id,
            Conversation.is_active == True
        ).first()
        
        if not conversation:
            conversation = Conversation(
                user_id=user.id,
                document_id=document.id,
                session_id=generate_unique_id(),
                is_active=True
            )
            db.add(conversation)
            db.commit()
            db.refresh(conversation)
        
        # Сохранить сообщения пользователя и ассистента
        user_message = Message(
            conversation_id=conversation.id,
            role="user",
            content=query_req.query
        )
        
        assistant_message = Message(
            conversation_id=conversation.id,
            role="assistant",
            content=result.get('answer', ''),
            response_time_ms=result.get('response_metadata', {}).get('total_processing_time_ms'),
            retrieved_chunks=[chunk['metadata'] for chunk in result.get('context_chunks', [])],
            similarity_scores=[chunk['similarity'] for chunk in result.get('context_chunks', [])]
        )
        
        db.add(user_message)
        db.add(assistant_message)
        
        # Обновить время последнего сообщения в беседе
        conversation.last_message_at = datetime.utcnow()
        db.commit()
        
        return QueryResponse(
            success=result['success'],
            query=result['query'],
            answer=result.get('answer'),
            error=result.get('error'),
            found_chunks=result.get('found_chunks', 0),
            response_metadata=result.get('response_metadata')
        )
        
    except HTTPException:
        raise
    except (RAGServiceError) as e:
        logger.error(f"Ошибка выполнения запроса: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка RAG сервиса: {e}")
    except Exception as e:
        logger.error(f"Неожиданная ошибка при выполнении запроса: {e}")
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {e}")


@app.get("/users/{user_id}/documents/")
async def get_user_documents(user_id: str, db: Session = Depends(get_db)):
    """Получить список документов пользователя"""
    try:
        user = db.query(User).filter(User.telegram_id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="Пользователь не найден")
        
        documents = db.query(Document).filter(Document.user_id == user.id).all()
        
        document_list = []
        for doc in documents:
            stats = rag_service.get_document_stats(user_id, doc.file_path.split('/')[-1].split('_')[0])
            
            document_list.append({
                "id": doc.id,
                "filename": doc.original_filename,
                "file_type": doc.file_type,
                "file_size_mb": doc.file_size_mb,
                "uploaded_at": doc.uploaded_at.isoformat(),
                "is_processed": doc.is_processed,
                "chunks_count": doc.total_chunks,
                "is_indexed": stats.get('is_indexed', False)
            })
        
        return {
            "user_id": user_id,
            "documents": document_list,
            "total_documents": len(document_list)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка получения списка документов: {e}")
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {e}")


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str, user_id: str, db: Session = Depends(get_db)):
    """Удалить документ"""
    try:
        user = db.query(User).filter(User.telegram_id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="Пользователь не найден")
        
        document = db.query(Document).filter(
            Document.user_id == user.id,
            Document.file_path.contains(document_id)
        ).first()
        
        if not document:
            raise HTTPException(status_code=404, detail="Документ не найден")
        
        # Удалить из векторной БД
        rag_service.delete_document(user_id, document_id)
        
        # Удалить файл
        if os.path.exists(document.file_path):
            os.remove(document.file_path)
        
        # Удалить запись из БД
        db.delete(document)
        db.commit()
        
        logger.info(f"Документ {document_id} удалён для пользователя {user_id}")
        
        return {"message": "Документ успешно удалён", "document_id": document_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка удаления документа: {e}")
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug
    ) 