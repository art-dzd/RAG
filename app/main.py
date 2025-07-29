"""FastAPI application with enhanced security and modern practices"""

import os
import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, ValidationError, Field
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
    get_file_stats,
    is_valid_file_extension,
    validate_user_id,
    sanitize_text_input,
    validate_and_resolve_path
)

# Setup logging
setup_logging(settings.log_level, settings.log_file, settings.enable_json_logs)
logger = get_logger(__name__)

# Security
security = HTTPBearer(auto_error=False)

# Rate limiting
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["100/minute"] if settings.rate_limit_enabled else []
)


# Pydantic models for API with enhanced validation
class UserCreate(BaseModel):
    """User creation request model"""
    telegram_id: str = Field(..., min_length=1, max_length=20)
    username: Optional[str] = Field(None, max_length=100)
    first_name: Optional[str] = Field(None, max_length=100)
    last_name: Optional[str] = Field(None, max_length=100)
    
    class Config:
        str_strip_whitespace = True


class QueryRequest(BaseModel):
    """Document query request model"""
    user_id: str = Field(..., min_length=1, max_length=20)
    document_id: str = Field(..., min_length=1, max_length=100)
    query: str = Field(..., min_length=1, max_length=1000)
    chat_history: Optional[List[Dict[str, str]]] = Field(default=None, max_items=20)
    
    class Config:
        str_strip_whitespace = True


class QueryResponse(BaseModel):
    """Document query response model"""
    success: bool
    query: str
    answer: Optional[str] = None
    error: Optional[str] = None
    found_chunks: int = 0
    response_metadata: Optional[Dict[str, Any]] = None


class DocumentProcessResponse(BaseModel):
    """Document processing response model"""
    success: bool
    document_id: str
    user_id: str
    chunks_count: int = 0
    processing_time_seconds: float = 0.0
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    timestamp: str
    version: str
    services: Dict[str, str]


# Application lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting FastAPI application")
    
    try:
        # Create database tables
        create_tables()
        logger.info("Database tables created successfully")
        
        # Ensure required directories exist
        directories = [
            settings.chroma_db_path,
            "./data/user_files",
            "./logs"
        ]
        
        for directory in directories:
            ensure_directory_exists(directory)
        
        logger.info("Required directories ensured")
        logger.info("FastAPI application started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down FastAPI application")
    
    # Close OpenAI service
    try:
        from app.services.openai_service import openai_service
        await openai_service.close()
        logger.info("OpenAI service closed")
    except Exception as e:
        logger.warning(f"Error closing OpenAI service: {e}")


# Create FastAPI application
app = FastAPI(
    title="RAG Telegram Bot API",
    description="Secure API for RAG service with Telegram bot integration",
    version="1.0.0",
    debug=settings.debug,
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None
)

# Security middleware
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["localhost", "127.0.0.1", settings.api_host]
)

# CORS middleware with restricted origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

# Rate limiting
if settings.rate_limit_enabled:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# Dependency for user validation
async def validate_user_input(user_id: str) -> str:
    """Validate user ID input"""
    if not validate_user_id(user_id):
        raise HTTPException(
            status_code=400, 
            detail="Invalid user ID format"
        )
    return user_id


# Error handlers
@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """Handle Pydantic validation errors"""
    logger.warning(f"Validation error on {request.url}: {exc}")
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Invalid input data",
            "errors": exc.errors()
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    logger.warning(f"HTTP error {exc.status_code} on {request.url}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    logger.error(f"Unhandled error on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error" if not settings.debug else str(exc),
            "status_code": 500
        }
    )


# API endpoints
@app.get("/", tags=["System"])
async def root():
    """Root endpoint"""
    return {
        "message": "RAG Telegram Bot API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "services": {}
    }
    
    # Check RAG service
    try:
        from app.services.rag_service import rag_service
        health_status["services"]["rag"] = "healthy"
    except Exception as e:
        health_status["services"]["rag"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check database
    try:
        db = next(get_db())
        db.execute("SELECT 1")
        health_status["services"]["database"] = "healthy"
        db.close()
    except Exception as e:
        health_status["services"]["database"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check OpenAI service
    try:
        from app.services.openai_service import openai_service
        if await openai_service.validate_api_key():
            health_status["services"]["openai"] = "healthy"
        else:
            health_status["services"]["openai"] = "api_key_invalid"
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["services"]["openai"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    return health_status


@app.get("/metrics", tags=["System"])
async def get_metrics(db: Session = Depends(get_db)):
    """Get service metrics"""
    try:
        # Calculate metrics
        total_users = db.query(User).count()
        total_documents = db.query(Document).count()
        processed_documents = db.query(Document).filter(Document.is_processed == True).count()
        total_conversations = db.query(Conversation).count()
        total_messages = db.query(Message).count()
        
        # 24h metrics
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
                "processing_success_rate": round(
                    processed_documents / total_documents * 100, 2
                ) if total_documents > 0 else 0
            },
            "last_24h_metrics": {
                "new_users": new_users_24h,
                "new_documents": new_documents_24h,
                "new_messages": new_messages_24h
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving metrics")


@app.post("/users/", tags=["Users"])
async def create_or_get_user(
    user_data: UserCreate, 
    db: Session = Depends(get_db),
    user_id: str = Depends(validate_user_input)
):
    """Create or get user"""
    try:
        # Validate user ID matches request
        if user_data.telegram_id != user_id:
            raise HTTPException(status_code=400, detail="User ID mismatch")
        
        # Check if user exists
        existing_user = db.query(User).filter(
            User.telegram_id == user_data.telegram_id
        ).first()
        
        if existing_user:
            # Update last activity
            existing_user.last_activity = datetime.utcnow()
            db.commit()
            logger.info(f"User {user_data.telegram_id} found")
            
            return {
                "id": existing_user.id,
                "telegram_id": existing_user.telegram_id,
                "username": existing_user.username,
                "created_at": existing_user.created_at.isoformat(),
                "is_new": False
            }
        
        # Create new user with sanitized data
        new_user = User(
            telegram_id=user_data.telegram_id,
            username=sanitize_text_input(user_data.username) if user_data.username else None,
            first_name=sanitize_text_input(user_data.first_name) if user_data.first_name else None,
            last_name=sanitize_text_input(user_data.last_name) if user_data.last_name else None
        )
        
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        logger.info(f"Created new user {user_data.telegram_id}")
        
        return {
            "id": new_user.id,
            "telegram_id": new_user.telegram_id,
            "username": new_user.username,
            "created_at": new_user.created_at.isoformat(),
            "is_new": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating/getting user: {e}")
        raise HTTPException(status_code=500, detail="Database error")


@app.post("/upload/", response_model=DocumentProcessResponse, tags=["Documents"])
@limiter.limit(f"{settings.max_requests_per_minute}/minute" if settings.rate_limit_enabled else "1000/minute")
async def upload_and_process_file(
    request: Request,
    user_id: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload and process document with enhanced security"""
    validated_user_id = await validate_user_input(user_id)
    
    try:
        logger.info(f"Processing file upload: {file.filename} from user {validated_user_id}")
        
        # Check user exists
        user = db.query(User).filter(User.telegram_id == validated_user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # File validation
        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename is required")
        
        # Validate file extension
        if not is_valid_file_extension(file.filename, settings.allowed_file_types):
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Allowed: {', '.join(settings.allowed_file_types)}"
            )
        
        # Check user's file limit
        user_files_count = db.query(Document).filter(Document.user_id == user.id).count()
        if user_files_count >= settings.max_files_per_user:
            raise HTTPException(
                status_code=400,
                detail=f"File limit exceeded. Maximum {settings.max_files_per_user} files per user"
            )
        
        # Create secure filename and path
        safe_filename = sanitize_filename(file.filename)
        document_id = generate_unique_id()
        
        # Secure path construction
        user_dir = f"./data/user_files/{validated_user_id}"
        safe_user_dir = validate_and_resolve_path(user_dir)
        ensure_directory_exists(str(safe_user_dir))
        
        file_path = safe_user_dir / f"{document_id}_{safe_filename}"
        
        # Read and validate file content
        content = await file.read()
        
        # Check file size
        file_size_mb = len(content) / (1024 * 1024)
        if file_size_mb > settings.max_file_size_mb:
            raise HTTPException(
                status_code=413, 
                detail=f"File too large: {file_size_mb:.2f}MB > {settings.max_file_size_mb}MB"
            )
        
        # Basic malware check (simple content validation)
        if b'\x00' in content[:1024]:  # Check for null bytes in header
            raise HTTPException(status_code=400, detail="Invalid file content detected")
        
        # Save file securely
        try:
            with open(file_path, "wb") as f:
                f.write(content)
        except OSError as e:
            logger.error(f"Error saving file: {e}")
            raise HTTPException(status_code=500, detail="Failed to save file")
        
        logger.info(f"File saved: {file_path} ({file_size_mb:.2f}MB)")
        
        # Process document through RAG service
        try:
            result = await rag_service.process_document(
                user_id=validated_user_id,
                file_path=str(file_path),
                document_id=document_id
            )
            
            # Save document record
            document_record = Document(
                user_id=user.id,
                filename=safe_filename,
                original_filename=sanitize_text_input(file.filename),
                file_path=str(file_path),
                file_type=file.filename.split('.')[-1].lower(),
                file_size_mb=file_size_mb,
                file_hash=result['file_info']['file_hash'],
                is_processed=True,
                total_chunks=result['chunks_count'],
                processed_at=datetime.utcnow()
            )
            
            db.add(document_record)
            db.commit()
            
            logger.info(f"Document {document_id} processed successfully for user {validated_user_id}")
            
            return DocumentProcessResponse(
                success=True,
                document_id=document_id,
                user_id=validated_user_id,
                chunks_count=result['chunks_count'],
                processing_time_seconds=result['processing_time_seconds']
            )
            
        except (RAGServiceError, FileParserError) as e:
            # Clean up file on processing error
            if file_path.exists():
                file_path.unlink()
            
            logger.error(f"Document processing error: {e}")
            raise HTTPException(status_code=500, detail=f"Processing failed: {e}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during file upload: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/query/", response_model=QueryResponse, tags=["Documents"])
@limiter.limit(f"{settings.max_requests_per_minute}/minute" if settings.rate_limit_enabled else "1000/minute")
async def query_document(
    request: Request, 
    query_req: QueryRequest, 
    db: Session = Depends(get_db)
):
    """Query document with enhanced validation"""
    validated_user_id = await validate_user_input(query_req.user_id)
    
    try:
        logger.info(f"Document query from user {validated_user_id}: {query_req.query[:100]}...")
        
        # Sanitize query input
        clean_query = sanitize_text_input(query_req.query, max_length=1000)
        if not clean_query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Validate user
        user = db.query(User).filter(User.telegram_id == validated_user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Find document
        document = db.query(Document).filter(
            Document.user_id == user.id,
            Document.file_path.contains(query_req.document_id)
        ).first()
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        if not document.is_processed:
            raise HTTPException(status_code=400, detail="Document not yet processed")
        
        # Sanitize chat history
        clean_history = []
        if query_req.chat_history:
            for msg in query_req.chat_history[-10:]:  # Limit to last 10 messages
                if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                    clean_content = sanitize_text_input(str(msg['content']), max_length=500)
                    if clean_content and msg['role'] in ['user', 'assistant']:
                        clean_history.append({'role': msg['role'], 'content': clean_content})
        
        # Execute query
        result = await rag_service.query_document(
            user_id=validated_user_id,
            document_id=query_req.document_id,
            query=clean_query,
            chat_history=clean_history
        )
        
        # Save conversation
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
        
        # Save messages
        user_message = Message(
            conversation_id=conversation.id,
            role="user",
            content=clean_query
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
        
        # Update conversation timestamp
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
    except RAGServiceError as e:
        logger.error(f"RAG service error: {e}")
        raise HTTPException(status_code=500, detail="Query processing failed")
    except Exception as e:
        logger.error(f"Unexpected error during query: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/users/{user_id}/documents/", tags=["Documents"])
async def get_user_documents(
    user_id: str, 
    db: Session = Depends(get_db)
):
    """Get user's documents list"""
    validated_user_id = await validate_user_input(user_id)
    
    try:
        user = db.query(User).filter(User.telegram_id == validated_user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        documents = db.query(Document).filter(Document.user_id == user.id).all()
        
        document_list = []
        for doc in documents:
            try:
                stats = rag_service.get_document_stats(
                    validated_user_id, 
                    doc.file_path.split('/')[-1].split('_')[0]
                )
                
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
            except Exception as e:
                logger.warning(f"Error getting stats for document {doc.id}: {e}")
                document_list.append({
                    "id": doc.id,
                    "filename": doc.original_filename,
                    "file_type": doc.file_type,
                    "file_size_mb": doc.file_size_mb,
                    "uploaded_at": doc.uploaded_at.isoformat(),
                    "is_processed": doc.is_processed,
                    "chunks_count": doc.total_chunks,
                    "is_indexed": False
                })
        
        return {
            "user_id": validated_user_id,
            "documents": document_list,
            "total_documents": len(document_list)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user documents: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.delete("/documents/{document_id}", tags=["Documents"])
async def delete_document(
    document_id: str, 
    user_id: str, 
    db: Session = Depends(get_db)
):
    """Delete document with security validation"""
    validated_user_id = await validate_user_input(user_id)
    
    try:
        user = db.query(User).filter(User.telegram_id == validated_user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        document = db.query(Document).filter(
            Document.user_id == user.id,
            Document.file_path.contains(document_id)
        ).first()
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Delete from vector store
        rag_service.delete_document(validated_user_id, document_id)
        
        # Delete physical file securely
        try:
            file_path = validate_and_resolve_path(document.file_path)
            if file_path.exists():
                file_path.unlink()
        except Exception as e:
            logger.warning(f"Error deleting file {document.file_path}: {e}")
        
        # Delete from database
        db.delete(document)
        db.commit()
        
        logger.info(f"Document {document_id} deleted for user {validated_user_id}")
        
        return {"message": "Document deleted successfully", "document_id": document_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        access_log=True
    ) 