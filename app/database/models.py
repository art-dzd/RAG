"""Модели базы данных"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, 
    Boolean, ForeignKey, Float, JSON, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class User(Base):
    """Модель пользователя"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    telegram_id = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, nullable=True)
    first_name = Column(String, nullable=True)
    last_name = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_activity = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Связи
    documents = relationship("Document", back_populates="user")
    conversations = relationship("Conversation", back_populates="user")


class Document(Base):
    """Модель документа"""
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    filename = Column(String, nullable=False)
    original_filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_type = Column(String, nullable=False)  # pdf, docx, txt
    file_size_mb = Column(Float, nullable=False)
    file_hash = Column(String, nullable=False)
    
    # Метаданные обработки
    is_processed = Column(Boolean, default=False)
    processing_error = Column(Text, nullable=True)
    total_chunks = Column(Integer, default=0)
    
    # Временные метки
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)
    
    # Связи
    user = relationship("User", back_populates="documents")
    conversations = relationship("Conversation", back_populates="document")
    
    # Уникальный индекс для предотвращения дубликатов файлов у одного пользователя
    __table_args__ = (
        Index('ix_documents_user_hash_unique', 'user_id', 'file_hash', unique=True),
    )


class Conversation(Base):
    """Модель диалога/беседы"""
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=True)
    
    # Контекст диалога
    session_id = Column(String, nullable=False, index=True)
    is_active = Column(Boolean, default=True)
    
    # Временные метки
    started_at = Column(DateTime, default=datetime.utcnow)
    last_message_at = Column(DateTime, default=datetime.utcnow)
    
    # Связи
    user = relationship("User", back_populates="conversations")
    document = relationship("Document", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation")


class Message(Base):
    """Модель сообщения в диалоге"""
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False)
    
    # Содержимое сообщения
    role = Column(String, nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)
    
    # Метаданные
    token_count = Column(Integer, nullable=True)
    response_time_ms = Column(Integer, nullable=True)
    
    # RAG контекст
    retrieved_chunks = Column(JSON, nullable=True)  # ID чанков использованных для ответа
    similarity_scores = Column(JSON, nullable=True)  # Скоры похожести чанков
    
    # Временные метки
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Связи
    conversation = relationship("Conversation", back_populates="messages")


class DocumentChunk(Base):
    """Модель чанка документа (для отслеживания индексированного контента)"""
    __tablename__ = "document_chunks"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    
    # Содержимое чанка
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    start_char = Column(Integer, nullable=False)
    end_char = Column(Integer, nullable=False)
    
    # Метаданные
    chunk_hash = Column(String, nullable=False)
    word_count = Column(Integer, nullable=False)
    
    # Векторная БД метаданные
    vector_id = Column(String, nullable=True)  # ID в Chroma
    
    # Временные метки
    created_at = Column(DateTime, default=datetime.utcnow)


class UserSession(Base):
    """Модель пользовательской сессии"""
    __tablename__ = "user_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    session_id = Column(String, unique=True, nullable=False)
    
    # Состояние сессии
    current_document_id = Column(Integer, ForeignKey("documents.id"), nullable=True)
    is_active = Column(Boolean, default=True)
    
    # Контекст
    context_data = Column(JSON, nullable=True)
    
    # Временные метки
    created_at = Column(DateTime, default=datetime.utcnow)
    last_activity = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True) 