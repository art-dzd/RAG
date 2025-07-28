"""Конфигурация приложения"""

import os
from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Основные настройки приложения"""
    
    # OpenAI Configuration
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    
    # Telegram Bot Configuration
    telegram_bot_token: str = Field(..., env="TELEGRAM_BOT_TOKEN")
    
    # Database Configuration
    database_url: str = Field("sqlite:///./data/app.db", env="DATABASE_URL")
    
    # Vector Store Configuration
    chroma_db_path: str = Field("./data/chroma_db", env="CHROMA_DB_PATH")
    
    # File Upload Configuration
    max_file_size_mb: int = Field(50, env="MAX_FILE_SIZE_MB")
    allowed_file_types: List[str] = Field(["pdf", "docx", "txt"], env="ALLOWED_FILE_TYPES")
    
    # RAG Configuration
    chunk_size: int = Field(1500, env="CHUNK_SIZE")
    chunk_overlap: int = Field(200, env="CHUNK_OVERLAP")
    top_k_results: int = Field(5, env="TOP_K_RESULTS")
    
    # Logging Configuration
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_file: str = Field("./logs/app.log", env="LOG_FILE")
    
    # FastAPI Configuration
    api_host: str = Field("localhost", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    debug: bool = Field(False, env="DEBUG")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Глобальная конфигурация
settings = Settings() 