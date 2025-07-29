"""Configuration settings with enhanced security and modern practices"""

import os
from typing import List, Optional
from pydantic import BaseSettings, Field, validator, SecretStr
import secrets


class Settings(BaseSettings):
    """Application settings with enhanced security and validation"""
    
    # OpenAI Configuration
    openai_api_key: SecretStr = Field(..., env="OPENAI_API_KEY")
    openai_model: str = Field("gpt-4o-mini", env="OPENAI_MODEL")
    openai_embedding_model: str = Field("text-embedding-3-small", env="OPENAI_EMBEDDING_MODEL")
    openai_max_tokens: int = Field(4096, env="OPENAI_MAX_TOKENS")
    openai_temperature: float = Field(0.7, env="OPENAI_TEMPERATURE")
    
    # Telegram Bot Configuration
    telegram_bot_token: SecretStr = Field(..., env="TELEGRAM_BOT_TOKEN")
    telegram_webhook_secret: Optional[str] = Field(None, env="TELEGRAM_WEBHOOK_SECRET")
    
    # Database Configuration
    database_url: str = Field("sqlite:///./data/app.db", env="DATABASE_URL")
    
    # Vector Store Configuration
    chroma_db_path: str = Field("./data/chroma_db", env="CHROMA_DB_PATH")
    
    # File Upload Configuration with enhanced security
    max_file_size_mb: int = Field(50, env="MAX_FILE_SIZE_MB")
    allowed_file_types: List[str] = Field(["pdf", "docx", "txt"], env="ALLOWED_FILE_TYPES")
    max_files_per_user: int = Field(100, env="MAX_FILES_PER_USER")
    file_scan_enabled: bool = Field(True, env="FILE_SCAN_ENABLED")
    
    # RAG Configuration
    chunk_size: int = Field(1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(200, env="CHUNK_OVERLAP")
    top_k_results: int = Field(5, env="TOP_K_RESULTS")
    min_similarity_threshold: float = Field(0.3, env="MIN_SIMILARITY_THRESHOLD")
    
    # Security Configuration
    secret_key: str = Field(default_factory=lambda: secrets.token_urlsafe(32), env="SECRET_KEY")
    cors_origins: List[str] = Field(["http://localhost:3000"], env="CORS_ORIGINS")
    rate_limit_enabled: bool = Field(True, env="RATE_LIMIT_ENABLED")
    max_requests_per_minute: int = Field(30, env="MAX_REQUESTS_PER_MINUTE")
    
    # Logging Configuration
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_file: str = Field("./logs/app.log", env="LOG_FILE")
    enable_json_logs: bool = Field(False, env="ENABLE_JSON_LOGS")
    
    # FastAPI Configuration
    api_host: str = Field("127.0.0.1", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    debug: bool = Field(False, env="DEBUG")
    
    # Performance Configuration
    max_concurrent_requests: int = Field(10, env="MAX_CONCURRENT_REQUESTS")
    request_timeout: int = Field(300, env="REQUEST_TIMEOUT")
    
    @validator('openai_api_key')
    def validate_openai_key(cls, v: SecretStr):
        """Validate OpenAI API key"""
        if not v or v.get_secret_value() == "your_openai_api_key_here":
            raise ValueError("OPENAI_API_KEY must be set")
        if not v.get_secret_value().startswith("sk-"):
            raise ValueError("OPENAI_API_KEY must start with 'sk-'")
        return v
    
    @validator('telegram_bot_token')
    def validate_telegram_token(cls, v: SecretStr):
        """Validate Telegram bot token"""
        if not v or v.get_secret_value() == "your_telegram_bot_token_here":
            raise ValueError("TELEGRAM_BOT_TOKEN must be set")
        token = v.get_secret_value()
        if len(token.split(':')) != 2:
            raise ValueError("Invalid TELEGRAM_BOT_TOKEN format")
        return v
    
    @validator('api_port')
    def validate_port(cls, v: int):
        """Validate API port"""
        if not 1 <= v <= 65535:
            raise ValueError("API_PORT must be between 1 and 65535")
        return v
    
    @validator('max_file_size_mb')
    def validate_file_size(cls, v: int):
        """Validate file size limit"""
        if v <= 0 or v > 100:
            raise ValueError("MAX_FILE_SIZE_MB must be between 1 and 100")
        return v
    
    @validator('chunk_size')
    def validate_chunk_size(cls, v: int):
        """Validate chunk size"""
        if v < 100 or v > 2000:
            raise ValueError("CHUNK_SIZE must be between 100 and 2000")
        return v
    
    @validator('chunk_overlap')
    def validate_chunk_overlap(cls, v: int):
        """Validate chunk overlap"""
        if v < 0 or v > 500:
            raise ValueError("CHUNK_OVERLAP must be between 0 and 500")
        return v
    
    @validator('openai_temperature')
    def validate_temperature(cls, v: float):
        """Validate OpenAI temperature"""
        if not 0.0 <= v <= 2.0:
            raise ValueError("OPENAI_TEMPERATURE must be between 0.0 and 2.0")
        return v
    
    @validator('min_similarity_threshold')
    def validate_similarity_threshold(cls, v: float):
        """Validate similarity threshold"""
        if not 0.0 <= v <= 1.0:
            raise ValueError("MIN_SIMILARITY_THRESHOLD must be between 0.0 and 1.0")
        return v
    
    @validator('allowed_file_types')
    def validate_file_types(cls, v: List[str]):
        """Validate allowed file types"""
        allowed_types = {"pdf", "docx", "txt", "doc", "rtf"}
        for file_type in v:
            if file_type.lower() not in allowed_types:
                raise ValueError(f"Unsupported file type: {file_type}")
        return [ft.lower() for ft in v]
    
    @validator('cors_origins')
    def validate_cors_origins(cls, v: List[str]):
        """Validate CORS origins"""
        for origin in v:
            if not origin.startswith(("http://", "https://")) and origin != "*":
                raise ValueError(f"Invalid CORS origin: {origin}")
        return v

    class Config:
        """Pydantic configuration"""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings() 