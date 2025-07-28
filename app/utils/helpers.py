"""Вспомогательные функции"""

import hashlib
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional


def generate_unique_id() -> str:
    """Генерация уникального ID"""
    return str(uuid.uuid4())


def get_file_hash(file_path: str) -> str:
    """
    Получить MD5 хеш файла
    
    Args:
        file_path: Путь к файлу
        
    Returns:
        MD5 хеш файла
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def ensure_directory_exists(directory_path: str) -> None:
    """
    Убедиться что директория существует, создать если нет
    
    Args:
        directory_path: Путь к директории
    """
    Path(directory_path).mkdir(parents=True, exist_ok=True)


def get_file_size_mb(file_path: str) -> float:
    """
    Получить размер файла в мегабайтах
    
    Args:
        file_path: Путь к файлу
        
    Returns:
        Размер файла в MB
    """
    size_bytes = os.path.getsize(file_path)
    return size_bytes / (1024 * 1024)


def is_valid_file_extension(filename: str, allowed_extensions: list) -> bool:
    """
    Проверить валидность расширения файла
    
    Args:
        filename: Имя файла
        allowed_extensions: Список разрешённых расширений
        
    Returns:
        True если расширение валидно
    """
    file_extension = filename.split('.')[-1].lower()
    return file_extension in [ext.lower() for ext in allowed_extensions]


def sanitize_filename(filename: str) -> str:
    """
    Очистить имя файла от небезопасных символов
    
    Args:
        filename: Исходное имя файла
        
    Returns:
        Безопасное имя файла
    """
    # Удалить или заменить небезопасные символы
    unsafe_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
    safe_filename = filename
    for char in unsafe_chars:
        safe_filename = safe_filename.replace(char, '_')
    
    # Ограничить длину
    if len(safe_filename) > 255:
        name, ext = os.path.splitext(safe_filename)
        safe_filename = name[:255-len(ext)] + ext
    
    return safe_filename


def format_timestamp(timestamp: Optional[datetime] = None) -> str:
    """
    Форматировать временную метку
    
    Args:
        timestamp: Временная метка (если None - текущее время)
        
    Returns:
        Отформатированная строка времени
    """
    if timestamp is None:
        timestamp = datetime.now()
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")


def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Обрезать текст до указанной длины
    
    Args:
        text: Исходный текст
        max_length: Максимальная длина
        
    Returns:
        Обрезанный текст
    """
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..." 