"""Utility functions with enhanced security and validation"""

import hashlib
import os
import re
import uuid
import mimetypes
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple
import secrets
import bleach


def generate_unique_id() -> str:
    """Generate a cryptographically secure unique ID"""
    return str(uuid.uuid4())


def generate_secure_token(length: int = 32) -> str:
    """Generate a secure random token"""
    return secrets.token_urlsafe(length)


def get_file_hash(file_path: str, algorithm: str = "sha256") -> str:
    """
    Get file hash using specified algorithm
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm (sha256, md5, sha1)
        
    Returns:
        File hash in hexadecimal format
        
    Raises:
        ValueError: If algorithm is not supported
        FileNotFoundError: If file doesn't exist
    """
    if algorithm not in ["sha256", "md5", "sha1"]:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    hash_func = getattr(hashlib, algorithm)()
    
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_func.update(chunk)
        return hash_func.hexdigest()
    except IOError as e:
        raise IOError(f"Error reading file {file_path}: {e}")


def ensure_directory_exists(directory_path: str) -> None:
    """
    Ensure directory exists with security checks
    
    Args:
        directory_path: Path to directory
        
    Raises:
        ValueError: If path is invalid or unsafe
    """
    # Resolve and validate path
    safe_path = validate_and_resolve_path(directory_path)
    safe_path.mkdir(parents=True, exist_ok=True)


def validate_and_resolve_path(path: str, base_path: Optional[str] = None) -> Path:
    """
    Validate and resolve path with security checks against path traversal
    
    Args:
        path: Path to validate
        base_path: Base path to restrict to (defaults to current working directory)
        
    Returns:
        Resolved Path object
        
    Raises:
        ValueError: If path is invalid or unsafe
    """
    if not path or not isinstance(path, str):
        raise ValueError("Path must be a non-empty string")
    
    # Remove null bytes and other dangerous characters
    clean_path = path.replace('\0', '').strip()
    if not clean_path:
        raise ValueError("Path is empty after cleaning")
    
    # Convert to Path object and resolve
    path_obj = Path(clean_path)
    
    try:
        resolved_path = path_obj.resolve()
    except (OSError, RuntimeError) as e:
        raise ValueError(f"Cannot resolve path: {e}")
    
    # Set base path for restriction
    if base_path is None:
        base_path = Path.cwd()
    else:
        base_path = Path(base_path).resolve()
    
    # Check if resolved path is within allowed base path
    try:
        resolved_path.relative_to(base_path)
    except ValueError:
        raise ValueError(f"Path traversal detected: {path}")
    
    return resolved_path


def get_file_size_mb(file_path: str) -> float:
    """
    Get file size in megabytes with validation
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in MB
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)
    except OSError as e:
        raise OSError(f"Error getting file size: {e}")


def is_valid_file_extension(filename: str, allowed_extensions: List[str]) -> bool:
    """
    Check if file extension is valid with enhanced security
    
    Args:
        filename: File name to check
        allowed_extensions: List of allowed extensions
        
    Returns:
        True if extension is valid
    """
    if not filename or not isinstance(filename, str):
        return False
    
    # Get extension using multiple methods for security
    file_extension = Path(filename).suffix.lower().lstrip('.')
    
    # Also check using splitext for edge cases
    _, ext_alt = os.path.splitext(filename.lower())
    ext_alt = ext_alt.lstrip('.')
    
    # Both methods should agree
    if file_extension != ext_alt:
        return False
    
    return file_extension in [ext.lower() for ext in allowed_extensions]


def get_safe_mime_type(file_path: str) -> str:
    """
    Get MIME type of file with security validation
    
    Args:
        file_path: Path to file
        
    Returns:
        MIME type string
    """
    mime_type, _ = mimetypes.guess_type(file_path)
    
    # Define allowed MIME types
    allowed_types = {
        'application/pdf',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'application/msword',
        'text/plain',
        'text/rtf'
    }
    
    if mime_type in allowed_types:
        return mime_type
    
    return 'application/octet-stream'


def sanitize_filename(filename: str, max_length: int = 200) -> str:
    """
    Sanitize filename with enhanced security
    
    Args:
        filename: Original filename
        max_length: Maximum allowed length
        
    Returns:
        Safe filename
    """
    if not filename or not isinstance(filename, str):
        return "unnamed_file"
    
    # Remove null bytes and control characters
    clean_name = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', filename)
    
    # Replace unsafe characters
    unsafe_chars = r'[<>:"/\\|?*]'
    clean_name = re.sub(unsafe_chars, '_', clean_name)
    
    # Remove multiple dots (potential security issue)
    clean_name = re.sub(r'\.{2,}', '.', clean_name)
    
    # Remove leading/trailing dots and spaces
    clean_name = clean_name.strip('. ')
    
    # Handle reserved names on Windows
    reserved_names = {
        'CON', 'PRN', 'AUX', 'NUL',
        'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
        'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
    }
    
    name_without_ext = Path(clean_name).stem.upper()
    if name_without_ext in reserved_names:
        clean_name = f"file_{clean_name}"
    
    # Limit length while preserving extension
    if len(clean_name) > max_length:
        name, ext = os.path.splitext(clean_name)
        max_name_length = max_length - len(ext)
        clean_name = name[:max_name_length] + ext
    
    # Ensure we have a valid filename
    if not clean_name or clean_name == '.':
        clean_name = f"unnamed_file_{generate_unique_id()[:8]}"
    
    return clean_name


def sanitize_text_input(text: str, max_length: int = 10000) -> str:
    """
    Sanitize text input to prevent XSS and other attacks
    
    Args:
        text: Input text to sanitize
        max_length: Maximum allowed length
        
    Returns:
        Sanitized text
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Limit length
    if len(text) > max_length:
        text = text[:max_length]
    
    # Remove null bytes and control characters (except newlines and tabs)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    
    # Use bleach to clean HTML/script content
    allowed_tags = []  # No HTML tags allowed
    text = bleach.clean(text, tags=allowed_tags, strip=True)
    
    return text.strip()


def validate_user_id(user_id: str) -> bool:
    """
    Validate user ID format for security
    
    Args:
        user_id: User ID to validate
        
    Returns:
        True if valid
    """
    if not user_id or not isinstance(user_id, str):
        return False
    
    # Telegram user IDs are numeric strings
    if not re.match(r'^\d{1,20}$', user_id):
        return False
    
    # Additional length check
    if len(user_id) > 20:
        return False
    
    return True


def format_timestamp(timestamp: Optional[datetime] = None, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format timestamp with validation
    
    Args:
        timestamp: Timestamp to format (defaults to current time)
        format_str: Format string
        
    Returns:
        Formatted timestamp string
    """
    if timestamp is None:
        timestamp = datetime.utcnow()
    
    if not isinstance(timestamp, datetime):
        raise ValueError("timestamp must be a datetime object")
    
    try:
        return timestamp.strftime(format_str)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid format string: {e}")


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text with validation
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add when truncating
        
    Returns:
        Truncated text
    """
    if not isinstance(text, str):
        return str(text)[:max_length]
    
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def is_safe_path(path: str, allowed_dirs: List[str]) -> bool:
    """
    Check if path is safe and within allowed directories
    
    Args:
        path: Path to check
        allowed_dirs: List of allowed base directories
        
    Returns:
        True if path is safe
    """
    try:
        resolved_path = Path(path).resolve()
        
        for allowed_dir in allowed_dirs:
            allowed_path = Path(allowed_dir).resolve()
            try:
                resolved_path.relative_to(allowed_path)
                return True
            except ValueError:
                continue
        
        return False
    except (OSError, ValueError):
        return False


def get_file_stats(file_path: str) -> dict:
    """
    Get comprehensive file statistics
    
    Args:
        file_path: Path to file
        
    Returns:
        Dictionary with file statistics
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        stat = os.stat(file_path)
        return {
            'size_bytes': stat.st_size,
            'size_mb': stat.st_size / (1024 * 1024),
            'created': datetime.fromtimestamp(stat.st_ctime),
            'modified': datetime.fromtimestamp(stat.st_mtime),
            'mime_type': get_safe_mime_type(file_path),
            'extension': Path(file_path).suffix.lower(),
            'is_readable': os.access(file_path, os.R_OK),
        }
    except OSError as e:
        raise OSError(f"Error getting file stats: {e}") 