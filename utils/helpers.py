"""
Utility Functions
Helper functions for the multimodal RAG chatbot.
"""

import os
import hashlib
from typing import List, Optional
from pathlib import Path

def get_file_hash(file_path: str) -> str:
    """Generate MD5 hash of a file for caching purposes."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hasher.update(chunk)
    return hasher.hexdigest()

def ensure_dir(path: str) -> str:
    """Ensure a directory exists, create if not."""
    os.makedirs(path, exist_ok=True)
    return path

def get_file_extension(file_path: str) -> str:
    """Get the file extension in lowercase."""
    return Path(file_path).suffix.lower()

def is_supported_file(file_path: str) -> bool:
    """Check if a file type is supported."""
    supported_extensions = {'.pdf', '.png', '.jpg', '.jpeg', '.txt'}
    return get_file_extension(file_path) in supported_extensions

def truncate_text(text: str, max_length: int = 500, suffix: str = "...") -> str:
    """Truncate text to a maximum length."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"

def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace and normalizing."""
    # Replace multiple whitespace with single space
    text = ' '.join(text.split())
    # Remove leading/trailing whitespace
    text = text.strip()
    return text

def batch_list(items: List, batch_size: int) -> List[List]:
    """Split a list into batches."""
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

def safe_filename(filename: str) -> str:
    """Create a safe filename by removing/replacing invalid characters."""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename
