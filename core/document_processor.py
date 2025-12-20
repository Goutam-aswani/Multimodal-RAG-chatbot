"""
Document Processing Module
Functions:
- load_document(file_path) → List[Chunk]
- extract_text_pymupdf(pdf_path) → str
- extract_images_pymupdf(pdf_path) → List[Image]
- extract_tables_pymupdf(pdf_path) → List[str]
- run_ocr(image) → str
- chunk_documents(documents) → List[Chunk]
- semantic_chunk(text) → List[str]
"""

import fitz
import easyocr
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dataclasses import dataclass, field
from typing import List, Optional
from PIL import Image
import io
import os
import base64

@dataclass
class Chunk:
    content: str
    chunk_type: str  # "text", "table", "image"
    page_number: int
    source_file: str
    metadata: dict = field(default_factory=dict)
    image_data: Optional[str] = None  # Base64-encoded image bytes (for image chunks)

_ocr_reader = None

def get_ocr_reader():
    """Get or initialize OCR reader."""
    global _ocr_reader
    if _ocr_reader is None:
        _ocr_reader = easyocr.Reader(['en'])
    return _ocr_reader

def load_document(file_path: str) -> List[Chunk]:
    """Load and process a document into chunks."""
    if file_path.endswith(".pdf"):
        return process_pdf(file_path)
    elif file_path.endswith((".png", ".jpg", ".jpeg")):
        return process_image(file_path)
    elif file_path.endswith(".txt"):
        return process_text(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

def process_pdf(pdf_path: str) -> List[Chunk]:
    """Extract text, images, and tables from PDF using PyMuPDF."""
    doc = fitz.open(pdf_path)
    chunks = []
    source_name = os.path.basename(pdf_path)
    
    for page_num, page in enumerate(doc):
        # Extract text
        text = page.get_text()
        if text.strip():
            text_chunks = chunk_text(text, page_num + 1, source_name)
            chunks.extend(text_chunks)
        
        # Extract images - store actual image data for CLIP embedding
        images = page.get_images()
        for img_idx, img in enumerate(images):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Convert to PIL Image
                pil_image = Image.open(io.BytesIO(image_bytes))
                
                # Convert image to base64 for storage
                img_buffer = io.BytesIO()
                # Convert to RGB if necessary (for JPEG compatibility)
                if pil_image.mode in ('RGBA', 'P'):
                    pil_image = pil_image.convert('RGB')
                pil_image.save(img_buffer, format='JPEG', quality=85)
                image_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
                
                # Run OCR on image for text content (used in prompts/display)
                ocr_text = run_ocr(pil_image)
                content_text = f"[Image Content]\n{ocr_text}" if ocr_text.strip() else "[Image Content]"
                
                chunks.append(Chunk(
                    content=content_text,
                    chunk_type="image",
                    page_number=page_num + 1,
                    source_file=source_name,
                    metadata={"image_index": img_idx, "width": pil_image.width, "height": pil_image.height},
                    image_data=image_base64  # Store actual image for CLIP embedding
                ))
            except Exception as e:
                continue
        
        # Extract tables
        try:
            tables = page.find_tables()
            for table_idx, table in enumerate(tables):
                # Convert table to markdown format
                table_md = table_to_markdown(table)
                if table_md.strip():
                    chunks.append(Chunk(
                        content=f"[Table Content]\n{table_md}",
                        chunk_type="table",
                        page_number=page_num + 1,
                        source_file=source_name,
                        metadata={"table_index": table_idx}
                    ))
        except Exception as e:
            # Tables extraction might not be available in all versions
            pass
    
    doc.close()
    return chunks

def process_image(image_path: str) -> List[Chunk]:
    """Process an image file - store image data for CLIP embedding."""
    source_name = os.path.basename(image_path)
    pil_image = Image.open(image_path)

    data = list(pil_image.getdata())
    pil_image_rgb = Image.new(pil_image.mode, pil_image.size)
    pil_image_rgb.putdata(data)
    
    # Convert image to base64 for storage
    img_buffer = io.BytesIO()

    if pil_image.mode in ('RGBA', 'P'):
        pil_image = pil_image.convert('RGB')
    pil_image.save(img_buffer, format='JPEG', quality=85)
    image_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    
    # Run OCR for text content
    ocr_text = run_ocr(pil_image)
    content_text = f"[Image Content]\n{ocr_text}" if ocr_text.strip() else "[Image Content]"
    
    return [Chunk(
        content=content_text,
        chunk_type="image",
        page_number=1,
        source_file=source_name,
        metadata={"width": pil_image.width, "height": pil_image.height},
        image_data=image_base64  # Store actual image for CLIP embedding
    )]

def process_text(text_path: str) -> List[Chunk]:
    """Process a text file."""
    source_name = os.path.basename(text_path)
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    return chunk_text(text, 1, source_name)

def run_ocr(image: Image.Image) -> str:
    """Run OCR on an image."""
    try:
        reader = get_ocr_reader()
        # Convert PIL Image to bytes for EasyOCR
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        results = reader.readtext(img_byte_arr)
        text_parts = [result[1] for result in results]
        return " ".join(text_parts)
    except Exception as e:
        return ""

def table_to_markdown(table) -> str:
    """Convert a PyMuPDF table to markdown format."""
    try:
        data = table.extract()
        if not data:
            return ""
        
        lines = []
        for row_idx, row in enumerate(data):
            cells = [str(cell) if cell else "" for cell in row]
            lines.append("| " + " | ".join(cells) + " |")
            if row_idx == 0:
                lines.append("| " + " | ".join(["---"] * len(cells)) + " |")
        
        return "\n".join(lines)
    except Exception:
        return ""

def chunk_text(text: str, page_num: int, source: str) -> List[Chunk]:
    """Split text into semantic chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " "]
    )
    splits = splitter.split_text(text)
    
    return [
        Chunk(
            content=split,
            chunk_type="text",
            page_number=page_num,
            source_file=source,
            metadata={}
        )
        for split in splits
    ]
