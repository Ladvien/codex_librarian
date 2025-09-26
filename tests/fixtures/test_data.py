"""
Test data constants and generators for PDF to Markdown MCP Server tests.

This module provides sample data for testing various components:
- PDF content and metadata
- Markdown conversion results
- Processing results and statistics
- Embeddings and vector data
- Database test data
"""

import hashlib
from datetime import datetime
from typing import Any

# Sample PDF content (minimal valid PDF)
SAMPLE_PDF_CONTENT = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>
endobj
4 0 obj
<< /Length 44 >>
stream
BT
/F1 12 Tf
100 700 Td
(Hello, World!) Tj
ET
endstream
endobj
xref
0 5
0000000000 65535 f
0000000010 00000 n
0000000053 00000 n
0000000110 00000 n
0000000191 00000 n
trailer
<< /Size 5 /Root 1 0 R >>
startxref
290
%%EOF"""

# Sample PDF metadata
SAMPLE_PDF_METADATA = {
    "file_name": "test_document.pdf",
    "file_size": len(SAMPLE_PDF_CONTENT),
    "file_hash": hashlib.sha256(SAMPLE_PDF_CONTENT).hexdigest(),
    "mime_type": "application/pdf",
    "page_count": 1,
    "language": "en",
    "confidence": 0.95,
}

# Sample Markdown content
SAMPLE_MARKDOWN_CONTENT = """# Test Document

This is a comprehensive test document that demonstrates various content types and formatting options.

## Introduction

This document contains multiple sections with different types of content to test the PDF processing pipeline.

## Main Content

### Text Processing

Regular paragraph text with **bold** and *italic* formatting. This section tests basic text extraction and formatting preservation.

### Lists

Unordered list:
- Item 1
- Item 2
- Item 3

Ordered list:
1. First item
2. Second item
3. Third item

### Code Blocks

```python
def process_pdf(file_path: str) -> ProcessingResult:
    \"\"\"Process a PDF file and return results.\"\"\"
    return ProcessingResult(success=True)
```

### Mathematical Formulas

Einstein's mass-energy equation: $E = mc^2$

Quadratic formula: $x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$

### Tables

| Header 1 | Header 2 | Header 3 |
|----------|----------|----------|
| Row 1, Col 1 | Row 1, Col 2 | Row 1, Col 3 |
| Row 2, Col 1 | Row 2, Col 2 | Row 2, Col 3 |
| Row 3, Col 1 | Row 3, Col 2 | Row 3, Col 3 |

### Images

![Sample Image](image1.png)
*Caption: This is a sample image extracted from the PDF*

## Conclusion

This test document covers various content types including text, lists, code, formulas, tables, and images. It serves as a comprehensive test case for the PDF processing pipeline.

---
*Document processed on {timestamp}*
"""

# Sample plain text content
SAMPLE_PLAIN_TEXT = """Test Document

This is a comprehensive test document that demonstrates various content types and formatting options.

Introduction

This document contains multiple sections with different types of content to test the PDF processing pipeline.

Main Content

Text Processing

Regular paragraph text with bold and italic formatting. This section tests basic text extraction and formatting preservation.

Lists

Unordered list:
- Item 1
- Item 2
- Item 3

Ordered list:
1. First item
2. Second item
3. Third item

Code Blocks

def process_pdf(file_path: str) -> ProcessingResult:
    \"\"\"Process a PDF file and return results.\"\"\"
    return ProcessingResult(success=True)

Mathematical Formulas

Einstein's mass-energy equation: E = mc^2

Quadratic formula: x = (-b ± √(b²-4ac)) / 2a

Tables

Header 1 | Header 2 | Header 3
Row 1, Col 1 | Row 1, Col 2 | Row 1, Col 3
Row 2, Col 1 | Row 2, Col 2 | Row 2, Col 3
Row 3, Col 1 | Row 3, Col 2 | Row 3, Col 3

Images

Sample Image
Caption: This is a sample image extracted from the PDF

Conclusion

This test document covers various content types including text, lists, code, formulas, tables, and images. It serves as a comprehensive test case for the PDF processing pipeline.

Document processed on {timestamp}
"""

# Sample text chunks for embedding
SAMPLE_CHUNKS = [
    {
        "text": "This is a comprehensive test document that demonstrates various content types and formatting options.",
        "start_char": 0,
        "end_char": 106,
        "token_count": 16,
        "chunk_index": 0,
    },
    {
        "text": "This document contains multiple sections with different types of content to test the PDF processing pipeline.",
        "start_char": 107,
        "end_char": 216,
        "token_count": 17,
        "chunk_index": 1,
    },
    {
        "text": "Regular paragraph text with bold and italic formatting. This section tests basic text extraction and formatting preservation.",
        "start_char": 217,
        "end_char": 341,
        "token_count": 18,
        "chunk_index": 2,
    },
    {
        "text": "Einstein's mass-energy equation: E = mc^2. Quadratic formula: x = (-b ± √(b²-4ac)) / 2a",
        "start_char": 342,
        "end_char": 431,
        "token_count": 19,
        "chunk_index": 3,
    },
]

# Sample extracted tables
SAMPLE_TABLES = [
    {
        "table_index": 0,
        "headers": ["Header 1", "Header 2", "Header 3"],
        "rows": [
            ["Row 1, Col 1", "Row 1, Col 2", "Row 1, Col 3"],
            ["Row 2, Col 1", "Row 2, Col 2", "Row 2, Col 3"],
            ["Row 3, Col 1", "Row 3, Col 2", "Row 3, Col 3"],
        ],
        "markdown": """| Header 1 | Header 2 | Header 3 |
|----------|----------|----------|
| Row 1, Col 1 | Row 1, Col 2 | Row 1, Col 3 |
| Row 2, Col 1 | Row 2, Col 2 | Row 2, Col 3 |
| Row 3, Col 1 | Row 3, Col 2 | Row 3, Col 3 |""",
        "position": {"page": 1, "x": 100, "y": 400, "width": 400, "height": 120},
        "confidence": 0.92,
    }
]

# Sample extracted formulas
SAMPLE_FORMULAS = [
    {
        "formula_index": 0,
        "latex": "E = mc^2",
        "text": "E = mc^2",
        "position": {"page": 1, "x": 150, "y": 300, "width": 80, "height": 20},
        "confidence": 0.98,
    },
    {
        "formula_index": 1,
        "latex": "x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}",
        "text": "x = (-b ± √(b²-4ac)) / 2a",
        "position": {"page": 1, "x": 120, "y": 250, "width": 200, "height": 30},
        "confidence": 0.94,
    },
]

# Sample extracted images
SAMPLE_IMAGES = [
    {
        "image_index": 0,
        "file_path": "/tmp/extracted_image_0.png",
        "description": "Sample image extracted from PDF",
        "ocr_text": "Sample Image Caption: This is a sample image extracted from the PDF",
        "position": {"page": 1, "x": 100, "y": 200, "width": 300, "height": 200},
        "format": "PNG",
        "size_bytes": 12345,
        "confidence": 0.89,
    }
]

# Sample embeddings (1536-dimensional vectors)
SAMPLE_EMBEDDINGS = [
    [0.1 + i * 0.001 for i in range(1536)],  # Chunk 1 embedding
    [0.2 + i * 0.001 for i in range(1536)],  # Chunk 2 embedding
    [0.3 + i * 0.001 for i in range(1536)],  # Chunk 3 embedding
    [0.4 + i * 0.001 for i in range(1536)],  # Chunk 4 embedding
]

# Sample image embeddings (512-dimensional CLIP vectors)
SAMPLE_IMAGE_EMBEDDINGS = [
    [0.5 + i * 0.002 for i in range(512)],  # Image 1 embedding
]


def create_sample_pdf_content(content: str | None = None) -> bytes:
    """Create sample PDF content with optional custom text."""
    if content is None:
        return SAMPLE_PDF_CONTENT

    # Simple PDF template with custom content
    template = f"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>
endobj
4 0 obj
<< /Length {len(content) + 20} >>
stream
BT
/F1 12 Tf
100 700 Td
({content[:100]}) Tj
ET
endstream
endobj
xref
0 5
0000000000 65535 f
trailer
<< /Size 5 /Root 1 0 R >>
startxref
300
%%EOF"""

    return template.encode("utf-8")


def create_sample_markdown(
    title: str = "Test Document", sections: list[str] | None = None
) -> str:
    """Create sample markdown content with optional customization."""
    if sections is None:
        return SAMPLE_MARKDOWN_CONTENT.format(timestamp=datetime.now().isoformat())

    content = f"# {title}\n\n"
    for i, section in enumerate(sections, 1):
        content += f"## Section {i}\n\n{section}\n\n"

    content += f"---\n*Document processed on {datetime.now().isoformat()}*\n"
    return content


def create_sample_processing_result(**overrides) -> dict[str, Any]:
    """Create a sample processing result with optional overrides."""
    result = {
        "success": True,
        "markdown_content": SAMPLE_MARKDOWN_CONTENT,
        "plain_text": SAMPLE_PLAIN_TEXT,
        "chunks": SAMPLE_CHUNKS.copy(),
        "tables": SAMPLE_TABLES.copy(),
        "formulas": SAMPLE_FORMULAS.copy(),
        "images": SAMPLE_IMAGES.copy(),
        "metadata": {
            "processing_time": 2.5,
            "page_count": 1,
            "word_count": 245,
            "language": "en",
            "confidence": 0.95,
            "file_size": len(SAMPLE_PDF_CONTENT),
            "file_hash": SAMPLE_PDF_METADATA["file_hash"],
        },
    }

    # Apply overrides
    result.update(overrides)
    return result


def create_sample_embeddings(
    count: int = 4, dimensions: int = 1536
) -> list[list[float]]:
    """Create sample embedding vectors."""
    embeddings = []
    for i in range(count):
        # Create deterministic but varied embeddings
        base_value = 0.1 * (i + 1)
        embedding = [base_value + j * 0.0001 for j in range(dimensions)]
        embeddings.append(embedding)

    return embeddings


def create_database_test_data() -> dict[str, list[dict[str, Any]]]:
    """Create comprehensive test data for database operations."""
    now = datetime.utcnow()

    documents = [
        {
            "id": 1,
            "file_path": "/tmp/test_doc_1.pdf",
            "file_name": "test_document_1.pdf",
            "file_size": 12345,
            "file_hash": "hash1",
            "mime_type": "application/pdf",
            "status": "completed",
            "created_at": now,
            "updated_at": now,
        },
        {
            "id": 2,
            "file_path": "/tmp/test_doc_2.pdf",
            "file_name": "test_document_2.pdf",
            "file_size": 23456,
            "file_hash": "hash2",
            "mime_type": "application/pdf",
            "status": "processing",
            "created_at": now,
            "updated_at": now,
        },
        {
            "id": 3,
            "file_path": "/tmp/test_doc_3.pdf",
            "file_name": "test_document_3.pdf",
            "file_size": 34567,
            "file_hash": "hash3",
            "mime_type": "application/pdf",
            "status": "failed",
            "created_at": now,
            "updated_at": now,
        },
    ]

    document_content = [
        {
            "id": 1,
            "document_id": 1,
            "markdown_content": SAMPLE_MARKDOWN_CONTENT,
            "plain_text": SAMPLE_PLAIN_TEXT,
            "word_count": 245,
            "language": "en",
            "created_at": now,
        },
        {
            "id": 2,
            "document_id": 2,
            "markdown_content": "# Document 2\n\nProcessing...",
            "plain_text": "Document 2\n\nProcessing...",
            "word_count": 3,
            "language": "en",
            "created_at": now,
        },
    ]

    document_embeddings = [
        {
            "id": 1,
            "document_id": 1,
            "chunk_index": 0,
            "chunk_text": SAMPLE_CHUNKS[0]["text"],
            "embedding": SAMPLE_EMBEDDINGS[0],
            "start_char": SAMPLE_CHUNKS[0]["start_char"],
            "end_char": SAMPLE_CHUNKS[0]["end_char"],
            "token_count": SAMPLE_CHUNKS[0]["token_count"],
            "created_at": now,
        },
        {
            "id": 2,
            "document_id": 1,
            "chunk_index": 1,
            "chunk_text": SAMPLE_CHUNKS[1]["text"],
            "embedding": SAMPLE_EMBEDDINGS[1],
            "start_char": SAMPLE_CHUNKS[1]["start_char"],
            "end_char": SAMPLE_CHUNKS[1]["end_char"],
            "token_count": SAMPLE_CHUNKS[1]["token_count"],
            "created_at": now,
        },
    ]

    document_images = [
        {
            "id": 1,
            "document_id": 1,
            "image_index": 0,
            "file_path": "/tmp/extracted_image_0.png",
            "description": "Sample image",
            "ocr_text": "Sample Image Caption",
            "embedding": SAMPLE_IMAGE_EMBEDDINGS[0],
            "page_number": 1,
            "position_data": SAMPLE_IMAGES[0]["position"],
            "created_at": now,
        }
    ]

    processing_queue = [
        {
            "id": 1,
            "task_id": "task-123",
            "document_id": 2,
            "task_type": "pdf_processing",
            "status": "running",
            "priority": 5,
            "progress": 50.0,
            "error_message": None,
            "retry_count": 0,
            "created_at": now,
            "updated_at": now,
        },
        {
            "id": 2,
            "task_id": "task-456",
            "document_id": 3,
            "task_type": "pdf_processing",
            "status": "failed",
            "priority": 5,
            "progress": 25.0,
            "error_message": "Processing failed: Invalid PDF format",
            "retry_count": 3,
            "created_at": now,
            "updated_at": now,
        },
    ]

    return {
        "documents": documents,
        "document_content": document_content,
        "document_embeddings": document_embeddings,
        "document_images": document_images,
        "processing_queue": processing_queue,
    }


# Error scenarios for testing
ERROR_SCENARIOS = {
    "invalid_pdf": {
        "content": b"Not a PDF file",
        "expected_error": "Invalid PDF format",
    },
    "corrupted_pdf": {
        "content": b"%PDF-1.4\n1 0 obj\n<< corrupted",
        "expected_error": "Corrupted PDF file",
    },
    "empty_pdf": {
        "content": b"",
        "expected_error": "Empty file",
    },
    "large_pdf": {
        "size": 600 * 1024 * 1024,  # 600MB
        "expected_error": "File too large",
    },
}
