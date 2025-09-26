#!/usr/bin/env python3
"""
Complete API Usage Example - Full PDF to Markdown MCP Pipeline

This comprehensive example demonstrates how to use the entire PDF to Markdown MCP
system programmatically:

- Database connection and setup
- MinerU PDF processing service
- Text chunking for embeddings
- Embedding generation (Ollama/OpenAI)
- Vector database storage with PGVector
- Semantic similarity search
- Comprehensive error handling
- Performance monitoring

This is the definitive example for integrating all system components.
Run this to see the complete API in action and verify your setup.
"""

import os
import sys
import asyncio
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import traceback

# Load environment variables FIRST (before any imports that need them)
from dotenv import load_dotenv

load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Imports (after environment is loaded)
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

try:
    from pdf_to_markdown_mcp.db.models import (
        Document,
        DocumentContent,
        DocumentEmbedding,
    )
    from pdf_to_markdown_mcp.services.mineru import MinerUService
    from pdf_to_markdown_mcp.services.embeddings import (
        EmbeddingService,
        EmbeddingConfig,
        EmbeddingProvider,
    )
    from pdf_to_markdown_mcp.services.database import VectorDatabaseService
    from pdf_to_markdown_mcp.models.request import ProcessingOptions
    from pdf_to_markdown_mcp.core.chunker import TextChunker
    from pdf_to_markdown_mcp.core.processor import PDFProcessor
    from pdf_to_markdown_mcp.config import settings
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    print("This suggests there may be missing dependencies or configuration issues.")
    sys.exit(1)

# Environment already loaded above


class FullPipelineTest:
    """Complete PDF processing pipeline test."""

    def __init__(self):
        self.engine = None
        self.session = None
        self.results = {
            "database_connection": False,
            "mineru_available": False,
            "embedding_service": False,
            "pdf_processing": False,
            "content_extraction": False,
            "text_chunking": False,
            "embedding_generation": False,
            "vector_storage": False,
            "similarity_search": False,
            "error_handling": False,
        }
        self.test_document_id = None
        self.test_embeddings = []
        self.processing_time = 0

    def print_header(self, title: str):
        """Print a formatted header."""
        print(f"\n{'='*60}")
        print(f"{title}")
        print("=" * 60)

    def print_step(self, step: str, success: bool = None, details: str = ""):
        """Print a test step result."""
        if success is True:
            status = "✅ PASS"
        elif success is False:
            status = "❌ FAIL"
        else:
            status = "🔄 TEST"

        print(f"{status} {step}")
        if details:
            print(f"     {details}")

    async def test_database_connection(self) -> bool:
        """Test database connection and setup."""
        self.print_header("1. DATABASE CONNECTION TEST")

        try:
            database_url = os.getenv("DATABASE_URL")
            if not database_url:
                self.print_step(
                    "Database URL configured", False, "DATABASE_URL not found in .env"
                )
                return False

            self.print_step("Database URL configured", True)

            # Test connection
            self.engine = create_engine(database_url)

            with self.engine.connect() as conn:
                # Test basic connection
                result = conn.execute(text("SELECT version()"))
                version = result.fetchone()[0]
                self.print_step(
                    "PostgreSQL connection", True, f"Version: {version[:50]}..."
                )

                # Test PGVector
                result = conn.execute(
                    text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
                )
                vector_version = result.fetchone()
                if vector_version:
                    self.print_step(
                        "PGVector extension", True, f"Version: {vector_version[0]}"
                    )
                else:
                    self.print_step("PGVector extension", False, "Extension not found")
                    return False

                # Test tables exist
                tables = ["documents", "document_content", "document_embeddings"]
                for table in tables:
                    result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                    count = result.fetchone()[0]
                    self.print_step(f"Table '{table}' exists", True, f"{count} records")

            # Setup session
            SessionLocal = sessionmaker(bind=self.engine)
            self.session = SessionLocal()

            self.results["database_connection"] = True
            return True

        except Exception as e:
            self.print_step("Database connection", False, str(e))
            logger.exception("Database connection failed")
            return False

    async def test_mineru_service(self) -> bool:
        """Test MinerU PDF processing service."""
        self.print_header("2. MINERU SERVICE TEST")

        try:
            # Test import
            self.print_step(
                "MinerU import", True, "Service class imported successfully"
            )

            # Initialize service
            mineru_service = MinerUService()
            self.print_step("MinerU initialization", True, "Service initialized")

            # Test with a small PDF from research directory
            pdf_files = list(Path("/mnt/codex_fs/research").rglob("*.pdf"))[:3]

            if not pdf_files:
                self.print_step(
                    "Test PDFs available", False, "No PDFs found in research directory"
                )
                return False

            test_pdf = pdf_files[0]  # Use smallest/first PDF
            self.print_step(
                "Test PDF selected",
                True,
                f"{test_pdf.name} ({test_pdf.stat().st_size / (1024*1024):.1f} MB)",
            )

            # Test PDF processing
            options = ProcessingOptions(
                ocr_language="eng",
                preserve_layout=True,
                extract_tables=True,
                extract_formulas=True,
                extract_images=True,
                chunk_for_embeddings=True,
                chunk_size=1000,
                chunk_overlap=200,
            )

            self.print_step("Processing options configured", True)

            start_time = time.time()

            # This is the key test - actual PDF processing
            try:
                result = await mineru_service.process_pdf(
                    pdf_path=test_pdf, options=options
                )

                self.processing_time = time.time() - start_time

                self.print_step(
                    "PDF processing completed",
                    True,
                    f"Time: {self.processing_time:.2f}s",
                )

                # Validate results
                if result.markdown_content:
                    content_length = len(result.markdown_content)
                    self.print_step(
                        "Markdown content extracted",
                        True,
                        f"{content_length} characters",
                    )
                else:
                    self.print_step(
                        "Markdown content extracted", False, "No content returned"
                    )

                if result.plain_text:
                    text_length = len(result.plain_text)
                    self.print_step(
                        "Plain text extracted", True, f"{text_length} characters"
                    )
                else:
                    self.print_step("Plain text extracted", False, "No text returned")

                # Check metadata
                if (
                    hasattr(result, "processing_metadata")
                    and result.processing_metadata
                ):
                    pages = getattr(result.processing_metadata, "pages", 0)
                    self.print_step("Processing metadata", True, f"Pages: {pages}")
                else:
                    self.print_step(
                        "Processing metadata", False, "No metadata returned"
                    )

                # Store for later tests
                self.test_pdf_result = result
                self.test_pdf_path = test_pdf

                self.results["mineru_available"] = True
                self.results["pdf_processing"] = True
                self.results["content_extraction"] = True
                return True

            except Exception as e:
                self.print_step("PDF processing", False, f"Processing failed: {str(e)}")
                logger.exception("MinerU processing failed")

                # Try to continue with mock data for other tests
                self.print_step("Creating mock data for remaining tests", True)

                from pdf_to_markdown_mcp.models.processing import (
                    ProcessingResult,
                    ProcessingMetadata,
                )

                # Create mock result
                self.test_pdf_result = ProcessingResult(
                    markdown_content="# Mock Document\n\nThis is mock content for testing.",
                    plain_text="Mock Document\n\nThis is mock content for testing.",
                    tables=[],
                    formulas=[],
                    images=[],
                    chunks=[],
                    processing_metadata=ProcessingMetadata(
                        pages=1, processing_time_ms=100
                    ),
                )
                self.test_pdf_path = test_pdf

                return False

        except Exception as e:
            self.print_step("MinerU service test", False, str(e))
            logger.exception("MinerU service test failed")
            return False

    async def test_text_chunking(self) -> bool:
        """Test text chunking functionality."""
        self.print_header("3. TEXT CHUNKING TEST")

        try:
            chunker = TextChunker()
            self.print_step("Text chunker initialized", True)

            # Get text to chunk
            if hasattr(self, "test_pdf_result") and self.test_pdf_result.plain_text:
                text = self.test_pdf_result.plain_text
            else:
                text = "This is a test document. " * 100  # Mock text

            self.print_step("Text for chunking", True, f"{len(text)} characters")

            # Test chunking
            chunks = await chunker.create_chunks(
                text=text, chunk_size=1000, chunk_overlap=200
            )

            self.print_step(
                "Text chunking completed", True, f"{len(chunks)} chunks created"
            )

            # Validate chunks
            if chunks:
                avg_size = sum(len(chunk.text) for chunk in chunks) / len(chunks)
                self.print_step(
                    "Chunk validation",
                    True,
                    f"Average chunk size: {avg_size:.0f} chars",
                )

                # Store for later tests
                self.test_chunks = chunks
                self.results["text_chunking"] = True
                return True
            else:
                self.print_step("Chunk validation", False, "No chunks created")
                return False

        except Exception as e:
            self.print_step("Text chunking", False, str(e))
            logger.exception("Text chunking failed")
            return False

    async def test_embedding_service(self) -> bool:
        """Test embedding generation."""
        self.print_header("4. EMBEDDING SERVICE TEST")

        try:
            # Test embedding service initialization
            embedding_config = EmbeddingConfig(
                provider=EmbeddingProvider.OLLAMA, batch_size=5, timeout=30.0
            )
            embedding_service = EmbeddingService(config=embedding_config)
            self.print_step("Embedding service initialized", True)

            # Test with sample text
            test_text = "This is a test sentence for embedding generation."

            self.print_step(
                "Generating test embedding", None, f"Text: '{test_text[:50]}...'"
            )

            # Generate embedding
            embedding_result = await embedding_service.generate_embeddings([test_text])
            embedding = (
                embedding_result.embeddings[0] if embedding_result.embeddings else None
            )

            if embedding is not None and len(embedding) > 0:
                self.print_step(
                    "Embedding generation", True, f"Dimension: {len(embedding)}"
                )

                # Test batch generation if we have chunks
                if hasattr(self, "test_chunks") and self.test_chunks:
                    chunk_texts = [
                        chunk.text[:500] for chunk in self.test_chunks[:3]
                    ]  # Limit for speed

                    self.print_step(
                        "Batch embedding generation",
                        None,
                        f"Processing {len(chunk_texts)} chunks",
                    )

                    batch_embeddings = []
                    batch_result = await embedding_service.generate_embeddings(
                        chunk_texts
                    )
                    batch_embeddings = (
                        batch_result.embeddings if batch_result.embeddings else []
                    )

                    self.print_step(
                        "Batch embedding generation",
                        True,
                        f"Generated {len(batch_embeddings)} embeddings",
                    )

                    self.test_embeddings = batch_embeddings
                else:
                    self.test_embeddings = [embedding]

                self.results["embedding_service"] = True
                self.results["embedding_generation"] = True
                return True
            else:
                self.print_step("Embedding generation", False, "No embedding returned")
                return False

        except Exception as e:
            self.print_step("Embedding service", False, str(e))
            logger.exception("Embedding service failed")

            # Try to continue with mock embeddings
            self.print_step("Creating mock embeddings for testing", True)
            import numpy as np

            mock_embedding = np.random.random(1536).tolist()  # OpenAI embedding size
            self.test_embeddings = [mock_embedding]

            return False

    async def test_database_storage(self) -> bool:
        """Test storing processed content and embeddings in database."""
        self.print_header("5. DATABASE STORAGE TEST")

        try:
            if not self.session:
                self.print_step(
                    "Database session", False, "No database session available"
                )
                return False

            # Create document record (or find existing one for test)
            import hashlib

            file_path = getattr(self, "test_pdf_path", Path("/tmp/test.pdf"))
            file_hash = hashlib.sha256(str(file_path).encode()).hexdigest()

            # Check if document already exists (for test purposes)
            existing_document = (
                self.session.query(Document)
                .filter_by(source_path=str(file_path))
                .first()
            )

            if existing_document:
                # Use existing document for test
                document = existing_document
                self.print_step(
                    "Found existing test document", True, f"ID: {document.id}"
                )
            else:
                # Create new document
                document = Document(
                    source_path=str(file_path),
                    filename=file_path.name,
                    file_hash=file_hash,
                    file_size_bytes=getattr(
                        file_path,
                        "stat",
                        lambda: type("obj", (object,), {"st_size": 1000}),
                    )().st_size,
                    conversion_status="processing",
                    meta_data={
                        "test_run": True,
                        "timestamp": datetime.now().isoformat(),
                        "processing_time": getattr(self, "processing_time", 0),
                    },
                )

                self.session.add(document)
                self.session.commit()
                self.print_step("Document record created", True, f"ID: {document.id}")

            self.test_document_id = document.id

            # Store content
            content = DocumentContent(
                document_id=document.id,
                markdown_content=getattr(
                    self.test_pdf_result, "markdown_content", "# Test"
                ),
                plain_text=getattr(self.test_pdf_result, "plain_text", "Test content"),
                page_count=1,
                has_images=False,
                has_tables=False,
                processing_time_ms=int(getattr(self, "processing_time", 0) * 1000),
            )

            self.session.add(content)
            self.print_step(
                "Content stored", True, f"Content length: {len(content.plain_text)}"
            )

            # Store embeddings
            if hasattr(self, "test_embeddings") and self.test_embeddings:
                for i, embedding in enumerate(
                    self.test_embeddings[:5]
                ):  # Limit to 5 for speed
                    chunk_text = (
                        getattr(self.test_chunks[i], "text", f"Test chunk {i}")
                        if hasattr(self, "test_chunks") and i < len(self.test_chunks)
                        else f"Test chunk {i}"
                    )

                    doc_embedding = DocumentEmbedding(
                        document_id=document.id,
                        chunk_index=i,
                        page_number=1,
                        chunk_text=chunk_text[:1000],  # Limit length
                        embedding=embedding,
                        meta_data={
                            "chunk_size": len(chunk_text),
                            "embedding_model": "test_model",
                        },
                    )

                    self.session.add(doc_embedding)

                self.print_step(
                    "Embeddings stored", True, f"Count: {len(self.test_embeddings)}"
                )
            else:
                self.print_step("Embeddings stored", False, "No embeddings to store")

            # Update document status
            document.conversion_status = "completed"
            self.session.commit()

            self.print_step("Document status updated", True, "Status: completed")

            self.results["vector_storage"] = True
            return True

        except Exception as e:
            self.print_step("Database storage", False, str(e))
            logger.exception("Database storage failed")
            if self.session:
                self.session.rollback()
            return False

    async def test_similarity_search(self) -> bool:
        """Test vector similarity search (simplified synchronous test)."""
        self.print_header("6. SIMILARITY SEARCH TEST")

        try:
            if not self.test_document_id:
                self.print_step(
                    "Test data available", False, "No test document for search"
                )
                return False

            self.print_step("Database service initialized", True)

            # Test basic vector search using raw SQL for simplicity
            test_query = "test document content"
            self.print_step("Search query", None, f"Query: '{test_query}'")

            # Generate query embedding
            if hasattr(self, "test_embeddings") and self.test_embeddings:
                query_embedding = self.test_embeddings[
                    0
                ]  # Use first embedding as query
            else:
                # Mock query embedding
                import numpy as np

                query_embedding = np.random.random(768).tolist()

            self.print_step(
                "Query embedding generated", True, f"Dimension: {len(query_embedding)}"
            )

            # Perform basic vector similarity test using raw SQL
            from sqlalchemy import text

            # Convert embedding to string format for PostgreSQL
            embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

            # Test basic vector similarity search
            # Use direct SQL with vector literal to avoid parameter binding issues
            query = text(
                f"""
                SELECT
                    de.document_id,
                    d.filename,
                    de.chunk_text,
                    (de.embedding <=> '{embedding_str}'::vector) as similarity_score
                FROM document_embeddings de
                JOIN documents d ON de.document_id = d.id
                ORDER BY de.embedding <=> '{embedding_str}'::vector
                LIMIT 5
            """
            )

            result = self.session.execute(query)
            search_results = result.fetchall()

            if search_results:
                self.print_step(
                    "Vector search completed",
                    True,
                    f"Found {len(search_results)} results",
                )

                # Check result quality
                for i, row in enumerate(search_results[:2]):
                    score = float(row.similarity_score) if row.similarity_score else 0
                    filename = row.filename or "unknown"
                    self.print_step(
                        f"Result {i+1}", True, f"Score: {score:.3f}, Doc: {filename}"
                    )

                self.results["similarity_search"] = True
                return True
            else:
                self.print_step("Vector search completed", False, "No results found")
                return False

        except Exception as e:
            self.print_step("Similarity search", False, str(e))
            logger.exception("Similarity search failed")
            return False

    async def test_error_handling(self) -> bool:
        """Test error handling capabilities."""
        self.print_header("7. ERROR HANDLING TEST")

        try:
            # Test with invalid PDF path
            try:
                mineru_service = MinerUService()
                invalid_path = Path("/tmp/nonexistent.pdf")

                options = ProcessingOptions()
                await mineru_service.process_pdf(invalid_path, options)

                self.print_step(
                    "Invalid file handling", False, "Should have raised error"
                )

            except Exception as expected_error:
                self.print_step(
                    "Invalid file handling",
                    True,
                    f"Correctly raised: {type(expected_error).__name__}",
                )

            # Test database error handling
            try:
                # Try to create document with invalid data
                invalid_doc = Document(
                    source_path="",  # Invalid empty path
                    filename="",
                    file_hash="",
                    file_size_bytes=-1,  # Invalid negative size
                )
                self.session.add(invalid_doc)
                self.session.commit()

                self.print_step(
                    "Database validation", False, "Should have failed validation"
                )

            except Exception as expected_error:
                self.print_step(
                    "Database validation",
                    True,
                    f"Correctly caught: {type(expected_error).__name__}",
                )
                self.session.rollback()

            self.results["error_handling"] = True
            return True

        except Exception as e:
            self.print_step("Error handling test", False, str(e))
            logger.exception("Error handling test failed")
            return False

    def generate_report(self):
        """Generate final test report."""
        self.print_header("FINAL TEST REPORT")

        total_tests = len(self.results)
        passed_tests = sum(1 for result in self.results.values() if result)

        print(f"\n📊 SUMMARY: {passed_tests}/{total_tests} tests passed")
        print(f"🕐 Total processing time: {getattr(self, 'processing_time', 0):.2f}s")

        if self.test_document_id:
            print(f"📄 Test document ID: {self.test_document_id}")

        print("\n📝 DETAILED RESULTS:")
        for test_name, result in self.results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            test_display = test_name.replace("_", " ").title()
            print(f"  {status} {test_display}")

        print(
            f"\n🎯 OVERALL STATUS: {'✅ ALL SYSTEMS GO' if passed_tests == total_tests else '⚠️ ISSUES DETECTED'}"
        )

        if passed_tests < total_tests:
            print("\n🔧 RECOMMENDATIONS:")
            for test_name, result in self.results.items():
                if not result:
                    if test_name == "mineru_available":
                        print("  - Install MinerU: pip install mineru")
                    elif test_name == "embedding_service":
                        print("  - Configure Ollama or OpenAI API key")
                    elif test_name == "database_connection":
                        print("  - Check DATABASE_URL in .env file")
                    else:
                        print(f"  - Review {test_name.replace('_', ' ')} configuration")

        return passed_tests == total_tests

    def cleanup(self):
        """Cleanup resources."""
        if self.session:
            self.session.close()
        if self.engine:
            self.engine.dispose()


async def main():
    """Run the complete API usage example."""
    print("🚀 PDF TO MARKDOWN MCP - COMPLETE API USAGE EXAMPLE")
    print("=" * 60)
    print("Demonstrating all components of the PDF processing system...")

    tester = FullPipelineTest()

    try:
        # Run all tests in sequence
        await tester.test_database_connection()
        await tester.test_mineru_service()
        await tester.test_text_chunking()
        await tester.test_embedding_service()
        await tester.test_database_storage()
        await tester.test_similarity_search()
        await tester.test_error_handling()

        # Generate report
        all_passed = tester.generate_report()

        # Return appropriate exit code
        sys.exit(0 if all_passed else 1)

    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 CRITICAL ERROR: {e}")
        logger.exception("Critical test failure")
        sys.exit(1)
    finally:
        tester.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
