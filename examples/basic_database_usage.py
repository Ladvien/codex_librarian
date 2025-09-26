#!/usr/bin/env python3
"""
Basic Database Usage Example - Simple PDF Database Operations

This example demonstrates basic database operations with the PDF to Markdown MCP:
- Database connection setup
- Document record creation
- Content storage and retrieval
- Basic statistics and querying

A simple introduction to the database layer without complex processing.
"""

import os
import sys
import logging
import hashlib
import time
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment
from dotenv import load_dotenv

load_dotenv()

# Import database stuff
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from pdf_to_markdown_mcp.db.models import Document, DocumentContent


def test_database_connection():
    """Test database connection."""
    print("\n=== Testing Database Connection ===")

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL not set in .env")
        return None

    try:
        engine = create_engine(database_url)

        # Test connection
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version()"))
            version = result.fetchone()[0]
            print(f"✓ Connected to PostgreSQL")
            print(f"  Version: {version}")

            # Check pgvector
            result = conn.execute(
                text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            )
            vector_version = result.fetchone()
            if vector_version:
                print(f"✓ PGVector extension: v{vector_version[0]}")
            else:
                print("✗ PGVector extension not found")

        return engine
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return None


def list_pdfs(directory="/mnt/codex_fs/research/", limit=5):
    """List PDFs in the research directory."""
    print(f"\n=== Listing PDFs in {directory} ===")

    pdf_files = []
    path = Path(directory)

    if not path.exists():
        print(f"Directory not found: {directory}")
        return []

    # Find PDFs
    for pdf in path.rglob("*.pdf"):
        if not pdf.name.startswith("."):
            pdf_files.append(pdf)
            if len(pdf_files) >= limit:
                break

    print(f"Found {len(pdf_files)} PDFs (showing first {limit}):")
    for i, pdf in enumerate(pdf_files, 1):
        size_mb = pdf.stat().st_size / (1024 * 1024)
        print(f"  {i}. {pdf.name} ({size_mb:.2f} MB)")
        print(f"     Path: {pdf}")

    return pdf_files


def process_simple_pdf(pdf_path, engine):
    """Simple PDF processing test."""
    print(f"\n=== Processing PDF ===")
    print(f"File: {pdf_path.name}")

    try:
        # Calculate file hash
        sha256 = hashlib.sha256()
        with open(pdf_path, "rb") as f:
            while chunk := f.read(65536):
                sha256.update(chunk)
        file_hash = sha256.hexdigest()

        print(f"File hash: {file_hash[:16]}...")
        print(f"File size: {pdf_path.stat().st_size / (1024*1024):.2f} MB")

        # Create database session
        SessionLocal = sessionmaker(bind=engine)
        session = SessionLocal()

        # Check if already processed
        existing = session.query(Document).filter_by(file_hash=file_hash).first()
        if existing:
            print(f"✓ Document already in database (ID: {existing.id})")
            print(f"  Status: {existing.conversion_status}")
            print(f"  Created: {existing.created_at}")
            session.close()
            return existing.id

        # Create new document record
        document = Document(
            source_path=str(pdf_path),
            filename=pdf_path.name,
            file_hash=file_hash,
            file_size_bytes=pdf_path.stat().st_size,
            conversion_status="pending",
            meta_data={"test_run": True, "timestamp": datetime.now().isoformat()},
        )

        session.add(document)
        session.commit()

        print(f"✓ Created document record (ID: {document.id})")

        # For now, just create a placeholder content
        # Real processing would use MinerU here
        content = DocumentContent(
            document_id=document.id,
            markdown_content="# Test Document\n\nThis is a placeholder for actual content.",
            plain_text="Test Document\nThis is a placeholder for actual content.",
            page_count=1,
            has_images=False,
            has_tables=False,
            processing_time_ms=100,
        )

        session.add(content)

        # Update status
        document.conversion_status = "completed"
        session.commit()

        print(f"✓ Document processed successfully")

        # Get the ID before closing session
        doc_id = document.id
        session.close()
        return doc_id

    except Exception as e:
        print(f"✗ Processing failed: {e}")
        logger.exception("Processing error")
        return None


def show_database_stats(engine):
    """Show database statistics."""
    print("\n=== Database Statistics ===")

    try:
        with engine.connect() as conn:
            # Count documents
            result = conn.execute(text("SELECT COUNT(*) FROM documents"))
            total_docs = result.fetchone()[0]

            # Count by status
            result = conn.execute(
                text(
                    """
                SELECT conversion_status, COUNT(*)
                FROM documents
                GROUP BY conversion_status
            """
                )
            )
            status_counts = {row[0]: row[1] for row in result}

            # Count embeddings
            result = conn.execute(text("SELECT COUNT(*) FROM document_embeddings"))
            total_embeddings = result.fetchone()[0]

            print(f"Total documents: {total_docs}")
            for status, count in status_counts.items():
                print(f"  {status}: {count}")
            print(f"Total embeddings: {total_embeddings}")

    except Exception as e:
        print(f"Failed to get stats: {e}")


def main():
    """Main example function."""
    print("=" * 50)
    print("PDF to Markdown - Basic Database Usage Example")
    print("=" * 50)

    # Test database
    engine = test_database_connection()
    if not engine:
        print("\nCannot proceed without database connection")
        return

    # List PDFs
    pdf_files = list_pdfs(limit=3)

    if pdf_files:
        # Process first PDF as a test
        print("\n" + "=" * 50)
        print("Testing PDF processing with first file...")
        print("=" * 50)

        document_id = process_simple_pdf(pdf_files[0], engine)

        if document_id:
            print(f"\n✓ Test successful! Document ID: {document_id}")

    # Show stats
    show_database_stats(engine)

    # Cleanup
    engine.dispose()
    print("\n✓ Test complete!")


if __name__ == "__main__":
    main()
