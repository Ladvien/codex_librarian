#!/usr/bin/env python3
"""
Kitchen Sink Example - Complete PDF to Markdown Pipeline

This comprehensive example demonstrates the ENTIRE PDF-to-Markdown system:
- Watch directories for PDFs (watchdog)
- Process with MinerU (real PDF extraction)
- Mirror directory structure to markdown
- Store in PostgreSQL database
- Generate embeddings with Ollama/OpenAI
- Store vectors in PGVector
- Semantic search over processed content
- Health checks and statistics

Prerequisites:
- PostgreSQL with PGVector extension running
- Ollama running (http://localhost:11434) OR OpenAI API key
- MinerU library installed
- Environment configured in .env file

Usage:
    # Watch and process files continuously:
    python examples/watch_and_mirror.py

    # Process existing files once and exit:
    python examples/watch_and_mirror.py --batch

    # Search processed documents:
    python examples/watch_and_mirror.py --search
"""

import asyncio
import logging
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sqlalchemy.orm import Session

from pdf_to_markdown_mcp.config import Settings
from pdf_to_markdown_mcp.core.mirror import DirectoryMirror, MirrorConfig
from pdf_to_markdown_mcp.core.watcher import DirectoryWatcher, WatcherConfig
from pdf_to_markdown_mcp.db.models import (
    Base,
    Document,
    DocumentContent,
    DocumentEmbedding,
)
from pdf_to_markdown_mcp.db.session import get_db_session, engine
from pdf_to_markdown_mcp.models.request import ProcessingOptions
from pdf_to_markdown_mcp.services.database import VectorDatabaseService
from pdf_to_markdown_mcp.services.embeddings import EmbeddingService
from pdf_to_markdown_mcp.services.mineru import MinerUService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class PDFProcessingPipeline:
    """Complete PDF processing pipeline with all services."""

    def __init__(self, settings: Settings):
        """Initialize all services."""
        from pdf_to_markdown_mcp.services.embeddings import EmbeddingConfig, EmbeddingProvider

        self.settings = settings
        self.mineru = MinerUService()
        self.db_service = VectorDatabaseService()

        embedding_config = EmbeddingConfig(
            provider=EmbeddingProvider(settings.embedding.provider),
            ollama_model=settings.embedding.model,
            ollama_base_url=settings.embedding.ollama_url,
            openai_api_key=settings.embedding.openai_api_key,
            openai_model="text-embedding-3-small",
            embedding_dimensions=settings.embedding.dimensions,
        )
        self.embedding_service = EmbeddingService(config=embedding_config)
        self.processed_count = 0
        self.failed_count = 0
        self.total_pages = 0

        logger.info("✅ MinerU service initialized")
        logger.info("✅ Database service initialized")
        logger.info("✅ Embedding service initialized")

    async def process_pdf_file(
        self, pdf_path: Path, output_path: Path, db_session: Session
    ) -> bool:
        """
        Process a single PDF through the complete pipeline.

        Args:
            pdf_path: Path to PDF file
            output_path: Path for markdown output
            db_session: Database session

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"📄 Processing: {pdf_path.name}")

            options = ProcessingOptions(
                extract_tables=False,
                extract_formulas=False,
                extract_images=False,
                preserve_layout=True,
                ocr_language="eng",
            )

            result = await self.mineru.process_pdf(pdf_path=pdf_path, options=options)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(result.markdown_content, encoding="utf-8")
            logger.info(f"   ✅ Markdown saved: {output_path}")

            doc = self._save_to_database(pdf_path, result, db_session)
            logger.info(f"   ✅ Database record created (ID: {doc.id})")

            await self._generate_and_store_embeddings(doc, result, db_session)
            logger.info(f"   ✅ Embeddings generated and stored")

            self.processed_count += 1
            self.total_pages += result.processing_metadata.pages
            logger.info(
                f"   📊 Pages: {result.processing_metadata.pages}, "
                f"Tables: {result.processing_metadata.tables_found}, "
                f"Formulas: {result.processing_metadata.formulas_found}"
            )

            return True

        except Exception as e:
            logger.error(f"   ❌ Error processing {pdf_path.name}: {e}")
            self.failed_count += 1
            return False

    def _save_to_database(
        self, pdf_path: Path, result: Any, db_session: Session
    ) -> Document:
        """Save document and content to database."""
        existing = (
            db_session.query(Document)
            .filter_by(file_hash=result.processing_metadata.file_hash)
            .first()
        )
        if existing:
            logger.info(f"   ℹ️  Document already exists (ID: {existing.id}), updating")
            doc = existing
            doc.conversion_status = "completed"
            doc.updated_at = datetime.utcnow()
        else:
            doc = Document(
                source_path=str(pdf_path.absolute()),
                filename=pdf_path.name,
                file_hash=result.processing_metadata.file_hash,
                file_size_bytes=result.processing_metadata.file_size_bytes,
                conversion_status="completed",
            )
            db_session.add(doc)
            db_session.flush()

        content = (
            db_session.query(DocumentContent).filter_by(document_id=doc.id).first()
        )
        if content:
            content.markdown_content = result.markdown_content
            content.plain_text = result.plain_text
        else:
            content = DocumentContent(
                document_id=doc.id,
                markdown_content=result.markdown_content,
                plain_text=result.plain_text,
            )
            db_session.add(content)

        db_session.commit()
        return doc

    async def _generate_and_store_embeddings(
        self, doc: Document, result: Any, db_session: Session
    ):
        """Generate embeddings for document chunks and store in PGVector."""
        db_session.query(DocumentEmbedding).filter_by(document_id=doc.id).delete()
        db_session.commit()

        text = result.plain_text
        chunk_size = 1000
        overlap = 200

        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end]

            if chunk_text.strip():
                chunks.append(
                    {
                        "chunk_index": chunk_index,
                        "text": chunk_text,
                        "start_char": start,
                        "end_char": end,
                    }
                )
                chunk_index += 1

            start += chunk_size - overlap

        if not chunks:
            return

        texts = [c["text"] for c in chunks]
        embedding_result = await self.embedding_service.generate_embeddings(texts)
        embeddings = embedding_result.embeddings

        for chunk_data, embedding_vector in zip(chunks, embeddings):
            chunk = DocumentEmbedding(
                document_id=doc.id,
                chunk_index=chunk_data["chunk_index"],
                chunk_text=chunk_data["text"],
                embedding=embedding_vector,
            )
            db_session.add(chunk)

        db_session.commit()
        logger.info(f"   ✅ Created {len(chunks)} chunks with embeddings")


class KitchenSinkProcessor:
    """Task queue adapter for DirectoryWatcher."""

    def __init__(self, pipeline: PDFProcessingPipeline, mirror: DirectoryMirror):
        self.pipeline = pipeline
        self.mirror = mirror

    def queue_pdf_processing(self, file_path: str, metadata: dict[str, Any]):
        """Process PDF file immediately (no actual queue)."""
        pdf_path = Path(file_path)
        mirror_paths = self.mirror.get_mirror_paths(pdf_path)
        output_path = mirror_paths['output_path']

        with get_db_session() as db_session:
            asyncio.run(
                self.pipeline.process_pdf_file(pdf_path, output_path, db_session)
            )


async def search_documents(query: str, pipeline: PDFProcessingPipeline, limit: int = 5):
    """Semantic search over processed documents."""
    logger.info(f"\n🔍 Searching: '{query}'")

    query_embedding = await pipeline.embedding_service.generate_embeddings([query])
    query_vector = query_embedding[0]

    results = await pipeline.db_service.vector_similarity_search(
        query_embedding=query_vector, top_k=limit, similarity_threshold=0.0
    )

    if not results:
        logger.info("   No results found.")
        return

    logger.info(f"\n📊 Found {len(results)} results:\n")

    for i, result in enumerate(results, 1):
        logger.info(f"{i}. {result.filename} (similarity: {result.similarity_score:.3f})")
        content_preview = result.content[:200] if result.content else ""
        logger.info(f"   {content_preview}...")
        logger.info("")


def show_stats(pipeline: PDFProcessingPipeline):
    """Display processing statistics."""
    print("\n" + "=" * 70)
    print("📊 PROCESSING STATISTICS")
    print("=" * 70)
    print(f"✅ Processed: {pipeline.processed_count} documents")
    print(f"❌ Failed: {pipeline.failed_count} documents")
    print(f"📄 Total pages: {pipeline.total_pages}")

    with get_db_session() as db_session:
        total_docs = db_session.query(Document).count()
        completed = (
            db_session.query(Document).filter_by(conversion_status="completed").count()
        )
        total_chunks = db_session.query(DocumentEmbedding).count()

        print(f"💾 Database: {total_docs} documents ({completed} completed)")
        print(f"🔢 Embeddings: {total_chunks} chunks")
    print("=" * 70 + "\n")


def show_health(pipeline: PDFProcessingPipeline):
    """Display service health status."""
    print("\n" + "=" * 70)
    print("🏥 SERVICE HEALTH")
    print("=" * 70)

    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("✅ PostgreSQL: Connected")
    except Exception as e:
        print(f"❌ PostgreSQL: {e}")

    try:
        mineru_ok = pipeline.mineru.validate_mineru_dependency()
        print(f"✅ MinerU: {'Available' if mineru_ok else 'Unavailable'}")
    except Exception as e:
        print(f"❌ MinerU: {e}")

    try:
        embedding_status = (
            "Ollama" if pipeline.settings.embedding.provider == "ollama" else "OpenAI"
        )
        print(f"✅ Embeddings: {embedding_status}")
    except Exception as e:
        print(f"❌ Embeddings: {e}")

    print("=" * 70 + "\n")


async def process_existing_files(
    watch_dir: Path, pipeline: PDFProcessingPipeline, mirror: DirectoryMirror
):
    """Process all existing PDF files in watch directory."""
    pdfs = sorted(watch_dir.rglob("*.pdf"))

    if not pdfs:
        logger.info("   No PDFs found.")
        return

    logger.info(f"   Found {len(pdfs)} PDF files")

    with get_db_session() as db_session:
        for pdf in pdfs:
            pdf_abs = pdf.absolute()
            rel_path = pdf.relative_to(watch_dir)
            mirror_paths = mirror.get_mirror_paths(pdf_abs)
            output_path = mirror_paths['output_path']

            logger.info(f"\n📄 {rel_path}")
            await pipeline.process_pdf_file(pdf, output_path, db_session)


def interactive_search(pipeline: PDFProcessingPipeline):
    """Interactive search mode."""
    print("\n" + "=" * 70)
    print("🔍 SEMANTIC SEARCH MODE")
    print("=" * 70)
    print("Type your search query (or 'quit' to exit)\n")

    while True:
        try:
            query = input("Search> ").strip()

            if query.lower() in ["quit", "exit", "q"]:
                break

            if not query:
                continue

            asyncio.run(search_documents(query, pipeline))

        except (KeyboardInterrupt, EOFError):
            break

    print("\n👋 Exiting search mode\n")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="PDF to Markdown Kitchen Sink Example")
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process existing files once and exit",
    )
    parser.add_argument(
        "--search",
        action="store_true",
        help="Enter interactive search mode",
    )
    parser.add_argument(
        "--watch-dir",
        type=str,
        default="watch_pdf_test",
        help="Directory to watch (default: watch_pdf_test)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="watch_pdf_test_output",
        help="Output directory (default: watch_pdf_test_output)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("🚀 PDF to Markdown - Kitchen Sink Example")
    print("=" * 70)
    print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")

    watch_dir = Path(args.watch_dir)
    output_dir = Path(args.output_dir)

    if not watch_dir.exists():
        logger.error(f"❌ Watch directory not found: {watch_dir}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"📁 Watch directory: {watch_dir}")
    logger.info(f"📁 Output directory: {output_dir}")
    logger.info("")

    try:
        settings = Settings()
        logger.info("✅ Settings loaded")

        Base.metadata.create_all(engine)
        logger.info("✅ Database initialized")

        pipeline = PDFProcessingPipeline(settings)
        logger.info("")

        show_health(pipeline)

        if args.search:
            interactive_search(pipeline)
            return 0

        logger.info("🔍 Scanning for existing PDFs...")
        mirror_config = MirrorConfig(
            watch_base_dir=watch_dir,
            output_base_dir=output_dir,
        )
        mirror = DirectoryMirror(config=mirror_config)

        asyncio.run(process_existing_files(watch_dir, pipeline, mirror))

        show_stats(pipeline)

        if args.batch:
            logger.info("✅ Batch processing complete")
            return 0

        logger.info("👀 Starting file watcher...")
        logger.info("   Press Ctrl+C to stop\n")

        processor = KitchenSinkProcessor(pipeline, mirror)
        config = WatcherConfig(
            watch_directories=[str(watch_dir)],
            recursive=True,
            patterns=["*.pdf", "*.PDF"],
        )
        watcher = DirectoryWatcher(
            task_queue=processor, config=config, directory_mirror=mirror
        )

        stop_event = asyncio.Event()

        def signal_handler(sig, frame):
            logger.info("\n\n🛑 Stopping watcher...")
            stop_event.set()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        watcher.start()

        try:
            while not stop_event.is_set():
                time.sleep(1)
        finally:
            watcher.stop()
            show_stats(pipeline)
            logger.info("✅ Watcher stopped")

        return 0

    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
