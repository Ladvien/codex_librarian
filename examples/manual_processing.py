#!/usr/bin/env python3
"""
Manual PDF Processing Example - Interactive PDF Processing Demonstration

This example shows how to process PDFs manually without Celery workers,
demonstrating:
- Interactive PDF selection from research directory
- Direct service integration
- Rich CLI interface for processing progress
- Manual control over each processing step

Perfect for learning the API step-by-step or processing individual files.
"""

import os
import sys
import asyncio
import logging
import hashlib
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)
from rich.prompt import Prompt, Confirm
from rich import print as rprint

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import application modules
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from pdf_to_markdown_mcp.db.models import (
    Base,
    Document,
    DocumentContent,
    DocumentEmbedding,
)
from pdf_to_markdown_mcp.services.mineru import MinerUService

# from pdf_to_markdown_mcp.services.embeddings import EmbeddingService
from pdf_to_markdown_mcp.services.database import VectorDatabaseService
from pdf_to_markdown_mcp.core.chunker import TextChunker
from pdf_to_markdown_mcp.core.processor import PDFProcessor
from pdf_to_markdown_mcp.models.request import ProcessingOptions
from pdf_to_markdown_mcp.config import settings

# Initialize Rich console
console = Console()

# Load environment variables
load_dotenv()


class ManualPDFProcessor:
    """Manual PDF processor for testing and debugging."""

    def __init__(self):
        """Initialize the processor with database connection."""
        self.console = console
        self.db_session = None
        self.engine = None
        self.mineru_service = None
        self.embedding_service = None
        self.database_service = None
        self.processor = None

    def setup_database(self) -> bool:
        """Setup database connection."""
        try:
            database_url = os.getenv("DATABASE_URL")
            if not database_url:
                self.console.print(
                    "[red]ERROR: DATABASE_URL not set in .env file[/red]"
                )
                return False

            self.console.print(f"[cyan]Connecting to database...[/cyan]")
            self.engine = create_engine(database_url)

            # Test connection
            with self.engine.connect() as conn:
                result = conn.execute("SELECT 1")
                result.fetchone()

            SessionLocal = sessionmaker(bind=self.engine)
            self.db_session = SessionLocal()

            self.console.print("[green]✓ Database connected successfully[/green]")
            return True

        except Exception as e:
            self.console.print(f"[red]Database connection failed: {e}[/red]")
            return False

    def initialize_services(self) -> bool:
        """Initialize processing services."""
        try:
            self.console.print("[cyan]Initializing services...[/cyan]")

            # Initialize MinerU service
            self.mineru_service = MinerUService()
            self.console.print("[green]✓ MinerU service initialized[/green]")

            # Initialize embedding service
            self.embedding_service = EmbeddingService()
            self.console.print("[green]✓ Embedding service initialized[/green]")

            # Initialize database service
            self.database_service = VectorDatabaseService(self.db_session)
            self.console.print("[green]✓ Database service initialized[/green]")

            # Initialize PDF processor
            self.processor = PDFProcessor(self.db_session)
            self.console.print("[green]✓ PDF processor initialized[/green]")

            return True

        except Exception as e:
            self.console.print(f"[red]Service initialization failed: {e}[/red]")
            return False

    def list_pdfs(self, directory: Path = None) -> List[Path]:
        """List all PDFs in the research directory."""
        if directory is None:
            directory = Path("/mnt/codex_fs/research/")

        pdf_files = []

        # Recursively find all PDFs
        for pdf_path in directory.rglob("*.pdf"):
            # Skip hidden files (starting with .)
            if not pdf_path.name.startswith("."):
                pdf_files.append(pdf_path)

        return sorted(pdf_files)

    def display_pdfs(self, pdf_files: List[Path]):
        """Display PDFs in a nice table."""
        table = Table(title="Available PDFs", show_lines=True)
        table.add_column("#", style="cyan", justify="right")
        table.add_column("Filename", style="green")
        table.add_column("Size", style="yellow", justify="right")
        table.add_column("Path", style="blue")

        for idx, pdf_path in enumerate(pdf_files, 1):
            size_mb = pdf_path.stat().st_size / (1024 * 1024)
            table.add_row(
                str(idx),
                pdf_path.name,
                f"{size_mb:.2f} MB",
                str(pdf_path.parent.relative_to("/mnt/codex_fs/research/")),
            )

        self.console.print(table)

    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                sha256.update(chunk)
        return sha256.hexdigest()

    async def process_single_pdf(
        self, pdf_path: Path, options: ProcessingOptions = None
    ) -> bool:
        """Process a single PDF file."""
        try:
            self.console.print(f"\n[cyan]Processing: {pdf_path.name}[/cyan]")

            # Check if already processed
            file_hash = self.calculate_file_hash(pdf_path)
            existing = (
                self.db_session.query(Document).filter_by(file_hash=file_hash).first()
            )

            if existing:
                self.console.print(
                    f"[yellow]⚠ File already processed (ID: {existing.id})[/yellow]"
                )
                if not Confirm.ask("Process anyway?"):
                    return False

            # Default processing options
            if options is None:
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

            start_time = time.time()

            # Progress tracking
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=self.console,
            ) as progress:

                # Main processing task
                task = progress.add_task("Processing PDF...", total=100)

                # Step 1: Extract content with MinerU
                progress.update(
                    task, advance=10, description="Extracting content with MinerU..."
                )

                # Synchronous wrapper for async processing
                loop = asyncio.get_event_loop()
                result = await self.mineru_service.process_pdf_async(
                    file_path=pdf_path,
                    options=options,
                    progress_callback=lambda current, total, msg: progress.update(
                        task, completed=current, total=total, description=msg
                    ),
                )

                progress.update(
                    task, advance=20, description="Extracted content successfully"
                )

                # Step 2: Create database record
                progress.update(
                    task, advance=10, description="Creating database record..."
                )

                document = Document(
                    source_path=str(pdf_path),
                    filename=pdf_path.name,
                    file_hash=file_hash,
                    file_size_bytes=pdf_path.stat().st_size,
                    conversion_status="processing",
                    meta_data={
                        "pages": result.processing_metadata.page_count,
                        "has_tables": len(result.tables) > 0,
                        "has_formulas": len(result.formulas) > 0,
                        "has_images": len(result.images) > 0,
                        "processing_time_ms": result.processing_metadata.processing_time_ms,
                    },
                )
                self.db_session.add(document)
                self.db_session.flush()

                progress.update(
                    task, advance=10, description=f"Document ID: {document.id}"
                )

                # Step 3: Store content
                progress.update(task, advance=10, description="Storing content...")

                content = DocumentContent(
                    document_id=document.id,
                    markdown_text=result.markdown_content,
                    plain_text=result.plain_text,
                    extracted_tables=(
                        {"tables": [t.dict() for t in result.tables]}
                        if result.tables
                        else None
                    ),
                    extracted_formulas=(
                        {"formulas": [f.dict() for f in result.formulas]}
                        if result.formulas
                        else None
                    ),
                )
                self.db_session.add(content)

                # Step 4: Generate and store embeddings
                if options.chunk_for_embeddings and result.chunks:
                    progress.update(
                        task, advance=10, description="Generating embeddings..."
                    )

                    total_chunks = len(result.chunks)
                    for idx, chunk in enumerate(result.chunks):
                        # Generate embedding
                        embedding_vector = (
                            await self.embedding_service.generate_embedding(chunk.text)
                        )

                        # Store embedding
                        embedding = DocumentEmbedding(
                            document_id=document.id,
                            chunk_index=idx,
                            page_number=chunk.page_number,
                            chunk_text=chunk.text,
                            embedding=embedding_vector,
                            metadata={
                                "start_char": chunk.start_char,
                                "end_char": chunk.end_char,
                                "chunk_size": len(chunk.text),
                            },
                        )
                        self.db_session.add(embedding)

                        progress.update(
                            task,
                            advance=30 / total_chunks,
                            description=f"Generated embedding {idx+1}/{total_chunks}",
                        )

                # Step 5: Finalize
                progress.update(task, advance=10, description="Finalizing...")

                document.conversion_status = "completed"
                self.db_session.commit()

                progress.update(
                    task, completed=100, description="✓ Processing complete!"
                )

            # Display results
            processing_time = time.time() - start_time
            self.console.print(
                f"\n[green]✓ Successfully processed {pdf_path.name}[/green]"
            )
            self.console.print(f"  Document ID: {document.id}")
            self.console.print(f"  Pages: {result.processing_metadata.page_count}")
            self.console.print(
                f"  Chunks: {len(result.chunks) if result.chunks else 0}"
            )
            self.console.print(f"  Processing time: {processing_time:.2f} seconds")

            return True

        except Exception as e:
            self.console.print(f"[red]✗ Processing failed: {e}[/red]")
            logger.exception("Processing error")

            # Rollback transaction
            self.db_session.rollback()

            # Update document status if it exists
            if "document" in locals():
                document.conversion_status = "failed"
                document.error_message = str(e)
                self.db_session.commit()

            return False

    async def batch_process(
        self, pdf_files: List[Path], options: ProcessingOptions = None
    ):
        """Process multiple PDFs in batch."""
        self.console.print(f"\n[cyan]Batch processing {len(pdf_files)} files...[/cyan]")

        success_count = 0
        failed_files = []

        for idx, pdf_path in enumerate(pdf_files, 1):
            self.console.print(f"\n[bold]Processing {idx}/{len(pdf_files)}[/bold]")

            success = await self.process_single_pdf(pdf_path, options)

            if success:
                success_count += 1
            else:
                failed_files.append(pdf_path)

        # Display summary
        self.console.print("\n" + "=" * 50)
        self.console.print(f"[bold]Batch Processing Complete[/bold]")
        self.console.print(f"  ✓ Successful: {success_count}")
        self.console.print(f"  ✗ Failed: {len(failed_files)}")

        if failed_files:
            self.console.print("\nFailed files:")
            for f in failed_files:
                self.console.print(f"  - {f.name}")

    def search_documents(self, query: str, limit: int = 10):
        """Search processed documents using vector similarity."""
        try:
            self.console.print(f"\n[cyan]Searching for: '{query}'[/cyan]")

            # Generate query embedding
            query_embedding = asyncio.run(
                self.embedding_service.generate_embedding(query)
            )

            # Perform vector search
            results = self.database_service.vector_search(
                query_embedding=query_embedding, top_k=limit, similarity_threshold=0.7
            )

            if not results:
                self.console.print("[yellow]No results found[/yellow]")
                return

            # Display results
            table = Table(title="Search Results", show_lines=True)
            table.add_column("#", style="cyan", justify="right")
            table.add_column("Document", style="green")
            table.add_column("Score", style="yellow")
            table.add_column("Excerpt", style="white", width=50)

            for idx, result in enumerate(results, 1):
                excerpt = (
                    result.content[:100] + "..."
                    if len(result.content) > 100
                    else result.content
                )
                table.add_row(
                    str(idx), result.filename, f"{result.similarity_score:.3f}", excerpt
                )

            self.console.print(table)

        except Exception as e:
            self.console.print(f"[red]Search failed: {e}[/red]")

    def show_statistics(self):
        """Display database statistics."""
        try:
            # Query statistics
            total_docs = self.db_session.query(Document).count()
            completed_docs = (
                self.db_session.query(Document)
                .filter_by(conversion_status="completed")
                .count()
            )
            failed_docs = (
                self.db_session.query(Document)
                .filter_by(conversion_status="failed")
                .count()
            )
            processing_docs = (
                self.db_session.query(Document)
                .filter_by(conversion_status="processing")
                .count()
            )
            total_embeddings = self.db_session.query(DocumentEmbedding).count()

            # Display statistics
            table = Table(title="Database Statistics", show_header=False)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="yellow", justify="right")

            table.add_row("Total Documents", str(total_docs))
            table.add_row("Completed", str(completed_docs))
            table.add_row("Failed", str(failed_docs))
            table.add_row("Processing", str(processing_docs))
            table.add_row("Total Embeddings", str(total_embeddings))

            if total_docs > 0:
                avg_embeddings = total_embeddings / total_docs
                table.add_row("Avg Embeddings/Doc", f"{avg_embeddings:.1f}")

            self.console.print(table)

        except Exception as e:
            self.console.print(f"[red]Failed to get statistics: {e}[/red]")

    def cleanup(self):
        """Cleanup resources."""
        if self.db_session:
            self.db_session.close()
        if self.engine:
            self.engine.dispose()


async def main():
    """Main entry point."""
    processor = ManualPDFProcessor()

    # Header
    console.print("\n[bold cyan]PDF to Markdown Manual Processor[/bold cyan]")
    console.print("=" * 50)

    # Setup database
    if not processor.setup_database():
        return

    # Initialize services
    if not processor.initialize_services():
        processor.cleanup()
        return

    try:
        while True:
            # Display menu
            console.print("\n[bold]Options:[/bold]")
            console.print("1. List PDFs")
            console.print("2. Process single PDF")
            console.print("3. Batch process directory")
            console.print("4. Search documents")
            console.print("5. Show statistics")
            console.print("6. Exit")

            choice = Prompt.ask("Select option", choices=["1", "2", "3", "4", "5", "6"])

            if choice == "1":
                # List PDFs
                pdf_files = processor.list_pdfs()
                if pdf_files:
                    processor.display_pdfs(pdf_files)
                else:
                    console.print("[yellow]No PDFs found[/yellow]")

            elif choice == "2":
                # Process single PDF
                pdf_files = processor.list_pdfs()
                if not pdf_files:
                    console.print("[yellow]No PDFs found[/yellow]")
                    continue

                processor.display_pdfs(pdf_files)

                idx_str = Prompt.ask("Enter PDF number to process")
                try:
                    idx = int(idx_str) - 1
                    if 0 <= idx < len(pdf_files):
                        await processor.process_single_pdf(pdf_files[idx])
                    else:
                        console.print("[red]Invalid selection[/red]")
                except ValueError:
                    console.print("[red]Invalid number[/red]")

            elif choice == "3":
                # Batch process
                subdirs = [
                    "scientific_articles",
                    "codex_articles",
                    "agentic_research_results",
                ]

                console.print("\n[bold]Select directory:[/bold]")
                for i, subdir in enumerate(subdirs, 1):
                    console.print(f"{i}. /mnt/codex_fs/research/{subdir}")
                console.print(f"{len(subdirs)+1}. All directories")

                dir_choice = Prompt.ask(
                    "Select directory",
                    choices=[str(i) for i in range(1, len(subdirs) + 2)],
                )

                if int(dir_choice) <= len(subdirs):
                    subdir = subdirs[int(dir_choice) - 1]
                    pdf_files = processor.list_pdfs(
                        Path(f"/mnt/codex_fs/research/{subdir}")
                    )
                else:
                    pdf_files = processor.list_pdfs()

                if pdf_files:
                    console.print(f"Found {len(pdf_files)} PDFs")
                    if Confirm.ask("Process all?"):
                        await processor.batch_process(pdf_files)
                else:
                    console.print("[yellow]No PDFs found[/yellow]")

            elif choice == "4":
                # Search documents
                query = Prompt.ask("Enter search query")
                processor.search_documents(query)

            elif choice == "5":
                # Show statistics
                processor.show_statistics()

            elif choice == "6":
                # Exit
                break

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        logger.exception("Unexpected error")
    finally:
        processor.cleanup()
        console.print("\n[cyan]Goodbye![/cyan]")


if __name__ == "__main__":
    asyncio.run(main())
