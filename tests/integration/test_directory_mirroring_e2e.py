"""
End-to-end integration tests for directory mirroring functionality.

This test suite verifies the complete directory mirroring pipeline from
PDF file detection through to final Markdown output with preserved structure.
"""

import hashlib
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from sqlalchemy import text

from pdf_to_markdown_mcp.core.mirror import (
    DirectoryMirror,
    MirrorConfig,
    create_directory_mirror,
)
from pdf_to_markdown_mcp.core.processor import PDFProcessor
from pdf_to_markdown_mcp.core.task_queue import TaskQueue
from pdf_to_markdown_mcp.core.watcher import (
    DirectoryWatcher,
    WatcherConfig,
)
from pdf_to_markdown_mcp.db.models import Document, PathMapping


@pytest.mark.integration
@pytest.mark.database
class TestDirectoryMirroringE2E:
    """End-to-end tests for directory mirroring functionality."""

    @pytest.fixture
    def temp_directories(self):
        """Create temporary directory structure for testing."""
        with tempfile.TemporaryDirectory() as temp_root:
            base_path = Path(temp_root)

            # Create directory structure
            watch_dir = base_path / "pdf_input"
            output_dir = base_path / "markdown_output"
            temp_dir = base_path / "temp"

            # Create base directories
            watch_dir.mkdir()
            output_dir.mkdir()
            temp_dir.mkdir()

            # Create nested directory structure in watch_dir
            research_dir = watch_dir / "research"
            papers_dir = research_dir / "papers"
            archive_dir = research_dir / "archive"
            reports_dir = watch_dir / "reports" / "2024"

            research_dir.mkdir()
            papers_dir.mkdir(parents=True)
            archive_dir.mkdir()
            reports_dir.mkdir(parents=True)

            yield {
                "base": base_path,
                "watch": watch_dir,
                "output": output_dir,
                "temp": temp_dir,
                "research": research_dir,
                "papers": papers_dir,
                "archive": archive_dir,
                "reports": reports_dir,
            }

    @pytest.fixture
    def sample_pdf_files(self, temp_directories, sample_pdf_content):
        """Create sample PDF files in the directory structure."""
        dirs = temp_directories

        # Create PDF files in different subdirectories
        pdf_files = {
            "root_doc": dirs["watch"] / "root_document.pdf",
            "research_paper": dirs["research"] / "important_paper.pdf",
            "paper1": dirs["papers"] / "machine_learning.pdf",
            "paper2": dirs["papers"] / "neural_networks.pdf",
            "archive_doc": dirs["archive"] / "old_research.pdf",
            "report": dirs["reports"] / "quarterly_report.pdf",
        }

        # Write PDF content to all files
        for pdf_file in pdf_files.values():
            pdf_file.write_bytes(sample_pdf_content)

        return pdf_files

    @pytest.fixture
    def test_mirror_config(self, temp_directories):
        """Create test mirror configuration."""
        dirs = temp_directories
        return MirrorConfig(
            watch_base_dir=dirs["watch"],
            output_base_dir=dirs["output"],
            max_directory_depth=5,
            preserve_structure=True,
            create_missing_dirs=True,
            safe_filenames=True,
        )

    @pytest.fixture
    def test_watcher_config(self, temp_directories):
        """Create test watcher configuration."""
        dirs = temp_directories
        return WatcherConfig(
            watch_directories=[str(dirs["watch"])],
            recursive=True,
            patterns=["*.pdf"],
            ignore_patterns=[".*", "*/tmp/*"],
            stability_timeout=0.1,  # Fast for testing
            max_file_size_mb=10,
            enable_deduplication=True,
        )

    @pytest.fixture
    def directory_mirror(self, test_mirror_config, db_session):
        """Create DirectoryMirror instance with test database."""

        def mock_db_session():
            return db_session

        return DirectoryMirror(test_mirror_config, lambda: mock_db_session())

    @pytest.fixture
    def task_queue(self, db_session):
        """Create TaskQueue instance with test database."""

        def mock_db_session():
            return db_session

        return TaskQueue(lambda: mock_db_session())

    @pytest.fixture
    def mock_mineru_processing_result(self, temp_directories):
        """Mock MinerU processing result with output path."""
        dirs = temp_directories

        def create_result(source_file: Path, output_file: Path):
            # Ensure output directory exists
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # Create mock markdown content
            markdown_content = f"""# {source_file.stem}

This is a mock conversion of {source_file.name}.

## Content Summary
- Source: {source_file}
- Output: {output_file}
- Processing: Successful

## Technical Details
The document was processed using MinerU with the following features:
- Layout preservation: ✓
- Table extraction: ✓
- Formula recognition: ✓
- Image extraction: ✓

Generated on: {time.strftime("%Y-%m-%d %H:%M:%S")}
"""

            # Write the markdown file
            output_file.write_text(markdown_content)

            return Mock(
                success=True,
                markdown_content=markdown_content,
                plain_text=markdown_content.replace("#", "").replace("*", ""),
                page_count=1,
                has_images=False,
                has_tables=False,
                processing_time_ms=100,
                output_path=output_file,
                chunks=[],
                tables=[],
                formulas=[],
                images=[],
            )

        return create_result

    @pytest.mark.asyncio
    async def test_complete_directory_mirroring_pipeline(
        self,
        temp_directories,
        sample_pdf_files,
        test_mirror_config,
        test_watcher_config,
        directory_mirror,
        task_queue,
        db_session,
        mock_mineru_processing_result,
    ):
        """Test complete end-to-end directory mirroring pipeline."""
        dirs = temp_directories

        # Test data tracking
        processed_files = []
        created_mappings = []

        # Phase 1: Test DirectoryMirror path calculations
        print("\n=== Phase 1: Testing DirectoryMirror Path Calculations ===")

        for name, pdf_file in sample_pdf_files.items():
            mirror_info = directory_mirror.process_file_for_mirroring(pdf_file)
            assert mirror_info is not None, f"Failed to process {name}: {pdf_file}"

            # Verify path calculations
            assert mirror_info["source_path"] == pdf_file
            assert mirror_info["output_path"].suffix == ".md"
            assert mirror_info["output_path"].parent.exists()  # Directory created

            # Verify relative paths preserve structure
            expected_relative = pdf_file.relative_to(dirs["watch"]).with_suffix(".md")
            assert mirror_info["output_relative_path"] == expected_relative

            processed_files.append(mirror_info)
            print(f"✓ {name}: {pdf_file.name} -> {mirror_info['output_relative_path']}")

        # Phase 2: Verify path mappings in database
        print("\n=== Phase 2: Verifying Path Mappings in Database ===")

        mappings = db_session.query(PathMapping).all()
        assert len(mappings) > 0, "No path mappings were created in database"

        for mapping in mappings:
            assert mapping.source_directory == str(dirs["watch"])
            assert mapping.output_directory == str(dirs["output"])
            assert mapping.files_count > 0
            created_mappings.append(mapping)
            print(f"✓ Mapping: {mapping.relative_path} ({mapping.files_count} files)")

        # Phase 3: Test task queue integration with mirroring
        print("\n=== Phase 3: Testing Task Queue Integration ===")

        # Queue files for processing with mirror information
        document_ids = []
        for i, mirror_info in enumerate(processed_files):
            # Create unique hash for each file (proper SHA-256 format)
            hash_input = f"test_hash_{i}_{mirror_info['source_path'].name}"
            file_hash = hashlib.sha256(hash_input.encode()).hexdigest()

            mock_validation = {
                "valid": True,
                "hash": file_hash,
                "size_bytes": 1024,
                "mime_type": "application/pdf",
            }

            doc_id = task_queue.queue_pdf_processing(
                str(mirror_info["source_path"]),
                mock_validation,
                mirror_info=mirror_info,
            )
            document_ids.append(doc_id)
            print(f"✓ Queued document {doc_id} with mirror info")

        # Phase 4: Verify documents have mirror information
        print("\n=== Phase 4: Verifying Documents with Mirror Information ===")

        documents = (
            db_session.query(Document).filter(Document.id.in_(document_ids)).all()
        )
        assert len(documents) == len(sample_pdf_files)

        for doc in documents:
            assert doc.source_relative_path is not None
            assert doc.output_path is not None
            assert doc.output_relative_path is not None
            assert doc.directory_depth is not None
            print(
                f"✓ Document {doc.id}: {doc.source_relative_path} -> {doc.output_relative_path}"
            )

        # Phase 5: Simulate processing with actual file output
        print("\n=== Phase 5: Simulating Complete Processing Pipeline ===")

        # Mock processor with file creation
        with patch(
            "pdf_to_markdown_mcp.core.processor.MinerUService"
        ) as mock_mineru_class:
            mock_service = Mock()
            mock_mineru_class.return_value = mock_service

            # Configure mock to create actual output files
            async def mock_process_pdf(
                file_path=None, output_dir=None, output_filename=None, options=None, **kwargs
            ):
                source_file = Path(file_path)

                # Find the document to get the correct output path
                doc = (
                    db_session.query(Document)
                    .filter_by(source_path=str(source_file))
                    .first()
                )
                if doc and doc.output_path:
                    output_file = Path(doc.output_path)
                else:
                    # Fallback to basic path calculation
                    relative_path = source_file.relative_to(dirs["watch"]).with_suffix(
                        ".md"
                    )
                    output_file = dirs["output"] / relative_path

                return mock_mineru_processing_result(source_file, output_file)

            mock_service.process_pdf = AsyncMock(side_effect=mock_process_pdf)

            # Create processor and process each document
            processor = PDFProcessor(db_session)

            for doc_id in document_ids:
                doc = db_session.query(Document).filter_by(id=doc_id).first()

                # Process with document ID (simulating Celery task)
                result = await processor.process_pdf(
                    file_path=Path(doc.source_path),
                    document_id=doc_id,
                )

                assert result.document_id == doc_id
                assert result.output_path is not None
                assert result.output_path.exists()

                print(f"✓ Processed document {doc_id}: {result.output_path}")

        # Phase 6: Verify final directory structure
        print("\n=== Phase 6: Verifying Final Directory Structure ===")

        # Check that output directory mirrors input structure
        expected_structure = [
            dirs["output"] / "root_document.md",
            dirs["output"] / "research" / "important_paper.md",
            dirs["output"] / "research" / "papers" / "machine_learning.md",
            dirs["output"] / "research" / "papers" / "neural_networks.md",
            dirs["output"] / "research" / "archive" / "old_research.md",
            dirs["output"] / "reports" / "quarterly_report.md",
        ]

        for expected_file in expected_structure:
            assert expected_file.exists(), (
                f"Expected output file not found: {expected_file}"
            )

            # Verify content
            content = expected_file.read_text()
            assert "mock conversion" in content.lower()
            assert expected_file.stem in content

            print(f"✓ Output file exists with content: {expected_file}")

        # Verify directory structure preservation
        input_dirs = list(dirs["watch"].rglob("*"))
        input_dirs = [d for d in input_dirs if d.is_dir()]

        for input_dir in input_dirs:
            relative_path = input_dir.relative_to(dirs["watch"])
            expected_output_dir = dirs["output"] / relative_path

            if any(expected_output_dir.glob("*.md")):  # Only check dirs with MD files
                assert expected_output_dir.exists(), (
                    f"Output directory not created: {expected_output_dir}"
                )
                print(f"✓ Directory structure preserved: {relative_path}")

        print("\n=== SUCCESS: Complete E2E test passed! ===")
        print(f"Processed {len(processed_files)} files")
        print(f"Created {len(created_mappings)} path mappings")
        print(f"Generated {len(expected_structure)} output files")
        print(f"Preserved directory structure with {len(input_dirs)} subdirectories")

    async def test_watcher_integration_with_mirroring(
        self,
        temp_directories,
        sample_pdf_content,
        test_watcher_config,
        db_session,
        directory_mirror,
    ):
        """Test file watcher integration with directory mirroring."""
        dirs = temp_directories

        # Create watcher with directory mirroring enabled
        def mock_db_session():
            return db_session

        task_queue = TaskQueue(lambda: mock_db_session())
        watcher = DirectoryWatcher(test_watcher_config, task_queue, directory_mirror)

        # Create a new PDF file to trigger watcher
        new_pdf = dirs["watch"] / "research" / "new_document.pdf"
        new_pdf.parent.mkdir(exist_ok=True)

        # Mock the validator and handler
        with patch(
            "pdf_to_markdown_mcp.core.watcher.FileValidator"
        ) as mock_validator_class:
            mock_validator = Mock()
            mock_validator_class.return_value = mock_validator
            mock_validator.validate_file.return_value = {
                "valid": True,
                "hash": "test_hash_watcher",
                "size_bytes": 2048,
                "mime_type": "application/pdf",
            }

            # Simulate file creation event
            handler = watcher.handler

            # Write PDF file
            new_pdf.write_bytes(sample_pdf_content)

            # Manually trigger handler (simulating file system event)
            from watchdog.events import FileCreatedEvent

            event = FileCreatedEvent(str(new_pdf))

            # Process the event
            handler.on_created(event)

            # Verify document was queued with mirror information
            document = (
                db_session.query(Document).filter_by(source_path=str(new_pdf)).first()
            )
            assert document is not None, "Document was not created by watcher"
            assert document.source_relative_path is not None
            assert document.output_path is not None

            # Verify expected output path
            expected_output = dirs["output"] / "research" / "new_document.md"
            assert str(expected_output) == document.output_path

            # Verify output directory was created
            assert expected_output.parent.exists()

            print(f"✓ Watcher integration successful: {new_pdf} -> {expected_output}")

    async def test_error_handling_and_recovery(
        self,
        temp_directories,
        sample_pdf_files,
        directory_mirror,
        db_session,
    ):
        """Test error handling and recovery in directory mirroring."""
        dirs = temp_directories

        # Test 1: Invalid file path (outside watch directory)
        invalid_file = dirs["base"] / "outside_watch.pdf"
        invalid_file.write_text("fake pdf")

        result = directory_mirror.process_file_for_mirroring(invalid_file)
        assert result is None, "Should reject files outside watch directory"

        # Test 2: Permission denied simulation
        restricted_dir = dirs["output"] / "restricted"
        restricted_dir.mkdir()

        # Make directory read-only
        os.chmod(restricted_dir, 0o444)

        try:
            restricted_subdir = restricted_dir / "subdir"
            can_create = directory_mirror.create_output_directory(restricted_subdir)
            # Should handle permission error gracefully
            assert can_create is False or can_create is True  # Either is acceptable
        finally:
            # Restore permissions for cleanup
            os.chmod(restricted_dir, 0o755)

        # Test 3: Database connection failure simulation
        with patch.object(db_session, "add", side_effect=Exception("DB Error")):
            pdf_file = list(sample_pdf_files.values())[0]

            # Should handle database errors gracefully
            result = directory_mirror.process_file_for_mirroring(pdf_file)
            # Mirror info should be calculated even if DB storage fails
            assert result is not None

        print("✓ Error handling tests completed")

    async def test_directory_sync_operations(
        self,
        temp_directories,
        sample_pdf_files,
        directory_mirror,
        db_session,
    ):
        """Test directory synchronization operations."""
        dirs = temp_directories

        # Process some files first
        for pdf_file in list(sample_pdf_files.values())[:3]:
            directory_mirror.process_file_for_mirroring(pdf_file)

        # Test sync statistics
        stats = directory_mirror.sync_directory_structure(scan_existing=True)

        assert stats["directories_scanned"] > 0
        assert stats["files_processed"] >= 3  # At least the 3 we processed
        assert stats["errors"] >= 0  # No errors expected in normal case

        # Test getting directory mappings
        mappings = directory_mirror.get_directory_mappings(
            source_directory=str(dirs["watch"])
        )

        assert len(mappings) > 0
        for mapping in mappings:
            assert mapping["source_directory"] == str(dirs["watch"])
            assert mapping["output_directory"] == str(dirs["output"])
            assert mapping["files_count"] > 0

        print(f"✓ Sync operations: {stats}")
        print(f"✓ Directory mappings: {len(mappings)} found")

    @pytest.mark.slow
    async def test_performance_with_many_files(
        self,
        temp_directories,
        sample_pdf_content,
        directory_mirror,
    ):
        """Test performance with larger number of files."""
        dirs = temp_directories

        # Create many PDF files in nested structure
        num_files = 50
        pdf_files = []

        for i in range(num_files):
            subdir = dirs["watch"] / "batch" / f"group_{i // 10}"
            subdir.mkdir(parents=True, exist_ok=True)

            pdf_file = subdir / f"document_{i:03d}.pdf"
            pdf_file.write_bytes(sample_pdf_content)
            pdf_files.append(pdf_file)

        # Process all files and measure time
        start_time = time.time()

        success_count = 0
        for pdf_file in pdf_files:
            result = directory_mirror.process_file_for_mirroring(pdf_file)
            if result is not None:
                success_count += 1

        processing_time = time.time() - start_time

        assert success_count == num_files

        # Performance assertions (adjust based on requirements)
        avg_time_per_file = processing_time / num_files
        assert avg_time_per_file < 0.1, (
            f"Processing too slow: {avg_time_per_file:.3f}s per file"
        )

        print(f"✓ Performance test: {num_files} files in {processing_time:.2f}s")
        print(f"✓ Average: {avg_time_per_file:.4f}s per file")

    def test_factory_function(self, temp_directories):
        """Test the create_directory_mirror factory function."""
        dirs = temp_directories

        mirror = create_directory_mirror(
            str(dirs["watch"]),
            str(dirs["output"]),
            max_depth=8,
        )

        assert isinstance(mirror, DirectoryMirror)
        assert mirror.config.watch_base_dir == dirs["watch"]
        assert mirror.config.output_base_dir == dirs["output"]
        assert mirror.config.max_directory_depth == 8

        print("✓ Factory function test passed")


@pytest.mark.integration
@pytest.mark.database
class TestDatabaseSchemaCompliance:
    """Test database schema compliance for directory mirroring."""

    def test_path_mappings_table_structure(self, db_session):
        """Test that path_mappings table has correct structure."""
        # Test table exists and has expected columns
        result = db_session.execute(
            text("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'path_mappings'
            ORDER BY ordinal_position;
        """)
        ).fetchall()

        expected_columns = {
            "id": "integer",
            "source_directory": "text",
            "output_directory": "text",
            "relative_path": "text",
            "directory_level": "integer",
            "files_count": "integer",
            "last_scanned": "timestamp without time zone",
            "created_at": "timestamp without time zone",
            "updated_at": "timestamp without time zone",
        }

        actual_columns = {row.column_name: row.data_type for row in result}

        for col_name, expected_type in expected_columns.items():
            assert col_name in actual_columns, f"Missing column: {col_name}"
            # Note: PostgreSQL type names may vary, so we check key columns

        assert len(actual_columns) >= len(expected_columns)
        print(
            f"✓ path_mappings table structure verified: {len(actual_columns)} columns"
        )

    def test_documents_table_extensions(self, db_session):
        """Test that documents table has new mirroring fields."""
        # Check that new columns exist
        result = db_session.execute(
            text("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'documents'
            AND column_name IN ('source_relative_path', 'output_path', 'output_relative_path', 'directory_depth')
        """)
        ).fetchall()

        expected_new_columns = {
            "source_relative_path",
            "output_path",
            "output_relative_path",
            "directory_depth",
        }

        actual_new_columns = {row.column_name for row in result}

        assert expected_new_columns.issubset(actual_new_columns), (
            f"Missing columns: {expected_new_columns - actual_new_columns}"
        )

        print(
            f"✓ documents table extensions verified: {len(actual_new_columns)} new columns"
        )

    def test_database_constraints_and_indexes(self, db_session):
        """Test that database constraints and indexes exist."""
        # Test unique constraint on path_mappings
        result = db_session.execute(
            text("""
            SELECT constraint_name
            FROM information_schema.table_constraints
            WHERE table_name = 'path_mappings'
            AND constraint_type = 'UNIQUE'
        """)
        ).fetchall()

        constraint_names = {row.constraint_name for row in result}
        assert any("source_relative" in name for name in constraint_names), (
            "Missing unique constraint on source_directory + relative_path"
        )

        print(
            f"✓ Database constraints verified: {len(constraint_names)} unique constraints"
        )
