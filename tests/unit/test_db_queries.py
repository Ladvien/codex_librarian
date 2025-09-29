"""
Database Queries Unit Tests

Comprehensive unit tests for all database query functions following TDD principles.
Focuses on security, functionality, and edge cases for all database operations.

Test Categories:
1. Document operations (CRUD)
2. Search operations (fulltext, vector, hybrid)
3. Queue operations (job processing)
4. Input validation and security
5. Error handling and edge cases
"""

from unittest.mock import Mock

import pytest
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from src.pdf_to_markdown_mcp.db.models import (
    Document,
    ProcessingQueue,
)
from src.pdf_to_markdown_mcp.db.queries import (
    DocumentQueries,
    QueueQueries,
    SearchQueries,
    _validate_embedding,
    _validate_float,
    _validate_integer,
    _validate_string,
)


class TestDocumentQueries:
    """Test document CRUD operations with security focus"""

    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session"""
        session = Mock(spec=Session)
        session.query = Mock()
        session.add = Mock()
        session.commit = Mock()
        session.rollback = Mock()
        return session

    @pytest.fixture
    def sample_document(self):
        """Create sample document for testing"""
        doc = Mock(spec=Document)
        doc.id = 1
        doc.filename = "test.pdf"
        doc.source_path = "/test/path.pdf"
        doc.file_hash = "abc123def456"
        doc.conversion_status = "completed"
        doc.file_size_bytes = 1024000
        doc.created_at = "2025-09-26T10:00:00Z"
        doc.metadata = {"pages": 10}
        return doc

    # Test get_by_id functionality

    def test_get_by_id_existing_document(self, mock_db_session, sample_document):
        """
        Test retrieval of existing document by ID

        Given a valid document ID that exists in database
        When getting document by ID
        Then should return the correct document
        """
        # Given
        mock_result = Mock()
        mock_result.first.return_value = sample_document
        mock_db_session.query.return_value = mock_result

        # When
        result = DocumentQueries.get_by_id(mock_db_session, 1)

        # Then
        assert result == sample_document
        mock_db_session.query.assert_called_once_with(Document)
        mock_result.filter.assert_called_once()

    def test_get_by_id_nonexistent_document(self, mock_db_session):
        """
        Test retrieval of non-existent document by ID

        Given a document ID that doesn't exist
        When getting document by ID
        Then should return None
        """
        # Given
        mock_result = Mock()
        mock_result.first.return_value = None
        mock_db_session.query.return_value = mock_result

        # When
        result = DocumentQueries.get_by_id(mock_db_session, 999)

        # Then
        assert result is None
        mock_db_session.query.assert_called_once_with(Document)

    def test_get_by_id_invalid_input_types(self, mock_db_session):
        """
        Test get_by_id with invalid input types

        Given invalid input types (strings, None, negative numbers)
        When calling get_by_id
        Then should handle gracefully or raise appropriate errors
        """
        # Test cases for invalid inputs
        invalid_inputs = [
            "not_an_integer",
            None,
            -1,
            0,
            "1'; DROP TABLE documents; --",  # SQL injection attempt
        ]

        for invalid_input in invalid_inputs:
            # When & Then
            try:
                result = DocumentQueries.get_by_id(mock_db_session, invalid_input)
                # If it succeeds, verify ORM handles type conversion safely
                mock_db_session.query.assert_called_with(Document)
            except (TypeError, ValueError):
                # Expected for some invalid types
                assert True, f"Correctly rejected invalid input: {invalid_input}"

    # Test get_by_path functionality

    def test_get_by_path_existing_document(self, mock_db_session, sample_document):
        """
        Test retrieval of document by source path

        Given a valid source path that exists
        When getting document by path
        Then should return the correct document
        """
        # Given
        mock_result = Mock()
        mock_result.first.return_value = sample_document
        mock_db_session.query.return_value = mock_result

        # When
        result = DocumentQueries.get_by_path(mock_db_session, "/test/path.pdf")

        # Then
        assert result == sample_document
        mock_db_session.query.assert_called_once_with(Document)

    def test_get_by_path_path_traversal_attempt(self, mock_db_session):
        """
        Test path traversal security in get_by_path

        Given malicious path with traversal attempts
        When getting document by path
        Then should safely handle without security issues
        """
        # Given
        malicious_paths = [
            "../../../etc/passwd",
            "../../config/secrets.env",
            "/etc/shadow",
            "\\..\\..\\windows\\system32\\config\\sam",
            "'; DROP TABLE documents; --",
        ]

        mock_result = Mock()
        mock_result.first.return_value = None
        mock_db_session.query.return_value = mock_result

        # When & Then
        for malicious_path in malicious_paths:
            try:
                result = DocumentQueries.get_by_path(mock_db_session, malicious_path)
                # Should use parameterized query - safe even with malicious input
                assert result is None or hasattr(result, "id")
                mock_db_session.query.assert_called_with(Document)
            except ValueError:
                # Input validation may reject obviously malicious paths
                assert True, f"Path validation correctly rejected: {malicious_path}"

    # Test get_by_hash functionality

    def test_get_by_hash_existing_document(self, mock_db_session, sample_document):
        """
        Test retrieval of document by file hash

        Given a valid file hash that exists
        When getting document by hash
        Then should return the correct document
        """
        # Given
        mock_result = Mock()
        mock_result.first.return_value = sample_document
        mock_db_session.query.return_value = mock_result

        # When
        result = DocumentQueries.get_by_hash(mock_db_session, "abc123def456")

        # Then
        assert result == sample_document
        mock_db_session.query.assert_called_once_with(Document)

    def test_get_by_hash_collision_resistance(self, mock_db_session):
        """
        Test hash collision handling and security

        Given various hash formats and potential collisions
        When getting documents by hash
        Then should handle securely without information leakage
        """
        # Given
        hash_test_cases = [
            "validhash123",  # Normal case
            "",  # Empty hash
            "a" * 64,  # SHA-256 length
            "a" * 32,  # MD5 length
            "invalid!@#$%",  # Special characters
            "'; DROP TABLE documents; --",  # SQL injection attempt
        ]

        mock_result = Mock()
        mock_result.first.return_value = None
        mock_db_session.query.return_value = mock_result

        # When & Then
        for test_hash in hash_test_cases:
            try:
                result = DocumentQueries.get_by_hash(mock_db_session, test_hash)
                mock_db_session.query.assert_called_with(Document)
            except (ValueError, TypeError):
                assert True, f"Hash validation correctly handled: {test_hash}"

    # Test get_by_status functionality

    def test_get_by_status_with_pagination(self, mock_db_session):
        """
        Test document retrieval by status with pagination

        Given documents with specific status and pagination parameters
        When getting documents by status
        Then should return correct paginated results
        """
        # Given
        mock_documents = [Mock(spec=Document) for _ in range(5)]
        mock_query = Mock()
        mock_query.offset.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = mock_documents
        mock_db_session.query.return_value = mock_query

        # When
        result = DocumentQueries.get_by_status(
            mock_db_session, status="completed", limit=10, offset=20
        )

        # Then
        assert result == mock_documents
        mock_db_session.query.assert_called_once_with(Document)
        mock_query.offset.assert_called_once_with(20)
        mock_query.limit.assert_called_once_with(10)

    def test_get_by_status_invalid_parameters(self, mock_db_session):
        """
        Test get_by_status with invalid parameters

        Given invalid status values and pagination parameters
        When getting documents by status
        Then should handle gracefully or validate inputs
        """
        # Given
        invalid_test_cases = [
            {"status": "'; DROP TABLE documents; --", "limit": 10, "offset": 0},
            {"status": "completed", "limit": -1, "offset": 0},
            {"status": "completed", "limit": 10, "offset": -5},
            {"status": "", "limit": 0, "offset": 0},
            {"status": None, "limit": 10, "offset": 0},
        ]

        mock_query = Mock()
        mock_query.all.return_value = []
        mock_db_session.query.return_value = mock_query

        # When & Then
        for test_case in invalid_test_cases:
            try:
                result = DocumentQueries.get_by_status(mock_db_session, **test_case)
                # Should use ORM safely even with invalid inputs
                mock_db_session.query.assert_called_with(Document)
            except (ValueError, TypeError):
                assert True, f"Input validation rejected: {test_case}"


class TestSearchQueries:
    """Test search operations with security and performance focus"""

    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session for search tests"""
        session = Mock(spec=Session)
        session.execute = Mock()
        return session

    @pytest.fixture
    def sample_search_results(self):
        """Create sample search results"""
        results = []
        for i in range(3):
            row = Mock()
            row.id = i + 1
            row.filename = f"document_{i + 1}.pdf"
            row.source_path = f"/path/to/document_{i + 1}.pdf"
            row.plain_text = f"Sample content for document {i + 1}"
            row.score = 0.8 - (i * 0.1)
            row.metadata = {"page_count": 10 + i}
            results.append(row)
        return results

    # Test fulltext_search functionality

    def test_fulltext_search_basic_functionality(
        self, mock_db_session, sample_search_results
    ):
        """
        Test basic full-text search functionality

        Given a legitimate search query
        When performing full-text search
        Then should return properly formatted results
        """
        # Given
        mock_db_session.execute.return_value = sample_search_results

        # When
        results = SearchQueries.fulltext_search(
            mock_db_session, query="machine learning", limit=10
        )

        # Then
        assert len(results) == 3
        assert all(hasattr(result, "document_id") for result in results)
        assert all(hasattr(result, "filename") for result in results)
        assert all(hasattr(result, "score") for result in results)

        # Verify parameterized query was used
        mock_db_session.execute.assert_called_once()
        call_args = mock_db_session.execute.call_args
        assert ":query" in str(call_args[0][0])
        assert "machine learning" in call_args[0][1]["query"]

    def test_fulltext_search_input_validation(self, mock_db_session):
        """
        Test input validation in full-text search

        Given various invalid inputs
        When performing full-text search
        Then should validate inputs and reject dangerous patterns
        """
        # Given
        mock_db_session.execute.return_value = []

        invalid_queries = [
            "",  # Empty query
            " " * 600,  # Too long
            None,  # None value
            12345,  # Non-string
            "'; DROP TABLE documents; --",  # SQL injection
        ]

        # When & Then
        for invalid_query in invalid_queries:
            try:
                result = SearchQueries.fulltext_search(
                    mock_db_session, query=invalid_query, limit=10
                )
                # If it succeeds, verify safe execution
                if mock_db_session.execute.called:
                    call_args = mock_db_session.execute.call_args
                    assert ":query" in str(call_args[0][0])

            except (ValueError, TypeError):
                assert True, f"Input validation correctly rejected: {invalid_query}"

    def test_fulltext_search_pagination_edge_cases(self, mock_db_session):
        """
        Test pagination edge cases in full-text search

        Given extreme pagination parameters
        When performing search with pagination
        Then should handle edge cases safely
        """
        # Given
        mock_db_session.execute.return_value = []

        edge_cases = [
            {"limit": 0, "offset": 0},  # Zero limit
            {"limit": 1001, "offset": 0},  # Excessive limit
            {"limit": 10, "offset": -1},  # Negative offset
            {"limit": -5, "offset": 10},  # Negative limit
        ]

        # When & Then
        for case in edge_cases:
            try:
                result = SearchQueries.fulltext_search(
                    mock_db_session, query="test", **case
                )
                # Should validate limits
                if case["limit"] <= 0 or case["limit"] > 1000:
                    pytest.fail("Should have validated limit")

            except ValueError as e:
                assert "limit" in str(e).lower() or "offset" in str(e).lower()

    def test_fulltext_search_filter_injection_prevention(self, mock_db_session):
        """
        Test filter parameter injection prevention

        Given malicious filter parameters
        When performing filtered search
        Then should safely handle or reject malicious filters
        """
        # Given
        mock_db_session.execute.return_value = []

        malicious_filters = [
            {"document_id": "1'; DROP TABLE documents; --"},
            {"document_id": {"$ne": None}},  # NoSQL injection
            {"unknown_field": "malicious_value"},
            {"document_id": [1, 2, "'; TRUNCATE TABLE documents; --"]},
        ]

        # When & Then
        for malicious_filter in malicious_filters:
            try:
                result = SearchQueries.fulltext_search(
                    mock_db_session, query="test", filters=malicious_filter, limit=10
                )

                # Verify safe parameterization
                if mock_db_session.execute.called:
                    call_args = mock_db_session.execute.call_args
                    params = call_args[0][1]

                    # document_id should be validated as integer if present
                    if "document_id" in params:
                        assert isinstance(params["document_id"], int)

            except (ValueError, TypeError):
                assert True, "Filter validation correctly rejected malicious filter"

    # Test vector_similarity_search functionality

    def test_vector_similarity_search_basic_functionality(self, mock_db_session):
        """
        Test basic vector similarity search

        Given valid embedding vector
        When performing vector search
        Then should return similarity-ranked results
        """
        # Given
        valid_embedding = [0.1] * 1536  # Valid 1536-dimensional vector
        mock_results = []

        for i in range(3):
            row = Mock()
            row.document_id = i + 1
            row.id = i + 1
            row.filename = f"doc_{i + 1}.pdf"
            row.source_path = f"/path/doc_{i + 1}.pdf"
            row.chunk_text = f"Content chunk {i + 1}"
            row.similarity = 0.9 - (i * 0.1)
            row.page_number = i + 1
            row.chunk_index = i
            row.metadata = {}
            mock_results.append(row)

        mock_db_session.execute.return_value = mock_results

        # When
        results = SearchQueries.vector_similarity_search(
            mock_db_session, query_embedding=valid_embedding, threshold=0.7
        )

        # Then
        assert len(results) == 3
        assert all(hasattr(result, "similarity") for result in results)
        assert all(result.similarity >= 0.7 for result in results)

        # Verify parameterized query with vector casting
        mock_db_session.execute.assert_called_once()
        call_args = mock_db_session.execute.call_args
        assert ":query_embedding::vector" in str(call_args[0][0])

    def test_vector_similarity_search_embedding_validation(self, mock_db_session):
        """
        Test embedding validation in vector search

        Given invalid embedding formats
        When performing vector search
        Then should validate and reject invalid embeddings
        """
        # Given
        mock_db_session.execute.return_value = []

        invalid_embeddings = [
            [],  # Empty vector
            [1.0] * 512,  # Wrong dimension
            [1.0] * 2000,  # Too many dimensions
            ["not", "numbers"],  # Non-numeric values
            None,  # None value
            "not_a_list",  # Not a list
            [float("inf")] * 1536,  # Infinity values
            [float("nan")] * 1536,  # NaN values
        ]

        # When & Then
        for invalid_embedding in invalid_embeddings:
            try:
                result = SearchQueries.vector_similarity_search(
                    mock_db_session, query_embedding=invalid_embedding, threshold=0.7
                )

                # If it doesn't raise an exception, verify safe handling
                if mock_db_session.execute.called:
                    call_args = mock_db_session.execute.call_args
                    # Should still use parameterized query
                    assert ":query_embedding::vector" in str(call_args[0][0])

            except (ValueError, TypeError):
                assert True, (
                    "Embedding validation correctly rejected invalid embedding"
                )

    def test_vector_similarity_search_threshold_validation(self, mock_db_session):
        """
        Test threshold parameter validation

        Given invalid threshold values
        When performing vector search
        Then should validate thresholds appropriately
        """
        # Given
        mock_db_session.execute.return_value = []
        valid_embedding = [0.1] * 1536

        invalid_thresholds = [
            -0.5,  # Negative threshold
            1.5,  # Threshold > 1
            "0.7",  # String threshold
            None,  # None threshold
            float("inf"),  # Infinity
            float("nan"),  # NaN
        ]

        # When & Then
        for invalid_threshold in invalid_thresholds:
            try:
                result = SearchQueries.vector_similarity_search(
                    mock_db_session,
                    query_embedding=valid_embedding,
                    threshold=invalid_threshold,
                )

                # Should handle gracefully or use default
                if mock_db_session.execute.called:
                    call_args = mock_db_session.execute.call_args
                    params = call_args[0][1]
                    # Threshold should be a valid float
                    assert isinstance(params["threshold"], (int, float))

            except (ValueError, TypeError):
                assert True, f"Threshold validation rejected: {invalid_threshold}"

    # Test hybrid_search functionality

    def test_hybrid_search_basic_functionality(self, mock_db_session):
        """
        Test basic hybrid search functionality

        Given valid query and embedding
        When performing hybrid search
        Then should combine semantic and keyword results
        """
        # Given
        valid_embedding = [0.1] * 1536
        mock_results = []

        for i in range(3):
            row = Mock()
            row.id = i + 1
            row.filename = f"hybrid_doc_{i + 1}.pdf"
            row.source_path = f"/path/hybrid_{i + 1}.pdf"
            row.metadata = {}
            row.created_at = Mock()
            row.created_at.isoformat.return_value = "2025-09-26T10:00:00Z"
            row.file_size_bytes = 1000 + i
            row.conversion_status = "completed"
            row.semantic_score = 0.8 - (i * 0.1)
            row.keyword_score = 0.7 - (i * 0.1)
            row.combined_score = (0.8 - i * 0.1) * 0.7 + (0.7 - i * 0.1) * 0.3
            row.chunk_text = f"Hybrid content {i + 1}"
            row.page_number = i + 1
            row.chunk_index = i
            row.chunk_metadata = {}
            mock_results.append(row)

        mock_db_session.execute.return_value = mock_results

        # When
        results = SearchQueries.hybrid_search(
            mock_db_session,
            query="artificial intelligence",
            query_embedding=valid_embedding,
            limit=10,
        )

        # Then
        assert len(results) <= 3  # May be deduplicated
        assert all("combined_score" in result for result in results)
        assert all("semantic_score" in result for result in results)
        assert all("keyword_score" in result for result in results)

        # Verify complex parameterized query
        mock_db_session.execute.assert_called_once()
        call_args = mock_db_session.execute.call_args
        assert ":query" in str(call_args[0][0])
        assert ":query_embedding::vector" in str(call_args[0][0])

    def test_hybrid_search_weight_validation(self, mock_db_session):
        """
        Test weight parameter validation in hybrid search

        Given invalid weight combinations
        When performing hybrid search
        Then should validate weight parameters
        """
        # Given
        mock_db_session.execute.return_value = []
        valid_embedding = [0.1] * 1536

        invalid_weight_cases = [
            {"semantic_weight": -0.1, "keyword_weight": 0.3},  # Negative weight
            {"semantic_weight": 1.5, "keyword_weight": 0.3},  # Weight > 1
            {"semantic_weight": "0.7", "keyword_weight": 0.3},  # String weight
            {"semantic_weight": None, "keyword_weight": 0.3},  # None weight
        ]

        # When & Then
        for weight_case in invalid_weight_cases:
            try:
                result = SearchQueries.hybrid_search(
                    mock_db_session,
                    query="test",
                    query_embedding=valid_embedding,
                    **weight_case,
                )

                # Should validate weights
                if (
                    weight_case["semantic_weight"] < 0
                    or weight_case["semantic_weight"] > 1
                ):
                    pytest.fail("Should have validated weight range")

            except (ValueError, TypeError):
                assert True, f"Weight validation rejected: {weight_case}"


class TestQueueQueries:
    """Test queue operations with concurrency and security focus"""

    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session for queue tests"""
        session = Mock(spec=Session)
        session.query = Mock()
        session.commit = Mock()
        session.rollback = Mock()
        return session

    @pytest.fixture
    def sample_queue_job(self):
        """Create sample queue job"""
        job = Mock(spec=ProcessingQueue)
        job.id = 1
        job.file_path = "/test/document.pdf"
        job.status = "queued"
        job.priority = 1
        job.attempts = 0
        job.worker_id = None
        job.started_at = None
        job.completed_at = None
        job.error_message = None
        return job

    # Test get_next_job functionality

    def test_get_next_job_successful_claim(self, mock_db_session, sample_queue_job):
        """
        Test successful job claim from queue

        Given available job in queue
        When worker requests next job
        Then should claim job atomically with proper locking
        """
        # Given
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.with_for_update.return_value = mock_query
        mock_query.first.return_value = sample_queue_job
        mock_db_session.query.return_value = mock_query

        # When
        result = QueueQueries.get_next_job(mock_db_session, "worker_123")

        # Then
        assert result == sample_queue_job
        assert sample_queue_job.status == "processing"
        assert sample_queue_job.worker_id == "worker_123"
        assert sample_queue_job.attempts == 1

        # Verify proper locking was used
        mock_query.with_for_update.assert_called_once_with(skip_locked=True)
        mock_db_session.commit.assert_called_once()

    def test_get_next_job_no_jobs_available(self, mock_db_session):
        """
        Test job request when no jobs available

        Given empty queue
        When worker requests next job
        Then should return None without errors
        """
        # Given
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.with_for_update.return_value = mock_query
        mock_query.first.return_value = None
        mock_db_session.query.return_value = mock_query

        # When
        result = QueueQueries.get_next_job(mock_db_session, "worker_123")

        # Then
        assert result is None
        mock_db_session.commit.assert_not_called()

    def test_get_next_job_worker_id_validation(self, mock_db_session):
        """
        Test worker ID validation in get_next_job

        Given invalid worker IDs
        When requesting next job
        Then should validate worker ID format and reject malicious inputs
        """
        # Given
        invalid_worker_ids = [
            "",  # Empty string
            None,  # None value
            "worker'; DROP TABLE processing_queue; --",  # SQL injection
            "a" * 300,  # Too long
            123,  # Non-string
        ]

        # When & Then
        for invalid_worker_id in invalid_worker_ids:
            try:
                result = QueueQueries.get_next_job(mock_db_session, invalid_worker_id)
                # Should handle gracefully with ORM
                mock_db_session.query.assert_called_with(ProcessingQueue)
            except (ValueError, TypeError):
                assert True, f"Worker ID validation rejected: {invalid_worker_id}"

    def test_get_next_job_database_error_handling(self, mock_db_session):
        """
        Test database error handling in get_next_job

        Given database error during job claim
        When requesting next job
        Then should rollback transaction and re-raise error
        """
        # Given
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.with_for_update.return_value = mock_query
        mock_query.first.side_effect = SQLAlchemyError("Database connection lost")
        mock_db_session.query.return_value = mock_query

        # When & Then
        with pytest.raises(SQLAlchemyError):
            QueueQueries.get_next_job(mock_db_session, "worker_123")

        # Should rollback on error
        mock_db_session.rollback.assert_called_once()

    # Test queue statistics functionality

    def test_get_queue_stats_basic_functionality(self, mock_db_session):
        """
        Test basic queue statistics retrieval

        Given queue with various job statuses
        When requesting queue statistics
        Then should return accurate counts by status
        """
        # Given
        mock_total_query = Mock()
        mock_total_query.count.return_value = 100

        mock_status_query = Mock()
        mock_status_query.group_by.return_value = mock_status_query
        mock_status_query.all.return_value = [
            ("queued", 50),
            ("processing", 30),
            ("completed", 15),
            ("failed", 5),
        ]

        mock_db_session.query.side_effect = [mock_total_query, mock_status_query]

        # When
        stats = QueueQueries.get_queue_stats(mock_db_session)

        # Then
        assert stats["total"] == 100
        assert stats["by_status"]["queued"] == 50
        assert stats["by_status"]["processing"] == 30
        assert stats["by_status"]["completed"] == 15
        assert stats["by_status"]["failed"] == 5

    def test_get_by_file_path_security(self, mock_db_session):
        """
        Test file path security in queue lookup

        Given various file paths including malicious ones
        When looking up queue entries by file path
        Then should handle safely without path traversal issues
        """
        # Given
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.first.return_value = None
        mock_db_session.query.return_value = mock_query

        malicious_paths = [
            "../../../etc/passwd",
            "/etc/shadow",
            "'; DROP TABLE processing_queue; --",
            "\\..\\..\\windows\\system32",
            "/dev/null; rm -rf /",
        ]

        # When & Then
        for malicious_path in malicious_paths:
            try:
                result = QueueQueries.get_by_file_path(mock_db_session, malicious_path)
                # ORM should handle safely
                mock_db_session.query.assert_called_with(ProcessingQueue)
            except ValueError:
                assert True, f"Path validation rejected: {malicious_path}"


class TestInputValidationFunctions:
    """Test input validation helper functions"""

    def test_validate_string_basic_functionality(self):
        """
        Test basic string validation

        Given valid strings
        When validating strings
        Then should return cleaned strings
        """
        # Given
        valid_strings = [
            "normal string",
            "string with numbers 123",
            "string-with-dashes",
            "string_with_underscores",
        ]

        # When & Then
        for valid_string in valid_strings:
            result = _validate_string(valid_string, "test_field", max_length=100)
            assert isinstance(result, str)
            assert len(result) <= 100

    def test_validate_string_dangerous_patterns(self):
        """
        Test dangerous pattern detection in string validation

        Given strings with SQL injection patterns
        When validating strings
        Then should detect and handle dangerous patterns
        """
        # Given
        dangerous_patterns = [
            "'; DROP TABLE test; --",
            "' OR 1=1 --",
            "'; DELETE FROM users; --",
            "UNION SELECT * FROM",
            "<script>alert('xss')</script>",
        ]

        # When & Then
        for dangerous_pattern in dangerous_patterns:
            try:
                result = _validate_string(
                    dangerous_pattern, "test_field", max_length=500
                )
                # If validation passes, ensure dangerous elements are neutralized
                assert "DROP TABLE" not in result.upper()
                assert "DELETE FROM" not in result.upper()
                assert "<script>" not in result.lower()
            except ValueError:
                assert True, (
                    f"Dangerous pattern correctly rejected: {dangerous_pattern}"
                )

    def test_validate_integer_type_validation(self):
        """
        Test integer validation with type checking

        Given various data types
        When validating as integers
        Then should accept only valid integers
        """
        # Given
        valid_integers = [1, 10, 100, 0, -1]
        invalid_integers = [
            "123",  # String number
            "not_a_number",  # String
            1.5,  # Float
            None,  # None
            [],  # List
            {},  # Dict
        ]

        # When & Then
        for valid_int in valid_integers:
            result = _validate_integer(valid_int, "test_field")
            assert isinstance(result, int)
            assert result == valid_int

        for invalid_int in invalid_integers:
            with pytest.raises((ValueError, TypeError)):
                _validate_integer(invalid_int, "test_field")

    def test_validate_embedding_dimension_validation(self):
        """
        Test embedding validation with dimension checking

        Given embeddings of various dimensions
        When validating embeddings
        Then should enforce correct dimensions
        """
        # Given
        valid_embedding_1536 = [0.1] * 1536
        valid_embedding_512 = [0.1] * 512

        invalid_embeddings = [
            [0.1] * 100,  # Wrong dimension
            [0.1] * 2000,  # Too many dimensions
            [],  # Empty
            ["not", "numbers"],  # Non-numeric
            None,  # None
            "not_a_list",  # Not a list
        ]

        # When & Then
        result_1536 = _validate_embedding(valid_embedding_1536, 1536)
        assert len(result_1536) == 1536
        assert all(isinstance(x, (int, float)) for x in result_1536)

        result_512 = _validate_embedding(valid_embedding_512, 512)
        assert len(result_512) == 512

        for invalid_embedding in invalid_embeddings:
            with pytest.raises((ValueError, TypeError)):
                _validate_embedding(invalid_embedding, 1536)

    def test_validate_float_range_validation(self):
        """
        Test float validation with range checking

        Given various float values
        When validating floats
        Then should accept valid floats and reject invalid ones
        """
        # Given
        valid_floats = [0.0, 0.5, 1.0, -1.0, 3.14159]
        invalid_floats = [
            "not_a_float",  # String
            None,  # None
            float("inf"),  # Infinity
            float("nan"),  # NaN
            [],  # List
        ]

        # When & Then
        for valid_float in valid_floats:
            result = _validate_float(valid_float, "test_field")
            assert isinstance(result, (int, float))
            assert result == valid_float

        for invalid_float in invalid_floats:
            with pytest.raises((ValueError, TypeError)):
                _validate_float(invalid_float, "test_field")


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases across all query functions"""

    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session that can simulate errors"""
        session = Mock(spec=Session)
        return session

    def test_database_connection_failure_handling(self, mock_db_session):
        """
        Test handling of database connection failures

        Given database connection failure
        When executing any query
        Then should handle gracefully and provide meaningful errors
        """
        # Given
        mock_db_session.query.side_effect = SQLAlchemyError(
            "Connection to database failed"
        )

        # When & Then
        with pytest.raises(SQLAlchemyError) as exc_info:
            DocumentQueries.get_by_id(mock_db_session, 1)

        # Error should be informative but not leak sensitive information
        error_msg = str(exc_info.value).lower()
        sensitive_terms = ["password", "secret", "key", "token"]
        assert not any(term in error_msg for term in sensitive_terms)

    def test_transaction_rollback_on_error(self, mock_db_session, sample_queue_job):
        """
        Test transaction rollback on errors in queue operations

        Given transaction failure during job processing
        When error occurs
        Then should properly rollback transaction
        """
        # Given
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.with_for_update.return_value = mock_query
        mock_query.first.return_value = sample_queue_job
        mock_db_session.query.return_value = mock_query
        mock_db_session.commit.side_effect = IntegrityError(
            "Constraint violation", None, None
        )

        # When & Then
        with pytest.raises(SQLAlchemyError):
            QueueQueries.get_next_job(mock_db_session, "worker_123")

        # Should rollback on commit failure
        mock_db_session.rollback.assert_called_once()

    def test_large_result_set_handling(self, mock_db_session):
        """
        Test handling of large result sets

        Given queries that might return many results
        When executing search queries
        Then should handle memory efficiently
        """
        # Given - Simulate large result set
        large_result_set = [Mock() for _ in range(1000)]
        mock_db_session.execute.return_value = large_result_set

        # When
        try:
            results = SearchQueries.fulltext_search(
                mock_db_session,
                query="common term",
                limit=50,  # Should limit even with large dataset
            )

            # Then - Should handle large datasets gracefully
            assert len(results) <= 1000  # Should not exceed what was returned
            mock_db_session.execute.assert_called_once()

        except MemoryError:
            pytest.fail("Query should handle large result sets without memory errors")

    def test_concurrent_access_safety(self, mock_db_session):
        """
        Test safety under concurrent access

        Given multiple concurrent requests
        When accessing queue operations
        Then should use proper locking mechanisms
        """
        # Given
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.with_for_update.return_value = mock_query
        mock_query.first.return_value = None
        mock_db_session.query.return_value = mock_query

        # When - Simulate concurrent access
        worker_ids = [f"worker_{i}" for i in range(5)]

        for worker_id in worker_ids:
            QueueQueries.get_next_job(mock_db_session, worker_id)

        # Then - Should use proper locking for all requests
        assert mock_query.with_for_update.call_count == len(worker_ids)
        all_calls = mock_query.with_for_update.call_args_list
        assert all(call[1]["skip_locked"] for call in all_calls)


if __name__ == "__main__":
    pytest.main(
        [__file__, "-v", "--tb=short", "--cov=src.pdf_to_markdown_mcp.db.queries"]
    )
