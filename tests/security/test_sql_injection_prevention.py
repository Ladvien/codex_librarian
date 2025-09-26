"""
SQL Injection Prevention Test Suite

This module contains comprehensive tests for SQL injection attack prevention
across all database query functions. Following TDD principles to ensure
security vulnerabilities are properly tested.

Security Test Categories:
1. Basic SQL injection attempts
2. Advanced SQL injection patterns
3. Parameterized query validation
4. Input sanitization verification
5. Error information leakage prevention
"""

from unittest.mock import Mock

import pytest
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from src.pdf_to_markdown_mcp.db.models import (
    Document,
    ProcessingQueue,
)
from src.pdf_to_markdown_mcp.db.queries import (
    DocumentQueries,
    QueueQueries,
    SearchQueries,
    _validate_integer,
    _validate_string,
)

# SQL Injection Attack Patterns for Testing
SQL_INJECTION_PAYLOADS = [
    # Basic injection attempts
    "'; DROP TABLE documents; --",
    "' OR '1'='1",
    "' OR 1=1 --",
    "' UNION SELECT * FROM users --",
    # Advanced injection patterns
    "'; INSERT INTO documents (filename) VALUES ('hacked'); --",
    "' OR EXISTS(SELECT * FROM information_schema.tables WHERE table_name='documents') --",
    "' AND (SELECT COUNT(*) FROM documents) > 0 --",
    "'; UPDATE documents SET filename='hacked' WHERE id=1; --",
    # Boolean-based blind injection
    "' AND (SELECT SUBSTRING(version(), 1, 1)) = '1",
    "' OR (SELECT COUNT(*) FROM documents WHERE filename LIKE '%a%') > 0 --",
    # Time-based blind injection
    "'; SELECT CASE WHEN (1=1) THEN pg_sleep(5) ELSE pg_sleep(0) END --",
    # Error-based injection
    "' AND EXTRACTVALUE(1, CONCAT(0x7e, (SELECT version()), 0x7e)) --",
    "' OR (SELECT * FROM (SELECT COUNT(*),CONCAT(version(),FLOOR(RAND(0)*2))x FROM information_schema.tables GROUP BY x)a) --",
    # NoSQL-style injection attempts
    "'; db.documents.drop(); --",
    "' || this.filename == 'admin' --",
    # XML injection attempts
    "' OR 1=1 <!--",
    "'><script>alert('xss')</script>",
    # PostgreSQL-specific injection
    "'; COPY documents TO '/tmp/output.txt'; --",
    "' OR (SELECT current_user) IS NOT NULL --",
    "'; CREATE ROLE hacker WITH LOGIN PASSWORD 'password'; --",
]

FILTER_INJECTION_PAYLOADS = [
    # Dictionary-based injection attempts for filters
    {"document_id": "1'; DROP TABLE documents; --"},
    {"document_id": "1 OR 1=1"},
    {"unknown_field": "'; INSERT INTO documents VALUES (1); --"},
    {"document_id": {"$ne": None}},  # NoSQL-style
    {"document_id": [1, 2, "'; DROP TABLE documents; --"]},
]


class TestSQLInjectionPrevention:
    """Test SQL injection prevention across all database queries"""

    @pytest.fixture
    def mock_db_session(self):
        """Create a mock database session for testing"""
        session = Mock(spec=Session)
        session.execute = Mock()
        session.query = Mock()
        return session

    @pytest.fixture
    def sample_document(self):
        """Create a sample document for testing"""
        doc = Mock(spec=Document)
        doc.id = 1
        doc.filename = "test.pdf"
        doc.source_path = "/test/path.pdf"
        doc.file_hash = "abc123"
        doc.conversion_status = "completed"
        return doc

    # CRITICAL: Full-text Search SQL Injection Tests

    @pytest.mark.parametrize("malicious_query", SQL_INJECTION_PAYLOADS)
    def test_fulltext_search_sql_injection_prevention(
        self, mock_db_session, malicious_query
    ):
        """
        Test that full-text search prevents SQL injection attacks

        Given a malicious SQL injection payload in search query
        When performing full-text search
        Then the query should be safely parameterized and not executed as SQL
        """
        # Given
        mock_db_session.execute.return_value = []

        # When & Then - Should not raise SQL injection error
        try:
            result = SearchQueries.fulltext_search(
                db=mock_db_session, query=malicious_query, limit=10
            )

            # Verify parameterized query was used
            mock_db_session.execute.assert_called_once()
            call_args = mock_db_session.execute.call_args

            # Ensure the query uses parameters, not string concatenation
            assert call_args[0][0].is_text()  # SQLAlchemy text object
            assert "plainto_tsquery('english', :query)" in str(call_args[0][0])
            assert (
                malicious_query in call_args[0][1]["query"]
            )  # In parameters, not query string

        except ValueError as e:
            # Input validation should catch malicious patterns
            assert "invalid" in str(e).lower() or "dangerous" in str(e).lower()
        except SQLAlchemyError:
            pytest.fail(
                "SQL injection attempt caused database error - parameterization failed"
            )

    @pytest.mark.parametrize("malicious_filters", FILTER_INJECTION_PAYLOADS)
    def test_fulltext_search_filter_injection_prevention(
        self, mock_db_session, malicious_filters
    ):
        """
        Test that search filters prevent SQL injection

        Given malicious filters containing SQL injection
        When performing filtered full-text search
        Then filters should be validated and rejected or parameterized
        """
        # Given
        mock_db_session.execute.return_value = []

        # When & Then
        try:
            result = SearchQueries.fulltext_search(
                db=mock_db_session,
                query="legitimate search",
                filters=malicious_filters,
                limit=10,
            )

            # Verify safe parameterization
            mock_db_session.execute.assert_called_once()
            call_args = mock_db_session.execute.call_args

            # Check that parameters are properly typed
            params = call_args[0][1]
            if "document_id" in params:
                assert isinstance(params["document_id"], int), (
                    "document_id should be validated as integer"
                )

        except (ValueError, TypeError):
            # Input validation should reject invalid filter types
            assert True, "Input validation correctly rejected malicious filter"
        except SQLAlchemyError:
            pytest.fail("Filter injection caused database error")

    # CRITICAL: Vector Search SQL Injection Tests

    def test_vector_search_embedding_injection_prevention(self, mock_db_session):
        """
        Test that vector search prevents injection through embedding parameters

        Given malicious data disguised as embedding vectors
        When performing vector similarity search
        Then the embedding should be validated and parameterized
        """
        # Given - Malicious embedding attempts
        malicious_embeddings = [
            "'; DROP TABLE document_embeddings; --",
            [1.0, 2.0, "'; DELETE FROM documents; --"],
            {"injection": "'; TRUNCATE TABLE documents; --"},
        ]

        mock_db_session.execute.return_value = []

        for malicious_embedding in malicious_embeddings:
            # When & Then
            try:
                result = SearchQueries.vector_similarity_search(
                    db=mock_db_session,
                    query_embedding=malicious_embedding,
                    threshold=0.7,
                )

                # Verify parameterized query execution
                mock_db_session.execute.assert_called()
                call_args = mock_db_session.execute.call_args
                assert ":query_embedding::vector" in str(call_args[0][0])

            except (ValueError, TypeError):
                # Should reject invalid embedding types
                assert True, "Input validation correctly rejected malicious embedding"
            except SQLAlchemyError:
                pytest.fail("Embedding injection caused database error")

    # CRITICAL: Queue Operations SQL Injection Tests

    def test_queue_operations_injection_prevention(self, mock_db_session):
        """
        Test that queue operations prevent SQL injection

        Given malicious input in queue operation parameters
        When performing queue operations
        Then parameters should be validated and safely handled
        """
        # Given
        malicious_inputs = [
            "'; DROP TABLE processing_queue; --",
            "1'; UPDATE processing_queue SET status='failed'; --",
            {"status": "'; DELETE FROM processing_queue; --"},
        ]

        mock_result = Mock()
        mock_result.first.return_value = None
        mock_db_session.query.return_value = mock_result

        for malicious_input in malicious_inputs:
            # When & Then
            try:
                # Test file path injection
                result = QueueQueries.get_by_file_path(
                    db=mock_db_session, file_path=malicious_input
                )

                # Verify ORM query was used (not raw SQL)
                mock_db_session.query.assert_called()
                query_calls = mock_db_session.query.call_args_list

                # Ensure filter uses parameterized comparison
                for call in query_calls:
                    assert call[0][0] == ProcessingQueue  # ORM model, not raw SQL

            except (ValueError, TypeError):
                assert True, "Input validation correctly rejected malicious input"
            except SQLAlchemyError:
                pytest.fail("Queue injection caused database error")

    # CRITICAL: Document Operations SQL Injection Tests

    @pytest.mark.parametrize("malicious_path", SQL_INJECTION_PAYLOADS)
    def test_document_operations_path_injection_prevention(
        self, mock_db_session, malicious_path
    ):
        """
        Test that document operations prevent path-based SQL injection

        Given malicious SQL in document path parameters
        When performing document queries
        Then paths should be safely parameterized
        """
        # Given
        mock_result = Mock()
        mock_result.first.return_value = None
        mock_db_session.query.return_value = mock_result

        # When & Then
        try:
            result = DocumentQueries.get_by_path(
                db=mock_db_session, source_path=malicious_path
            )

            # Verify ORM parameterized query
            mock_db_session.query.assert_called_with(Document)
            # ORM automatically parameterizes filter conditions

        except ValueError:
            # Input validation may reject extremely malicious paths
            assert True, "Input validation correctly handled malicious path"
        except SQLAlchemyError:
            pytest.fail("Path injection caused database error")

    # Input Validation Tests

    def test_validate_string_injection_prevention(self):
        """
        Test string validation prevents injection patterns

        Given various malicious string inputs
        When validating strings
        Then dangerous patterns should be detected and rejected
        """
        # Given - Test each injection payload
        for payload in SQL_INJECTION_PAYLOADS[:10]:  # Test subset for performance
            # When & Then
            try:
                result = _validate_string(payload, "test_field", max_length=500)
                # If validation passes, ensure dangerous characters are sanitized
                assert "DROP TABLE" not in result.upper()
                assert "DELETE FROM" not in result.upper()
                assert "UNION SELECT" not in result.upper()

            except ValueError as e:
                # Expected for obviously malicious patterns
                assert "invalid" in str(e).lower() or "dangerous" in str(e).lower()

    def test_validate_integer_injection_prevention(self):
        """
        Test integer validation prevents injection through type confusion

        Given malicious inputs disguised as integers
        When validating integers
        Then non-integer types should be rejected
        """
        # Given
        malicious_integer_attempts = [
            "1'; DROP TABLE documents; --",
            {"$ne": None},
            [1, "'; DELETE FROM documents; --"],
            "1 OR 1=1",
            "1); DROP TABLE documents; --",
        ]

        # When & Then
        for malicious_input in malicious_integer_attempts:
            with pytest.raises((ValueError, TypeError)):
                _validate_integer(malicious_input, "test_field")

    # Error Information Leakage Prevention Tests

    def test_error_handling_prevents_information_disclosure(self, mock_db_session):
        """
        Test that database errors don't leak sensitive information

        Given database operations that cause errors
        When SQL injection attempts trigger database errors
        Then error messages should not reveal database structure
        """
        # Given - Simulate database error
        mock_db_session.execute.side_effect = SQLAlchemyError(
            "relation 'secret_table' does not exist"
        )

        # When & Then
        with pytest.raises(SQLAlchemyError) as exc_info:
            SearchQueries.fulltext_search(
                db=mock_db_session, query="legitimate query", limit=10
            )

        # Error should be generic, not reveal internal structure
        error_message = str(exc_info.value).lower()
        sensitive_keywords = ["password", "admin", "user", "secret", "config", "env"]

        for keyword in sensitive_keywords:
            assert keyword not in error_message, (
                f"Error message leaked sensitive keyword: {keyword}"
            )

    # Integration Tests for Comprehensive Security

    def test_chained_injection_attempts_prevention(self, mock_db_session):
        """
        Test prevention of chained SQL injection attempts

        Given multiple chained injection attempts in single request
        When performing complex searches with multiple parameters
        Then all parameters should be safely handled
        """
        # Given
        mock_db_session.execute.return_value = []

        chained_attack = {
            "query": "'; DROP TABLE documents; --",
            "filters": {"document_id": "1'; DELETE FROM users; --"},
            "limit": "10'; TRUNCATE TABLE processing_queue; --",
        }

        # When & Then
        with pytest.raises((ValueError, TypeError, SQLAlchemyError)):
            SearchQueries.fulltext_search(
                db=mock_db_session,
                query=chained_attack["query"],
                filters=chained_attack["filters"],
                limit=chained_attack["limit"],
            )

    def test_parameterized_query_verification(self, mock_db_session):
        """
        Test that all queries use proper parameterization

        Given legitimate database operations
        When executing queries
        Then all user input should be in parameters, not query strings
        """
        # Given
        mock_db_session.execute.return_value = []
        legitimate_query = "test search"

        # When
        SearchQueries.fulltext_search(
            db=mock_db_session, query=legitimate_query, limit=10
        )

        # Then
        mock_db_session.execute.assert_called_once()
        call_args = mock_db_session.execute.call_args

        # Verify query structure
        sql_query = str(call_args[0][0])
        parameters = call_args[0][1]

        # Query should contain parameter placeholders, not actual values
        assert ":query" in sql_query
        assert ":limit" in sql_query
        assert legitimate_query not in sql_query  # Value should be in parameters
        assert legitimate_query in parameters["query"]  # Value should be in parameters


class TestAdvancedSQLInjectionScenarios:
    """Advanced SQL injection attack prevention tests"""

    @pytest.fixture
    def mock_db_session(self):
        """Create a mock database session for testing"""
        session = Mock(spec=Session)
        session.execute = Mock()
        session.query = Mock()
        return session

    def test_second_order_injection_prevention(self, mock_db_session):
        """
        Test prevention of second-order SQL injection attacks

        Given malicious data stored in database that could be used in subsequent queries
        When retrieving and using stored data in new queries
        Then stored data should be treated as untrusted and parameterized
        """
        # Given - Simulate stored malicious data
        mock_doc = Mock()
        mock_doc.filename = "'; DROP TABLE documents; --"
        mock_doc.id = 1

        mock_result = Mock()
        mock_result.first.return_value = mock_doc
        mock_db_session.query.return_value = mock_result

        # When - Using stored data in new query (common second-order scenario)
        doc = DocumentQueries.get_by_id(mock_db_session, 1)

        # Then - Any subsequent use of doc.filename should be safe
        if doc:
            # Simulate using filename in another query
            try:
                SearchQueries.fulltext_search(
                    db=mock_db_session,
                    query=doc.filename,  # This contains malicious data
                    limit=10,
                )
                # Should be safely parameterized
                mock_db_session.execute.assert_called()

            except ValueError:
                # Input validation should catch this
                assert True, "Second-order injection attempt properly rejected"

    def test_blind_sql_injection_prevention(self, mock_db_session):
        """
        Test prevention of blind SQL injection attacks

        Given time-based and boolean-based blind injection attempts
        When executing queries with injection payloads
        Then responses should not vary based on injection success/failure
        """
        # Given
        blind_payloads = [
            "' AND (SELECT COUNT(*) FROM documents) > 0 --",  # Boolean-based
            "'; WAITFOR DELAY '00:00:05'; --",  # Time-based
            "' OR BENCHMARK(1000000,MD5(1)) --",  # Time-based
        ]

        mock_db_session.execute.return_value = []

        # When & Then
        for payload in blind_payloads:
            try:
                result = SearchQueries.fulltext_search(
                    db=mock_db_session, query=payload, limit=10
                )

                # Verify safe execution
                mock_db_session.execute.assert_called()

            except ValueError:
                # Input validation correctly rejects malicious patterns
                assert True, (
                    f"Blind injection payload correctly rejected: {payload[:50]}"
                )

    def test_union_based_injection_prevention(self, mock_db_session):
        """
        Test prevention of UNION-based SQL injection attacks

        Given UNION injection attempts to retrieve unauthorized data
        When executing search queries
        Then UNION statements should be safely parameterized or rejected
        """
        # Given
        union_payloads = [
            "' UNION SELECT id, filename FROM documents --",
            "' UNION ALL SELECT username, password FROM users --",
            "test' UNION SELECT 1,2,3,4,5,6,7,8,9,10 --",
        ]

        mock_db_session.execute.return_value = []

        # When & Then
        for payload in union_payloads:
            try:
                result = SearchQueries.fulltext_search(
                    db=mock_db_session, query=payload, limit=10
                )

                # Verify parameterized execution
                call_args = mock_db_session.execute.call_args
                query_text = str(call_args[0][0])

                # Should not contain actual UNION in query structure
                assert "UNION" not in query_text.upper() or ":query" in query_text

            except ValueError:
                assert True, f"UNION injection correctly rejected: {payload[:50]}"


# Performance Impact Tests for Security Measures


class TestSecurityPerformanceImpact:
    """Test that security measures don't significantly impact performance"""

    @pytest.fixture
    def mock_db_session(self):
        session = Mock(spec=Session)
        session.execute = Mock(return_value=[])
        return session

    def test_input_validation_performance(self, mock_db_session):
        """
        Test that input validation doesn't create significant performance overhead

        Given legitimate queries with input validation
        When performing many search operations
        Then validation should complete efficiently
        """
        import time

        # Given
        legitimate_queries = [
            "machine learning",
            "artificial intelligence",
            "data science",
            "natural language processing",
            "computer vision",
        ]

        # When
        start_time = time.time()

        for query in legitimate_queries * 20:  # 100 queries total
            try:
                SearchQueries.fulltext_search(db=mock_db_session, query=query, limit=10)
            except Exception:
                pass  # Focus on validation performance, not execution

        end_time = time.time()
        execution_time = end_time - start_time

        # Then - Should complete within reasonable time (< 1 second for 100 validations)
        assert execution_time < 1.0, (
            f"Input validation took too long: {execution_time:.3f}s"
        )

    def test_parameterization_overhead(self, mock_db_session):
        """
        Test that parameterized queries don't have excessive overhead

        Given parameterized vs direct query execution
        When measuring execution time
        Then parameterization should have minimal overhead
        """
        # This test ensures our security measures maintain good performance
        # In real implementation, parameterized queries are actually faster
        # due to query plan caching

        start_time = time.time()

        # Execute multiple parameterized queries
        for i in range(50):
            try:
                SearchQueries.fulltext_search(
                    db=mock_db_session, query=f"test query {i}", limit=10
                )
            except Exception:
                pass

        end_time = time.time()
        execution_time = end_time - start_time

        # Should be fast enough for production use
        assert execution_time < 0.5, (
            f"Parameterized queries too slow: {execution_time:.3f}s"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
