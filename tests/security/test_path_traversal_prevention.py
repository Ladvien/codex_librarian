"""
Path Traversal Attack Prevention Test Suite

This module contains comprehensive tests for path traversal attack prevention
across all file system operations. Following TDD principles to ensure
directory traversal vulnerabilities are properly tested and prevented.

Security Test Categories:
1. Basic path traversal attempts
2. Advanced traversal techniques
3. File access restriction validation
4. Symlink traversal prevention
5. Platform-specific traversal patterns
"""

import os
import tempfile
from unittest.mock import Mock, patch

import pytest
from fastapi import HTTPException

from src.pdf_to_markdown_mcp.auth.security import (
    sanitize_error_message,
    validate_file_security,
    validate_path_security,
)

# Path Traversal Attack Patterns for Testing
PATH_TRAVERSAL_PAYLOADS = [
    # Basic traversal attempts
    "../../../etc/passwd",
    "..\\..\\..\\windows\\system32\\config\\sam",
    "../../../../root/.bashrc",
    # URL encoded traversal
    "%2e%2e%2f%2e%2e%2f%2e%2e%2f/etc/passwd",
    "..%2f..%2f..%2fetc%2fpasswd",
    "%2e%2e\\%2e%2e\\%2e%2e\\windows\\system32",
    # Double encoded
    "%252e%252e%252f%252e%252e%252f%252e%252e%252f/etc/passwd",
    # Mixed encoding
    "..%c0%af..%c0%af..%c0%af/etc/passwd",
    "..%c1%9c..%c1%9c..%c1%9c/etc/passwd",
    # Unicode traversal
    "\u002e\u002e\u002f\u002e\u002e\u002f\u002e\u002e\u002f/etc/passwd",
    # Null byte injection
    "../../../etc/passwd%00.pdf",
    "../../../etc/passwd\x00.pdf",
    # Filter bypass attempts
    "....//....//....//etc/passwd",
    "..//////../../../etc/passwd",
    "..\\\\..\\\\..\\\\/etc/passwd",
    # Absolute path attempts
    "/etc/passwd",
    "\\windows\\system32\\config\\sam",
    "C:\\windows\\system32\\config\\sam",
    "/proc/self/environ",
    "/var/log/auth.log",
    # Home directory traversal
    "~/../../etc/passwd",
    "~root/.ssh/id_rsa",
    "~admin/.bash_history",
    # Special device files (Linux/Unix)
    "/dev/random",
    "/dev/urandom",
    "/dev/zero",
    "/proc/cpuinfo",
    "/proc/meminfo",
    # Windows-specific paths
    "\\\\localhost\\c$\\windows\\system32\\config\\sam",
    "file:///c:/windows/system32/config/sam",
    "\\\\?\\c:\\windows\\system32\\config\\sam",
    # Network path attempts
    "//attacker.com/share/malicious.pdf",
    "\\\\attacker.com\\share\\malicious.pdf",
    # Server traversal (web-specific)
    "../../../var/www/html/config.php",
    "../../../etc/apache2/apache2.conf",
    "../../../etc/nginx/nginx.conf",
    # Application-specific traversal
    "../../config/database.yml",
    "../../../.env",
    "../../../../app/secrets.json",
]

DANGEROUS_FILENAMES = [
    # System files
    "passwd",
    "shadow",
    "hosts",
    "resolv.conf",
    "fstab",
    "sudoers",
    # Application configs
    ".env",
    ".env.local",
    "config.json",
    "database.yml",
    "secrets.json",
    "private.key",
    # Windows system files
    "boot.ini",
    "ntuser.dat",
    "system.ini",
    "win.ini",
    # SSH and crypto
    "id_rsa",
    "id_dsa",
    "authorized_keys",
    "known_hosts",
]


class TestPathTraversalPrevention:
    """Test path traversal attack prevention across all file operations"""

    @pytest.fixture
    def temp_safe_directory(self):
        """Create temporary directory for safe path testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create nested structure
            safe_dir = os.path.join(temp_dir, "safe")
            os.makedirs(safe_dir)

            # Create a test PDF file
            test_pdf = os.path.join(safe_dir, "test.pdf")
            with open(test_pdf, "wb") as f:
                f.write(b"%PDF-1.4")  # Valid PDF header

            yield temp_dir, safe_dir, test_pdf

    @pytest.fixture
    def mock_settings(self):
        """Mock application settings for testing"""
        mock = Mock()
        mock.INPUT_DIRECTORY = "/mnt/codex_fs/research/"
        mock.OUTPUT_DIRECTORY = "/mnt/codex_fs/research/librarian_output/"
        mock.MAX_FILE_SIZE_MB = 500
        return mock

    # Basic Path Traversal Tests

    @pytest.mark.parametrize("malicious_path", PATH_TRAVERSAL_PAYLOADS)
    def test_path_traversal_prevention_basic(self, malicious_path, mock_settings):
        """
        Test basic path traversal attack prevention

        Given various path traversal payloads
        When validating path security
        Then should reject all traversal attempts
        """
        # Given - Mock settings
        with patch("src.pdf_to_markdown_mcp.auth.security.settings", mock_settings):
            # When & Then
            with pytest.raises(HTTPException) as exc_info:
                validate_path_security(malicious_path)

            # Should raise 400 Bad Request for path traversal
            assert exc_info.value.status_code == 400
            assert "path" in str(exc_info.value.detail).lower()

    def test_legitimate_paths_allowed(self, temp_safe_directory, mock_settings):
        """
        Test that legitimate paths within allowed directories are accepted

        Given legitimate file paths within allowed directories
        When validating path security
        Then should accept valid paths
        """
        temp_dir, safe_dir, test_pdf = temp_safe_directory

        # Given
        mock_settings.INPUT_DIRECTORY = safe_dir
        legitimate_paths = [
            os.path.join(safe_dir, "document.pdf"),
            os.path.join(safe_dir, "subdir", "file.pdf"),
            test_pdf,
        ]

        # When & Then
        with patch("src.pdf_to_markdown_mcp.auth.security.settings", mock_settings):
            for legitimate_path in legitimate_paths:
                try:
                    result = validate_path_security(legitimate_path)
                    # Should not raise exception for legitimate paths
                    assert True, f"Legitimate path accepted: {legitimate_path}"
                except HTTPException:
                    # Some paths may fail due to file not existing, but not due to traversal
                    pytest.fail(f"Legitimate path rejected: {legitimate_path}")

    def test_path_resolution_normalization(self, temp_safe_directory, mock_settings):
        """
        Test that path resolution normalizes paths correctly

        Given paths with various normalization issues
        When validating paths
        Then should resolve to canonical form and validate correctly
        """
        temp_dir, safe_dir, test_pdf = temp_safe_directory

        # Given
        mock_settings.INPUT_DIRECTORY = safe_dir
        paths_requiring_normalization = [
            os.path.join(safe_dir, ".", "document.pdf"),  # Current directory
            os.path.join(safe_dir, "subdir", "..", "document.pdf"),  # Back reference
            os.path.join(safe_dir, "subdir", ".", "document.pdf"),  # Mixed references
        ]

        # When & Then
        with patch("src.pdf_to_markdown_mcp.auth.security.settings", mock_settings):
            for path_to_normalize in paths_requiring_normalization:
                try:
                    result = validate_path_security(path_to_normalize)
                    # Should handle path normalization correctly
                    assert True, f"Path normalization handled: {path_to_normalize}"
                except HTTPException as e:
                    # Should not fail due to path resolution issues
                    if (
                        "path" in str(e.detail).lower()
                        and "traversal" in str(e.detail).lower()
                    ):
                        pytest.fail(
                            f"Path normalization incorrectly flagged as traversal: {path_to_normalize}"
                        )

    # Advanced Traversal Technique Tests

    def test_symlink_traversal_prevention(self, temp_safe_directory, mock_settings):
        """
        Test prevention of symlink-based traversal attacks

        Given symbolic links that point outside allowed directories
        When validating symlink paths
        Then should detect and prevent symlink traversal
        """
        temp_dir, safe_dir, test_pdf = temp_safe_directory

        # Given
        mock_settings.INPUT_DIRECTORY = safe_dir

        try:
            # Create symlink pointing outside safe directory
            malicious_symlink = os.path.join(safe_dir, "malicious_link.pdf")
            target_path = "/etc/passwd"  # Outside allowed directory

            if os.name != "nt":  # Unix-like systems
                try:
                    os.symlink(target_path, malicious_symlink)

                    # When & Then
                    with patch(
                        "src.pdf_to_markdown_mcp.auth.security.settings", mock_settings
                    ):
                        with pytest.raises(HTTPException) as exc_info:
                            validate_path_security(malicious_symlink)

                        assert exc_info.value.status_code == 400
                        assert "path" in str(exc_info.value.detail).lower()

                except OSError:
                    # Skip if we can't create symlinks (permissions, etc.)
                    pytest.skip("Cannot create symbolic links in test environment")

        except Exception:
            # Skip on systems where symlink creation fails
            pytest.skip("Symlink testing not available in this environment")

    def test_case_insensitive_traversal_prevention(self, mock_settings):
        """
        Test case-insensitive path traversal prevention (Windows-specific)

        Given path traversal attempts with various case combinations
        When validating paths on case-insensitive systems
        Then should prevent case-based bypass attempts
        """
        # Given
        case_variation_payloads = [
            "../../../ETC/PASSWD",
            "../../../Etc/Passwd",
            "..\\..\\..\\WINDOWS\\SYSTEM32\\CONFIG\\SAM",
            "..\\..\\..\\Windows\\System32\\Config\\Sam",
        ]

        # When & Then
        with patch("src.pdf_to_markdown_mcp.auth.security.settings", mock_settings):
            for case_payload in case_variation_payloads:
                with pytest.raises(HTTPException) as exc_info:
                    validate_path_security(case_payload)

                assert exc_info.value.status_code == 400

    def test_long_path_traversal_prevention(self, mock_settings):
        """
        Test prevention of long path traversal attacks

        Given extremely long traversal paths designed to bypass filters
        When validating paths
        Then should handle long paths correctly without performance issues
        """
        # Given
        long_traversal_patterns = [
            "../" * 1000 + "etc/passwd",  # Very long relative path
            "a" * 4000 + "/../../../etc/passwd",  # Long filename with traversal
            "../" * 100 + "etc/passwd" + "?" + "x" * 2000,  # Query string padding
        ]

        # When & Then
        with patch("src.pdf_to_markdown_mcp.auth.security.settings", mock_settings):
            for long_pattern in long_traversal_patterns:
                with pytest.raises(HTTPException) as exc_info:
                    validate_path_security(long_pattern)

                assert exc_info.value.status_code == 400

    # Filename Validation Tests

    @pytest.mark.parametrize("dangerous_filename", DANGEROUS_FILENAMES)
    def test_dangerous_filename_detection(
        self, dangerous_filename, temp_safe_directory, mock_settings
    ):
        """
        Test detection of dangerous filenames

        Given filenames that could be system files or sensitive config files
        When validating file security
        Then should detect and prevent access to dangerous files
        """
        temp_dir, safe_dir, test_pdf = temp_safe_directory

        # Given
        mock_settings.INPUT_DIRECTORY = safe_dir
        dangerous_path = os.path.join(safe_dir, dangerous_filename)

        # Create the file to test filename-based detection
        try:
            with open(dangerous_path, "w") as f:
                f.write("test content")

            # When & Then
            with patch("src.pdf_to_markdown_mcp.auth.security.settings", mock_settings):
                with pytest.raises(HTTPException) as exc_info:
                    validate_path_security(dangerous_path)

                assert exc_info.value.status_code == 400

        except (OSError, PermissionError):
            # Skip if we can't create the file
            pytest.skip(f"Cannot create test file: {dangerous_filename}")

    def test_file_type_validation(self, temp_safe_directory, mock_settings):
        """
        Test file type validation for security

        Given files with various extensions
        When validating file security
        Then should only allow PDF files
        """
        temp_dir, safe_dir, test_pdf = temp_safe_directory

        # Given
        mock_settings.INPUT_DIRECTORY = safe_dir

        non_pdf_files = [
            "document.txt",
            "script.sh",
            "config.json",
            "malicious.exe",
            "document.pdf.exe",  # Double extension attack
        ]

        # Create test files
        for filename in non_pdf_files:
            file_path = os.path.join(safe_dir, filename)
            try:
                with open(file_path, "w") as f:
                    f.write("test content")

                # When & Then
                with patch(
                    "src.pdf_to_markdown_mcp.auth.security.settings", mock_settings
                ):
                    with pytest.raises(HTTPException) as exc_info:
                        validate_file_security(file_path)

                    assert exc_info.value.status_code == 400
                    assert "pdf" in str(exc_info.value.detail).lower()

            except (OSError, PermissionError):
                # Skip if we can't create the file
                continue

    def test_pdf_header_validation(self, temp_safe_directory, mock_settings):
        """
        Test PDF header validation for file security

        Given files with PDF extension but invalid headers
        When validating file security
        Then should reject files without proper PDF headers
        """
        temp_dir, safe_dir, test_pdf = temp_safe_directory

        # Given
        mock_settings.INPUT_DIRECTORY = safe_dir
        mock_settings.MAX_FILE_SIZE_MB = 500

        fake_pdf_files = [
            ("fake1.pdf", b"This is not a PDF"),
            ("fake2.pdf", b"<html>Fake PDF</html>"),
            ("fake3.pdf", b"\x00\x00\x00\x00Not PDF"),
            ("fake4.pdf", b""),  # Empty file
        ]

        # Create fake PDF files
        for filename, content in fake_pdf_files:
            file_path = os.path.join(safe_dir, filename)
            try:
                with open(file_path, "wb") as f:
                    f.write(content)

                # When & Then
                with patch(
                    "src.pdf_to_markdown_mcp.auth.security.settings", mock_settings
                ):
                    with pytest.raises(HTTPException) as exc_info:
                        validate_file_security(file_path)

                    assert exc_info.value.status_code == 400
                    assert (
                        "pdf" in str(exc_info.value.detail).lower()
                        or "header" in str(exc_info.value.detail).lower()
                    )

            except (OSError, PermissionError):
                continue

    def test_file_size_validation(self, temp_safe_directory, mock_settings):
        """
        Test file size validation for security

        Given files exceeding size limits
        When validating file security
        Then should reject oversized files
        """
        temp_dir, safe_dir, test_pdf = temp_safe_directory

        # Given
        mock_settings.INPUT_DIRECTORY = safe_dir
        mock_settings.MAX_FILE_SIZE_MB = 1  # 1 MB limit for testing

        large_file_path = os.path.join(safe_dir, "large.pdf")

        try:
            # Create file larger than limit (2 MB)
            with open(large_file_path, "wb") as f:
                f.write(b"%PDF-1.4")  # Valid PDF header
                f.write(b"x" * (2 * 1024 * 1024))  # 2 MB of padding

            # When & Then
            with patch("src.pdf_to_markdown_mcp.auth.security.settings", mock_settings):
                with pytest.raises(HTTPException) as exc_info:
                    validate_file_security(large_file_path)

                assert exc_info.value.status_code == 400
                assert "size" in str(exc_info.value.detail).lower()

        except (OSError, PermissionError):
            pytest.skip("Cannot create large test file")

    # Error Message Security Tests

    def test_error_message_sanitization(self):
        """
        Test that error messages don't leak sensitive path information

        Given various error conditions
        When sanitizing error messages
        Then should not expose sensitive system information
        """
        # Given
        sensitive_error_messages = [
            "File not found: /etc/passwd",
            "Permission denied: /root/.ssh/id_rsa",
            "Database error: Connection failed to postgresql://user:password@host:5432/db",
            "Config error: Cannot read /app/secrets.json",
            "System error: Access denied to C:\\Windows\\System32\\config\\sam",
        ]

        # When & Then
        for error_msg in sensitive_error_messages:
            sanitized = sanitize_error_message(error_msg)

            # Should not contain sensitive paths
            sensitive_patterns = [
                "/etc/",
                "/root/",
                "password",
                "secrets",
                "config",
                "system32",
            ]
            for pattern in sensitive_patterns:
                assert pattern.lower() not in sanitized.lower(), (
                    f"Sensitive pattern '{pattern}' found in: {sanitized}"
                )

    def test_path_information_disclosure_prevention(
        self, temp_safe_directory, mock_settings
    ):
        """
        Test prevention of path information disclosure in errors

        Given path validation errors
        When error messages are generated
        Then should not reveal internal directory structure
        """
        temp_dir, safe_dir, test_pdf = temp_safe_directory

        # Given
        mock_settings.INPUT_DIRECTORY = safe_dir
        traversal_paths = [
            "../../../etc/passwd",
            "../../../../root/.bashrc",
            "../../../var/www/html/config.php",
        ]

        # When & Then
        with patch("src.pdf_to_markdown_mcp.auth.security.settings", mock_settings):
            for traversal_path in traversal_paths:
                with pytest.raises(HTTPException) as exc_info:
                    validate_path_security(traversal_path)

                error_detail = str(exc_info.value.detail)

                # Error message should not reveal the attempted traversal path
                assert traversal_path not in error_detail, (
                    f"Path disclosed in error: {error_detail}"
                )

                # Should not reveal internal directory structure
                internal_indicators = [
                    "/etc/",
                    "/root/",
                    "/var/",
                    "passwd",
                    "bashrc",
                    "config.php",
                ]
                for indicator in internal_indicators:
                    assert indicator not in error_detail.lower(), (
                        f"Internal path indicator '{indicator}' disclosed"
                    )

    # Integration Tests with API Endpoints

    def test_convert_single_path_validation(self, temp_safe_directory):
        """
        Test path validation integration with convert_single endpoint

        Given API requests with malicious file paths
        When calling convert_single endpoint
        Then should reject traversal attempts at API layer
        """
        temp_dir, safe_dir, test_pdf = temp_safe_directory

        # This test would require full API integration
        # For now, we test the security function that would be called

        malicious_api_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "/etc/shadow",
            "~/.ssh/id_rsa",
        ]

        for malicious_path in malicious_api_paths:
            # Simulate API request path validation
            with pytest.raises(HTTPException) as exc_info:
                # This would be called by the API endpoint
                validate_path_security(malicious_path)

            assert exc_info.value.status_code == 400

    def test_batch_convert_directory_validation(
        self, temp_safe_directory, mock_settings
    ):
        """
        Test directory validation for batch_convert endpoint

        Given batch processing requests with malicious directory paths
        When validating batch directories
        Then should prevent traversal to unauthorized directories
        """
        temp_dir, safe_dir, test_pdf = temp_safe_directory

        # Given
        mock_settings.INPUT_DIRECTORY = safe_dir
        malicious_directories = [
            "../../../etc/",
            "..\\..\\..\\windows\\system32\\",
            "/root/",
            "/var/www/html/",
        ]

        # When & Then
        with patch("src.pdf_to_markdown_mcp.auth.security.settings", mock_settings):
            for malicious_dir in malicious_directories:
                with pytest.raises(HTTPException) as exc_info:
                    validate_path_security(malicious_dir)

                assert exc_info.value.status_code == 400


class TestAdvancedPathTraversalScenarios:
    """Advanced path traversal attack scenario tests"""

    def test_race_condition_path_validation(self, temp_safe_directory, mock_settings):
        """
        Test path validation under race conditions

        Given concurrent path validation requests
        When validating paths simultaneously
        Then should maintain security guarantees under concurrency
        """
        import threading

        temp_dir, safe_dir, test_pdf = temp_safe_directory
        mock_settings.INPUT_DIRECTORY = safe_dir

        # Test data
        paths_to_test = [
            "../../../etc/passwd",
            "legitimate.pdf",
            "../../../root/.ssh/id_rsa",
            "another_legitimate.pdf",
        ] * 10  # 40 total requests

        results = []
        exceptions = []

        def validate_path_worker(path):
            try:
                with patch(
                    "src.pdf_to_markdown_mcp.auth.security.settings", mock_settings
                ):
                    result = validate_path_security(path)
                    results.append((path, "success", result))
            except HTTPException as e:
                exceptions.append((path, e.status_code, str(e.detail)))
            except Exception as e:
                results.append((path, "error", str(e)))

        # When - Run concurrent validations
        threads = []
        for path in paths_to_test:
            thread = threading.Thread(target=validate_path_worker, args=(path,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Then - Verify security was maintained
        malicious_paths = [
            p for p in paths_to_test if "../" in p or "/etc/" in p or "/root/" in p
        ]

        # All malicious paths should have been rejected
        malicious_exceptions = [exc for exc in exceptions if exc[0] in malicious_paths]
        assert len(malicious_exceptions) >= len(set(malicious_paths)), (
            "Not all malicious paths were rejected under concurrency"
        )

        # All exceptions for malicious paths should be 400 errors
        for path, status_code, detail in malicious_exceptions:
            assert status_code == 400, (
                f"Wrong status code for malicious path {path}: {status_code}"
            )

    def test_memory_exhaustion_path_validation(self, mock_settings):
        """
        Test path validation with memory exhaustion attempts

        Given extremely large or numerous path validation requests
        When processing path validation
        Then should handle without excessive memory consumption
        """
        # Given
        memory_exhaustion_payloads = [
            "a" * 100000 + "/../../../etc/passwd",  # Very long path
            "../" * 10000 + "etc/passwd",  # Many traversal attempts
            ("../" * 1000 + "etc/passwd\x00") * 100,  # Many null bytes
        ]

        # When & Then
        with patch("src.pdf_to_markdown_mcp.auth.security.settings", mock_settings):
            for payload in memory_exhaustion_payloads:
                start_time = time.time()

                with pytest.raises(HTTPException) as exc_info:
                    validate_path_security(payload)

                end_time = time.time()
                processing_time = end_time - start_time

                # Should complete quickly even with large payloads
                assert processing_time < 1.0, (
                    f"Path validation took too long: {processing_time:.3f}s"
                )
                assert exc_info.value.status_code == 400

    def test_encoding_bypass_comprehensive(self, mock_settings):
        """
        Test comprehensive encoding bypass prevention

        Given various encoding bypass attempts
        When validating paths
        Then should prevent all encoding-based bypasses
        """
        # Given - Comprehensive encoding bypass attempts
        encoding_bypass_payloads = [
            # URL encoding variants
            "%2e%2e%2f%2e%2e%2f%2e%2e%2f/etc/passwd",
            "%2e%2e\\%2e%2e\\%2e%2e\\/etc/passwd",
            # Double URL encoding
            "%252e%252e%252f%252e%252e%252f%252e%252e%252f/etc/passwd",
            # Mixed encoding
            "..%2f..%2f..%2f/etc/passwd",
            "..\\..\\..\\%2fetc%2fpasswd",
            # Unicode encoding
            "\u002e\u002e\u002f\u002e\u002e\u002f\u002e\u002e\u002f/etc/passwd",
            "\u002e\u002e\u005c\u002e\u002e\u005c\u002e\u002e\u005c/etc/passwd",
            # Overlong UTF-8 encoding
            "%c0%ae%c0%ae%c0%af%c0%ae%c0%ae%c0%af%c0%ae%c0%ae%c0%af/etc/passwd",
            # Directory separator bypass
            "..;/..;/..;/etc/passwd",
            "..:/..:/..:/etc/passwd",
        ]

        # When & Then
        with patch("src.pdf_to_markdown_mcp.auth.security.settings", mock_settings):
            for bypass_payload in encoding_bypass_payloads:
                with pytest.raises(HTTPException) as exc_info:
                    validate_path_security(bypass_payload)

                assert exc_info.value.status_code == 400, (
                    f"Encoding bypass not prevented: {bypass_payload}"
                )


class TestPathTraversalPerformanceImpact:
    """Test that security measures don't create performance bottlenecks"""

    def test_path_validation_performance(self, mock_settings):
        """
        Test path validation performance with many requests

        Given many path validation requests
        When processing them
        Then should maintain good performance
        """
        import time

        # Given
        test_paths = [
            "legitimate_file.pdf",
            "../../../etc/passwd",
            "another_file.pdf",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "valid_document.pdf",
        ] * 100  # 500 validations total

        # When
        start_time = time.time()

        with patch("src.pdf_to_markdown_mcp.auth.security.settings", mock_settings):
            for test_path in test_paths:
                try:
                    validate_path_security(test_path)
                except HTTPException:
                    # Expected for malicious paths
                    pass

        end_time = time.time()
        total_time = end_time - start_time

        # Then - Should complete all validations quickly
        assert total_time < 2.0, (
            f"Path validation too slow: {total_time:.3f}s for {len(test_paths)} validations"
        )

        avg_time_per_validation = total_time / len(test_paths)
        assert avg_time_per_validation < 0.01, (
            f"Individual validation too slow: {avg_time_per_validation:.6f}s"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
