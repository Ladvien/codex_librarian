"""
Tests for security middleware including headers and CORS configuration.

Following TDD approach: RED-GREEN-REFACTOR
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from pdf_to_markdown_mcp.config import settings
from pdf_to_markdown_mcp.main import app


class TestSecurityHeaders:
    """Test security headers middleware implementation."""

    def test_security_headers_present_in_response(self):
        """Test that all required security headers are present."""
        client = TestClient(app)

        response = client.get("/")

        # Expected security headers
        expected_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'",
            "Referrer-Policy": "strict-origin-when-cross-origin",
        }

        for header_name, expected_value in expected_headers.items():
            assert header_name in response.headers, (
                f"Missing security header: {header_name}"
            )
            assert response.headers[header_name] == expected_value

    def test_sse_endpoint_has_proper_headers(self):
        """Test that Server-Sent Events endpoints have appropriate security headers."""
        client = TestClient(app)

        # SSE endpoint would be something like /api/v1/stream_progress
        response = client.get("/api/v1/stream_progress/test-job")

        # Should have security headers but modified CSP for SSE
        assert "Content-Security-Policy" in response.headers
        assert "X-Frame-Options" in response.headers

        # CSP might be different for SSE to allow event streams
        csp_header = response.headers["Content-Security-Policy"]
        assert "default-src 'self'" in csp_header

    def test_cors_headers_in_development(self):
        """Test CORS headers in development environment."""
        with patch.dict("os.environ", {"ENVIRONMENT": "development"}):
            client = TestClient(app)

            # OPTIONS request to test CORS
            response = client.options("/api/v1/convert_single")

            # Should allow development origins
            assert "Access-Control-Allow-Origin" in response.headers

    def test_cors_headers_in_production(self):
        """Test CORS headers are restricted in production."""
        with patch.dict(
            "os.environ",
            {"ENVIRONMENT": "production", "CORS_ORIGINS": "https://myapp.com"},
        ):
            client = TestClient(app)

            # Request with allowed origin
            headers = {"Origin": "https://myapp.com"}
            response = client.options("/api/v1/convert_single", headers=headers)

            # Should allow specific origins only
            assert (
                response.headers.get("Access-Control-Allow-Origin")
                == "https://myapp.com"
            )

    def test_cors_headers_block_unauthorized_origins(self):
        """Test CORS blocks unauthorized origins in production."""
        with patch.dict(
            "os.environ",
            {"ENVIRONMENT": "production", "CORS_ORIGINS": "https://myapp.com"},
        ):
            client = TestClient(app)

            # Request with disallowed origin
            headers = {"Origin": "https://evil.com"}
            response = client.options("/api/v1/convert_single", headers=headers)

            # Should not allow unauthorized origins
            assert (
                response.headers.get("Access-Control-Allow-Origin")
                != "https://evil.com"
            )

    def test_security_headers_not_expose_server_info(self):
        """Test that security headers don't expose server information."""
        client = TestClient(app)

        response = client.get("/")

        # Should not have server identification headers
        assert (
            "Server" not in response.headers
            or "FastAPI" not in response.headers.get("Server", "")
        )
        assert "X-Powered-By" not in response.headers

    def test_correlation_id_header_present(self):
        """Test that correlation ID is present in response headers."""
        client = TestClient(app)

        response = client.get("/")

        assert "X-Correlation-ID" in response.headers
        assert (
            len(response.headers["X-Correlation-ID"]) > 10
        )  # Should be a UUID-like string


class TestCORSConfiguration:
    """Test CORS configuration based on environment."""

    def test_development_cors_permissive(self):
        """Test that development allows permissive CORS."""
        with patch.object(settings, "environment", "development"):
            with patch.object(settings, "cors_origins", ["*"]):
                client = TestClient(app)

                response = client.options(
                    "/api/v1/convert_single",
                    headers={"Origin": "http://localhost:3000"},
                )

                # Development should be more permissive
                assert response.status_code in [200, 204]

    def test_production_cors_restrictive(self):
        """Test that production restricts CORS to specific origins."""
        production_origins = ["https://myapp.com", "https://api.myapp.com"]

        with patch.object(settings, "environment", "production"):
            with patch.object(settings, "cors_origins", production_origins):
                client = TestClient(app)

                # Test allowed origin
                response = client.options(
                    "/api/v1/convert_single", headers={"Origin": "https://myapp.com"}
                )
                assert response.status_code in [200, 204]

                # Test disallowed origin
                response = client.options(
                    "/api/v1/convert_single", headers={"Origin": "https://evil.com"}
                )
                # Should still work but without CORS headers for evil.com

    def test_cors_credentials_handling(self):
        """Test CORS credentials are handled properly."""
        client = TestClient(app)

        response = client.options(
            "/api/v1/convert_single", headers={"Origin": "http://localhost:3000"}
        )

        # Check if credentials are properly configured
        credentials_header = response.headers.get("Access-Control-Allow-Credentials")
        if credentials_header:
            assert credentials_header.lower() in ["true", "false"]


if __name__ == "__main__":
    pytest.main([__file__])
