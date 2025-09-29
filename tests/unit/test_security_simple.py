"""
Simple security tests to verify current state without complex dependencies.
"""

import os
import sys

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from fastapi.testclient import TestClient


def test_security_headers_missing():
    """Test that security headers are currently missing (should fail)."""
    # Import here to avoid dependency issues
    from pdf_to_markdown_mcp.main import app

    client = TestClient(app)
    response = client.get("/")

    # These should currently be missing
    expected_headers = [
        "X-Content-Type-Options",
        "X-Frame-Options",
        "X-XSS-Protection",
        "Strict-Transport-Security",
        "Content-Security-Policy",
    ]

    missing_headers = []
    for header in expected_headers:
        if header not in response.headers:
            missing_headers.append(header)

    # Should have missing headers currently (this will pass initially, then fail after we implement)
    assert len(missing_headers) > 0, (
        f"Expected missing headers but found: {[h for h in expected_headers if h in response.headers]}"
    )
    print(f"Missing security headers: {missing_headers}")


def test_cors_too_permissive():
    """Test that CORS is currently too permissive (should fail)."""
    from pdf_to_markdown_mcp.config import settings

    # Should be permissive currently
    assert "*" in settings.cors_origins, (
        f"CORS origins should be permissive but got: {settings.cors_origins}"
    )
    print(f"CORS origins: {settings.cors_origins}")


if __name__ == "__main__":
    test_security_headers_missing()
    test_cors_too_permissive()
    print("Current security state confirmed - ready to implement fixes!")
