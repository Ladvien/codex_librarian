"""
Integration test configuration and fixtures.

This conftest is automatically loaded by pytest for integration and e2e tests.
It imports all real resource fixtures (PostgreSQL, GPU, Redis, Ollama, etc.)
that connect to actual services from .env configuration.

DO NOT import this for unit tests - unit tests should use tests/conftest.py
which provides mocked services and SQLite database.

Usage:
    # In integration or e2e tests
    @pytest.mark.integration
    @pytest.mark.database
    @require_postgresql
    def test_with_real_database(real_db_session):
        # This uses real PostgreSQL, not SQLite
        pass
"""

import os
import sys
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import all real resource fixtures
# These fixtures connect to actual services (PostgreSQL, Redis, Ollama, GPU)
pytest_plugins = [
    "tests.fixtures.real_database",
    "tests.fixtures.real_gpu",
    "tests.fixtures.real_services",
]

# Re-export commonly used fixtures and markers for convenience
from tests.fixtures.real_database import (
    is_pgvector_available,
    is_postgresql_available,
    require_pgvector,
    require_postgresql,
)
from tests.fixtures.real_gpu import (
    is_cuda_available,
    get_gpu_info,
    require_gpu,
)
from tests.fixtures.real_services import (
    is_ollama_available,
    is_redis_available,
    require_ollama,
    require_ollama_model,
    require_redis,
    require_mineru,
    require_celery,
)


@pytest.fixture(scope="session", autouse=True)
def integration_test_environment():
    """
    Validate integration test environment at session start.

    Automatically runs for all integration/e2e tests.
    Provides warnings about missing services.
    """
    print("\n" + "=" * 80)
    print("INTEGRATION TEST ENVIRONMENT CHECK")
    print("=" * 80)

    # Check PostgreSQL
    postgres_ok = is_postgresql_available()
    print(f"PostgreSQL:     {'✅ Available' if postgres_ok else '❌ Not Available'}")

    # Check PGVector
    if postgres_ok:
        pgvector_ok = is_pgvector_available()
        print(f"PGVector:       {'✅ Available' if pgvector_ok else '❌ Not Available'}")

    # Check GPU/CUDA
    cuda_ok = is_cuda_available()
    print(f"CUDA/GPU:       {'✅ Available' if cuda_ok else '❌ Not Available'}")

    if cuda_ok:
        gpu_info = get_gpu_info()
        print(f"  - GPU Name:   {gpu_info.gpu_name}")
        print(f"  - GPU Memory: {gpu_info.total_memory_mb} MB")
        print(f"  - Device Mode: {gpu_info.device_mode}")

    # Check Redis
    redis_ok = is_redis_available()
    print(f"Redis:          {'✅ Available' if redis_ok else '❌ Not Available'}")

    # Check Ollama
    ollama_ok = is_ollama_available()
    print(f"Ollama:         {'✅ Available' if ollama_ok else '❌ Not Available'}")

    print("=" * 80)

    # Warnings for missing critical services
    missing_services = []
    if not postgres_ok:
        missing_services.append("PostgreSQL")
    if not cuda_ok:
        missing_services.append("GPU/CUDA")
    if not redis_ok:
        missing_services.append("Redis")
    if not ollama_ok:
        missing_services.append("Ollama")

    if missing_services:
        print(f"\n⚠️  WARNING: Missing services: {', '.join(missing_services)}")
        print("   Integration tests requiring these services will be skipped.")
        print("   See tests/README.md for setup instructions.\n")

    yield

    print("\n" + "=" * 80)
    print("INTEGRATION TEST SESSION COMPLETE")
    print("=" * 80)


@pytest.fixture
def integration_test_pdf():
    """
    Provide a real PDF file for integration testing.

    Returns path to a sample PDF file in the test fixtures directory.
    """
    # Use a sample PDF from fixtures if available
    fixtures_dir = Path(__file__).parent / "fixtures"
    sample_pdf = fixtures_dir / "sample.pdf"

    if sample_pdf.exists():
        return sample_pdf

    # Otherwise create a minimal valid PDF for testing
    import tempfile

    # Minimal PDF structure
    pdf_content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
endobj
4 0 obj
<< /Length 44 >>
stream
BT
/F1 12 Tf
100 700 Td
(Test PDF Document) Tj
ET
endstream
endobj
5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000267 00000 n
0000000361 00000 n
trailer
<< /Size 6 /Root 1 0 R >>
startxref
441
%%EOF
"""

    # Create temporary PDF file
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(pdf_content)
        temp_pdf_path = Path(f.name)

    return temp_pdf_path


@pytest.fixture
def integration_output_dir(tmp_path):
    """
    Provide temporary output directory for integration tests.

    Creates a clean directory for test outputs (markdown files, etc.)
    """
    output_dir = tmp_path / "integration_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture(scope="session")
def load_env_config():
    """
    Load environment configuration for integration tests.

    Returns dict with all relevant environment variables.
    """
    config = {
        # Database
        "database_url": os.getenv("DATABASE_URL"),
        "db_host": os.getenv("DB_HOST"),
        "db_port": os.getenv("DB_PORT"),
        "db_name": os.getenv("DB_NAME"),
        "db_user": os.getenv("DB_USER"),

        # Redis
        "redis_host": os.getenv("REDIS_HOST", "localhost"),
        "redis_port": int(os.getenv("REDIS_PORT", "6379")),
        "redis_db": int(os.getenv("REDIS_DB", "0")),

        # Ollama
        "ollama_url": os.getenv("OLLAMA_URL", "http://localhost:11434"),
        "ollama_model": os.getenv("OLLAMA_MODEL", "nomic-embed-text"),

        # GPU
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "mineru_device_mode": os.getenv("MINERU_DEVICE_MODE", "cpu"),

        # Celery
        "celery_broker_url": os.getenv("CELERY_BROKER_URL"),
        "celery_result_backend": os.getenv("CELERY_RESULT_BACKEND"),

        # Directories
        "watch_directories": os.getenv("WATCH_DIRECTORIES"),
        "output_directory": os.getenv("OUTPUT_DIRECTORY"),
    }

    return config


# Pytest hooks for integration tests
def pytest_collection_modifyitems(config, items):
    """
    Automatically add markers to integration and e2e tests.

    Tests in integration/ directory get @pytest.mark.integration
    Tests in e2e/ directory get @pytest.mark.e2e
    """
    for item in items:
        # Add integration marker to tests in integration/ directory
        if "integration" in str(item.fspath) and "test_" in item.name:
            if "integration" not in [marker.name for marker in item.iter_markers()]:
                item.add_marker(pytest.mark.integration)

        # Add e2e marker to tests in e2e/ directory
        if "e2e" in str(item.fspath) and "test_" in item.name:
            if "e2e" not in [marker.name for marker in item.iter_markers()]:
                item.add_marker(pytest.mark.e2e)

        # Add slow marker to e2e tests
        if "e2e" in [marker.name for marker in item.iter_markers()]:
            if "slow" not in [marker.name for marker in item.iter_markers()]:
                item.add_marker(pytest.mark.slow)


# Example usage documentation
"""
INTEGRATION TEST EXAMPLES:

1. Test with real PostgreSQL database:
   @pytest.mark.integration
   @pytest.mark.database
   @require_postgresql
   def test_database_operation(real_db_session):
       # Use real PostgreSQL
       pass

2. Test with real GPU:
   @pytest.mark.integration
   @pytest.mark.gpu
   @require_gpu
   def test_gpu_processing(verify_gpu_available, assert_gpu_used):
       # Use real CUDA GPU
       pass

3. Test with real Ollama embeddings:
   @pytest.mark.integration
   @pytest.mark.embeddings
   @require_ollama
   @require_ollama_model
   async def test_embeddings(test_ollama_embedding):
       # Use real Ollama service
       pass

4. Test with real Redis:
   @pytest.mark.integration
   @pytest.mark.redis
   @require_redis
   def test_redis(real_redis_client):
       # Use real Redis
       pass

For complete examples, see tests/integration/ and tests/e2e/ directories.
"""
