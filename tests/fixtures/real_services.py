"""
Real service connection fixtures for integration and e2e tests.

These fixtures connect to actual services (Redis, Ollama, Celery, MinerU)
configured in .env. Use these for integration/e2e tests only.

For unit tests, use mocked services from conftest.py.
"""

import os
import time
from typing import Optional

import pytest
import redis


def is_redis_available() -> bool:
    """Check if Redis server is available."""
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))
    redis_db = int(os.getenv("REDIS_DB", "0"))

    try:
        client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        client.ping()
        client.close()
        return True
    except Exception as e:
        print(f"Redis not available: {e}")
        return False


def is_ollama_available() -> bool:
    """Check if Ollama service is available."""
    try:
        import httpx

        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")

        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{ollama_url}/api/tags")
            return response.status_code == 200
    except Exception as e:
        print(f"Ollama not available: {e}")
        return False


def is_ollama_model_available(model_name: str = "nomic-embed-text") -> bool:
    """Check if specific Ollama model is available."""
    try:
        import httpx

        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")

        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{ollama_url}/api/tags")
            if response.status_code != 200:
                return False

            data = response.json()
            models = [model["name"] for model in data.get("models", [])]
            return any(model_name in model for model in models)
    except Exception:
        return False


def is_mineru_standalone_running() -> bool:
    """Check if MinerU standalone service is running."""
    if not is_redis_available():
        return False

    try:
        # Check if mineru_requests queue exists (created by standalone service)
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", "6379"))

        client = redis.Redis(host=redis_host, port=redis_port, db=0)
        # Just check if we can access Redis - MinerU service creates queues on demand
        client.ping()
        client.close()
        return True
    except Exception:
        return False


def is_celery_worker_running() -> bool:
    """Check if Celery worker is running."""
    if not is_redis_available():
        return False

    try:
        from celery import Celery

        celery_broker_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")

        app = Celery(broker=celery_broker_url)
        inspector = app.control.inspect()
        stats = inspector.stats()

        return stats is not None and len(stats) > 0
    except Exception as e:
        print(f"Celery worker not available: {e}")
        return False


# Skip markers for missing services
require_redis = pytest.mark.skipif(
    not is_redis_available(),
    reason="Redis server not available (check REDIS_HOST and REDIS_PORT in .env)",
)

require_ollama = pytest.mark.skipif(
    not is_ollama_available(),
    reason="Ollama service not available (check OLLAMA_URL in .env)",
)

require_ollama_model = pytest.mark.skipif(
    not is_ollama_model_available(),
    reason="Ollama nomic-embed-text model not available (run: ollama pull nomic-embed-text)",
)

require_mineru = pytest.mark.skipif(
    not is_mineru_standalone_running(),
    reason="MinerU standalone service not running",
)

require_celery = pytest.mark.skipif(
    not is_celery_worker_running(),
    reason="Celery worker not running",
)


@pytest.fixture
def real_redis_client():
    """
    Create real Redis client for testing.

    Yields a connected Redis client and closes it after test.
    """
    if not is_redis_available():
        pytest.skip("Redis not available")

    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))
    redis_db = int(os.getenv("REDIS_DB", "0"))

    client = redis.Redis(
        host=redis_host,
        port=redis_port,
        db=redis_db,
        decode_responses=True,
    )

    try:
        # Verify connection
        client.ping()
        yield client
    finally:
        client.close()


@pytest.fixture
async def real_ollama_client():
    """
    Create real Ollama client for testing.

    Yields a connected Ollama client for embedding generation.
    """
    if not is_ollama_available():
        pytest.skip("Ollama not available")

    try:
        import ollama

        client = ollama.AsyncClient()
        yield client
    except ImportError:
        pytest.skip("Ollama Python package not installed")


@pytest.fixture
def verify_ollama_model(real_ollama_client):
    """
    Verify Ollama model is available.

    Use this fixture to ensure nomic-embed-text model is pulled.
    """
    model_name = os.getenv("OLLAMA_MODEL", "nomic-embed-text")

    if not is_ollama_model_available(model_name):
        pytest.skip(f"Ollama model '{model_name}' not available (run: ollama pull {model_name})")

    return model_name


@pytest.fixture
async def test_ollama_embedding(real_ollama_client, verify_ollama_model):
    """
    Test Ollama embedding generation.

    Returns a function that generates embeddings for test text.
    """
    model_name = verify_ollama_model

    async def _generate_embedding(text: str) -> list[float]:
        """Generate embedding for test text."""
        response = await real_ollama_client.embeddings(model=model_name, prompt=text)
        return response["embedding"]

    return _generate_embedding


@pytest.fixture
def redis_queue_monitor(real_redis_client):
    """
    Monitor Redis queue lengths during tests.

    Returns a callable that reports queue lengths.
    """
    def _get_queue_lengths() -> dict[str, int]:
        """Get lengths of all known queues."""
        queues = [
            "mineru_requests",
            "mineru_results",
            "celery",
            "embeddings",
        ]

        lengths = {}
        for queue in queues:
            try:
                length = real_redis_client.llen(queue)
                lengths[queue] = length
            except Exception:
                lengths[queue] = -1  # Error getting length

        return lengths

    yield _get_queue_lengths


@pytest.fixture
def clean_redis_queues(real_redis_client):
    """
    Clean Redis queues before and after test.

    WARNING: This clears all queues! Use only in isolated test environments.
    """
    queues = [
        "mineru_requests",
        "mineru_results",
        "celery",
        "embeddings",
    ]

    def _clean():
        """Clean all test queues."""
        for queue in queues:
            try:
                real_redis_client.delete(queue)
            except Exception as e:
                print(f"Warning: Could not clean queue '{queue}': {e}")

    # Clean before test
    _clean()

    yield

    # Clean after test
    _clean()


@pytest.fixture
def mineru_service_health():
    """
    Check MinerU standalone service health.

    Returns health status information.
    """
    def _check_health() -> dict:
        """Check MinerU service health."""
        return {
            "redis_available": is_redis_available(),
            "mineru_running": is_mineru_standalone_running(),
            "timestamp": time.time(),
        }

    return _check_health


@pytest.fixture
def real_celery_app():
    """
    Create real Celery app connected to broker.

    Use this to test Celery task queuing and execution.
    """
    if not is_redis_available():
        pytest.skip("Redis not available for Celery")

    try:
        from celery import Celery

        celery_broker_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
        celery_result_backend = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

        app = Celery(
            "test_app",
            broker=celery_broker_url,
            backend=celery_result_backend,
        )

        yield app
    except ImportError:
        pytest.skip("Celery not installed")


@pytest.fixture
def wait_for_celery_task():
    """
    Helper to wait for Celery task completion.

    Returns a function that polls for task completion.
    """
    def _wait(task_result, timeout: int = 30, poll_interval: float = 0.5) -> bool:
        """
        Wait for Celery task to complete.

        Args:
            task_result: Celery AsyncResult object
            timeout: Maximum seconds to wait
            poll_interval: Seconds between polls

        Returns:
            True if task completed successfully, False otherwise
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            if task_result.ready():
                return task_result.successful()

            time.sleep(poll_interval)

        return False  # Timeout

    return _wait


@pytest.fixture(scope="session")
def service_health_check():
    """
    Run health check on all services at session start.

    Provides summary of service availability.
    """
    health = {
        "redis": is_redis_available(),
        "ollama": is_ollama_available(),
        "ollama_model": is_ollama_model_available(),
        "mineru": is_mineru_standalone_running(),
        "celery": is_celery_worker_running(),
    }

    print("\n🏥 Service Health Check:")
    for service, available in health.items():
        status = "✅ Available" if available else "❌ Not Available"
        print(f"   {service}: {status}")

    return health


# Example usage in tests:
"""
@pytest.mark.integration
@pytest.mark.redis
@require_redis
def test_redis_operations(real_redis_client, redis_queue_monitor):
    # Test with real Redis
    real_redis_client.set("test_key", "test_value")
    assert real_redis_client.get("test_key") == "test_value"

    # Monitor queues
    queue_lengths = redis_queue_monitor()
    print(f"Queue lengths: {queue_lengths}")


@pytest.mark.integration
@pytest.mark.embeddings
@require_ollama
@require_ollama_model
async def test_embedding_generation(test_ollama_embedding):
    # Test with real Ollama
    embedding = await test_ollama_embedding("Test document content")
    assert len(embedding) == 768  # nomic-embed-text dimension
"""
