"""
Unit tests for health API endpoints.

Following TDD principles, these tests define the expected behavior
of the enhanced health check API endpoints.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from pdf_to_markdown_mcp.api.health import router
from pdf_to_markdown_mcp.core.monitoring import (
    ComponentHealth,
    HealthStatus,
    MetricsCollector,
)


class TestHealthEndpoints:
    """Test health API endpoints following TDD"""

    @pytest.fixture
    def mock_health_monitor(self):
        """Mock health monitor"""
        mock = AsyncMock()
        return mock

    @pytest.fixture
    def mock_metrics_collector(self):
        """Mock metrics collector"""
        return Mock(spec=MetricsCollector)

    @pytest.fixture
    def test_client(self, mock_health_monitor, mock_metrics_collector):
        """Create test client with mocked dependencies"""
        from fastapi import FastAPI
        from pdf_to_markdown_mcp.core.monitoring import metrics_collector

        app = FastAPI()
        app.include_router(router)

        # Inject mocked dependencies
        app.dependency_overrides = {}

        # Record some histogram observations so bucket metrics are generated
        metrics_collector.processing_duration_histogram.labels(processing_type="pdf").observe(1.5)
        metrics_collector.search_response_histogram.labels(search_type="fulltext").observe(0.25)

        return TestClient(app)

    def test_detailed_health_endpoint_returns_comprehensive_status(self, test_client):
        """Test that /health returns detailed health status for all components"""
        # Given
        expected_components = [
            "database",
            "embeddings",
            "celery",
            "system",
            "redis",
            "mineru",
        ]

        # When
        response = test_client.get("/health")

        # Then
        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert "service" in data
        assert "version" in data
        assert "timestamp" in data
        assert "uptime_seconds" in data
        assert "components" in data

        # Should include all major components
        for component in expected_components:
            assert component in data["components"]
            component_data = data["components"][component]
            assert "status" in component_data
            assert "last_check" in component_data

    def test_health_endpoint_returns_unhealthy_when_critical_component_fails(
        self, test_client
    ):
        """Test that overall status is unhealthy when critical components fail"""
        # Given - mock database failure
        with patch(
            "pdf_to_markdown_mcp.api.health.health_monitor.check_database_health"
        ) as mock_db_check:
            mock_db_check.return_value = ComponentHealth(
                status=HealthStatus.UNHEALTHY,
                last_check=123456789,
                details={"error": "Connection failed"},
            )

            # When
            response = test_client.get("/health")

            # Then
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "unhealthy"

    def test_readiness_endpoint_checks_essential_services(self, test_client):
        """Test that /ready endpoint checks only essential services for readiness"""
        # Given - all services ready
        with patch(
            "pdf_to_markdown_mcp.api.health.health_monitor.check_readiness"
        ) as mock_readiness:
            mock_readiness.return_value = {
                "ready": True,
                "checks": {"database": "ready", "configuration": "ready"},
            }

            # When
            response = test_client.get("/ready")

            # Then
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ready"
            assert "checks" in data
            assert "database" in data["checks"]

    def test_readiness_endpoint_returns_not_ready_when_essential_service_unavailable(
        self, test_client
    ):
        """Test that readiness fails when essential services are unavailable"""
        # Given - database not ready
        with patch(
            "pdf_to_markdown_mcp.api.health.health_monitor.check_readiness"
        ) as mock_readiness:
            mock_readiness.return_value = {
                "ready": False,
                "checks": {"database": "not_ready", "configuration": "ready"},
            }

            # When
            response = test_client.get("/ready")

            # Then
            assert response.status_code == 503  # Service Unavailable
            data = response.json()
            assert data["status"] == "not_ready"

    def test_metrics_endpoint_returns_prometheus_format(self, test_client):
        """Test that /metrics endpoint returns Prometheus-compatible metrics"""
        # Given
        expected_metrics = [
            "documents_processed_total",
            "document_processing_duration_seconds",
            "search_queries_total",
            "system_resource_usage",
            "celery_active_tasks",
            "processing_queue_depth",
        ]

        # When
        response = test_client.get("/metrics")

        # Then
        assert response.status_code == 200
        assert (
            response.headers["content-type"]
            == "text/plain; version=0.0.4; charset=utf-8"
        )

        metrics_text = response.text
        for metric in expected_metrics:
            assert metric in metrics_text

    def test_metrics_endpoint_includes_help_and_type_information(self, test_client):
        """Test that metrics endpoint includes proper Prometheus help and type info"""
        # When
        response = test_client.get("/metrics")

        # Then
        assert response.status_code == 200
        metrics_text = response.text

        # Should include HELP comments
        assert "# HELP" in metrics_text
        # Should include TYPE comments
        assert "# TYPE" in metrics_text
        # Should include counter metrics
        assert "_total" in metrics_text
        # Should include histogram metrics
        assert "_bucket" in metrics_text

    def test_health_endpoint_includes_response_times(self, test_client):
        """Test that health checks include response time measurements"""
        # When
        response = test_client.get("/health")

        # Then
        assert response.status_code == 200
        data = response.json()

        # Database health should include response time
        if "database" in data["components"]:
            db_component = data["components"]["database"]
            if db_component["status"] in ["healthy", "degraded"]:
                assert "response_time_ms" in db_component

    def test_health_endpoint_handles_partial_failures_gracefully(self, test_client):
        """Test that health endpoint continues working even if some checks fail"""
        # Given - embedding service fails but others work
        with patch(
            "pdf_to_markdown_mcp.api.health.health_monitor.check_embedding_health"
        ) as mock_embed_check:
            mock_embed_check.side_effect = Exception("Service unavailable")

            # When
            response = test_client.get("/health")

            # Then
            assert response.status_code == 200
            data = response.json()

            # Should still return valid response
            assert "status" in data
            assert "components" in data

            # Failed component should be marked as unhealthy
            if "embeddings" in data["components"]:
                assert data["components"]["embeddings"]["status"] == "unhealthy"

    def test_metrics_endpoint_handles_database_unavailability(self, test_client):
        """Test that metrics endpoint returns valid metrics even when some components fail"""
        # When - get metrics (some components may be unhealthy)
        response = test_client.get("/metrics")

        # Then - should still return valid Prometheus format
        assert response.status_code == 200
        metrics_text = response.text

        # Should include Prometheus format headers
        assert "# HELP" in metrics_text
        assert "# TYPE" in metrics_text
        # Should still include system-level metrics regardless of component health
        assert "python_" in metrics_text or "process_" in metrics_text

    def test_health_checks_include_correlation_ids(self, test_client):
        """Test that health check responses include correlation IDs for tracing"""
        # When
        response = test_client.get("/health")

        # Then
        assert response.status_code == 200

        # Should include correlation ID in response headers
        assert "X-Correlation-ID" in response.headers

        # Response should include correlation ID
        data = response.json()
        assert "correlation_id" in data or "X-Correlation-ID" in response.headers

    def test_detailed_health_includes_dependency_versions(self, test_client):
        """Test that detailed health check includes version information for dependencies"""
        # When
        response = test_client.get("/health/detailed")

        # Then
        assert response.status_code == 200
        data = response.json()

        # Should include version information
        assert "service_version" in data
        assert "dependencies" in data

        # Dependencies should include version info
        deps = data["dependencies"]
        expected_deps = ["postgresql", "redis", "python"]
        for dep in expected_deps:
            if dep in deps:
                assert "version" in deps[dep]
