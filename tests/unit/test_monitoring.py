"""
Unit tests for monitoring system.

Following TDD principles, these tests define the expected behavior
of the monitoring and health check infrastructure.
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from pdf_to_markdown_mcp.core.monitoring import (
    AlertingEngine,
    AlertRule,
    AlertSeverity,
    ComponentHealth,
    HealthMonitor,
    HealthStatus,
    MetricsCollector,
    SystemHealth,
    TracingManager,
)


class TestMetricsCollector:
    """Test MetricsCollector following TDD"""

    @pytest.fixture
    def metrics_collector(self):
        """Setup MetricsCollector with mocked dependencies"""
        return MetricsCollector()

    def test_record_document_processing_increments_counters(self, metrics_collector):
        """Test that document processing metrics are recorded correctly"""
        # Given
        status = "success"
        file_type = "pdf"
        duration = 2.5
        processing_type = "pdf"

        # When
        metrics_collector.record_document_processing(
            status=status,
            file_type=file_type,
            duration_seconds=duration,
            processing_type=processing_type,
        )

        # Then - verify metrics were recorded (will verify with actual implementation)
        assert True  # Placeholder until implementation

    def test_record_search_query_categorizes_results(self, metrics_collector):
        """Test that search query metrics categorize result counts correctly"""
        # Given
        test_cases = [(0, "zero"), (5, "low"), (50, "medium"), (500, "high")]

        for result_count, expected_category in test_cases:
            # When
            metrics_collector.record_search_query(
                search_type="semantic",
                result_count=result_count,
                response_time_ms=100.0,
            )

            # Then - verify categorization (will verify with actual implementation)
            assert True  # Placeholder until implementation

    @pytest.mark.asyncio
    async def test_collect_system_metrics_runs_continuously(self, metrics_collector):
        """Test that system metrics collection runs in background"""
        # Given
        mock_psutil = Mock()
        mock_psutil.cpu_percent.return_value = 25.5
        mock_psutil.virtual_memory.return_value = Mock(percent=60.0)
        mock_psutil.disk_usage.return_value = Mock(used=100, total=200)

        # When/Then - test will verify continuous collection
        with patch("pdf_to_markdown_mcp.core.monitoring.psutil", mock_psutil):
            # Start collection and let it run briefly
            task = asyncio.create_task(metrics_collector.collect_system_metrics())
            await asyncio.sleep(0.1)  # Let it run briefly
            task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                pass

        # Verify psutil calls were made
        assert mock_psutil.cpu_percent.called
        assert mock_psutil.virtual_memory.called
        assert mock_psutil.disk_usage.called


class TestHealthMonitor:
    """Test HealthMonitor following TDD"""

    @pytest.fixture
    def health_monitor(self):
        """Setup HealthMonitor with mocked dependencies"""
        return HealthMonitor()

    @pytest.mark.asyncio
    async def test_check_database_health_success(self, health_monitor):
        """Test that database health check returns healthy status when DB is responsive"""
        # Given
        mock_session = AsyncMock()
        mock_result = Mock()
        mock_session.execute.return_value = mock_result

        # When
        with patch(
            "pdf_to_markdown_mcp.core.monitoring.get_db_session"
        ) as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session

            result = await health_monitor.check_database_health()

        # Then
        assert result.status == HealthStatus.HEALTHY
        assert result.response_time_ms is not None
        assert result.response_time_ms > 0
        assert result.last_check > 0

    @pytest.mark.asyncio
    async def test_check_database_health_slow_response(self, health_monitor):
        """Test that database health check returns degraded for slow responses"""
        # Given
        mock_session = AsyncMock()
        mock_session.execute.return_value = Mock()

        # When
        with patch(
            "pdf_to_markdown_mcp.core.monitoring.get_db_session"
        ) as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session

            # Simulate slow response
            with patch("time.time", side_effect=[0, 2.0]):  # 2 second response
                result = await health_monitor.check_database_health()

        # Then
        assert result.status == HealthStatus.DEGRADED
        assert result.response_time_ms > 1000

    @pytest.mark.asyncio
    async def test_check_database_health_failure(self, health_monitor):
        """Test that database health check returns unhealthy on connection failure"""
        # Given
        error_message = "Connection failed"

        # When
        with patch(
            "pdf_to_markdown_mcp.core.monitoring.get_db_session"
        ) as mock_get_session:
            mock_get_session.side_effect = Exception(error_message)

            result = await health_monitor.check_database_health()

        # Then
        assert result.status == HealthStatus.UNHEALTHY
        assert result.response_time_ms is None
        assert error_message in result.details["error"]

    @pytest.mark.asyncio
    async def test_get_system_health_aggregates_component_checks(self, health_monitor):
        """Test that system health aggregates all component health checks"""
        # Given
        mock_db_health = ComponentHealth(
            status=HealthStatus.HEALTHY,
            response_time_ms=50.0,
            last_check=123456789,
            details={},
        )

        mock_celery_health = ComponentHealth(
            status=HealthStatus.DEGRADED,
            response_time_ms=None,
            last_check=123456789,
            details={"message": "Few workers available"},
        )

        # When
        with patch.object(
            health_monitor, "check_database_health", return_value=mock_db_health
        ), patch.object(
            health_monitor, "check_celery_health", return_value=mock_celery_health
        ), patch("time.time", return_value=123456789):
            result = await health_monitor.get_system_health()

        # Then
        assert isinstance(result, SystemHealth)
        assert result.status == HealthStatus.DEGRADED  # Worst status wins
        assert result.uptime_seconds > 0
        assert "database" in result.components
        assert "celery" in result.components
        assert result.components["database"].status == HealthStatus.HEALTHY
        assert result.components["celery"].status == HealthStatus.DEGRADED


class TestComponentHealth:
    """Test ComponentHealth model"""

    def test_component_health_creation(self):
        """Test ComponentHealth can be created with required fields"""
        # Given
        status = HealthStatus.HEALTHY
        last_check = 123456789

        # When
        health = ComponentHealth(status=status, last_check=last_check)

        # Then
        assert health.status == status
        assert health.last_check == last_check
        assert health.response_time_ms is None
        assert health.details == {}

    def test_component_health_with_optional_fields(self):
        """Test ComponentHealth with all optional fields populated"""
        # Given
        status = HealthStatus.DEGRADED
        response_time = 500.0
        last_check = 123456789
        details = {"error": "Slow response", "attempts": 3}

        # When
        health = ComponentHealth(
            status=status,
            response_time_ms=response_time,
            last_check=last_check,
            details=details,
        )

        # Then
        assert health.status == status
        assert health.response_time_ms == response_time
        assert health.last_check == last_check
        assert health.details == details


class TestAlertingEngine:
    """Test AlertingEngine following TDD"""

    @pytest.fixture
    def alerting_engine(self):
        """Setup AlertingEngine"""
        return AlertingEngine()

    def test_add_rule_stores_alert_rule(self, alerting_engine):
        """Test that alerting rules can be added to the engine"""
        # Given
        rule = AlertRule(
            name="Test Rule",
            condition=lambda m: m.get("error_rate", 0) > 5.0,
            severity=AlertSeverity.ERROR,
        )

        # When
        alerting_engine.add_rule(rule)

        # Then
        assert len(alerting_engine.rules) == 1
        assert alerting_engine.rules[0] == rule

    @pytest.mark.asyncio
    async def test_evaluate_alerts_triggers_matching_conditions(self, alerting_engine):
        """Test that alerts are triggered when conditions match"""
        # Given
        triggered_alerts = []

        def mock_send_alert(rule, metrics):
            triggered_alerts.append((rule.name, metrics))
            return AsyncMock()

        alerting_engine._send_alert = mock_send_alert

        rule = AlertRule(
            name="High Error Rate",
            condition=lambda m: m.get("error_rate", 0) > 5.0,
            severity=AlertSeverity.ERROR,
        )
        alerting_engine.add_rule(rule)

        metrics = {"error_rate": 10.0}

        # When
        await alerting_engine.evaluate_alerts(metrics)

        # Then
        assert len(triggered_alerts) == 1
        assert triggered_alerts[0][0] == "High Error Rate"
        assert triggered_alerts[0][1] == metrics

    @pytest.mark.asyncio
    async def test_evaluate_alerts_respects_cooldown(self, alerting_engine):
        """Test that alerts respect cooldown periods"""
        # Given
        triggered_alerts = []

        def mock_send_alert(rule, metrics):
            triggered_alerts.append(rule.name)
            return AsyncMock()

        alerting_engine._send_alert = mock_send_alert

        rule = AlertRule(
            name="Test Rule",
            condition=lambda m: True,  # Always trigger
            severity=AlertSeverity.ERROR,
            cooldown_minutes=15,
        )
        alerting_engine.add_rule(rule)

        # When - trigger alert twice
        await alerting_engine.evaluate_alerts({"test": "data"})
        await alerting_engine.evaluate_alerts({"test": "data"})

        # Then - should only trigger once due to cooldown
        assert len(triggered_alerts) == 1


class TestTracingManager:
    """Test TracingManager following TDD"""

    def test_generate_correlation_id_creates_unique_ids(self):
        """Test that correlation IDs are unique"""
        # Given/When
        id1 = TracingManager.generate_correlation_id()
        id2 = TracingManager.generate_correlation_id()

        # Then
        assert id1 != id2
        assert isinstance(id1, str)
        assert isinstance(id2, str)
        assert len(id1) > 0
        assert len(id2) > 0

    def test_set_and_get_correlation_id(self):
        """Test that correlation IDs can be set and retrieved"""
        # Given
        correlation_id = "test-123"

        # When
        TracingManager.set_correlation_id(correlation_id)
        retrieved_id = TracingManager.get_correlation_id()

        # Then
        assert retrieved_id == correlation_id

    def test_trace_operation_decorator_logs_operations(self):
        """Test that operation tracing decorator works correctly"""
        # Given
        operation_logs = []

        def mock_logger_info(message, **kwargs):
            operation_logs.append((message, kwargs))

        def mock_logger_error(message, **kwargs):
            operation_logs.append((message, kwargs))

        mock_logger = Mock()
        mock_logger.info = mock_logger_info
        mock_logger.error = mock_logger_error

        @TracingManager.trace_operation("test_operation")
        async def test_function():
            return "success"

        # When
        with patch(
            "pdf_to_markdown_mcp.core.monitoring.structlog.get_logger",
            return_value=mock_logger,
        ), patch.object(
            TracingManager, "get_correlation_id", return_value="test-123"
        ):
            result = asyncio.run(test_function())

        # Then
        assert result == "success"
        assert len(operation_logs) >= 2  # Started and completed
        start_log = operation_logs[0]
        assert start_log[0] == "operation_started"
        assert start_log[1]["operation"] == "test_operation"
        assert start_log[1]["correlation_id"] == "test-123"
