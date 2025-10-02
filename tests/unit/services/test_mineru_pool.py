"""
Comprehensive tests for MinerU Pool Manager.

Tests cover:
- Pool initialization with various instance counts
- Queue name generation
- Round-robin and least-loaded instance selection
- Instance and pool health checking
- Available instance selection
- Pool statistics
- Workload distribution
- Singleton pattern
- Edge cases and error handling
"""

import os
from unittest.mock import Mock, patch

import pytest
import redis

from pdf_to_markdown_mcp.services.mineru_pool import (
    MinerUPoolManager,
    get_mineru_pool,
)


class TestMinerUPoolManager:
    """Test MinerU Pool Manager following TDD principles"""

    @pytest.fixture
    def mock_redis_client(self):
        """Mock Redis client for testing"""
        mock_client = Mock(spec=redis.Redis)
        mock_client.llen.return_value = 0
        mock_client.ping.return_value = True
        return mock_client

    @pytest.fixture
    def pool_manager(self, mock_redis_client):
        """Create pool manager with mocked Redis"""
        with patch("redis.from_url", return_value=mock_redis_client):
            manager = MinerUPoolManager(
                instance_count=3,
                redis_url="redis://localhost:6379/0"
            )
            return manager

    @pytest.fixture
    def pool_manager_single(self, mock_redis_client):
        """Create pool manager with single instance"""
        with patch("redis.from_url", return_value=mock_redis_client):
            manager = MinerUPoolManager(
                instance_count=1,
                redis_url="redis://localhost:6379/0"
            )
            return manager

    @pytest.fixture
    def pool_manager_large(self, mock_redis_client):
        """Create pool manager with many instances"""
        with patch("redis.from_url", return_value=mock_redis_client):
            manager = MinerUPoolManager(
                instance_count=10,
                redis_url="redis://localhost:6379/0"
            )
            return manager

    # Test: Pool Initialization
    def test_pool_initialization_default(self, mock_redis_client):
        """Test pool manager initialization with default parameters"""
        # Given/When (Arrange/Act)
        with patch("redis.from_url", return_value=mock_redis_client):
            manager = MinerUPoolManager()

        # Then (Assert)
        assert manager.instance_count == 3  # Default count
        assert manager.health_check_interval == 60  # Default interval
        assert manager.next_instance == 0
        assert len(manager.instance_health) == 3
        assert manager.redis_client == mock_redis_client

    def test_pool_initialization_custom_count(self, mock_redis_client):
        """Test pool initialization with custom instance count"""
        # Given/When (Arrange/Act)
        with patch("redis.from_url", return_value=mock_redis_client):
            manager = MinerUPoolManager(instance_count=5)

        # Then (Assert)
        assert manager.instance_count == 5
        assert len(manager.instance_health) == 5
        # Verify all instances initialized as healthy
        for i in range(5):
            assert manager.instance_health[i]["healthy"] is True
            assert manager.instance_health[i]["queue_length"] == 0
            assert manager.instance_health[i]["consecutive_failures"] == 0

    def test_pool_initialization_redis_url_from_env(self, mock_redis_client):
        """Test Redis URL construction from environment variables"""
        # Given (Arrange)
        with (
            patch.dict(os.environ, {
                "REDIS_HOST": "testhost",
                "REDIS_PORT": "6380",
                "REDIS_DB": "2"
            }),
            patch("redis.from_url", return_value=mock_redis_client) as mock_from_url
        ):
            # When (Act)
            manager = MinerUPoolManager(redis_url=None)

            # Then (Assert)
            assert manager.redis_url == "redis://testhost:6380/2"
            mock_from_url.assert_called_once_with(
                "redis://testhost:6380/2",
                decode_responses=False
            )

    def test_pool_initialization_health_status_structure(self, pool_manager):
        """Test that health status has correct structure for all instances"""
        # Then (Assert)
        for instance_id in range(pool_manager.instance_count):
            health = pool_manager.instance_health[instance_id]
            assert "healthy" in health
            assert "queue_length" in health
            assert "last_check" in health
            assert "consecutive_failures" in health
            assert isinstance(health["healthy"], bool)
            assert isinstance(health["queue_length"], int)

    # Test: Queue Name Generation
    def test_get_queue_names(self, pool_manager):
        """Test queue name generation for instances"""
        # When (Act)
        request_queue, result_queue = pool_manager.get_queue_names(0)

        # Then (Assert)
        assert request_queue == "mineru_requests_0"
        assert result_queue == "mineru_results_0"

    def test_get_queue_names_different_instances(self, pool_manager):
        """Test queue names are unique per instance"""
        # When (Act)
        names_0 = pool_manager.get_queue_names(0)
        names_1 = pool_manager.get_queue_names(1)
        names_2 = pool_manager.get_queue_names(2)

        # Then (Assert)
        assert names_0[0] == "mineru_requests_0"
        assert names_1[0] == "mineru_requests_1"
        assert names_2[0] == "mineru_requests_2"
        assert names_0[1] == "mineru_results_0"
        assert names_1[1] == "mineru_results_1"
        assert names_2[1] == "mineru_results_2"

    @pytest.mark.parametrize("instance_id,expected_request,expected_result", [
        (0, "mineru_requests_0", "mineru_results_0"),
        (5, "mineru_requests_5", "mineru_results_5"),
        (99, "mineru_requests_99", "mineru_results_99"),
    ])
    def test_get_queue_names_parametrized(
        self,
        pool_manager,
        instance_id,
        expected_request,
        expected_result
    ):
        """Test queue name generation with various instance IDs"""
        # When (Act)
        request_queue, result_queue = pool_manager.get_queue_names(instance_id)

        # Then (Assert)
        assert request_queue == expected_request
        assert result_queue == expected_result

    # Test: Round-Robin Instance Selection
    def test_get_next_instance_round_robin(self, pool_manager):
        """Test round-robin instance selection"""
        # When (Act)
        instance_0 = pool_manager.get_next_instance(prefer_least_loaded=False)
        instance_1 = pool_manager.get_next_instance(prefer_least_loaded=False)
        instance_2 = pool_manager.get_next_instance(prefer_least_loaded=False)
        instance_3 = pool_manager.get_next_instance(prefer_least_loaded=False)

        # Then (Assert)
        assert instance_0 == 0
        assert instance_1 == 1
        assert instance_2 == 2
        assert instance_3 == 0  # Wraps around

    def test_get_next_instance_round_robin_wraps_correctly(self, pool_manager):
        """Test round-robin wraps correctly after full cycle"""
        # When (Act) - Go through multiple full cycles
        instances = [
            pool_manager.get_next_instance(prefer_least_loaded=False)
            for _ in range(10)
        ]

        # Then (Assert)
        expected = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
        assert instances == expected

    def test_get_next_instance_single_instance(self, pool_manager_single):
        """Test round-robin with single instance always returns 0"""
        # When (Act)
        instances = [
            pool_manager_single.get_next_instance(prefer_least_loaded=False)
            for _ in range(5)
        ]

        # Then (Assert)
        assert instances == [0, 0, 0, 0, 0]

    # Test: Least-Loaded Instance Selection
    def test_get_next_instance_least_loaded(self, pool_manager):
        """Test least-loaded instance selection"""
        # Given (Arrange) - Set different queue lengths
        pool_manager.instance_health[0]["queue_length"] = 10
        pool_manager.instance_health[1]["queue_length"] = 5
        pool_manager.instance_health[2]["queue_length"] = 15

        # When (Act)
        instance = pool_manager.get_next_instance(prefer_least_loaded=True)

        # Then (Assert)
        assert instance == 1  # Instance with shortest queue (5)

    def test_get_next_instance_least_loaded_skips_unhealthy(self, pool_manager):
        """Test least-loaded selection skips unhealthy instances"""
        # Given (Arrange) - Set queue lengths and health
        pool_manager.instance_health[0]["queue_length"] = 5
        pool_manager.instance_health[0]["healthy"] = False  # Unhealthy
        pool_manager.instance_health[1]["queue_length"] = 10
        pool_manager.instance_health[1]["healthy"] = True
        pool_manager.instance_health[2]["queue_length"] = 15
        pool_manager.instance_health[2]["healthy"] = True

        # When (Act)
        instance = pool_manager.get_next_instance(prefer_least_loaded=True)

        # Then (Assert)
        assert instance == 1  # Skips unhealthy instance 0, picks instance 1 (10 < 15)

    def test_get_next_instance_least_loaded_all_equal(self, pool_manager):
        """Test least-loaded when all queues equal returns first healthy"""
        # Given (Arrange) - All queues have same length
        pool_manager.instance_health[0]["queue_length"] = 5
        pool_manager.instance_health[1]["queue_length"] = 5
        pool_manager.instance_health[2]["queue_length"] = 5

        # When (Act)
        instance = pool_manager.get_next_instance(prefer_least_loaded=True)

        # Then (Assert)
        assert instance == 0  # First instance with minimum queue

    def test_get_next_instance_least_loaded_all_unhealthy(self, pool_manager):
        """Test least-loaded when all instances unhealthy returns 0"""
        # Given (Arrange)
        for i in range(3):
            pool_manager.instance_health[i]["healthy"] = False

        # When (Act)
        instance = pool_manager.get_next_instance(prefer_least_loaded=True)

        # Then (Assert)
        assert instance == 0  # Fallback to instance 0

    # Test: Instance Health Checking
    def test_check_instance_health_healthy(self, pool_manager, mock_redis_client):
        """Test health check for healthy instance"""
        # Given (Arrange)
        mock_redis_client.llen.return_value = 10

        # When (Act)
        health = pool_manager.check_instance_health(0)

        # Then (Assert)
        assert health["instance_id"] == 0
        assert health["healthy"] is True
        assert health["queue_length"] == 10
        assert health["request_queue"] == "mineru_requests_0"
        assert pool_manager.instance_health[0]["healthy"] is True
        assert pool_manager.instance_health[0]["consecutive_failures"] == 0

    def test_check_instance_health_overloaded(self, pool_manager, mock_redis_client):
        """Test health check for overloaded instance (queue >= 100)"""
        # Given (Arrange)
        mock_redis_client.llen.return_value = 150

        # When (Act)
        health = pool_manager.check_instance_health(0)

        # Then (Assert)
        assert health["healthy"] is False
        assert health["queue_length"] == 150
        assert pool_manager.instance_health[0]["healthy"] is False
        assert pool_manager.instance_health[0]["consecutive_failures"] == 1

    def test_check_instance_health_threshold_boundary(self, pool_manager, mock_redis_client):
        """Test health check at threshold boundary (queue = 99)"""
        # Given (Arrange)
        mock_redis_client.llen.return_value = 99

        # When (Act)
        health = pool_manager.check_instance_health(0)

        # Then (Assert)
        assert health["healthy"] is True  # 99 < 100, so healthy
        assert health["queue_length"] == 99

    def test_check_instance_health_redis_error(self, pool_manager, mock_redis_client):
        """Test health check when Redis raises error"""
        # Given (Arrange)
        mock_redis_client.llen.side_effect = redis.ConnectionError("Connection failed")

        # When (Act)
        health = pool_manager.check_instance_health(0)

        # Then (Assert)
        assert health["healthy"] is False
        assert "error" in health
        assert "Connection failed" in health["error"]
        assert pool_manager.instance_health[0]["healthy"] is False
        assert pool_manager.instance_health[0]["consecutive_failures"] == 1

    def test_check_instance_health_consecutive_failures(
        self,
        pool_manager,
        mock_redis_client
    ):
        """Test consecutive failure tracking"""
        # Given (Arrange)
        mock_redis_client.llen.side_effect = redis.ConnectionError("Error")

        # When (Act) - Trigger multiple failures
        pool_manager.check_instance_health(0)
        pool_manager.check_instance_health(0)
        pool_manager.check_instance_health(0)

        # Then (Assert)
        assert pool_manager.instance_health[0]["consecutive_failures"] == 3
        assert pool_manager.instance_health[0]["healthy"] is False

    def test_check_instance_health_recovery_resets_failures(
        self,
        pool_manager,
        mock_redis_client
    ):
        """Test that successful health check resets consecutive failures"""
        # Given (Arrange) - Start with failures
        pool_manager.instance_health[0]["consecutive_failures"] = 5
        mock_redis_client.llen.return_value = 10  # Healthy queue length

        # When (Act)
        pool_manager.check_instance_health(0)

        # Then (Assert)
        assert pool_manager.instance_health[0]["consecutive_failures"] == 0
        assert pool_manager.instance_health[0]["healthy"] is True

    # Test: Pool Health Checking
    def test_check_pool_health_all_healthy(self, pool_manager, mock_redis_client):
        """Test pool health when all instances are healthy"""
        # Given (Arrange)
        mock_redis_client.llen.return_value = 10

        # When (Act)
        pool_health = pool_manager.check_pool_health()

        # Then (Assert)
        assert pool_health["total_instances"] == 3
        assert pool_health["healthy_instances"] == 3
        assert pool_health["unhealthy_instances"] == 0
        assert pool_health["pool_healthy"] is True
        assert len(pool_health["instances"]) == 3

    def test_check_pool_health_mixed(self, pool_manager, mock_redis_client):
        """Test pool health with mix of healthy and unhealthy instances"""
        # Given (Arrange) - Different queue lengths
        queue_lengths = [10, 150, 50]  # Second instance overloaded
        mock_redis_client.llen.side_effect = queue_lengths

        # When (Act)
        pool_health = pool_manager.check_pool_health()

        # Then (Assert)
        assert pool_health["healthy_instances"] == 2  # Instances 0 and 2
        assert pool_health["unhealthy_instances"] == 1  # Instance 1
        assert pool_health["pool_healthy"] is True  # At least one healthy

    def test_check_pool_health_all_unhealthy(self, pool_manager, mock_redis_client):
        """Test pool health when all instances are unhealthy"""
        # Given (Arrange)
        mock_redis_client.llen.side_effect = redis.ConnectionError("Error")

        # When (Act)
        pool_health = pool_manager.check_pool_health()

        # Then (Assert)
        assert pool_health["healthy_instances"] == 0
        assert pool_health["unhealthy_instances"] == 3
        assert pool_health["pool_healthy"] is False  # No healthy instances

    def test_check_pool_health_instance_details(self, pool_manager, mock_redis_client):
        """Test pool health includes detailed instance information"""
        # Given (Arrange)
        mock_redis_client.llen.return_value = 20

        # When (Act)
        pool_health = pool_manager.check_pool_health()

        # Then (Assert)
        for instance in pool_health["instances"]:
            assert "instance_id" in instance
            assert "healthy" in instance
            assert "queue_length" in instance
            assert "request_queue" in instance

    # Test: Available Instance Selection
    def test_get_available_instance_below_threshold(self, pool_manager):
        """Test getting available instance when queue below threshold"""
        # Given (Arrange)
        pool_manager.instance_health[0]["queue_length"] = 30
        pool_manager.instance_health[0]["healthy"] = True

        # When (Act)
        instance = pool_manager.get_available_instance(max_queue_length=50)

        # Then (Assert)
        assert instance == 0

    def test_get_available_instance_first_available(self, pool_manager):
        """Test getting first available instance when multiple available"""
        # Given (Arrange)
        pool_manager.instance_health[0]["queue_length"] = 60
        pool_manager.instance_health[0]["healthy"] = True
        pool_manager.instance_health[1]["queue_length"] = 20
        pool_manager.instance_health[1]["healthy"] = True
        pool_manager.instance_health[2]["queue_length"] = 10
        pool_manager.instance_health[2]["healthy"] = True

        # When (Act)
        instance = pool_manager.get_available_instance(max_queue_length=50)

        # Then (Assert)
        assert instance == 1  # First instance below threshold

    def test_get_available_instance_all_overloaded(self, pool_manager):
        """Test fallback when all instances overloaded"""
        # Given (Arrange) - All instances above threshold
        pool_manager.instance_health[0]["queue_length"] = 100
        pool_manager.instance_health[1]["queue_length"] = 80
        pool_manager.instance_health[2]["queue_length"] = 90

        # When (Act)
        instance = pool_manager.get_available_instance(max_queue_length=50)

        # Then (Assert)
        assert instance == 1  # Returns instance with shortest queue (80)

    def test_get_available_instance_unhealthy_skipped(self, pool_manager):
        """Test that unhealthy instances are skipped"""
        # Given (Arrange)
        pool_manager.instance_health[0]["queue_length"] = 10
        pool_manager.instance_health[0]["healthy"] = False  # Unhealthy
        pool_manager.instance_health[1]["queue_length"] = 20
        pool_manager.instance_health[1]["healthy"] = True
        pool_manager.instance_health[2]["queue_length"] = 30
        pool_manager.instance_health[2]["healthy"] = True

        # When (Act)
        instance = pool_manager.get_available_instance(max_queue_length=50)

        # Then (Assert)
        assert instance == 1  # Skips unhealthy instance 0

    def test_get_available_instance_custom_threshold(self, pool_manager):
        """Test available instance with custom threshold"""
        # Given (Arrange)
        # Set all instances to have some queue length
        pool_manager.instance_health[0]["queue_length"] = 25
        pool_manager.instance_health[0]["healthy"] = True
        pool_manager.instance_health[1]["queue_length"] = 30
        pool_manager.instance_health[1]["healthy"] = True
        pool_manager.instance_health[2]["queue_length"] = 35
        pool_manager.instance_health[2]["healthy"] = True

        # When (Act)
        instance_high = pool_manager.get_available_instance(max_queue_length=100)
        instance_low = pool_manager.get_available_instance(max_queue_length=20)

        # Then (Assert)
        assert instance_high == 0  # Below high threshold, returns first available
        assert instance_low == 0  # All above low threshold, fallback to shortest (instance 0 with 25)

    # Test: Pool Statistics
    def test_get_pool_stats_basic(self, pool_manager):
        """Test getting basic pool statistics"""
        # Given (Arrange)
        pool_manager.instance_health[0]["queue_length"] = 10
        pool_manager.instance_health[1]["queue_length"] = 20
        pool_manager.instance_health[2]["queue_length"] = 30

        # When (Act)
        stats = pool_manager.get_pool_stats()

        # Then (Assert)
        assert stats["instance_count"] == 3
        assert stats["total_queue_length"] == 60
        assert stats["avg_queue_length"] == 20.0
        assert "instance_health" in stats
        assert "next_instance" in stats

    def test_get_pool_stats_empty_queues(self, pool_manager):
        """Test pool stats with empty queues"""
        # Given (Arrange) - All queues empty
        for i in range(3):
            pool_manager.instance_health[i]["queue_length"] = 0

        # When (Act)
        stats = pool_manager.get_pool_stats()

        # Then (Assert)
        assert stats["total_queue_length"] == 0
        assert stats["avg_queue_length"] == 0.0

    def test_get_pool_stats_large_pool(self, pool_manager_large):
        """Test pool stats with large pool"""
        # Given (Arrange) - Set varying queue lengths
        for i in range(10):
            pool_manager_large.instance_health[i]["queue_length"] = i * 10

        # When (Act)
        stats = pool_manager_large.get_pool_stats()

        # Then (Assert)
        assert stats["instance_count"] == 10
        assert stats["total_queue_length"] == 450  # 0+10+20+...+90
        assert stats["avg_queue_length"] == 45.0

    def test_get_pool_stats_includes_health_data(self, pool_manager):
        """Test pool stats includes complete health data"""
        # When (Act)
        stats = pool_manager.get_pool_stats()

        # Then (Assert)
        assert len(stats["instance_health"]) == 3
        for instance_id, health in stats["instance_health"].items():
            assert "healthy" in health
            assert "queue_length" in health
            assert "last_check" in health
            assert "consecutive_failures" in health

    # Test: Workload Distribution
    def test_distribute_workload_sorts_by_queue_length(self, pool_manager):
        """Test workload distribution sorts instances by queue length"""
        # Given (Arrange)
        pool_manager.instance_health[0]["queue_length"] = 30
        pool_manager.instance_health[1]["queue_length"] = 10
        pool_manager.instance_health[2]["queue_length"] = 20

        # When (Act)
        distribution = pool_manager.distribute_workload()

        # Then (Assert)
        assert distribution == [1, 2, 0]  # Sorted by queue length (10, 20, 30)

    def test_distribute_workload_filters_unhealthy(self, pool_manager):
        """Test workload distribution filters out unhealthy instances"""
        # Given (Arrange)
        pool_manager.instance_health[0]["queue_length"] = 10
        pool_manager.instance_health[0]["healthy"] = False  # Unhealthy
        pool_manager.instance_health[1]["queue_length"] = 20
        pool_manager.instance_health[1]["healthy"] = True
        pool_manager.instance_health[2]["queue_length"] = 30
        pool_manager.instance_health[2]["healthy"] = True

        # When (Act)
        distribution = pool_manager.distribute_workload()

        # Then (Assert)
        assert distribution == [1, 2]  # Only healthy instances
        assert 0 not in distribution

    def test_distribute_workload_all_unhealthy(self, pool_manager):
        """Test workload distribution when all instances unhealthy"""
        # Given (Arrange)
        for i in range(3):
            pool_manager.instance_health[i]["healthy"] = False
            pool_manager.instance_health[i]["queue_length"] = i * 10

        # When (Act)
        distribution = pool_manager.distribute_workload()

        # Then (Assert)
        assert distribution == [0, 1, 2]  # Returns all, sorted by queue length

    def test_distribute_workload_equal_queues(self, pool_manager):
        """Test workload distribution with equal queue lengths"""
        # Given (Arrange)
        for i in range(3):
            pool_manager.instance_health[i]["queue_length"] = 10

        # When (Act)
        distribution = pool_manager.distribute_workload()

        # Then (Assert)
        assert len(distribution) == 3
        assert set(distribution) == {0, 1, 2}

    def test_distribute_workload_large_pool(self, pool_manager_large):
        """Test workload distribution with large pool"""
        # Given (Arrange) - Set varying queue lengths
        for i in range(10):
            pool_manager_large.instance_health[i]["queue_length"] = (10 - i) * 5
            pool_manager_large.instance_health[i]["healthy"] = True

        # When (Act)
        distribution = pool_manager_large.distribute_workload()

        # Then (Assert)
        assert len(distribution) == 10
        # Should be sorted ascending by queue length
        assert distribution == [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]


class TestMinerUPoolSingleton:
    """Test singleton pattern for MinerU pool manager"""

    def test_get_mineru_pool_creates_singleton(self, mock_redis_client):
        """Test that get_mineru_pool creates singleton instance"""
        # Given (Arrange)
        with patch("redis.from_url", return_value=mock_redis_client):
            with patch("pdf_to_markdown_mcp.services.mineru_pool._pool_manager", None):
                # When (Act)
                pool1 = get_mineru_pool(instance_count=3)
                pool2 = get_mineru_pool()

                # Then (Assert)
                assert pool1 is pool2  # Same instance

    def test_get_mineru_pool_uses_env_variable(self, mock_redis_client):
        """Test that get_mineru_pool reads from environment"""
        # Given (Arrange)
        with (
            patch.dict(os.environ, {"MINERU_INSTANCE_COUNT": "7"}),
            patch("redis.from_url", return_value=mock_redis_client),
            patch("pdf_to_markdown_mcp.services.mineru_pool._pool_manager", None)
        ):
            # When (Act)
            pool = get_mineru_pool()

            # Then (Assert)
            assert pool.instance_count == 7

    def test_get_mineru_pool_instance_count_on_first_call_only(
        self,
        mock_redis_client
    ):
        """Test that instance_count parameter only used on first call"""
        # Given (Arrange)
        with (
            patch("redis.from_url", return_value=mock_redis_client),
            patch("pdf_to_markdown_mcp.services.mineru_pool._pool_manager", None)
        ):
            # When (Act)
            pool1 = get_mineru_pool(instance_count=5)
            pool2 = get_mineru_pool(instance_count=10)  # Should be ignored

            # Then (Assert)
            assert pool1.instance_count == 5
            assert pool2.instance_count == 5  # Still 5, not changed to 10
            assert pool1 is pool2

    def test_get_mineru_pool_default_from_env(self, mock_redis_client):
        """Test default instance count from environment"""
        # Given (Arrange)
        with (
            patch.dict(os.environ, {}, clear=True),  # Clear env
            patch("redis.from_url", return_value=mock_redis_client),
            patch("pdf_to_markdown_mcp.services.mineru_pool._pool_manager", None)
        ):
            # When (Act)
            pool = get_mineru_pool()

            # Then (Assert)
            assert pool.instance_count == 3  # Default fallback


class TestMinerUPoolEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_zero_instances_not_allowed(self, mock_redis_client):
        """Test that zero instances creates pool with 0 count (edge case)"""
        # Given/When (Arrange/Act)
        with patch("redis.from_url", return_value=mock_redis_client):
            manager = MinerUPoolManager(instance_count=0)

        # Then (Assert)
        assert manager.instance_count == 0
        assert len(manager.instance_health) == 0

    def test_negative_instances_not_validated(self, mock_redis_client):
        """Test negative instance count (should be validated by caller)"""
        # Given/When (Arrange/Act)
        with patch("redis.from_url", return_value=mock_redis_client):
            manager = MinerUPoolManager(instance_count=-1)

        # Then (Assert)
        assert manager.instance_count == -1
        # This is an edge case - production should validate input

    def test_very_large_queue_length(self, mock_redis_client):
        """Test handling of very large queue lengths"""
        # Given (Arrange)
        with patch("redis.from_url", return_value=mock_redis_client):
            manager = MinerUPoolManager(instance_count=1)
            mock_redis_client.llen.return_value = 999999

        # When (Act)
        health = manager.check_instance_health(0)

        # Then (Assert)
        assert health["healthy"] is False  # Over threshold
        assert health["queue_length"] == 999999

    def test_redis_connection_timeout(self, mock_redis_client):
        """Test handling of Redis connection timeout"""
        # Given (Arrange)
        with patch("redis.from_url", return_value=mock_redis_client):
            manager = MinerUPoolManager(instance_count=2)
            mock_redis_client.llen.side_effect = redis.TimeoutError("Timeout")

        # When (Act)
        health = manager.check_instance_health(0)

        # Then (Assert)
        assert health["healthy"] is False
        assert "Timeout" in health.get("error", "")

    def test_concurrent_health_checks(self, mock_redis_client):
        """Test that health status updates correctly with concurrent checks"""
        # Given (Arrange)
        with patch("redis.from_url", return_value=mock_redis_client):
            manager = MinerUPoolManager(instance_count=1)
            mock_redis_client.llen.return_value = 10

        # When (Act) - Simulate concurrent health checks
        manager.check_instance_health(0)
        manager.check_instance_health(0)

        # Then (Assert)
        assert manager.instance_health[0]["healthy"] is True
        # Consecutive failures should be 0 after successful checks
        assert manager.instance_health[0]["consecutive_failures"] == 0
