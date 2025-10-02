"""
MinerU Pool Manager.

Manages multiple MinerU service instances for parallel PDF processing.
Distributes requests across instances using round-robin and health monitoring.
"""

import logging
import os
from typing import Dict, List, Optional

import redis

logger = logging.getLogger(__name__)


class MinerUPoolManager:
    """
    Pool manager for multiple MinerU instances.

    Distributes PDF processing requests across multiple MinerU service instances
    to maximize GPU utilization and throughput.
    """

    def __init__(
        self,
        instance_count: int = 3,
        redis_url: str = None,
        health_check_interval: int = 60
    ):
        """
        Initialize pool manager.

        Args:
            instance_count: Number of MinerU instances in the pool
            redis_url: Redis connection URL
            health_check_interval: Seconds between health checks
        """
        self.instance_count = instance_count
        self.health_check_interval = health_check_interval

        # Get redis URL from settings if not provided
        if redis_url is None:
            redis_host = os.getenv("REDIS_HOST", "localhost")
            redis_port = os.getenv("REDIS_PORT", "6379")
            redis_db = os.getenv("REDIS_DB", "0")
            redis_url = f"redis://{redis_host}:{redis_port}/{redis_db}"

        self.redis_url = redis_url
        self.redis_client = redis.from_url(redis_url, decode_responses=False)

        # Track if we need cleanup
        self._cleanup_required = True

        # Round-robin counter for load distribution
        self.next_instance = 0

        # Health status tracking
        self.instance_health: Dict[int, dict] = {}
        for i in range(instance_count):
            self.instance_health[i] = {
                "healthy": True,
                "queue_length": 0,
                "last_check": 0,
                "consecutive_failures": 0
            }

        logger.info(
            f"MinerU Pool Manager initialized: {instance_count} instances"
        )

    def cleanup(self):
        """
        Clean up Redis connection resources.

        Critical: Prevents connection leaks in long-running processes.
        """
        if self._cleanup_required and self.redis_client:
            try:
                self.redis_client.close()
                logger.info("MinerU Pool Manager: Redis connection closed")
            except Exception as e:
                logger.error(f"Error closing Redis connection: {e}")
            finally:
                self.redis_client = None
                self._cleanup_required = False

    def __del__(self):
        """Destructor to ensure cleanup on garbage collection."""
        self.cleanup()

    def get_queue_names(self, instance_id: int) -> tuple[str, str]:
        """
        Get request and result queue names for an instance.

        Args:
            instance_id: Instance ID

        Returns:
            Tuple of (request_queue, result_queue) names
        """
        return (
            f"mineru_requests_{instance_id}",
            f"mineru_results_{instance_id}"
        )

    def get_next_instance(self, prefer_least_loaded: bool = True) -> int:
        """
        Get next instance to use for processing.

        Args:
            prefer_least_loaded: If True, prefer instance with shortest queue

        Returns:
            Instance ID to use
        """
        if prefer_least_loaded:
            # Find healthy instance with shortest queue
            min_queue_length = float('inf')
            best_instance = 0

            for instance_id in range(self.instance_count):
                health = self.instance_health[instance_id]
                if health["healthy"] and health["queue_length"] < min_queue_length:
                    min_queue_length = health["queue_length"]
                    best_instance = instance_id

            return best_instance
        else:
            # Simple round-robin
            instance_id = self.next_instance
            self.next_instance = (self.next_instance + 1) % self.instance_count
            return instance_id

    def check_instance_health(self, instance_id: int) -> dict:
        """
        Check health of a specific instance.

        Args:
            instance_id: Instance ID to check

        Returns:
            Health status dictionary
        """
        try:
            request_queue, _ = self.get_queue_names(instance_id)

            # Check queue length as proxy for health
            queue_length = self.redis_client.llen(request_queue)

            # Instance is healthy if queue exists and is not overloaded
            healthy = queue_length < 100  # Threshold for "overloaded"

            # Update health status
            self.instance_health[instance_id].update({
                "healthy": healthy,
                "queue_length": queue_length,
                "consecutive_failures": 0 if healthy else self.instance_health[instance_id]["consecutive_failures"] + 1
            })

            return {
                "instance_id": instance_id,
                "healthy": healthy,
                "queue_length": queue_length,
                "request_queue": request_queue
            }

        except Exception as e:
            logger.error(f"Health check failed for instance {instance_id}: {e}")
            self.instance_health[instance_id]["consecutive_failures"] += 1
            self.instance_health[instance_id]["healthy"] = False

            return {
                "instance_id": instance_id,
                "healthy": False,
                "error": str(e)
            }

    def check_pool_health(self) -> dict:
        """
        Check health of all instances in the pool.

        Returns:
            Pool health status
        """
        instance_statuses = []
        healthy_count = 0

        for instance_id in range(self.instance_count):
            status = self.check_instance_health(instance_id)
            instance_statuses.append(status)
            if status.get("healthy", False):
                healthy_count += 1

        return {
            "total_instances": self.instance_count,
            "healthy_instances": healthy_count,
            "unhealthy_instances": self.instance_count - healthy_count,
            "instances": instance_statuses,
            "pool_healthy": healthy_count > 0  # At least one instance must be healthy
        }

    def get_available_instance(self, max_queue_length: int = 50) -> Optional[int]:
        """
        Get an available instance with queue below threshold.

        Args:
            max_queue_length: Maximum acceptable queue length

        Returns:
            Instance ID if available, None if all overloaded
        """
        for instance_id in range(self.instance_count):
            health = self.instance_health[instance_id]
            if health["healthy"] and health["queue_length"] < max_queue_length:
                return instance_id

        # Fallback: return instance with shortest queue even if over threshold
        min_queue = float('inf')
        best_instance = 0

        for instance_id in range(self.instance_count):
            queue_len = self.instance_health[instance_id]["queue_length"]
            if queue_len < min_queue:
                min_queue = queue_len
                best_instance = instance_id

        logger.warning(
            f"All instances overloaded, using instance {best_instance} "
            f"with queue length {min_queue}"
        )
        return best_instance

    def get_pool_stats(self) -> dict:
        """
        Get comprehensive pool statistics.

        Returns:
            Pool statistics dictionary
        """
        total_queue_length = sum(
            health["queue_length"]
            for health in self.instance_health.values()
        )

        avg_queue_length = total_queue_length / self.instance_count

        return {
            "instance_count": self.instance_count,
            "total_queue_length": total_queue_length,
            "avg_queue_length": round(avg_queue_length, 2),
            "instance_health": self.instance_health,
            "next_instance": self.next_instance
        }

    def distribute_workload(self) -> List[int]:
        """
        Get recommended workload distribution across instances.

        Returns:
            List of instance IDs sorted by recommended usage priority
        """
        # Sort instances by queue length (least loaded first)
        sorted_instances = sorted(
            range(self.instance_count),
            key=lambda i: self.instance_health[i]["queue_length"]
        )

        # Filter out unhealthy instances
        healthy_instances = [
            i for i in sorted_instances
            if self.instance_health[i]["healthy"]
        ]

        if not healthy_instances:
            logger.warning("No healthy instances available!")
            return sorted_instances  # Return all even if unhealthy

        return healthy_instances


# Global singleton instance
_pool_manager: Optional[MinerUPoolManager] = None


def get_mineru_pool(instance_count: int = None) -> MinerUPoolManager:
    """
    Get or create global MinerU pool manager.

    Args:
        instance_count: Number of instances (only used on first call)

    Returns:
        MinerUPoolManager instance
    """
    global _pool_manager

    if _pool_manager is None:
        if instance_count is None:
            instance_count = int(os.getenv("MINERU_INSTANCE_COUNT", "3"))

        _pool_manager = MinerUPoolManager(instance_count=instance_count)
        logger.info(f"Created global MinerU pool manager with {instance_count} instances")

    return _pool_manager
