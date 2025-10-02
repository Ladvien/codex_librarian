"""
GPU Memory Management Utilities.

Provides utilities for checking GPU memory availability to prevent
CUDA OOM errors when multiple processes share the GPU.
"""

import logging
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)


def get_gpu_memory_info() -> tuple[float, float]:
    """
    Get GPU memory usage information.

    Returns:
        Tuple of (used_memory_gb, total_memory_gb)

    Raises:
        RuntimeError: If nvidia-smi command fails
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )

        # Parse output: "used_mb, total_mb"
        output = result.stdout.strip()
        used_mb, total_mb = map(float, output.split(","))

        # Convert to GB
        used_gb = used_mb / 1024
        total_gb = total_mb / 1024

        return used_gb, total_gb

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"nvidia-smi command failed: {e.stderr}")
    except subprocess.TimeoutExpired:
        raise RuntimeError("nvidia-smi command timed out")
    except (ValueError, IndexError) as e:
        raise RuntimeError(f"Failed to parse nvidia-smi output: {e}")


def get_available_gpu_memory() -> float:
    """
    Get available GPU memory in GB.

    Returns:
        Available memory in GB

    Raises:
        RuntimeError: If GPU memory check fails
    """
    used_gb, total_gb = get_gpu_memory_info()
    available_gb = total_gb - used_gb

    logger.debug(
        f"GPU memory: {used_gb:.2f} GB used, {total_gb:.2f} GB total, "
        f"{available_gb:.2f} GB available"
    )

    return available_gb


def has_sufficient_gpu_memory(required_gb: float = 7.0) -> bool:
    """
    Check if GPU has sufficient available memory.

    Args:
        required_gb: Required memory in GB (default: 7.0 for MinerU)

    Returns:
        True if sufficient memory available, False otherwise
    """
    try:
        available_gb = get_available_gpu_memory()
        sufficient = available_gb >= required_gb

        if not sufficient:
            logger.warning(
                f"Insufficient GPU memory: {available_gb:.2f} GB available, "
                f"{required_gb:.2f} GB required"
            )
        else:
            logger.debug(
                f"Sufficient GPU memory: {available_gb:.2f} GB available, "
                f"{required_gb:.2f} GB required"
            )

        return sufficient

    except RuntimeError as e:
        logger.error(f"Failed to check GPU memory: {e}")
        # Assume sufficient memory if check fails (fail-open to avoid blocking)
        return True


def wait_for_gpu_memory(
    required_gb: float = 7.0, timeout_seconds: Optional[int] = None
) -> bool:
    """
    Wait until sufficient GPU memory is available.

    Args:
        required_gb: Required memory in GB
        timeout_seconds: Maximum time to wait (None = no timeout)

    Returns:
        True if memory became available, False if timed out
    """
    import time

    start_time = time.time()

    while True:
        if has_sufficient_gpu_memory(required_gb):
            return True

        if timeout_seconds is not None:
            elapsed = time.time() - start_time
            if elapsed >= timeout_seconds:
                logger.warning(
                    f"Timeout waiting for GPU memory after {elapsed:.1f}s"
                )
                return False

        # Wait before checking again
        time.sleep(5)
