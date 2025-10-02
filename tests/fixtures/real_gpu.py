"""
Real GPU/CUDA fixtures for integration and e2e tests.

These fixtures validate GPU availability and monitor GPU usage during tests.
Use these for integration/e2e tests that require GPU acceleration.

For unit tests, use mocked MinerU services from conftest.py.
"""

import os
import subprocess
from dataclasses import dataclass
from typing import Optional

import pytest


@dataclass
class GPUInfo:
    """GPU information and capabilities."""

    cuda_available: bool
    cuda_version: Optional[str]
    gpu_count: int
    gpu_name: Optional[str]
    total_memory_mb: Optional[int]
    device_mode: str  # "cuda" or "cpu"


@dataclass
class GPUMemorySnapshot:
    """GPU memory usage snapshot."""

    used_mb: float
    free_mb: float
    total_mb: float
    utilization_percent: float


def is_cuda_available() -> bool:
    """Check if CUDA is available via PyTorch."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def get_gpu_info() -> GPUInfo:
    """Get GPU information and capabilities."""
    cuda_available = is_cuda_available()

    if not cuda_available:
        return GPUInfo(
            cuda_available=False,
            cuda_version=None,
            gpu_count=0,
            gpu_name=None,
            total_memory_mb=None,
            device_mode="cpu",
        )

    try:
        import torch

        cuda_version = torch.version.cuda
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else None
        total_memory = torch.cuda.get_device_properties(0).total_memory if gpu_count > 0 else None
        total_memory_mb = int(total_memory / 1024 / 1024) if total_memory else None

        device_mode = os.getenv("MINERU_DEVICE_MODE", "cpu")

        return GPUInfo(
            cuda_available=True,
            cuda_version=cuda_version,
            gpu_count=gpu_count,
            gpu_name=gpu_name,
            total_memory_mb=total_memory_mb,
            device_mode=device_mode,
        )
    except Exception as e:
        print(f"Error getting GPU info: {e}")
        return GPUInfo(
            cuda_available=False,
            cuda_version=None,
            gpu_count=0,
            gpu_name=None,
            total_memory_mb=None,
            device_mode="cpu",
        )


def get_gpu_memory_usage() -> Optional[GPUMemorySnapshot]:
    """
    Get current GPU memory usage.

    Returns None if GPU not available or query fails.
    """
    if not is_cuda_available():
        return None

    try:
        import torch

        if torch.cuda.device_count() == 0:
            return None

        torch.cuda.synchronize()  # Wait for all operations to complete

        allocated = torch.cuda.memory_allocated(0) / 1024 / 1024  # MB
        reserved = torch.cuda.memory_reserved(0) / 1024 / 1024  # MB
        total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024  # MB

        used = reserved  # Use reserved memory as "used"
        free = total - used
        utilization = (used / total) * 100 if total > 0 else 0.0

        return GPUMemorySnapshot(
            used_mb=used,
            free_mb=free,
            total_mb=total,
            utilization_percent=utilization,
        )
    except Exception as e:
        print(f"Error getting GPU memory usage: {e}")
        return None


def get_nvidia_smi_memory() -> Optional[dict]:
    """
    Get GPU memory from nvidia-smi command.

    Fallback method when PyTorch not available.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode != 0:
            return None

        used, free, total = map(float, result.stdout.strip().split(","))

        return {
            "used_mb": used,
            "free_mb": free,
            "total_mb": total,
            "utilization_percent": (used / total) * 100 if total > 0 else 0.0,
        }
    except Exception:
        return None


# Skip GPU tests if CUDA not available
require_gpu = pytest.mark.skipif(
    not is_cuda_available(),
    reason="GPU/CUDA not available (check CUDA_VISIBLE_DEVICES and GPU drivers)",
)


@pytest.fixture(scope="session")
def gpu_info() -> GPUInfo:
    """
    Get GPU information for the test session.

    Skips tests if GPU not available.
    """
    info = get_gpu_info()

    if not info.cuda_available:
        pytest.skip("GPU/CUDA not available for testing")

    return info


@pytest.fixture(scope="session")
def verify_gpu_available(gpu_info):
    """
    Verify GPU is available and properly configured.

    This fixture will skip the test if GPU is not available.
    """
    assert gpu_info.cuda_available, "CUDA must be available"
    assert gpu_info.gpu_count > 0, "At least one GPU must be available"
    assert gpu_info.device_mode == "cuda", (
        f"MINERU_DEVICE_MODE should be 'cuda', got '{gpu_info.device_mode}'"
    )
    return True


@pytest.fixture
def gpu_memory_monitor():
    """
    Monitor GPU memory usage before and after test.

    Returns a callable that captures GPU memory snapshots.
    """
    snapshots = []

    def _capture(label: str = "snapshot") -> Optional[GPUMemorySnapshot]:
        """Capture GPU memory snapshot with a label."""
        snapshot = get_gpu_memory_usage()
        if snapshot:
            snapshots.append((label, snapshot))
        return snapshot

    # Capture initial state
    _capture("before_test")

    yield _capture

    # Capture final state
    _capture("after_test")

    # Report if there's significant memory increase (potential leak)
    if len(snapshots) >= 2:
        before = snapshots[0][1]
        after = snapshots[-1][1]
        delta_mb = after.used_mb - before.used_mb

        if delta_mb > 500:  # More than 500MB increase
            print(f"\n⚠️  GPU memory increased by {delta_mb:.1f} MB during test")
            print(f"   Before: {before.used_mb:.1f} MB")
            print(f"   After: {after.used_mb:.1f} MB")


@pytest.fixture
def assert_gpu_used():
    """
    Assert that GPU was actually used during the test.

    Call this fixture's returned function after GPU operations
    to verify GPU was utilized (not CPU fallback).
    """
    def _assert_gpu_was_used():
        """
        Verify GPU memory increased, indicating GPU was used.

        This is a simple heuristic: if GPU memory usage increased,
        then GPU was likely used for computation.
        """
        snapshot = get_gpu_memory_usage()

        if snapshot is None:
            pytest.fail("Cannot verify GPU usage - GPU not available")

        # Check that there's some GPU memory usage
        # Real GPU processing should use at least 100MB
        assert snapshot.used_mb > 100, (
            f"GPU memory usage too low ({snapshot.used_mb:.1f} MB). "
            "Test may have fallen back to CPU processing."
        )

        return snapshot

    return _assert_gpu_was_used


@pytest.fixture
def measure_gpu_performance():
    """
    Measure GPU performance metrics during test execution.

    Returns a context manager that tracks timing and memory.
    """
    import time
    from contextlib import contextmanager

    @contextmanager
    def _measure(operation_name: str = "operation"):
        """Measure GPU performance for an operation."""
        start_memory = get_gpu_memory_usage()
        start_time = time.time()

        try:
            yield
        finally:
            end_time = time.time()
            end_memory = get_gpu_memory_usage()

            duration = end_time - start_time
            print(f"\n📊 GPU Performance - {operation_name}:")
            print(f"   Duration: {duration:.2f} seconds")

            if start_memory and end_memory:
                memory_delta = end_memory.used_mb - start_memory.used_mb
                print(f"   Memory delta: {memory_delta:+.1f} MB")
                print(f"   Peak utilization: {end_memory.utilization_percent:.1f}%")

    return _measure


@pytest.fixture(scope="session")
def check_mineru_gpu_config():
    """
    Check MinerU GPU configuration from environment.

    Validates that GPU is properly configured for MinerU.
    """
    cuda_devices = os.getenv("CUDA_VISIBLE_DEVICES")
    device_mode = os.getenv("MINERU_DEVICE_MODE", "cpu")

    if cuda_devices is None:
        pytest.skip("CUDA_VISIBLE_DEVICES not set")

    if device_mode != "cuda":
        pytest.skip(f"MINERU_DEVICE_MODE is '{device_mode}', not 'cuda'")

    if not is_cuda_available():
        pytest.skip("CUDA not available even though configured")

    return {
        "cuda_devices": cuda_devices,
        "device_mode": device_mode,
        "gpu_info": get_gpu_info(),
    }


@pytest.fixture
def print_gpu_info(gpu_info):
    """
    Print GPU information at test start.

    Useful for debugging GPU-related test failures.
    """
    print(f"\n🖥️  GPU Configuration:")
    print(f"   CUDA Available: {gpu_info.cuda_available}")
    print(f"   CUDA Version: {gpu_info.cuda_version}")
    print(f"   GPU Count: {gpu_info.gpu_count}")
    print(f"   GPU Name: {gpu_info.gpu_name}")
    print(f"   Total Memory: {gpu_info.total_memory_mb} MB")
    print(f"   Device Mode: {gpu_info.device_mode}")

    snapshot = get_gpu_memory_usage()
    if snapshot:
        print(f"   Current Usage: {snapshot.used_mb:.1f} / {snapshot.total_mb:.1f} MB")
        print(f"   Utilization: {snapshot.utilization_percent:.1f}%")

    yield


# Example usage in tests:
"""
@pytest.mark.integration
@pytest.mark.gpu
@require_gpu
def test_gpu_processing(verify_gpu_available, gpu_memory_monitor, assert_gpu_used):
    # Capture before processing
    gpu_memory_monitor("before_processing")

    # Do GPU processing here
    result = process_with_gpu()

    # Capture after processing
    gpu_memory_monitor("after_processing")

    # Assert GPU was actually used
    assert_gpu_used()

    assert result is not None
"""
