"""
GPU Resource Manager for Celery Workers.

Manages GPU resources across worker processes to prevent contention and optimize performance.
"""

import logging
import os
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class GPUResourceManager:
    """Manages GPU resources for optimal worker performance."""

    def __init__(self):
        self._gpu_lock = threading.Lock()
        self._gpu_in_use = False
        self._current_process_id = None
        self._models_loaded = False
        self._model_cache = {}

    @contextmanager
    def acquire_gpu(self, process_id: Optional[str] = None):
        """Context manager for exclusive GPU access."""
        process_id = process_id or f"worker_{os.getpid()}"

        with self._gpu_lock:
            if self._gpu_in_use and self._current_process_id != process_id:
                logger.warning(f"GPU already in use by {self._current_process_id}, waiting...")
                # In single worker mode, this shouldn't happen
                time.sleep(0.1)

            self._gpu_in_use = True
            self._current_process_id = process_id
            logger.debug(f"GPU acquired by {process_id}")

        try:
            yield
        finally:
            with self._gpu_lock:
                self._gpu_in_use = False
                self._current_process_id = None
                logger.debug(f"GPU released by {process_id}")

    def setup_gpu_environment(self) -> Dict[str, Any]:
        """Configure optimal GPU environment settings."""
        config = {}

        try:
            import torch

            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                current_device = torch.cuda.current_device()
                device_name = torch.cuda.get_device_name(current_device)

                # Clear GPU cache
                torch.cuda.empty_cache()

                # Set memory management
                torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of GPU memory

                config.update({
                    "gpu_available": True,
                    "device_count": device_count,
                    "current_device": current_device,
                    "device_name": device_name,
                    "memory_allocated": torch.cuda.memory_allocated(),
                    "memory_reserved": torch.cuda.memory_reserved(),
                })

                logger.info(f"GPU configured: {device_name} (Device {current_device})")

            else:
                config["gpu_available"] = False
                logger.warning("CUDA not available, using CPU mode")

        except ImportError:
            config["gpu_available"] = False
            logger.warning("PyTorch not available, cannot configure GPU")
        except Exception as e:
            config["gpu_available"] = False
            logger.error(f"GPU setup failed: {e}")

        return config

    def preload_models(self) -> bool:
        """Preload GPU models for faster processing."""
        if self._models_loaded:
            return True

        try:
            logger.info("Preloading GPU models...")

            # Set environment for GPU mode
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            os.environ["MINERU_DEVICE_MODE"] = "cuda"

            # Preload MinerU models
            from ..services.mineru import get_shared_mineru_instance
            mineru_service = get_shared_mineru_instance()

            if mineru_service.validate_mineru_dependency():
                self._models_loaded = True
                logger.info("GPU models preloaded successfully")
                return True
            else:
                logger.warning("MinerU dependency not available")
                return False

        except Exception as e:
            logger.error(f"Model preloading failed: {e}")
            return False

    def get_gpu_status(self) -> Dict[str, Any]:
        """Get current GPU status and utilization."""
        status = {
            "gpu_lock_acquired": self._gpu_in_use,
            "current_process": self._current_process_id,
            "models_loaded": self._models_loaded,
        }

        try:
            import torch

            if torch.cuda.is_available():
                status.update({
                    "gpu_available": True,
                    "memory_allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
                    "memory_reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024,
                    "memory_cached_mb": torch.cuda.memory_cached() / 1024 / 1024,
                })
            else:
                status["gpu_available"] = False

        except ImportError:
            status["gpu_available"] = False

        return status

    def cleanup_gpu_resources(self):
        """Clean up GPU resources and cache."""
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info("GPU cache cleared")

        except (ImportError, Exception) as e:
            logger.debug(f"GPU cleanup skipped: {e}")

    def optimize_memory_usage(self):
        """Optimize GPU memory usage during processing."""
        try:
            import torch

            if torch.cuda.is_available():
                # Force garbage collection
                import gc
                gc.collect()
                torch.cuda.empty_cache()

                # Get memory stats
                allocated = torch.cuda.memory_allocated() / 1024 / 1024
                reserved = torch.cuda.memory_reserved() / 1024 / 1024

                logger.debug(f"GPU memory optimized: {allocated:.1f}MB allocated, {reserved:.1f}MB reserved")

        except Exception as e:
            logger.debug(f"Memory optimization failed: {e}")


# Global GPU manager instance
_gpu_manager = None
_manager_lock = threading.Lock()


def get_gpu_manager() -> GPUResourceManager:
    """Get or create global GPU manager instance."""
    global _gpu_manager

    if _gpu_manager is None:
        with _manager_lock:
            if _gpu_manager is None:
                _gpu_manager = GPUResourceManager()

    return _gpu_manager