#!/usr/bin/env python3
"""
GPU Optimization Status Report

This script generates a comprehensive report on GPU acceleration status
and provides performance optimization recommendations.
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def check_system_requirements():
    """Check system requirements for GPU acceleration."""
    print("=== System Requirements Check ===")

    requirements = {
        "nvidia_driver": False,
        "cuda_toolkit": False,
        "pytorch_cuda": False,
        "gpu_memory": False,
        "python_version": False,
    }

    # Check NVIDIA driver
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            requirements["nvidia_driver"] = True
            print("✅ NVIDIA Driver: Installed")

            # Extract driver version
            lines = result.stdout.split('\n')
            for line in lines:
                if "Driver Version:" in line:
                    driver_version = line.split("Driver Version:")[1].split()[0]
                    print(f"   Driver Version: {driver_version}")
                    break
        else:
            print("❌ NVIDIA Driver: Not found")
    except FileNotFoundError:
        print("❌ NVIDIA Driver: nvidia-smi not found")

    # Check CUDA toolkit
    try:
        import torch
        if torch.cuda.is_available():
            requirements["cuda_toolkit"] = True
            requirements["pytorch_cuda"] = True
            print("✅ CUDA Toolkit: Available")
            print(f"   CUDA Version: {torch.version.cuda}")
            print("✅ PyTorch CUDA: Available")
            print(f"   PyTorch Version: {torch.__version__}")

            # Check GPU memory
            if torch.cuda.device_count() > 0:
                total_memory = torch.cuda.get_device_properties(0).total_memory
                memory_gb = total_memory / (1024**3)
                if memory_gb >= 8:  # Minimum 8GB for efficient processing
                    requirements["gpu_memory"] = True
                    print(f"✅ GPU Memory: {memory_gb:.1f} GB (sufficient)")
                else:
                    print(f"⚠️ GPU Memory: {memory_gb:.1f} GB (may be insufficient)")
        else:
            print("❌ CUDA Toolkit: Not available")
            print("❌ PyTorch CUDA: Not available")
    except ImportError:
        print("❌ PyTorch: Not installed")

    # Check Python version
    python_version = sys.version_info
    if python_version >= (3, 11):
        requirements["python_version"] = True
        print(f"✅ Python Version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        print(f"⚠️ Python Version: {python_version.major}.{python_version.minor}.{python_version.micro} (recommend 3.11+)")

    return requirements

def check_application_configuration():
    """Check application-specific GPU configuration."""
    print("\n=== Application Configuration Check ===")

    config = {
        "mineru_service": False,
        "gpu_manager": False,
        "worker_tasks": False,
        "environment_vars": False,
    }

    try:
        # Check MinerU service
        from pdf_to_markdown_mcp.services.mineru import get_shared_mineru_instance
        service = get_shared_mineru_instance()
        if service:
            config["mineru_service"] = True
            print("✅ MinerU Service: Available")
        else:
            print("❌ MinerU Service: Failed to initialize")
    except Exception as e:
        print(f"❌ MinerU Service: {e}")

    try:
        # Check GPU manager
        from pdf_to_markdown_mcp.worker.gpu_manager import get_gpu_manager
        gpu_manager = get_gpu_manager()
        status = gpu_manager.get_gpu_status()
        if status["gpu_available"]:
            config["gpu_manager"] = True
            print("✅ GPU Manager: Available")
            print(f"   GPU Available: {status['gpu_available']}")
            print(f"   Memory Allocated: {status['memory_allocated_mb']:.1f} MB")
        else:
            print("❌ GPU Manager: GPU not available")
    except Exception as e:
        print(f"❌ GPU Manager: {e}")

    try:
        # Check worker tasks
        from pdf_to_markdown_mcp.worker.tasks import process_pdf_document
        config["worker_tasks"] = True
        print("✅ Worker Tasks: Available")
    except Exception as e:
        print(f"❌ Worker Tasks: {e}")

    # Check environment variables
    env_vars = ["CUDA_VISIBLE_DEVICES"]
    env_status = {}
    for var in env_vars:
        value = os.environ.get(var)
        env_status[var] = value
        if value:
            print(f"✅ Environment: {var}={value}")
        else:
            print(f"ℹ️  Environment: {var} not set (using defaults)")

    if any(env_status.values()):
        config["environment_vars"] = True

    return config

def benchmark_gpu_performance():
    """Benchmark GPU performance for baseline measurements."""
    print("\n=== GPU Performance Benchmark ===")

    try:
        import torch

        if not torch.cuda.is_available():
            print("❌ Cannot benchmark: CUDA not available")
            return {}

        device = torch.device('cuda:0')
        print(f"Benchmarking on: {torch.cuda.get_device_name(0)}")

        # Benchmark matrix multiplication
        sizes = [1000, 2000, 3000]
        results = {}

        for size in sizes:
            print(f"Testing {size}x{size} matrix multiplication...")

            # Warm up
            for _ in range(3):
                x = torch.randn(size, size, device=device)
                y = torch.randn(size, size, device=device)
                _ = torch.mm(x, y)
                torch.cuda.synchronize()

            # Benchmark
            times = []
            for _ in range(5):
                start_time = time.time()
                x = torch.randn(size, size, device=device)
                y = torch.randn(size, size, device=device)
                z = torch.mm(x, y)
                torch.cuda.synchronize()
                elapsed = time.time() - start_time
                times.append(elapsed)

            avg_time = sum(times) / len(times)
            ops_per_sec = (2 * size**3) / avg_time / 1e9  # GFLOPS

            results[f"matrix_{size}x{size}"] = {
                "avg_time_seconds": avg_time,
                "gflops": ops_per_sec,
            }

            print(f"   Average time: {avg_time:.3f}s")
            print(f"   Performance: {ops_per_sec:.1f} GFLOPS")

        return results

    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return {}

def check_memory_optimization():
    """Check GPU memory optimization settings."""
    print("\n=== Memory Optimization Check ===")

    try:
        import torch

        if not torch.cuda.is_available():
            print("❌ Cannot check memory: CUDA not available")
            return {}

        # Check current memory usage
        device = torch.device('cuda:0')
        allocated = torch.cuda.memory_allocated(device) / (1024**2)  # MB
        reserved = torch.cuda.memory_reserved(device) / (1024**2)   # MB
        total = torch.cuda.get_device_properties(device).total_memory / (1024**2)  # MB

        print(f"Current GPU Memory Usage:")
        print(f"   Allocated: {allocated:.1f} MB")
        print(f"   Reserved: {reserved:.1f} MB")
        print(f"   Total: {total:.1f} MB")
        print(f"   Free: {total - reserved:.1f} MB")
        print(f"   Utilization: {(reserved / total) * 100:.1f}%")

        # Check for memory fragmentation
        if reserved > allocated * 1.5:
            print("⚠️ Memory fragmentation detected (reserved >> allocated)")
            print("   Recommendation: Implement regular torch.cuda.empty_cache() calls")

        # Check available memory for processing
        available_gb = (total - reserved) / 1024
        if available_gb >= 16:
            print(f"✅ Excellent memory availability: {available_gb:.1f} GB free")
        elif available_gb >= 8:
            print(f"✅ Good memory availability: {available_gb:.1f} GB free")
        elif available_gb >= 4:
            print(f"⚠️ Limited memory availability: {available_gb:.1f} GB free")
        else:
            print(f"❌ Critical memory availability: {available_gb:.1f} GB free")

        return {
            "allocated_mb": allocated,
            "reserved_mb": reserved,
            "total_mb": total,
            "free_mb": total - reserved,
            "utilization_percent": (reserved / total) * 100,
            "available_gb": available_gb,
        }

    except Exception as e:
        print(f"❌ Memory check failed: {e}")
        return {}

def generate_optimization_recommendations(requirements, config, performance, memory):
    """Generate specific optimization recommendations."""
    print("\n=== Optimization Recommendations ===")

    recommendations = []

    # System-level recommendations
    if not requirements.get("nvidia_driver"):
        recommendations.append({
            "priority": "CRITICAL",
            "category": "System",
            "issue": "NVIDIA driver not installed",
            "recommendation": "Install latest NVIDIA driver compatible with RTX 3090",
            "command": "Download from NVIDIA website or use system package manager"
        })

    if not requirements.get("pytorch_cuda"):
        recommendations.append({
            "priority": "CRITICAL",
            "category": "Dependencies",
            "issue": "PyTorch CUDA support missing",
            "recommendation": "Install PyTorch with CUDA support",
            "command": "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
        })

    # Application-level recommendations
    if not config.get("mineru_service"):
        recommendations.append({
            "priority": "HIGH",
            "category": "Application",
            "issue": "MinerU service not available",
            "recommendation": "Check MinerU installation and configuration",
            "command": "pip install mineru[core] or check service initialization"
        })

    # Performance recommendations
    if memory.get("utilization_percent", 0) > 90:
        recommendations.append({
            "priority": "HIGH",
            "category": "Performance",
            "issue": "High GPU memory utilization",
            "recommendation": "Reduce batch sizes or implement memory cleanup",
            "command": "Adjust processing batch sizes in configuration"
        })

    if memory.get("available_gb", 0) < 4:
        recommendations.append({
            "priority": "MEDIUM",
            "category": "Performance",
            "issue": "Limited GPU memory available",
            "recommendation": "Clear GPU cache between operations",
            "command": "Add torch.cuda.empty_cache() calls in processing pipeline"
        })

    # Configuration recommendations
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        recommendations.append({
            "priority": "LOW",
            "category": "Configuration",
            "issue": "CUDA_VISIBLE_DEVICES not set",
            "recommendation": "Set environment variable to control GPU visibility",
            "command": "export CUDA_VISIBLE_DEVICES=0"
        })

    # Display recommendations by priority
    for priority in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        priority_recs = [r for r in recommendations if r["priority"] == priority]
        if priority_recs:
            print(f"\n{priority} Priority:")
            for rec in priority_recs:
                print(f"  📋 {rec['category']}: {rec['issue']}")
                print(f"     💡 {rec['recommendation']}")
                if rec.get("command"):
                    print(f"     💻 Command: {rec['command']}")
                print()

    if not recommendations:
        print("✅ No critical issues found! GPU acceleration is optimally configured.")

    return recommendations

def generate_report():
    """Generate comprehensive GPU optimization report."""
    print("🚀 GPU Acceleration Optimization Report")
    print("=" * 60)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"System: {os.uname().sysname} {os.uname().release}")
    print("=" * 60)

    # Run all checks
    requirements = check_system_requirements()
    config = check_application_configuration()
    performance = benchmark_gpu_performance()
    memory = check_memory_optimization()

    # Generate recommendations
    recommendations = generate_optimization_recommendations(requirements, config, performance, memory)

    # Overall status
    print("\n=== Overall Status ===")

    critical_issues = len([r for r in recommendations if r["priority"] == "CRITICAL"])
    high_issues = len([r for r in recommendations if r["priority"] == "HIGH"])

    if critical_issues > 0:
        print(f"❌ CRITICAL: {critical_issues} critical issues need immediate attention")
        status = "CRITICAL"
    elif high_issues > 0:
        print(f"⚠️ WARNING: {high_issues} high-priority issues found")
        status = "WARNING"
    elif recommendations:
        print(f"ℹ️ INFO: {len(recommendations)} minor optimizations available")
        status = "GOOD"
    else:
        print("✅ EXCELLENT: GPU acceleration is optimally configured!")
        status = "EXCELLENT"

    # Save detailed report
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "status": status,
        "requirements": requirements,
        "configuration": config,
        "performance": performance,
        "memory": memory,
        "recommendations": recommendations,
    }

    report_file = Path("gpu_optimization_report.json")
    with open(report_file, "w") as f:
        json.dump(report_data, f, indent=2)

    print(f"\n📊 Detailed report saved to: {report_file}")

    return status == "EXCELLENT" or status == "GOOD"

if __name__ == "__main__":
    try:
        success = generate_report()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nReport generation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nReport generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)