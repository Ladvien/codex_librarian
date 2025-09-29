#!/usr/bin/env python3
"""
Comprehensive System Diagnostic Tool for PDF to Markdown MCP
Checks all critical components and failure points
"""

import os
import sys
import time
import json
import psutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add project to path
sys.path.insert(0, '/mnt/datadrive_m2/codex_librarian/src')

from pdf_to_markdown_mcp.db.session import get_db_session
from pdf_to_markdown_mcp.db.models import Document, DocumentContent, DocumentEmbedding
from pdf_to_markdown_mcp.config import settings

class SystemDiagnostic:
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "checks": {},
            "failures": [],
            "warnings": [],
            "summary": {}
        }

    def run_command(self, cmd: str) -> Tuple[int, str, str]:
        """Execute shell command and return exit code, stdout, stderr."""
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=10
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "Command timed out"
        except Exception as e:
            return -1, "", str(e)

    def check_gpu(self) -> Dict:
        """Check GPU availability and usage."""
        print("🔍 Checking GPU...")
        gpu_info = {
            "available": False,
            "processes": [],
            "memory_used_mb": 0,
            "errors": []
        }

        try:
            # Check nvidia-smi
            code, stdout, stderr = self.run_command(
                "nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader"
            )

            if code == 0:
                gpu_info["available"] = True
                lines = stdout.strip().split('\n')
                if lines:
                    parts = lines[0].split(',')
                    gpu_info["name"] = parts[0].strip()
                    gpu_info["memory_used_mb"] = parts[1].strip()
                    gpu_info["memory_total_mb"] = parts[2].strip()

                # Get GPU processes
                code, stdout, _ = self.run_command(
                    "nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv,noheader"
                )
                if code == 0 and stdout:
                    for line in stdout.strip().split('\n'):
                        if line:
                            parts = line.split(',')
                            gpu_info["processes"].append({
                                "pid": parts[0].strip(),
                                "memory_mb": parts[1].strip(),
                                "name": parts[2].strip() if len(parts) > 2 else "unknown"
                            })
            else:
                gpu_info["errors"].append("nvidia-smi not available")

        except Exception as e:
            gpu_info["errors"].append(str(e))

        self.results["checks"]["gpu"] = gpu_info
        return gpu_info

    def check_services(self) -> Dict:
        """Check status of all required services."""
        print("🔍 Checking services...")
        services = {
            "redis": {"port": 6379, "running": False},
            "postgresql": {"port": 5432, "running": False},
            "mineru_standalone": {"process": "mineru_standalone.py", "running": False, "pid": None},
            "celery_worker": {"service": "pdf-celery-worker", "running": False},
            "celery_beat": {"service": "pdf-celery-beat", "running": False},
            "ollama": {"port": 11434, "running": False},
            "fastapi": {"port": 8000, "running": False}
        }

        # Check ports
        for service in ["redis", "postgresql", "ollama", "fastapi"]:
            port = services[service]["port"]
            code, stdout, _ = self.run_command(f"ss -tln | grep -q :{port} && echo 'running'")
            services[service]["running"] = (stdout.strip() == "running")

        # Check MinerU standalone
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if 'mineru_standalone.py' in cmdline:
                    services["mineru_standalone"]["running"] = True
                    services["mineru_standalone"]["pid"] = proc.info['pid']
                    break
            except:
                pass

        # Check systemd services
        for service in ["celery_worker", "celery_beat"]:
            service_name = services[service]["service"]
            code, stdout, _ = self.run_command(
                f"systemctl is-active {service_name}"
            )
            services[service]["running"] = (stdout.strip() == "active")

        self.results["checks"]["services"] = services
        return services

    def check_database(self) -> Dict:
        """Check database connectivity and data."""
        print("🔍 Checking database...")
        db_info = {
            "connected": False,
            "stats": {},
            "errors": []
        }

        try:
            with get_db_session() as db:
                db_info["connected"] = True

                # Get statistics
                db_info["stats"]["total_documents"] = db.query(Document).count()
                db_info["stats"]["documents_pending"] = db.query(Document).filter(
                    Document.conversion_status == "pending"
                ).count()
                db_info["stats"]["documents_completed"] = db.query(Document).filter(
                    Document.conversion_status == "completed"
                ).count()
                db_info["stats"]["documents_failed"] = db.query(Document).filter(
                    Document.conversion_status == "failed"
                ).count()

                # Content stats
                db_info["stats"]["documents_with_content"] = db.query(DocumentContent).count()

                # Embedding stats
                db_info["stats"]["total_embeddings"] = db.query(DocumentEmbedding).count()
                db_info["stats"]["documents_with_embeddings"] = db.query(
                    DocumentEmbedding.document_id
                ).distinct().count()

                # Recent activity
                one_hour_ago = datetime.utcnow() - timedelta(hours=1)
                db_info["stats"]["documents_processed_last_hour"] = db.query(Document).filter(
                    Document.updated_at >= one_hour_ago,
                    Document.conversion_status == "completed"
                ).count()

        except Exception as e:
            db_info["errors"].append(str(e))

        self.results["checks"]["database"] = db_info
        return db_info

    def check_file_system(self) -> Dict:
        """Check file system paths and permissions."""
        print("🔍 Checking file system...")
        fs_info = {
            "paths": {},
            "errors": []
        }

        # Key paths to check
        paths = {
            "output_directory": os.getenv("OUTPUT_DIRECTORY", "/mnt/codex_fs/research/librarian_output"),
            "source_directory": "/mnt/codex_fs/research/codex_articles",
            "project_root": "/mnt/datadrive_m2/codex_librarian",
            "venv": "/mnt/datadrive_m2/codex_librarian/.venv",
            "mineru_log": "/tmp/mineru.log",
            "celery_log": "/var/log/celery-worker.log"
        }

        for name, path in paths.items():
            path_info = {
                "path": path,
                "exists": os.path.exists(path),
                "readable": False,
                "writable": False
            }

            if path_info["exists"]:
                path_info["readable"] = os.access(path, os.R_OK)
                path_info["writable"] = os.access(path, os.W_OK)

                if name == "output_directory" and os.path.isdir(path):
                    # Count recent markdown files
                    recent_files = []
                    try:
                        for file in Path(path).glob("*.md"):
                            if file.stat().st_mtime > time.time() - 3600:  # Last hour
                                recent_files.append(file.name)
                        path_info["recent_files_count"] = len(recent_files)
                        path_info["recent_files"] = recent_files[:5]  # First 5
                    except:
                        pass

            fs_info["paths"][name] = path_info

        self.results["checks"]["file_system"] = fs_info
        return fs_info

    def check_redis_queues(self) -> Dict:
        """Check Redis queue status."""
        print("🔍 Checking Redis queues...")
        redis_info = {
            "connected": False,
            "queues": {},
            "errors": []
        }

        try:
            import redis
            r = redis.Redis(host='localhost', port=6379, decode_responses=True)
            r.ping()
            redis_info["connected"] = True

            # Check Celery queues
            queues = ["celery", "pdf_processing", "embeddings", "mineru_requests", "mineru_results"]
            for queue in queues:
                try:
                    length = r.llen(queue)
                    redis_info["queues"][queue] = length
                except:
                    redis_info["queues"][queue] = 0

        except Exception as e:
            redis_info["errors"].append(str(e))

        self.results["checks"]["redis"] = redis_info
        return redis_info

    def check_mineru_processing(self) -> Dict:
        """Check MinerU processing capability."""
        print("🔍 Checking MinerU processing...")
        mineru_info = {
            "service_running": False,
            "gpu_enabled": False,
            "recent_processing": [],
            "errors": []
        }

        # Check if MinerU log exists and has recent activity
        log_path = "/tmp/mineru.log"
        if os.path.exists(log_path):
            try:
                with open(log_path, 'r') as f:
                    lines = f.readlines()[-100:]  # Last 100 lines

                for line in reversed(lines):
                    if "GPU available" in line:
                        mineru_info["gpu_enabled"] = True
                    if "Completed job" in line and "success: True" in line:
                        # Extract timestamp if possible
                        parts = line.split(' - ')
                        if parts:
                            timestamp = parts[0].strip()
                            mineru_info["recent_processing"].append(timestamp)
                            if len(mineru_info["recent_processing"]) >= 5:
                                break

                # Check if service is responding
                if lines:
                    last_line_time = lines[-1].split(' ')[0]  # Get timestamp
                    mineru_info["service_running"] = True
                    mineru_info["last_activity"] = last_line_time

            except Exception as e:
                mineru_info["errors"].append(str(e))
        else:
            mineru_info["errors"].append("MinerU log file not found")

        self.results["checks"]["mineru"] = mineru_info
        return mineru_info

    def check_embeddings(self) -> Dict:
        """Check embedding generation status."""
        print("🔍 Checking embeddings...")
        embedding_info = {
            "ollama_running": False,
            "model_available": False,
            "recent_embeddings": 0,
            "errors": []
        }

        # Check Ollama
        code, stdout, _ = self.run_command("ollama list | grep nomic-embed-text")
        if code == 0 and stdout:
            embedding_info["model_available"] = True

        # Check if Ollama is responding
        code, stdout, _ = self.run_command(
            "curl -s http://localhost:11434/api/tags | grep -q nomic && echo 'running'"
        )
        embedding_info["ollama_running"] = (stdout.strip() == "running")

        # Check recent embeddings in DB
        try:
            with get_db_session() as db:
                one_hour_ago = datetime.utcnow() - timedelta(hours=1)
                recent = db.query(DocumentEmbedding).filter(
                    DocumentEmbedding.created_at >= one_hour_ago
                ).count()
                embedding_info["recent_embeddings"] = recent
        except Exception as e:
            embedding_info["errors"].append(str(e))

        self.results["checks"]["embeddings"] = embedding_info
        return embedding_info

    def run_diagnostics(self) -> Dict:
        """Run all diagnostic checks."""
        print("=" * 60)
        print("🏥 SYSTEM DIAGNOSTIC STARTING")
        print("=" * 60)

        # Run all checks
        gpu = self.check_gpu()
        services = self.check_services()
        database = self.check_database()
        file_system = self.check_file_system()
        redis = self.check_redis_queues()
        mineru = self.check_mineru_processing()
        embeddings = self.check_embeddings()

        # Analyze results
        self.analyze_results()

        # Print summary
        self.print_summary()

        return self.results

    def analyze_results(self):
        """Analyze results and identify issues."""
        failures = []
        warnings = []

        # GPU checks
        if not self.results["checks"]["gpu"]["available"]:
            failures.append("❌ GPU not available")
        elif not self.results["checks"]["gpu"]["processes"]:
            warnings.append("⚠️ No GPU processes running")

        # Service checks
        services = self.results["checks"]["services"]
        for service, info in services.items():
            if not info["running"]:
                if service in ["redis", "postgresql", "celery_worker"]:
                    failures.append(f"❌ Critical service not running: {service}")
                else:
                    warnings.append(f"⚠️ Service not running: {service}")

        # Database checks
        db = self.results["checks"]["database"]
        if not db["connected"]:
            failures.append("❌ Database connection failed")
        elif db["stats"].get("documents_pending", 0) > 100:
            warnings.append(f"⚠️ High number of pending documents: {db['stats']['documents_pending']}")

        # File system checks
        fs = self.results["checks"]["file_system"]
        for name, info in fs["paths"].items():
            if not info["exists"]:
                if name in ["output_directory", "project_root"]:
                    failures.append(f"❌ Critical path missing: {name}")
                else:
                    warnings.append(f"⚠️ Path missing: {name}")
            elif not info["writable"] and name == "output_directory":
                failures.append(f"❌ Output directory not writable")

        # Redis checks
        if not self.results["checks"]["redis"]["connected"]:
            failures.append("❌ Redis connection failed")

        # MinerU checks
        mineru = self.results["checks"]["mineru"]
        if not mineru["service_running"]:
            failures.append("❌ MinerU service not running")
        elif not mineru["gpu_enabled"]:
            warnings.append("⚠️ MinerU not using GPU")

        # Embedding checks
        emb = self.results["checks"]["embeddings"]
        if not emb["ollama_running"]:
            warnings.append("⚠️ Ollama not running")
        elif not emb["model_available"]:
            warnings.append("⚠️ Embedding model not available")

        self.results["failures"] = failures
        self.results["warnings"] = warnings
        self.results["summary"]["status"] = "FAILED" if failures else ("WARNING" if warnings else "HEALTHY")
        self.results["summary"]["failure_count"] = len(failures)
        self.results["summary"]["warning_count"] = len(warnings)

    def print_summary(self):
        """Print diagnostic summary."""
        print("\n" + "=" * 60)
        print("📊 DIAGNOSTIC SUMMARY")
        print("=" * 60)

        status = self.results["summary"]["status"]
        if status == "HEALTHY":
            print("✅ SYSTEM STATUS: HEALTHY")
        elif status == "WARNING":
            print("⚠️ SYSTEM STATUS: WARNING")
        else:
            print("❌ SYSTEM STATUS: FAILED")

        print(f"\nFailures: {self.results['summary']['failure_count']}")
        print(f"Warnings: {self.results['summary']['warning_count']}")

        # Print failures
        if self.results["failures"]:
            print("\n🔴 FAILURES:")
            for failure in self.results["failures"]:
                print(f"  {failure}")

        # Print warnings
        if self.results["warnings"]:
            print("\n🟡 WARNINGS:")
            for warning in self.results["warnings"]:
                print(f"  {warning}")

        # Print key metrics
        print("\n📈 KEY METRICS:")

        # GPU
        gpu = self.results["checks"]["gpu"]
        if gpu["available"]:
            print(f"  GPU: {gpu.get('name', 'Unknown')} - {gpu.get('memory_used_mb', 0)}/{gpu.get('memory_total_mb', 0)}")
            for proc in gpu["processes"][:3]:
                print(f"    - PID {proc['pid']}: {proc['memory_mb']} ({proc['name']})")

        # Database
        db = self.results["checks"]["database"]
        if db["connected"]:
            stats = db["stats"]
            print(f"  Database:")
            print(f"    - Documents: {stats.get('total_documents', 0)} total")
            print(f"    - Status: {stats.get('documents_completed', 0)} completed, {stats.get('documents_pending', 0)} pending, {stats.get('documents_failed', 0)} failed")
            print(f"    - Embeddings: {stats.get('total_embeddings', 0)} total across {stats.get('documents_with_embeddings', 0)} documents")
            print(f"    - Recent: {stats.get('documents_processed_last_hour', 0)} processed in last hour")

        # File system
        fs = self.results["checks"]["file_system"]
        output = fs["paths"].get("output_directory", {})
        if output.get("recent_files_count", 0) > 0:
            print(f"  Output Directory:")
            print(f"    - Recent files: {output['recent_files_count']} in last hour")

        # Redis
        redis = self.results["checks"]["redis"]
        if redis["connected"] and redis["queues"]:
            print(f"  Redis Queues:")
            for queue, length in redis["queues"].items():
                if length > 0:
                    print(f"    - {queue}: {length} items")

        # MinerU
        mineru = self.results["checks"]["mineru"]
        if mineru["recent_processing"]:
            print(f"  MinerU:")
            print(f"    - Recent jobs: {len(mineru['recent_processing'])}")
            print(f"    - GPU enabled: {mineru['gpu_enabled']}")

        print("\n" + "=" * 60)

        # Save results to file
        output_file = f"/tmp/diagnostic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\n📁 Full results saved to: {output_file}")

        return self.results

def main():
    """Run system diagnostic."""
    diagnostic = SystemDiagnostic()
    results = diagnostic.run_diagnostics()

    # Return exit code based on status
    if results["summary"]["status"] == "FAILED":
        sys.exit(1)
    elif results["summary"]["status"] == "WARNING":
        sys.exit(0)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()