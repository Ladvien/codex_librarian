#!/usr/bin/env python3
"""
Celery monitoring script for PDF to Markdown MCP Server.

This script provides monitoring and management capabilities for Celery workers
and tasks including queue inspection, task status, and worker health checks.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pdf_to_markdown_mcp.worker.celery import app as celery_app, get_worker_stats, get_queue_length


def print_json(data: Dict[str, Any], indent: int = 2):
    """Pretty print JSON data."""
    print(json.dumps(data, indent=indent, default=str))


def show_worker_status():
    """Show current worker status and statistics."""
    print("=" * 60)
    print("CELERY WORKER STATUS")
    print("=" * 60)

    try:
        stats = get_worker_stats()

        if 'error' in stats:
            print(f"❌ Error getting worker stats: {stats['error']}")
            return

        print(f"📊 Active Queues: {len(stats.get('active_queues', []))}")
        for queue in stats.get('active_queues', []):
            queue_length = get_queue_length(queue)
            length_str = f"({queue_length} tasks)" if queue_length >= 0 else "(unable to check)"
            print(f"  • {queue} {length_str}")

        print(f"\n🔧 Registered Tasks: {len(stats.get('registered_tasks', []))}")
        for task in stats.get('registered_tasks', []):
            if task.startswith('pdf_to_markdown_mcp'):
                print(f"  • {task.split('.')[-1]}")

        active_tasks = stats.get('active_tasks', {})
        total_active = sum(len(tasks) for tasks in active_tasks.values())
        print(f"\n⚡ Active Tasks: {total_active}")

        scheduled_tasks = stats.get('scheduled_tasks', {})
        total_scheduled = sum(len(tasks) for tasks in scheduled_tasks.values())
        print(f"⏰ Scheduled Tasks: {total_scheduled}")

        reserved_tasks = stats.get('reserved_tasks', {})
        total_reserved = sum(len(tasks) for tasks in reserved_tasks.values())
        print(f"📥 Reserved Tasks: {total_reserved}")

        print(f"\n🖥️  Worker Statistics:")
        worker_stats = stats.get('worker_stats', {})
        for worker_name, worker_data in worker_stats.items():
            print(f"  Worker: {worker_name}")
            if isinstance(worker_data, dict):
                print(f"    Total tasks: {worker_data.get('total', 'N/A')}")
                print(f"    Pool: {worker_data.get('pool', {}).get('max-concurrency', 'N/A')} workers")

    except Exception as e:
        print(f"❌ Error retrieving worker status: {e}")


def show_queue_status():
    """Show detailed queue status."""
    print("=" * 60)
    print("QUEUE STATUS")
    print("=" * 60)

    queues = ['pdf_processing', 'embeddings', 'maintenance', 'monitoring']

    for queue in queues:
        length = get_queue_length(queue)
        status_icon = "🟢" if length == 0 else "🟡" if length < 10 else "🔴"
        length_str = str(length) if length >= 0 else "ERROR"
        print(f"{status_icon} {queue:20} {length_str:>6} tasks")


def show_active_tasks():
    """Show currently active tasks."""
    print("=" * 60)
    print("ACTIVE TASKS")
    print("=" * 60)

    try:
        inspect = celery_app.control.inspect()
        active_tasks = inspect.active()

        if not active_tasks:
            print("No active tasks")
            return

        for worker_name, tasks in active_tasks.items():
            if tasks:
                print(f"\n🖥️  Worker: {worker_name}")
                for task in tasks:
                    task_name = task.get('name', 'Unknown').split('.')[-1]
                    task_id = task.get('id', 'Unknown')[:8]
                    worker_pid = task.get('worker_pid', 'N/A')
                    args_str = str(task.get('args', []))[:50] + "..." if len(str(task.get('args', []))) > 50 else str(task.get('args', []))

                    print(f"  📋 {task_name} [{task_id}] (PID: {worker_pid})")
                    print(f"     Args: {args_str}")

    except Exception as e:
        print(f"❌ Error retrieving active tasks: {e}")


def show_failed_tasks():
    """Show recently failed tasks."""
    print("=" * 60)
    print("FAILED TASKS")
    print("=" * 60)

    try:
        # This would require additional setup to track failed tasks
        # For now, just show that the feature is available
        print("Failed task tracking requires additional configuration.")
        print("Consider setting up task result backend and failure tracking.")

    except Exception as e:
        print(f"❌ Error retrieving failed tasks: {e}")


def purge_queue(queue_name: str):
    """Purge all tasks from a specific queue."""
    try:
        result = celery_app.control.purge()
        print(f"✅ Purged tasks from queue '{queue_name}': {result}")
    except Exception as e:
        print(f"❌ Error purging queue '{queue_name}': {e}")


def run_health_check():
    """Run a health check task."""
    print("=" * 60)
    print("RUNNING HEALTH CHECK")
    print("=" * 60)

    try:
        from pdf_to_markdown_mcp.worker.tasks import health_check

        print("🏥 Queuing health check task...")
        result = health_check.delay()

        print(f"📋 Task ID: {result.id}")
        print("⏳ Waiting for result...")

        # Wait for result with timeout
        timeout = 30
        start_time = time.time()

        while not result.ready() and (time.time() - start_time) < timeout:
            time.sleep(0.5)
            print(".", end="", flush=True)

        print()

        if result.ready():
            health_result = result.get()
            print("✅ Health check completed!")
            print_json(health_result)
        else:
            print("⏰ Health check timed out")

    except Exception as e:
        print(f"❌ Health check failed: {e}")


def monitor_continuously(interval: int = 5):
    """Monitor worker status continuously."""
    print(f"🔄 Monitoring workers every {interval} seconds (Press Ctrl+C to stop)")

    try:
        while True:
            # Clear screen
            os.system('clear' if os.name == 'posix' else 'cls')

            # Show timestamp
            print(f"📅 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            # Show status
            show_queue_status()
            print()
            show_worker_status()

            # Wait for next update
            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped")


def main():
    """Main entry point for monitoring script."""
    parser = argparse.ArgumentParser(description="Monitor PDF to Markdown MCP Celery Workers")

    parser.add_argument(
        "command",
        choices=["status", "queues", "active", "failed", "health", "monitor", "purge"],
        help="Monitoring command to run"
    )

    parser.add_argument(
        "--queue",
        help="Queue name (for purge command)"
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="Monitoring interval in seconds (for monitor command)"
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format"
    )

    args = parser.parse_args()

    if args.command == "status":
        show_worker_status()

    elif args.command == "queues":
        show_queue_status()

    elif args.command == "active":
        show_active_tasks()

    elif args.command == "failed":
        show_failed_tasks()

    elif args.command == "health":
        run_health_check()

    elif args.command == "monitor":
        monitor_continuously(args.interval)

    elif args.command == "purge":
        if not args.queue:
            print("❌ --queue argument required for purge command")
            sys.exit(1)
        purge_queue(args.queue)

    else:
        print(f"❌ Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()