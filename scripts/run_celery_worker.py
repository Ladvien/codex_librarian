#!/usr/bin/env python3
"""
Celery worker startup script for PDF to Markdown MCP Server.

This script provides an easy way to start Celery workers with proper configuration
and monitoring for the PDF processing pipeline.
"""

import os
import sys
import argparse
import signal
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pdf_to_markdown_mcp.worker.celery import app as celery_app
from pdf_to_markdown_mcp.config import configure_logging


def setup_logging():
    """Set up logging for the worker process."""
    configure_logging()
    logger = logging.getLogger(__name__)
    logger.info("Celery worker logging configured")


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logger = logging.getLogger(__name__)
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    sys.exit(0)


def main():
    """Main entry point for Celery worker."""
    parser = argparse.ArgumentParser(description="Start PDF to Markdown MCP Celery Worker")

    parser.add_argument(
        "--concurrency", "-c",
        type=int,
        default=4,
        help="Number of worker processes (default: 4)"
    )

    parser.add_argument(
        "--loglevel", "-l",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Log level (default: INFO)"
    )

    parser.add_argument(
        "--queues", "-Q",
        default="pdf_processing,embeddings,maintenance,monitoring",
        help="Comma-separated list of queues to process (default: all)"
    )

    parser.add_argument(
        "--hostname", "-n",
        default="worker@%h",
        help="Worker hostname (default: worker@hostname)"
    )

    parser.add_argument(
        "--pool",
        choices=["prefork", "eventlet", "gevent", "threads"],
        default="prefork",
        help="Worker pool implementation (default: prefork)"
    )

    parser.add_argument(
        "--autoscale",
        help="Enable autoscaling (format: max,min)"
    )

    parser.add_argument(
        "--beat",
        action="store_true",
        help="Also run beat scheduler for periodic tasks"
    )

    parser.add_argument(
        "--flower",
        action="store_true",
        help="Start Flower monitoring web interface"
    )

    parser.add_argument(
        "--flower-port",
        type=int,
        default=5555,
        help="Flower web interface port (default: 5555)"
    )

    args = parser.parse_args()

    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)

    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("Starting PDF to Markdown MCP Celery Worker")
    logger.info(f"Queues: {args.queues}")
    logger.info(f"Concurrency: {args.concurrency}")
    logger.info(f"Log level: {args.loglevel}")

    # Prepare worker arguments
    worker_args = [
        "worker",
        f"--loglevel={args.loglevel.lower()}",
        f"--concurrency={args.concurrency}",
        f"--hostname={args.hostname}",
        f"--queues={args.queues}",
        f"--pool={args.pool}",
    ]

    # Add autoscaling if specified
    if args.autoscale:
        worker_args.append(f"--autoscale={args.autoscale}")

    # Add beat if requested
    if args.beat:
        worker_args.append("--beat")
        logger.info("Beat scheduler enabled for periodic tasks")

    try:
        # Start Flower monitoring if requested
        if args.flower:
            import subprocess
            flower_cmd = [
                sys.executable, "-m", "flower",
                "--app=pdf_to_markdown_mcp.worker.celery:app",
                f"--port={args.flower_port}",
                "--url_prefix=flower"
            ]

            logger.info(f"Starting Flower monitoring on port {args.flower_port}")
            flower_process = subprocess.Popen(flower_cmd)

            # Register cleanup for flower process
            def cleanup_flower(signum, frame):
                flower_process.terminate()
                signal_handler(signum, frame)

            signal.signal(signal.SIGINT, cleanup_flower)
            signal.signal(signal.SIGTERM, cleanup_flower)

        # Start the Celery worker
        logger.info("Starting Celery worker...")
        celery_app.start(worker_args)

    except KeyboardInterrupt:
        logger.info("Worker interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Worker failed to start: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()