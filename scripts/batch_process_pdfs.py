#!/usr/bin/env python3
"""
Batch process all pending PDFs through MinerU GPU service.
"""

import json
import os
import time
import uuid
from pathlib import Path
import redis
import psycopg2
from typing import List, Dict

# Configuration
REDIS_HOST = 'localhost'
REDIS_PORT = 6380
DB_CONFIG = {
    'host': '192.168.1.104',
    'port': 5432,
    'database': 'codex_librarian',
    'user': 'codex_librarian',
    'password': os.getenv('DB_PASSWORD', 'YOUR_PASSWORD_HERE')
}
BATCH_SIZE = 50  # Number of PDFs to queue at once
CHECK_INTERVAL = 30  # Seconds between checks

def get_pending_pdfs(limit: int = 100) -> List[Dict]:
    """Get pending PDFs from database."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    query = """
        SELECT id, source_path, filename
        FROM documents
        WHERE conversion_status = 'pending'
        LIMIT %s
    """

    cur.execute(query, (limit,))
    results = []
    for row in cur.fetchall():
        results.append({
            'id': row[0],
            'path': row[1],
            'filename': row[2]
        })

    cur.close()
    conn.close()
    return results

def queue_pdf_for_processing(r: redis.Redis, pdf_path: str) -> str:
    """Queue a single PDF for MinerU processing."""
    request_id = str(uuid.uuid4())
    request = {
        'request_id': request_id,
        'pdf_path': pdf_path,
        'options': {
            'extract_tables': True,
            'extract_formulas': True
        },
        'callback_queue': 'mineru_results'
    }

    r.lpush('mineru_requests', json.dumps(request))
    return request_id

def get_queue_length(r: redis.Redis) -> int:
    """Get current queue length."""
    return r.llen('mineru_requests')

def main():
    """Main batch processing loop."""
    print("Starting batch PDF processing with GPU acceleration...")

    # Connect to Redis
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)

    try:
        r.ping()
        print(f"Connected to Redis on port {REDIS_PORT}")
    except redis.ConnectionError:
        print(f"Error: Cannot connect to Redis on port {REDIS_PORT}")
        print("Please start Redis with: redis-server --port 6380 --daemonize yes")
        return

    # Get all pending PDFs
    pending_pdfs = get_pending_pdfs(limit=1000)
    total_pdfs = len(pending_pdfs)
    print(f"Found {total_pdfs} pending PDFs to process")

    if total_pdfs == 0:
        print("No pending PDFs found")
        return

    # Process in batches
    processed = 0
    while processed < total_pdfs:
        # Check current queue length
        queue_length = get_queue_length(r)

        if queue_length < 10:  # Keep queue filled but not overloaded
            # Queue next batch
            batch_end = min(processed + BATCH_SIZE, total_pdfs)
            batch = pending_pdfs[processed:batch_end]

            print(f"\nQueueing batch {processed//BATCH_SIZE + 1} ({len(batch)} PDFs)...")
            for pdf in batch:
                if os.path.exists(pdf['path']):
                    request_id = queue_pdf_for_processing(r, pdf['path'])
                    print(f"  Queued: {pdf['filename']} (ID: {pdf['id']})")
                else:
                    print(f"  Skipped (not found): {pdf['filename']}")

            processed = batch_end
            print(f"Progress: {processed}/{total_pdfs} PDFs queued")
        else:
            print(f"Queue has {queue_length} items, waiting for processing...")

        # Wait before next check
        time.sleep(CHECK_INTERVAL)

        # Show current queue status
        queue_length = get_queue_length(r)
        print(f"Current queue: {queue_length} PDFs waiting")

    print("\nAll PDFs queued! Waiting for processing to complete...")

    # Monitor until queue is empty
    while True:
        queue_length = get_queue_length(r)
        if queue_length == 0:
            print("Queue empty! All PDFs processed.")
            break

        print(f"Remaining in queue: {queue_length} PDFs")
        time.sleep(CHECK_INTERVAL)

    # Final statistics
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("""
        SELECT conversion_status, COUNT(*)
        FROM documents
        GROUP BY conversion_status
        ORDER BY COUNT DESC
    """)

    print("\nFinal statistics:")
    for status, count in cur.fetchall():
        print(f"  {status}: {count} documents")

    cur.close()
    conn.close()

if __name__ == '__main__':
    main()