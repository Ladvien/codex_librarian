#!/usr/bin/env python3
"""
Export markdown files from database to disk.

This script triggers the export_markdown_to_disk Celery task to write
markdown content from the database to the configured output paths.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pdf_to_markdown_mcp.worker.tasks import export_markdown_to_disk


def main():
    """Run the export task synchronously."""
    print("Starting markdown export from database to disk...")
    print("This will export all completed documents with output_path set.\n")

    # Run the task directly (not async via Celery)
    result = export_markdown_to_disk.apply().get()

    # Print results
    print(f"\nExport completed!")
    print(f"Status: {result['status']}")

    if result["status"] == "failed":
        print(f"Error: {result.get('error', 'Unknown error')}")
        return 1

    print(f"Files written: {result['files_written']}")
    print(f"Files skipped: {result['files_skipped']}")
    print(f"Errors: {result['errors']}")

    if result["details"]:
        print("\nDetails:")
        for detail in result["details"]:
            doc_id = detail["document_id"]
            status = detail["status"]
            output_path = detail.get("output_path", "N/A")

            if status == "written":
                size = detail.get("size_bytes", 0)
                print(f"  [✓] Document {doc_id}: {output_path} ({size:,} bytes)")
            elif status == "skipped":
                reason = detail.get("reason", "unknown")
                print(f"  [⊘] Document {doc_id}: {output_path} - {reason}")
            elif status == "error":
                error = detail.get("error", "unknown error")
                print(f"  [✗] Document {doc_id}: {error}")

    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    sys.exit(main())