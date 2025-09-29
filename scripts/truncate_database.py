#!/usr/bin/env python3
"""
Database truncation script for PDF to Markdown MCP Server.

This script safely truncates document-related tables while preserving
configuration and system tables. Use this to reset the database state
before performing a full directory index.

CAUTION: This operation is irreversible!
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sqlalchemy import text
from pdf_to_markdown_mcp.db.session import SessionLocal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_table_counts(session) -> dict[str, int]:
    """Get current record counts for all tables."""
    tables = {
        "document_embeddings": "SELECT COUNT(*) FROM document_embeddings",
        "document_images": "SELECT COUNT(*) FROM document_images",
        "document_content": "SELECT COUNT(*) FROM document_content",
        "processing_queue": "SELECT COUNT(*) FROM processing_queue",
        "path_mappings": "SELECT COUNT(*) FROM path_mappings",
        "documents": "SELECT COUNT(*) FROM documents",
    }

    counts = {}
    for table_name, query in tables.items():
        try:
            result = session.execute(text(query))
            count = result.scalar()
            counts[table_name] = count
        except Exception as e:
            logger.warning(f"Could not get count for {table_name}: {e}")
            counts[table_name] = 0

    return counts


def truncate_tables(session, confirm: bool = False) -> dict[str, int]:
    """
    Truncate all document-related tables.

    Args:
        session: Database session
        confirm: If True, proceed with truncation. If False, dry run.

    Returns:
        Dictionary with record counts before truncation
    """
    # Get counts before truncation
    counts_before = get_table_counts(session)

    logger.info("Current table record counts:")
    for table, count in counts_before.items():
        logger.info(f"  {table}: {count:,} records")

    total_records = sum(counts_before.values())
    logger.info(f"Total records to be deleted: {total_records:,}")

    if not confirm:
        logger.info("DRY RUN - No changes made. Use --confirm to proceed.")
        return counts_before

    # Confirm with user if running interactively
    if sys.stdin.isatty():
        response = input("\n⚠️  This will permanently delete all document data. Continue? (yes/no): ")
        if response.lower() not in ["yes", "y"]:
            logger.info("Truncation cancelled by user.")
            return counts_before

    logger.info("Starting truncation...")

    try:
        # Truncate in correct order (respecting foreign keys)
        # CASCADE will automatically truncate dependent tables
        session.execute(text("TRUNCATE document_embeddings CASCADE"))
        logger.info("✓ Truncated document_embeddings")

        session.execute(text("TRUNCATE document_images CASCADE"))
        logger.info("✓ Truncated document_images")

        session.execute(text("TRUNCATE document_content CASCADE"))
        logger.info("✓ Truncated document_content")

        session.execute(text("TRUNCATE processing_queue CASCADE"))
        logger.info("✓ Truncated processing_queue")

        session.execute(text("TRUNCATE path_mappings CASCADE"))
        logger.info("✓ Truncated path_mappings")

        session.execute(text("TRUNCATE documents CASCADE"))
        logger.info("✓ Truncated documents")

        session.commit()
        logger.info("✅ All tables truncated successfully")

        # Verify truncation
        counts_after = get_table_counts(session)
        logger.info("\nTable record counts after truncation:")
        for table, count in counts_after.items():
            logger.info(f"  {table}: {count:,} records")

    except Exception as e:
        session.rollback()
        logger.error(f"❌ Truncation failed: {e}")
        raise

    return counts_before


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Truncate document tables in PDF to Markdown database"
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Actually perform truncation (without this, runs in dry-run mode)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip interactive confirmation prompt",
    )

    args = parser.parse_args()

    if args.force and not sys.stdin.isatty():
        # Allow non-interactive execution with --force
        pass

    logger.info("=" * 70)
    logger.info("PDF to Markdown MCP Server - Database Truncation")
    logger.info("=" * 70)

    try:
        with SessionLocal() as session:
            counts = truncate_tables(session, confirm=args.confirm)

            if args.confirm:
                logger.info(f"\n✅ Truncation completed. Deleted {sum(counts.values()):,} total records.")
                logger.info("\nNext steps:")
                logger.info("  1. Restart services: ./scripts/restart_services.sh")
                logger.info("  2. The file watcher will automatically index all PDFs on startup")
            else:
                logger.info("\n💡 Run with --confirm to actually truncate the tables")

        return 0

    except Exception as e:
        logger.error(f"❌ Script failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())