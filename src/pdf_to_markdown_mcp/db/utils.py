"""
Database utility functions for PDF to Markdown MCP Server.

This module provides maintenance, backup, and administrative
utilities for database operations.
"""

import logging
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import func, text
from sqlalchemy.orm import Session

from .models import (
    Document,
    DocumentContent,
    DocumentEmbedding,
    DocumentImage,
    ProcessingQueue,
)

logger = logging.getLogger(__name__)


class DatabaseUtils:
    """Database utility functions for maintenance and administration."""

    @staticmethod
    def cleanup_old_records(
        db: Session, days_old: int = 30, dry_run: bool = True
    ) -> dict[str, int]:
        """
        Clean up old processing queue records and failed documents.

        Args:
            db: Database session
            days_old: Age threshold in days
            dry_run: If True, only count records without deleting

        Returns:
            Dictionary with cleanup statistics
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)
        stats = {}

        # Clean up completed processing queue records
        completed_jobs = db.query(ProcessingQueue).filter(
            ProcessingQueue.status == "completed",
            ProcessingQueue.completed_at < cutoff_date,
        )
        stats["completed_queue_jobs"] = completed_jobs.count()
        if not dry_run:
            completed_jobs.delete(synchronize_session=False)

        # Clean up failed jobs older than threshold
        failed_jobs = db.query(ProcessingQueue).filter(
            ProcessingQueue.status == "failed", ProcessingQueue.created_at < cutoff_date
        )
        stats["failed_queue_jobs"] = failed_jobs.count()
        if not dry_run:
            failed_jobs.delete(synchronize_session=False)

        # Clean up orphaned embeddings (documents that were deleted)
        orphaned_embeddings = db.query(DocumentEmbedding).filter(
            ~DocumentEmbedding.document_id.in_(db.query(Document.id))
        )
        stats["orphaned_embeddings"] = orphaned_embeddings.count()
        if not dry_run:
            orphaned_embeddings.delete(synchronize_session=False)

        if not dry_run:
            db.commit()
            logger.info(f"Cleanup completed: {stats}")

        return stats

    @staticmethod
    def optimize_indexes(db: Session) -> list[str]:
        """
        Optimize database indexes and update statistics.

        Args:
            db: Database session

        Returns:
            List of executed optimization commands
        """
        commands = []

        # Vacuum and analyze main tables
        tables = [
            "documents",
            "document_content",
            "document_embeddings",
            "document_images",
            "processing_queue",
        ]

        for table in tables:
            try:
                # Use VACUUM ANALYZE for better performance
                command = f"VACUUM ANALYZE {table};"
                db.execute(text(command))
                commands.append(command)
                logger.info(f"Optimized table: {table}")
            except Exception as e:
                logger.error(f"Failed to optimize table {table}: {e}")

        # Update PGVector index statistics (if available)
        try:
            db.execute(text("SELECT pg_stat_reset();"))
            commands.append("pg_stat_reset()")
        except Exception as e:
            logger.warning(f"Could not reset statistics: {e}")

        db.commit()
        return commands

    @staticmethod
    def check_database_health(db: Session) -> dict[str, Any]:
        """
        Perform comprehensive database health check.

        Args:
            db: Database session

        Returns:
            Dictionary with health check results
        """
        health = {"status": "healthy", "checks": {}, "warnings": []}

        try:
            # Check basic connectivity
            db.execute(text("SELECT 1"))
            health["checks"]["connectivity"] = "OK"
        except Exception as e:
            health["status"] = "unhealthy"
            health["checks"]["connectivity"] = f"FAILED: {e}"

        try:
            # Check PGVector extension
            result = db.execute(
                text("SELECT * FROM pg_extension WHERE extname = 'vector';")
            )
            if result.fetchone():
                health["checks"]["pgvector"] = "OK"
            else:
                health["checks"]["pgvector"] = "NOT_INSTALLED"
                health["warnings"].append("PGVector extension not found")
        except Exception as e:
            health["checks"]["pgvector"] = f"ERROR: {e}"

        try:
            # Check table sizes and row counts
            table_stats = {}
            tables = [
                ("documents", Document),
                ("document_content", DocumentContent),
                ("document_embeddings", DocumentEmbedding),
                ("document_images", DocumentImage),
                ("processing_queue", ProcessingQueue),
            ]

            for table_name, model in tables:
                count = db.query(model).count()
                size_result = db.execute(
                    text(
                        f"SELECT pg_size_pretty(pg_total_relation_size('{table_name}'));"
                    )
                )
                size = size_result.fetchone()[0] if size_result else "unknown"

                table_stats[table_name] = {"row_count": count, "size": size}

            health["checks"]["table_stats"] = table_stats
        except Exception as e:
            health["checks"]["table_stats"] = f"ERROR: {e}"

        try:
            # Check index usage
            index_query = text(
                """
                SELECT schemaname, tablename, attname, n_distinct, correlation
                FROM pg_stats
                WHERE schemaname = 'public'
                ORDER BY tablename, attname;
            """
            )
            result = db.execute(index_query)
            health["checks"]["index_stats"] = [dict(row._mapping) for row in result]
        except Exception as e:
            health["checks"]["index_stats"] = f"ERROR: {e}"

        try:
            # Check for long-running queries
            long_queries = db.execute(
                text(
                    """
                SELECT pid, now() - pg_stat_activity.query_start AS duration, query
                FROM pg_stat_activity
                WHERE (now() - pg_stat_activity.query_start) > interval '5 minutes'
                AND state = 'active';
            """
                )
            )
            active_long_queries = long_queries.fetchall()
            if active_long_queries:
                health["warnings"].append(
                    f"Found {len(active_long_queries)} long-running queries"
                )
                health["checks"]["long_queries"] = len(active_long_queries)
            else:
                health["checks"]["long_queries"] = 0
        except Exception as e:
            health["checks"]["long_queries"] = f"ERROR: {e}"

        return health

    @staticmethod
    def get_embedding_stats(db: Session) -> dict[str, Any]:
        """
        Get statistics about vector embeddings.

        Args:
            db: Database session

        Returns:
            Dictionary with embedding statistics
        """
        stats = {}

        try:
            # Count embeddings by document
            doc_embedding_counts = (
                db.query(
                    DocumentEmbedding.document_id,
                    func.count(DocumentEmbedding.id).label("embedding_count"),
                )
                .group_by(DocumentEmbedding.document_id)
                .all()
            )

            stats["documents_with_embeddings"] = len(doc_embedding_counts)
            stats["total_text_embeddings"] = sum(
                count.embedding_count for count in doc_embedding_counts
            )
            stats["avg_embeddings_per_document"] = (
                stats["total_text_embeddings"] / stats["documents_with_embeddings"]
                if stats["documents_with_embeddings"] > 0
                else 0
            )

            # Count image embeddings
            stats["total_image_embeddings"] = db.query(DocumentImage).count()

            # Embedding distribution by page count
            page_distribution = (
                db.query(
                    DocumentEmbedding.page_number,
                    func.count(DocumentEmbedding.id).label("count"),
                )
                .group_by(DocumentEmbedding.page_number)
                .order_by(DocumentEmbedding.page_number)
                .all()
            )
            stats["embeddings_by_page"] = dict(page_distribution)

        except Exception as e:
            logger.error(f"Error getting embedding stats: {e}")
            stats["error"] = str(e)

        return stats

    @staticmethod
    def backup_table_schema(db: Session, table_name: str) -> str:
        """
        Generate CREATE TABLE statement for backup purposes.

        Args:
            db: Database session
            table_name: Name of table to backup

        Returns:
            SQL CREATE TABLE statement
        """
        try:
            result = db.execute(
                text(
                    f"""
                SELECT
                    'CREATE TABLE ' || quote_ident(schemaname) || '.' || quote_ident(tablename) || E' (\\n' ||
                    string_agg(
                        '  ' || quote_ident(attname) || ' ' || format_type(atttypid, atttypmod) ||
                        CASE WHEN attnotnull THEN ' NOT NULL' ELSE '' END,
                        E',\\n'
                        ORDER BY attnum
                    ) || E'\\n);' as create_statement
                FROM pg_attribute
                JOIN pg_class ON pg_class.oid = pg_attribute.attrelid
                JOIN pg_namespace ON pg_namespace.oid = pg_class.relnamespace
                WHERE pg_class.relname = '{table_name}'
                  AND pg_attribute.attnum > 0
                  AND NOT pg_attribute.attisdropped
                  AND pg_namespace.nspname = 'public'
                GROUP BY schemaname, tablename;
            """
                )
            )

            result_row = result.fetchone()
            if result_row:
                return result_row[0]
            else:
                return f"-- Table {table_name} not found"

        except Exception as e:
            logger.error(f"Error backing up schema for table {table_name}: {e}")
            return f"-- Error backing up table {table_name}: {e}"

    @staticmethod
    def reset_failed_jobs(
        db: Session, older_than_hours: int = 24, max_attempts: int = 3
    ) -> int:
        """
        Reset failed jobs to queued status for retry.

        Args:
            db: Database session
            older_than_hours: Only reset jobs older than this many hours
            max_attempts: Only reset jobs with fewer than this many attempts

        Returns:
            Number of jobs reset
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=older_than_hours)

        failed_jobs = db.query(ProcessingQueue).filter(
            ProcessingQueue.status == "failed",
            ProcessingQueue.created_at < cutoff_time,
            ProcessingQueue.attempts < max_attempts,
        )

        count = failed_jobs.count()
        failed_jobs.update(
            {
                "status": "queued",
                "worker_id": None,
                "started_at": None,
                "error_message": None,
            },
            synchronize_session=False,
        )

        db.commit()
        logger.info(f"Reset {count} failed jobs to queued status")
        return count
