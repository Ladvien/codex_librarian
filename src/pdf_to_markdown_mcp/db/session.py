"""
Database session management for PDF to Markdown MCP Server.

This module provides SQLAlchemy session configuration, connection pooling,
and database utilities for PostgreSQL with PGVector extension.
"""

import logging
import os
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import DisconnectionError, OperationalError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

# Database configuration from environment variables - NO DEFAULT CREDENTIALS
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError(
        "DATABASE_URL environment variable is required. "
        "Example: postgresql://username:password@host:port/database"
    )

# Optimized connection pool configuration
POOL_SIZE = int(os.environ.get("DB_POOL_SIZE", "20"))  # Increased from 15
MAX_OVERFLOW = int(
    os.environ.get("DB_MAX_OVERFLOW", "10")
)  # Reduced from 30 to prevent exhaustion
POOL_PRE_PING = os.environ.get("DB_POOL_PRE_PING", "true").lower() == "true"
POOL_RECYCLE = int(
    os.environ.get("DB_POOL_RECYCLE", "1800")
)  # 30 minutes instead of 1 hour
POOL_TIMEOUT = int(os.environ.get("DB_POOL_TIMEOUT", "20"))  # Reduced from 30
CONNECT_TIMEOUT = int(os.environ.get("DB_CONNECT_TIMEOUT", "10"))
SQL_ECHO = os.environ.get("SQL_ECHO", "false").lower() == "true"

# Connection retry configuration
MAX_RETRIES = int(os.environ.get("DB_MAX_RETRIES", "3"))
RETRY_DELAY = float(os.environ.get("DB_RETRY_DELAY", "1.0"))
RETRY_BACKOFF_FACTOR = float(os.environ.get("DB_RETRY_BACKOFF_FACTOR", "2.0"))

# Determine connection arguments based on database type
connect_args = {}
if DATABASE_URL.startswith("postgresql"):
    # PostgreSQL-specific connection arguments
    connect_args = {
        "connect_timeout": CONNECT_TIMEOUT,
        "application_name": "pdf_to_markdown_mcp",
        "options": "-c statement_timeout=300000",  # 5 minutes
    }
elif DATABASE_URL.startswith("sqlite"):
    # SQLite-specific connection arguments
    connect_args = {
        "check_same_thread": False,  # Allow SQLite to be used across threads
    }

# Build engine configuration based on database type
engine_kwargs = {
    "echo": SQL_ECHO,
    "connect_args": connect_args,
    "future": True,
}

if DATABASE_URL.startswith("postgresql"):
    # PostgreSQL-specific configuration
    engine_kwargs.update({
        "poolclass": QueuePool,
        "pool_size": POOL_SIZE,
        "max_overflow": MAX_OVERFLOW,
        "pool_pre_ping": POOL_PRE_PING,
        "pool_recycle": POOL_RECYCLE,
        "pool_timeout": POOL_TIMEOUT,
        "isolation_level": "READ_COMMITTED",
    })
elif DATABASE_URL.startswith("sqlite"):
    # SQLite-specific configuration
    engine_kwargs.update({
        "pool_pre_ping": POOL_PRE_PING,
    })

# Create the engine
engine = create_engine(DATABASE_URL, **engine_kwargs)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """
    Enable PGVector extension on database connection.

    This ensures the vector extension is available for all database
    operations involving vector similarity search.
    """
    if "postgresql" in DATABASE_URL:
        with dbapi_connection.cursor() as cursor:
            try:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                dbapi_connection.commit()
                logger.info("PGVector extension enabled")
            except Exception as e:
                logger.warning(f"Could not enable PGVector extension: {e}")
                dbapi_connection.rollback()


def get_db() -> Generator[Session, None, None]:
    """
    Dependency function to get database session with retry logic.

    This function provides a database session for dependency injection
    in FastAPI endpoints. It ensures proper session cleanup, error handling,
    and connection retry logic for resilient database operations.

    Yields:
        Session: SQLAlchemy database session with retry support
    """
    session = None
    retry_count = 0

    while retry_count < MAX_RETRIES:
        try:
            session = SessionLocal()
            # Test the connection
            session.execute(text("SELECT 1"))
            yield session
            break
        except (DisconnectionError, OperationalError) as e:
            if session:
                session.close()
                session = None

            retry_count += 1
            if retry_count < MAX_RETRIES:
                wait_time = RETRY_DELAY * (RETRY_BACKOFF_FACTOR ** (retry_count - 1))
                logger.warning(
                    f"Database connection failed (attempt {retry_count}/{MAX_RETRIES}), "
                    f"retrying in {wait_time:.2f} seconds: {e}"
                )
                time.sleep(wait_time)
            else:
                logger.error(
                    f"Database connection failed after {MAX_RETRIES} attempts: {e}"
                )
                raise
        except Exception as e:
            if session:
                logger.error(f"Database session error: {e}")
                session.rollback()
                raise
        finally:
            if session:
                session.close()


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Context manager for database sessions with automatic retry and cleanup.

    This provides a clean way to use database sessions outside of FastAPI
    dependency injection with built-in retry logic and proper cleanup.

    Yields:
        Session: SQLAlchemy database session
    """
    session = None
    retry_count = 0

    while retry_count < MAX_RETRIES:
        try:
            session = SessionLocal()
            # Test the connection
            session.execute(text("SELECT 1"))
            yield session
            session.commit()
            break
        except (DisconnectionError, OperationalError) as e:
            if session:
                session.close()
                session = None

            retry_count += 1
            if retry_count < MAX_RETRIES:
                wait_time = RETRY_DELAY * (RETRY_BACKOFF_FACTOR ** (retry_count - 1))
                logger.warning(
                    f"Database connection failed (attempt {retry_count}/{MAX_RETRIES}), "
                    f"retrying in {wait_time:.2f} seconds: {e}"
                )
                time.sleep(wait_time)
            else:
                logger.error(
                    f"Database connection failed after {MAX_RETRIES} attempts: {e}"
                )
                raise
        except Exception as e:
            if session:
                logger.error(f"Database session error: {e}")
                session.rollback()
                raise
        finally:
            if session:
                session.close()


def create_tables():
    """
    Create all database tables.

    This function creates all tables defined in the models module.
    It should be used during initial database setup or testing.
    """
    from .models import Base

    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        raise


def drop_tables():
    """
    Drop all database tables.

    This function drops all tables defined in the models module.
    Use with caution - this will delete all data!
    """
    from .models import Base

    try:
        Base.metadata.drop_all(bind=engine)
        logger.info("Database tables dropped successfully")
    except Exception as e:
        logger.error(f"Error dropping database tables: {e}")
        raise


class DatabaseManager:
    """
    Advanced database manager for handling database operations.

    This class provides utilities for database health checking,
    connection testing, maintenance operations, and advanced
    connection management with retry logic.
    """

    def __init__(self):
        self.engine = engine
        self.SessionLocal = SessionLocal

    def health_check(self) -> dict[str, Any]:
        """
        Comprehensive database connectivity and health check.

        Returns:
            dict: Detailed health check results including timing and pool status
        """
        health_info = {
            "status": "healthy",
            "database_connection": False,
            "pgvector_available": False,
            "response_time_ms": None,
            "connection_pool": {},
            "active_connections": 0,
            "error": None,
        }

        try:
            start_time = time.time()

            # Test basic connectivity with retry
            with get_db_session() as session:
                session.execute(text("SELECT 1"))
                health_info["database_connection"] = True

                # Check PGVector extension availability
                try:
                    result = session.execute(
                        text(
                            """
                        SELECT EXISTS(
                            SELECT 1 FROM pg_extension WHERE extname = 'vector'
                        )
                    """
                        )
                    ).scalar()
                    health_info["pgvector_available"] = bool(result)
                except Exception:
                    health_info["pgvector_available"] = False

                # Get active connections count
                try:
                    active_conn_result = session.execute(
                        text(
                            """
                        SELECT count(*)
                        FROM pg_stat_activity
                        WHERE datname = current_database()
                        AND application_name = 'pdf_to_markdown_mcp'
                        AND state = 'active'
                    """
                        )
                    ).scalar()
                    health_info["active_connections"] = active_conn_result or 0
                except Exception:
                    pass

            # Calculate response time
            end_time = time.time()
            health_info["response_time_ms"] = round((end_time - start_time) * 1000, 2)

            # Get connection pool information
            health_info["connection_pool"] = self.get_connection_info()

            return health_info

        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            health_info.update({"status": "unhealthy", "error": str(e)})
            return health_info

    def get_connection_info(self) -> dict[str, Any]:
        """
        Get detailed database connection information.

        Returns:
            dict: Comprehensive connection pool statistics and database info
        """
        pool = self.engine.pool
        connection_info = {
            "pool_size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "invalid": pool.invalid(),
            "url": (
                str(self.engine.url).replace(f":{self.engine.url.password}@", ":***@")
                if self.engine.url.password
                else str(self.engine.url)
            ),
            "pool_class": str(type(pool).__name__),
            "pool_timeout": POOL_TIMEOUT,
            "max_overflow": MAX_OVERFLOW,
            "pool_recycle": POOL_RECYCLE,
            "pool_pre_ping": POOL_PRE_PING,
        }

        # Add pool utilization metrics
        total_capacity = pool.size() + pool.overflow()
        if total_capacity > 0:
            utilization = (pool.checkedout() / total_capacity) * 100
            connection_info["pool_utilization_percent"] = round(utilization, 2)

        return connection_info

    def test_connection_with_retry(self) -> bool:
        """
        Test database connection with automatic retry logic.

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            with get_db_session() as session:
                session.execute(text("SELECT 1"))
                return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False

    def execute_maintenance_sql(self, operation: str) -> list:
        """
        Execute pre-approved maintenance SQL operations only.

        This method only allows specific whitelisted maintenance operations
        to prevent SQL injection vulnerabilities.

        Args:
            operation: One of allowed operations: 'vacuum', 'analyze', 'reindex'

        Returns:
            list: Query results

        Raises:
            ValueError: If operation is not in whitelist
        """
        # Whitelist of allowed operations
        allowed_operations = {
            "vacuum": "SELECT schemaname, tablename FROM pg_tables WHERE schemaname = 'public'",
            "analyze": "ANALYZE",
            "stats": """SELECT
                schemaname, tablename,
                n_tup_ins, n_tup_upd, n_tup_del
                FROM pg_stat_user_tables WHERE schemaname = 'public'""",
        }

        if operation not in allowed_operations:
            raise ValueError(
                f"Operation '{operation}' not allowed. Allowed: {list(allowed_operations.keys())}"
            )

        try:
            with get_db_session() as session:
                result = session.execute(text(allowed_operations[operation]))
                return result.fetchall()
        except Exception as e:
            logger.error(f"Maintenance SQL execution failed for {operation}: {e}")
            raise

    def get_database_size(self) -> str | None:
        """
        Get current database size.

        Returns:
            str: Human-readable database size, or None if query fails
        """
        try:
            with get_db_session() as session:
                result = session.execute(
                    text(
                        """
                    SELECT pg_size_pretty(pg_database_size(current_database()))
                """
                    )
                ).scalar()
                return result
        except Exception as e:
            logger.error(f"Failed to get database size: {e}")
            return None

    def get_table_sizes(self) -> dict[str, str]:
        """
        Get sizes of all tables in the database.

        Returns:
            dict: Table names mapped to their human-readable sizes
        """
        try:
            with get_db_session() as session:
                result = session.execute(
                    text(
                        """
                    SELECT
                        schemaname,
                        tablename,
                        pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
                    FROM pg_tables
                    WHERE schemaname = 'public'
                    ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
                """
                    )
                ).fetchall()

                return {row.tablename: row.size for row in result}
        except Exception as e:
            logger.error(f"Failed to get table sizes: {e}")
            return {}

    def optimize_database(self) -> dict[str, Any]:
        """
        Perform database optimization operations.

        Returns:
            dict: Results of optimization operations
        """
        results = {
            "vacuum_completed": False,
            "analyze_completed": False,
            "reindex_completed": False,
            "errors": [],
        }

        try:
            with get_db_session() as session:
                # Run VACUUM ANALYZE on main tables
                tables_to_optimize = [
                    "documents",
                    "document_content",
                    "document_embeddings",
                    "document_images",
                    "processing_queue",
                ]

                for table in tables_to_optimize:
                    try:
                        session.execute(text(f"VACUUM ANALYZE {table}"))
                        logger.info(f"Optimized table: {table}")
                    except Exception as e:
                        error_msg = f"Failed to optimize table {table}: {e}"
                        results["errors"].append(error_msg)
                        logger.error(error_msg)

                results["vacuum_completed"] = True
                results["analyze_completed"] = True

                # Update statistics for better query planning
                try:
                    session.execute(text("ANALYZE"))
                    logger.info("Database statistics updated")
                except Exception as e:
                    error_msg = f"Failed to update statistics: {e}"
                    results["errors"].append(error_msg)
                    logger.error(error_msg)

        except Exception as e:
            error_msg = f"Database optimization failed: {e}"
            results["errors"].append(error_msg)
            logger.error(error_msg)

        return results

    def close_idle_connections(self) -> int:
        """
        Close idle database connections.

        Returns:
            int: Number of connections closed
        """
        try:
            with get_db_session() as session:
                result = session.execute(
                    text(
                        """
                    SELECT pg_terminate_backend(pid)
                    FROM pg_stat_activity
                    WHERE datname = current_database()
                    AND application_name = 'pdf_to_markdown_mcp'
                    AND state = 'idle'
                    AND query_start < NOW() - INTERVAL '30 minutes'
                """
                    )
                ).fetchall()

                closed_count = len(result)
                logger.info(f"Closed {closed_count} idle database connections")
                return closed_count

        except Exception as e:
            logger.error(f"Failed to close idle connections: {e}")
            return 0


# Global database manager instance
db_manager = DatabaseManager()
