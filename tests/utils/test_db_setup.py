"""
Test database setup utilities for E2E testing.

This module provides utilities for setting up and managing test databases
that mirror the production schema for comprehensive integration testing.
"""

import logging
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)


class TestDatabaseManager:
    """Manager for test database setup, teardown, and management."""

    def __init__(self, test_db_url: str):
        """
        Initialize test database manager.

        Args:
            test_db_url: Test database connection URL
        """
        self.test_db_url = test_db_url
        self.parsed_url = urlparse(test_db_url)
        self.db_name = self.parsed_url.path.lstrip("/")

    def create_test_database(self) -> bool:
        """
        Create test database if it doesn't exist.

        Returns:
            bool: True if database created or already exists, False on failure
        """
        try:
            # Connect to postgres database to create test database
            admin_url = self.test_db_url.replace(f"/{self.db_name}", "/postgres")

            # Extract connection parameters
            conn_params = {
                "host": self.parsed_url.hostname,
                "port": self.parsed_url.port or 5432,
                "user": self.parsed_url.username,
                "password": self.parsed_url.password,
                "database": "postgres",
            }

            with psycopg2.connect(**conn_params) as conn:
                conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
                cursor = conn.cursor()

                # Check if database exists
                cursor.execute(
                    "SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s",
                    (self.db_name,),
                )

                if cursor.fetchone():
                    logger.info(f"Test database '{self.db_name}' already exists")
                    return True

                # Create test database
                cursor.execute(f'CREATE DATABASE "{self.db_name}"')
                logger.info(f"Created test database '{self.db_name}'")
                return True

        except Exception as e:
            logger.error(f"Failed to create test database '{self.db_name}': {e}")
            return False

    def setup_test_database(self) -> bool:
        """
        Set up test database with schema and extensions.

        Returns:
            bool: True if setup successful, False otherwise
        """
        try:
            # Create database if needed
            if not self.create_test_database():
                return False

            # Connect to test database and set up schema
            engine = create_engine(self.test_db_url)

            with engine.connect() as conn:
                # Enable pgvector extension
                try:
                    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                    logger.info("Enabled pgvector extension")
                except Exception as e:
                    logger.warning(f"Could not enable pgvector extension: {e}")

                # Create all tables
                from pdf_to_markdown_mcp.db.models import Base

                Base.metadata.create_all(engine)
                logger.info("Created all database tables")

                # Run any additional setup queries
                self._run_additional_setup(conn)

            return True

        except Exception as e:
            logger.error(f"Failed to setup test database: {e}")
            return False

    def _run_additional_setup(self, conn) -> None:
        """
        Run additional setup queries for test database.

        Args:
            conn: Database connection
        """
        # Create test user permissions if needed
        setup_queries = [
            # Grant permissions for test operations
            "GRANT ALL ON SCHEMA public TO CURRENT_USER",
            "GRANT ALL ON ALL TABLES IN SCHEMA public TO CURRENT_USER",
            "GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO CURRENT_USER",
            "GRANT ALL ON ALL FUNCTIONS IN SCHEMA public TO CURRENT_USER",
        ]

        for query in setup_queries:
            try:
                conn.execute(text(query))
            except Exception as e:
                logger.debug(f"Setup query failed (may be expected): {query} - {e}")

    def drop_test_database(self) -> bool:
        """
        Drop test database completely.

        Returns:
            bool: True if dropped successfully, False otherwise
        """
        try:
            # Connect to postgres database to drop test database
            admin_url = self.test_db_url.replace(f"/{self.db_name}", "/postgres")

            conn_params = {
                "host": self.parsed_url.hostname,
                "port": self.parsed_url.port or 5432,
                "user": self.parsed_url.username,
                "password": self.parsed_url.password,
                "database": "postgres",
            }

            with psycopg2.connect(**conn_params) as conn:
                conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
                cursor = conn.cursor()

                # Terminate active connections to test database
                cursor.execute(
                    """
                    SELECT pg_terminate_backend(pid)
                    FROM pg_stat_activity
                    WHERE datname = %s AND pid <> pg_backend_pid()
                """,
                    (self.db_name,),
                )

                # Drop test database
                cursor.execute(f'DROP DATABASE IF EXISTS "{self.db_name}"')
                logger.info(f"Dropped test database '{self.db_name}'")
                return True

        except Exception as e:
            logger.error(f"Failed to drop test database '{self.db_name}': {e}")
            return False

    def clean_test_database(self) -> bool:
        """
        Clean test database by truncating all tables.

        Returns:
            bool: True if cleaned successfully, False otherwise
        """
        try:
            engine = create_engine(self.test_db_url)

            with engine.begin() as conn:
                # Get all table names
                result = conn.execute(
                    text("""
                    SELECT tablename FROM pg_tables
                    WHERE schemaname = 'public'
                """)
                )

                tables = [row.tablename for row in result]

                if tables:
                    # Disable foreign key checks temporarily
                    conn.execute(text("SET session_replication_role = replica"))

                    # Truncate all tables
                    for table in tables:
                        conn.execute(
                            text(f'TRUNCATE TABLE "{table}" RESTART IDENTITY CASCADE')
                        )

                    # Re-enable foreign key checks
                    conn.execute(text("SET session_replication_role = DEFAULT"))

                    logger.info(f"Cleaned {len(tables)} tables in test database")

                return True

        except Exception as e:
            logger.error(f"Failed to clean test database: {e}")
            return False

    def verify_test_database(self) -> dict[str, Any]:
        """
        Verify test database setup and return status information.

        Returns:
            dict: Status information about test database
        """
        status = {
            "database_exists": False,
            "pgvector_available": False,
            "tables_created": False,
            "table_count": 0,
            "error": None,
        }

        try:
            engine = create_engine(self.test_db_url)

            with engine.connect() as conn:
                status["database_exists"] = True

                # Check pgvector extension
                result = conn.execute(
                    text("""
                    SELECT EXISTS(
                        SELECT 1 FROM pg_extension WHERE extname = 'vector'
                    )
                """)
                ).scalar()
                status["pgvector_available"] = bool(result)

                # Check tables
                result = conn.execute(
                    text("""
                    SELECT COUNT(*) FROM pg_tables WHERE schemaname = 'public'
                """)
                ).scalar()
                status["table_count"] = result
                status["tables_created"] = result > 0

        except Exception as e:
            status["error"] = str(e)

        return status


def setup_test_database_from_env() -> TestDatabaseManager | None:
    """
    Set up test database using environment variables.

    Returns:
        TestDatabaseManager instance if successful, None otherwise
    """
    test_db_url = os.getenv("TEST_DATABASE_URL")

    if not test_db_url:
        logger.warning(
            "TEST_DATABASE_URL not set, skipping PostgreSQL test database setup"
        )
        return None

    if "sqlite" in test_db_url:
        logger.info("SQLite test database detected, skipping PostgreSQL setup")
        return None

    try:
        manager = TestDatabaseManager(test_db_url)

        if manager.setup_test_database():
            logger.info("Test database setup completed successfully")
            return manager
        else:
            logger.error("Test database setup failed")
            return None

    except Exception as e:
        logger.error(f"Test database setup error: {e}")
        return None


def create_test_database_setup_script():
    """Create a standalone script for test database setup."""
    script_content = '''#!/usr/bin/env python3
"""
Standalone test database setup script.

This script sets up the test database for E2E testing.
Run this before executing integration tests.
"""

import sys
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from tests.utils.test_db_setup import setup_test_database_from_env

def main():
    """Main setup function."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    print("Setting up test database for E2E testing...")

    manager = setup_test_database_from_env()
    if manager:
        status = manager.verify_test_database()
        print("\\nTest Database Status:")
        for key, value in status.items():
            print(f"  {key}: {value}")

        if status["tables_created"]:
            print("\\n✅ Test database setup completed successfully!")
            print("You can now run E2E tests with: pytest tests/integration/test_directory_mirroring_e2e.py")
        else:
            print("\\n❌ Test database setup incomplete")
            sys.exit(1)
    else:
        print("\\n❌ Test database setup failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''

    script_path = Path(__file__).parent.parent.parent / "scripts" / "setup_test_db.py"
    script_path.parent.mkdir(exist_ok=True)
    script_path.write_text(script_content)
    script_path.chmod(0o755)

    print(f"Created test database setup script: {script_path}")


if __name__ == "__main__":
    create_test_database_setup_script()

    # Also run setup if called directly
    logging.basicConfig(level=logging.INFO)
    manager = setup_test_database_from_env()
    if manager:
        status = manager.verify_test_database()
        print("Test Database Status:", status)
