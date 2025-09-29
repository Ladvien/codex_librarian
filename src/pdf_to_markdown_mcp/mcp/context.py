"""
Database connection pool management for MCP server.

Provides async connection pool lifecycle management using asyncpg.
"""

import asyncpg
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from .config import MCPConfig

logger = logging.getLogger(__name__)


class DatabasePool:
    """
    Manages PostgreSQL connection pool for MCP server.

    Uses asyncpg for high-performance async database access.
    Handles pool lifecycle (creation, connection acquisition, cleanup).
    """

    def __init__(self, config: MCPConfig) -> None:
        """
        Initialize database pool manager.

        Args:
            config: MCP configuration with database settings
        """
        self.config = config
        self.pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        """
        Initialize the connection pool.

        Raises:
            ConnectionError: If unable to connect to database
        """
        try:
            logger.info(
                f"Creating database pool (min={self.config.db_pool_min_size}, "
                f"max={self.config.db_pool_max_size})"
            )

            self.pool = await asyncpg.create_pool(
                dsn=self.config.database_url,
                min_size=self.config.db_pool_min_size,
                max_size=self.config.db_pool_max_size,
                timeout=self.config.db_pool_timeout,
                command_timeout=60,  # Command execution timeout
                # Connection initialization
                init=self._init_connection,
            )

            logger.info("Database pool created successfully")

            # Test connection
            async with self.acquire() as conn:
                version = await conn.fetchval("SELECT version()")
                logger.info(f"Connected to PostgreSQL: {version[:50]}...")

        except Exception as e:
            logger.error(f"Failed to create database pool: {e}")
            raise ConnectionError(f"Database connection failed: {e}") from e

    async def _init_connection(self, conn: asyncpg.Connection) -> None:
        """
        Initialize a new connection from the pool.

        Registers custom types and sets up connection parameters.

        Args:
            conn: New connection to initialize
        """
        # Set statement timeout
        await conn.execute("SET statement_timeout = '60s'")

        # Register pgvector type if available
        try:
            await conn.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
            logger.debug("PGVector extension is available")
        except Exception as e:
            logger.warning(f"PGVector extension not available: {e}")

    async def disconnect(self) -> None:
        """
        Close the connection pool and clean up resources.
        """
        if self.pool:
            logger.info("Closing database pool")
            await self.pool.close()
            self.pool = None
            logger.info("Database pool closed")

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[asyncpg.Connection]:
        """
        Acquire a connection from the pool.

        Usage:
            async with db_pool.acquire() as conn:
                result = await conn.fetch("SELECT * FROM documents")

        Yields:
            Database connection from the pool

        Raises:
            RuntimeError: If pool is not initialized
            asyncpg.PostgresError: If connection acquisition fails
        """
        if not self.pool:
            raise RuntimeError(
                "Database pool not initialized. Call connect() first."
            )

        async with self.pool.acquire() as conn:
            yield conn

    async def execute_query(self, query: str, *args: object) -> list[asyncpg.Record]:
        """
        Execute a query and return results.

        Convenience method for simple queries.

        Args:
            query: SQL query string
            *args: Query parameters

        Returns:
            List of query result records

        Raises:
            RuntimeError: If pool is not initialized
            asyncpg.PostgresError: If query execution fails
        """
        async with self.acquire() as conn:
            return await conn.fetch(query, *args)

    async def execute_one(
        self, query: str, *args: object
    ) -> asyncpg.Record | None:
        """
        Execute a query and return a single result.

        Args:
            query: SQL query string
            *args: Query parameters

        Returns:
            Single query result record or None

        Raises:
            RuntimeError: If pool is not initialized
            asyncpg.PostgresError: If query execution fails
        """
        async with self.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def execute_value(self, query: str, *args: object) -> object:
        """
        Execute a query and return a single value.

        Args:
            query: SQL query string
            *args: Query parameters

        Returns:
            Single value from query result

        Raises:
            RuntimeError: If pool is not initialized
            asyncpg.PostgresError: If query execution fails
        """
        async with self.acquire() as conn:
            return await conn.fetchval(query, *args)

    async def health_check(self) -> bool:
        """
        Check if database connection is healthy.

        Returns:
            True if database is accessible, False otherwise
        """
        try:
            if not self.pool:
                return False

            async with self.acquire() as conn:
                await conn.fetchval("SELECT 1")
                return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False

    def is_connected(self) -> bool:
        """
        Check if pool is initialized.

        Returns:
            True if pool exists, False otherwise
        """
        return self.pool is not None

    async def get_pool_stats(self) -> dict[str, object]:
        """
        Get connection pool statistics.

        Returns:
            Dictionary with pool statistics
        """
        if not self.pool:
            return {"status": "not_connected"}

        return {
            "status": "connected",
            "min_size": self.pool.get_min_size(),
            "max_size": self.pool.get_max_size(),
            "size": self.pool.get_size(),
            "free_size": self.pool.get_idle_size(),
        }