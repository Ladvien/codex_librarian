"""
Tests for MCP server database context module.

Tests connection pool management, session lifecycle, and error handling.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import asyncpg
import pytest

from src.pdf_to_markdown_mcp.mcp.config import MCPConfig
from src.pdf_to_markdown_mcp.mcp.context import DatabasePool


@pytest.fixture
def mock_config() -> MCPConfig:
    """Create mock MCPConfig for testing."""
    return MCPConfig(
        database_url="postgresql://user:pass@localhost:5432/test_db",
        ollama_url="http://localhost:11434",
        db_pool_min_size=2,
        db_pool_max_size=10,
        db_pool_timeout=30,
    )


@pytest.fixture
def database_pool(mock_config: MCPConfig) -> DatabasePool:
    """Create DatabasePool instance for testing."""
    return DatabasePool(mock_config)


class TestDatabasePoolInit:
    """Test DatabasePool initialization."""

    def test_init_creates_pool_instance(self, mock_config: MCPConfig) -> None:
        """Test initialization creates DatabasePool with config."""
        pool = DatabasePool(mock_config)

        assert pool.config == mock_config
        assert pool.pool is None

    def test_init_stores_config(self, mock_config: MCPConfig) -> None:
        """Test initialization stores configuration."""
        pool = DatabasePool(mock_config)

        assert pool.config.database_url == "postgresql://user:pass@localhost:5432/test_db"
        assert pool.config.db_pool_min_size == 2
        assert pool.config.db_pool_max_size == 10


class TestDatabasePoolConnect:
    """Test database pool connection."""

    @pytest.mark.asyncio
    async def test_connect_creates_pool(self, database_pool: DatabasePool) -> None:
        """Test connect() creates asyncpg pool."""
        mock_pool = AsyncMock(spec=asyncpg.Pool)
        mock_conn = AsyncMock(spec=asyncpg.Connection)
        mock_conn.fetchval = AsyncMock(return_value="PostgreSQL 17.0")

        # Mock connection context manager
        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch("asyncpg.create_pool", AsyncMock(return_value=mock_pool)):
            await database_pool.connect()

            assert database_pool.pool is not None
            mock_conn.fetchval.assert_called_once_with("SELECT version()")

    @pytest.mark.asyncio
    async def test_connect_uses_config_parameters(self, database_pool: DatabasePool) -> None:
        """Test connect() uses configuration parameters."""
        mock_pool = AsyncMock(spec=asyncpg.Pool)
        mock_conn = AsyncMock(spec=asyncpg.Connection)
        mock_conn.fetchval = AsyncMock(return_value="PostgreSQL 17.0")

        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        mock_create = AsyncMock(return_value=mock_pool)
        with patch("asyncpg.create_pool", mock_create):
            await database_pool.connect()

            mock_create.assert_called_once()
            call_kwargs = mock_create.call_args[1]
            assert call_kwargs["dsn"] == "postgresql://user:pass@localhost:5432/test_db"
            assert call_kwargs["min_size"] == 2
            assert call_kwargs["max_size"] == 10
            assert call_kwargs["timeout"] == 30
            assert call_kwargs["command_timeout"] == 60

    @pytest.mark.asyncio
    async def test_connect_initializes_connections(self, database_pool: DatabasePool) -> None:
        """Test connect() sets up connection initialization callback."""
        mock_pool = AsyncMock(spec=asyncpg.Pool)
        mock_conn = AsyncMock(spec=asyncpg.Connection)
        mock_conn.fetchval = AsyncMock(return_value="PostgreSQL 17.0")

        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        mock_create = AsyncMock(return_value=mock_pool)
        with patch("asyncpg.create_pool", mock_create):
            await database_pool.connect()

            # Verify init callback was provided
            assert "init" in mock_create.call_args[1]
            assert callable(mock_create.call_args[1]["init"])

    @pytest.mark.asyncio
    async def test_connect_raises_on_connection_error(self, database_pool: DatabasePool) -> None:
        """Test connect() raises ConnectionError on failure."""
        with patch("asyncpg.create_pool", side_effect=Exception("Connection failed")):
            with pytest.raises(ConnectionError, match="Database connection failed"):
                await database_pool.connect()

    @pytest.mark.asyncio
    async def test_init_connection_sets_timeout(self, database_pool: DatabasePool) -> None:
        """Test _init_connection sets statement timeout."""
        mock_conn = AsyncMock(spec=asyncpg.Connection)

        await database_pool._init_connection(mock_conn)

        mock_conn.execute.assert_any_call("SET statement_timeout = '60s'")

    @pytest.mark.asyncio
    async def test_init_connection_checks_pgvector(self, database_pool: DatabasePool) -> None:
        """Test _init_connection checks for PGVector extension."""
        mock_conn = AsyncMock(spec=asyncpg.Connection)

        await database_pool._init_connection(mock_conn)

        # Should check for pgvector extension
        calls = [str(call) for call in mock_conn.execute.call_args_list]
        pgvector_check = any("pg_extension" in str(call) for call in calls)
        assert pgvector_check


class TestDatabasePoolDisconnect:
    """Test database pool disconnection."""

    @pytest.mark.asyncio
    async def test_disconnect_closes_pool(self, database_pool: DatabasePool) -> None:
        """Test disconnect() closes the pool."""
        mock_pool = AsyncMock(spec=asyncpg.Pool)
        database_pool.pool = mock_pool

        await database_pool.disconnect()

        mock_pool.close.assert_called_once()
        assert database_pool.pool is None

    @pytest.mark.asyncio
    async def test_disconnect_when_no_pool(self, database_pool: DatabasePool) -> None:
        """Test disconnect() handles no pool gracefully."""
        database_pool.pool = None

        # Should not raise
        await database_pool.disconnect()


class TestDatabasePoolAcquire:
    """Test connection acquisition from pool."""

    @pytest.mark.asyncio
    async def test_acquire_returns_connection(self, database_pool: DatabasePool) -> None:
        """Test acquire() returns a connection from pool."""
        mock_pool = AsyncMock(spec=asyncpg.Pool)
        mock_conn = AsyncMock(spec=asyncpg.Connection)

        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        database_pool.pool = mock_pool

        async with database_pool.acquire() as conn:
            assert conn == mock_conn

    @pytest.mark.asyncio
    async def test_acquire_raises_when_pool_not_initialized(
        self, database_pool: DatabasePool
    ) -> None:
        """Test acquire() raises RuntimeError when pool not initialized."""
        database_pool.pool = None

        with pytest.raises(RuntimeError, match="Database pool not initialized"):
            async with database_pool.acquire():
                pass


class TestDatabasePoolExecuteQuery:
    """Test query execution convenience methods."""

    @pytest.mark.asyncio
    async def test_execute_query_returns_results(self, database_pool: DatabasePool) -> None:
        """Test execute_query() returns query results."""
        mock_pool = AsyncMock(spec=asyncpg.Pool)
        mock_conn = AsyncMock(spec=asyncpg.Connection)
        mock_results = [{"id": 1, "name": "test"}]
        mock_conn.fetch = AsyncMock(return_value=mock_results)

        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        database_pool.pool = mock_pool

        results = await database_pool.execute_query("SELECT * FROM documents")

        assert results == mock_results
        mock_conn.fetch.assert_called_once_with("SELECT * FROM documents")

    @pytest.mark.asyncio
    async def test_execute_query_with_parameters(self, database_pool: DatabasePool) -> None:
        """Test execute_query() passes parameters correctly."""
        mock_pool = AsyncMock(spec=asyncpg.Pool)
        mock_conn = AsyncMock(spec=asyncpg.Connection)
        mock_conn.fetch = AsyncMock(return_value=[])

        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        database_pool.pool = mock_pool

        await database_pool.execute_query("SELECT * FROM documents WHERE id = $1", 123)

        mock_conn.fetch.assert_called_once_with("SELECT * FROM documents WHERE id = $1", 123)

    @pytest.mark.asyncio
    async def test_execute_one_returns_single_result(self, database_pool: DatabasePool) -> None:
        """Test execute_one() returns single row."""
        mock_pool = AsyncMock(spec=asyncpg.Pool)
        mock_conn = AsyncMock(spec=asyncpg.Connection)
        mock_result = {"id": 1, "name": "test"}
        mock_conn.fetchrow = AsyncMock(return_value=mock_result)

        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        database_pool.pool = mock_pool

        result = await database_pool.execute_one("SELECT * FROM documents WHERE id = $1", 1)

        assert result == mock_result
        mock_conn.fetchrow.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_one_returns_none_when_no_result(
        self, database_pool: DatabasePool
    ) -> None:
        """Test execute_one() returns None when no results."""
        mock_pool = AsyncMock(spec=asyncpg.Pool)
        mock_conn = AsyncMock(spec=asyncpg.Connection)
        mock_conn.fetchrow = AsyncMock(return_value=None)

        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        database_pool.pool = mock_pool

        result = await database_pool.execute_one("SELECT * FROM documents WHERE id = $1", 999)

        assert result is None

    @pytest.mark.asyncio
    async def test_execute_value_returns_single_value(self, database_pool: DatabasePool) -> None:
        """Test execute_value() returns single value."""
        mock_pool = AsyncMock(spec=asyncpg.Pool)
        mock_conn = AsyncMock(spec=asyncpg.Connection)
        mock_conn.fetchval = AsyncMock(return_value=42)

        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        database_pool.pool = mock_pool

        result = await database_pool.execute_value("SELECT COUNT(*) FROM documents")

        assert result == 42
        mock_conn.fetchval.assert_called_once()


class TestDatabasePoolHealthCheck:
    """Test database health check functionality."""

    @pytest.mark.asyncio
    async def test_health_check_returns_true_when_healthy(
        self, database_pool: DatabasePool
    ) -> None:
        """Test health_check() returns True when database is accessible."""
        mock_pool = AsyncMock(spec=asyncpg.Pool)
        mock_conn = AsyncMock(spec=asyncpg.Connection)
        mock_conn.fetchval = AsyncMock(return_value=1)

        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        database_pool.pool = mock_pool

        is_healthy = await database_pool.health_check()

        assert is_healthy is True
        mock_conn.fetchval.assert_called_once_with("SELECT 1")

    @pytest.mark.asyncio
    async def test_health_check_returns_false_when_no_pool(
        self, database_pool: DatabasePool
    ) -> None:
        """Test health_check() returns False when pool not initialized."""
        database_pool.pool = None

        is_healthy = await database_pool.health_check()

        assert is_healthy is False

    @pytest.mark.asyncio
    async def test_health_check_returns_false_on_error(
        self, database_pool: DatabasePool
    ) -> None:
        """Test health_check() returns False when query fails."""
        mock_pool = AsyncMock(spec=asyncpg.Pool)
        mock_conn = AsyncMock(spec=asyncpg.Connection)
        mock_conn.fetchval = AsyncMock(side_effect=Exception("Connection lost"))

        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        database_pool.pool = mock_pool

        is_healthy = await database_pool.health_check()

        assert is_healthy is False


class TestDatabasePoolConnectionStatus:
    """Test connection status checking."""

    def test_is_connected_returns_true_when_pool_exists(
        self, database_pool: DatabasePool
    ) -> None:
        """Test is_connected() returns True when pool exists."""
        database_pool.pool = AsyncMock(spec=asyncpg.Pool)

        assert database_pool.is_connected() is True

    def test_is_connected_returns_false_when_no_pool(
        self, database_pool: DatabasePool
    ) -> None:
        """Test is_connected() returns False when no pool."""
        database_pool.pool = None

        assert database_pool.is_connected() is False


class TestDatabasePoolStats:
    """Test pool statistics retrieval."""

    @pytest.mark.asyncio
    async def test_get_pool_stats_returns_stats_when_connected(
        self, database_pool: DatabasePool
    ) -> None:
        """Test get_pool_stats() returns statistics when connected."""
        mock_pool = AsyncMock(spec=asyncpg.Pool)
        mock_pool.get_min_size = MagicMock(return_value=2)
        mock_pool.get_max_size = MagicMock(return_value=10)
        mock_pool.get_size = MagicMock(return_value=5)
        mock_pool.get_idle_size = MagicMock(return_value=3)

        database_pool.pool = mock_pool

        stats = await database_pool.get_pool_stats()

        assert stats["status"] == "connected"
        assert stats["min_size"] == 2
        assert stats["max_size"] == 10
        assert stats["size"] == 5
        assert stats["free_size"] == 3

    @pytest.mark.asyncio
    async def test_get_pool_stats_returns_not_connected(
        self, database_pool: DatabasePool
    ) -> None:
        """Test get_pool_stats() returns not_connected when no pool."""
        database_pool.pool = None

        stats = await database_pool.get_pool_stats()

        assert stats == {"status": "not_connected"}
