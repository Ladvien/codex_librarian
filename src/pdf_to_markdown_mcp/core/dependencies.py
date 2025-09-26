"""
Dependency injection for services.

Provides FastAPI dependencies for service layer access,
maintaining proper architecture boundaries and testability.

Following TDD and architecture principles from CLAUDE.md.
"""

from collections.abc import Generator

from fastapi import Depends
from sqlalchemy.orm import Session

from ..db.session import get_db
from ..services.database import VectorDatabaseService


def get_database_service(
    db: Session = Depends(get_db),
) -> Generator[VectorDatabaseService, None, None]:
    """
    Dependency injection for VectorDatabaseService.

    Provides proper service layer access with database session injection.
    Used by API endpoints to maintain architecture boundaries.

    Args:
        db: Database session from FastAPI dependency

    Yields:
        VectorDatabaseService instance with injected database session
    """
    service = VectorDatabaseService(db_session=db)
    try:
        yield service
    finally:
        # Service cleanup if needed
        pass


# Type aliases for cleaner imports
DatabaseServiceDep = Depends(get_database_service)


# Export dependencies
__all__ = [
    "DatabaseServiceDep",
    "get_database_service",
]
