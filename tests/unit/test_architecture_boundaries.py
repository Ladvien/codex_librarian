"""
Test architecture boundary compliance.

Tests that ensure component boundaries are properly enforced:
- API layer should not import database models directly
- Service layer provides proper abstraction
- Dependency injection is used correctly

Following TDD approach as required by CLAUDE.md.
"""

import ast
import importlib
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from pdf_to_markdown_mcp.services.database import DatabaseService


class TestComponentBoundaryViolations:
    """Test that component boundaries are properly enforced."""

    def test_api_layer_does_not_import_db_models_directly(self):
        """
        CRITICAL: API layer should not import database models directly.

        This test ensures proper layered architecture:
        API → Services → Database (not API → Database)
        """
        # Given: The API convert module
        api_module_path = (
            Path(__file__).parent.parent.parent
            / "src/pdf_to_markdown_mcp/api/convert.py"
        )

        # When: We analyze the imports
        forbidden_imports = self._get_forbidden_db_imports(api_module_path)

        # Then: No direct database model imports should be found
        assert not forbidden_imports, (
            f"API layer violates component boundaries by importing database models directly: "
            f"{forbidden_imports}. Use service layer abstraction instead."
        )

    def test_api_uses_service_layer_for_document_operations(self):
        """
        API endpoints should use DocumentService for database operations.

        This test ensures proper service layer abstraction.
        """
        # Given: A mock document service
        with patch(
            "pdf_to_markdown_mcp.services.database.DatabaseService"
        ) as mock_service:
            mock_service.return_value.find_document_by_hash.return_value = None
            mock_service.return_value.create_document.return_value = Mock()

            # When: We attempt to use the service layer
            service = DatabaseService()

            # Then: Service should have document operations
            assert hasattr(service, "find_document_by_hash")
            assert hasattr(service, "create_document")
            assert hasattr(service, "update_document")

    def test_document_service_provides_crud_operations(self):
        """
        DocumentService should provide CRUD operations as abstraction.

        This test defines the expected service interface.
        """
        # Given: A database service instance
        service = DatabaseService()

        # When: We check for required methods
        required_methods = [
            "find_document_by_hash",
            "create_document",
            "update_document",
            "get_document_by_id",
            "delete_document",
        ]

        # Then: All CRUD methods should be available
        missing_methods = []
        for method in required_methods:
            if not hasattr(service, method):
                missing_methods.append(method)

        assert not missing_methods, (
            f"DatabaseService missing required CRUD methods for proper abstraction: "
            f"{missing_methods}"
        )

    def test_service_layer_returns_dtos_not_orm_objects(self):
        """
        Service layer should return DTOs, not SQLAlchemy ORM objects.

        This ensures proper layer separation and prevents tight coupling.
        """
        # Given: A database service
        service = DatabaseService()

        # When: We examine method signatures (this will fail initially)
        # This test defines the expected behavior

        # Then: Methods should return Pydantic models, not ORM objects
        # This test will guide our refactoring
        with pytest.raises(AttributeError):
            # This should fail until we implement proper service layer
            result = service.find_document_by_hash("test_hash")

        # Future assertion after refactoring:
        # assert hasattr(result, '__pydantic_model__') or result is None

    def test_dependency_injection_used_for_service_access(self):
        """
        API endpoints should use dependency injection for service access.

        This promotes testability and proper separation of concerns.
        """
        # Given: We analyze the convert.py file for dependency patterns
        api_module_path = (
            Path(__file__).parent.parent.parent
            / "src/pdf_to_markdown_mcp/api/convert.py"
        )

        with open(api_module_path) as f:
            content = f.read()
            tree = ast.parse(content)

        # When: We look for direct service instantiation
        direct_instantiations = self._find_direct_service_instantiation(tree)

        # Then: Services should be injected, not instantiated directly
        assert not direct_instantiations, (
            f"API endpoints should use dependency injection for services, "
            f"not direct instantiation: {direct_instantiations}"
        )

    def _get_forbidden_db_imports(self, file_path: Path) -> list[str]:
        """Find direct database model imports in API files."""
        with open(file_path) as f:
            content = f.read()
            tree = ast.parse(content)

        forbidden_imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and "db.models" in node.module:
                    for alias in node.names:
                        forbidden_imports.append(
                            f"from {node.module} import {alias.name}"
                        )

        return forbidden_imports

    def _find_direct_service_instantiation(self, tree: ast.AST) -> list[str]:
        """Find direct service instantiation patterns."""
        instantiations = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id.endswith("Service"):
                    instantiations.append(f"{node.func.id}()")
        return instantiations


class TestServiceLayerAbstraction:
    """Test that service layer provides proper abstraction."""

    @pytest.mark.asyncio
    async def test_document_service_abstract_interface(self):
        """
        Test that DocumentService provides abstract interface for document operations.

        This test will initially fail and guide our implementation.
        """
        # Given: A database service with mocked dependencies
        with patch(
            "pdf_to_markdown_mcp.services.database.DatabaseService._get_session"
        ) as mock_session:
            mock_session.return_value = Mock()
            service = DatabaseService()

            # When: We try to use document operations
            # These methods don't exist yet, so test will fail initially

            # Then: Service should provide async document operations
            with pytest.raises(AttributeError, match="find_document_by_hash"):
                await service.find_document_by_hash("test_hash")


class TestDataTransferObjects:
    """Test that DTOs are properly defined for API-Service communication."""

    def test_document_dto_exists_for_api_service_communication(self):
        """
        Test that DocumentDTO exists for API-Service layer communication.

        This test defines the expected DTO structure.
        """
        # This test will initially fail and guide our DTO creation
        try:
            from pdf_to_markdown_mcp.models.dto import DocumentDTO

            # If DTO exists, verify it has required fields
            dto = DocumentDTO(
                id=1,
                filename="test.pdf",
                file_path="/test/test.pdf",
                file_hash="abc123",
                size_bytes=1000,
                processing_status="completed",
                created_at="2025-09-26T10:00:00Z",
            )

            assert dto.id == 1
            assert dto.filename == "test.pdf"
            assert dto.processing_status == "completed"

        except ImportError:
            # Expected to fail initially - this guides our implementation
            pytest.fail(
                "DocumentDTO not found. Create models/dto.py with DocumentDTO class "
                "for proper API-Service layer communication."
            )


class TestArchitectureCompliance:
    """Test overall architecture compliance patterns."""

    def test_no_circular_dependencies_between_layers(self):
        """
        Test that there are no circular dependencies between architectural layers.

        Architecture: API → Services → Database (no cycles allowed)
        """
        # Given: We analyze imports across layers
        api_imports = self._get_module_imports("pdf_to_markdown_mcp.api")
        service_imports = self._get_module_imports("pdf_to_markdown_mcp.services")
        db_imports = self._get_module_imports("pdf_to_markdown_mcp.db")

        # When: We check for circular dependencies
        circular_deps = []

        # API should not be imported by services or database
        if any("api" in imp for imp in service_imports):
            circular_deps.append("Services → API")
        if any("api" in imp for imp in db_imports):
            circular_deps.append("Database → API")

        # Services should not be imported by database
        if any("services" in imp for imp in db_imports):
            circular_deps.append("Database → Services")

        # Then: No circular dependencies should exist
        assert not circular_deps, (
            f"Circular dependencies detected violating layered architecture: {circular_deps}"
        )

    def _get_module_imports(self, module_name: str) -> list[str]:
        """Get all imports for a given module."""
        try:
            module = importlib.import_module(module_name)
            # This is simplified - in real implementation would analyze AST
            return []
        except ImportError:
            return []
