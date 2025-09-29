#!/bin/bash
# Setup script for PDF to Markdown MCP Server
# This script performs initial setup and configuration

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_PATH="$PROJECT_DIR/.venv"

echo -e "${GREEN}PDF to Markdown MCP Server - Setup Script${NC}"
echo "==========================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to print step header
print_step() {
    echo -e "${BLUE}[STEP] $1${NC}"
}

# Function to print success
print_success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

# Function to print warning
print_warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

# Function to print error
print_error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

# Step 1: Check prerequisites
print_step "Checking prerequisites..."

if ! command_exists python3; then
    print_error "Python 3 is not installed. Please install Python 3.11 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if [[ $(echo "$PYTHON_VERSION >= 3.11" | bc -l 2>/dev/null || echo "0") -eq 1 ]]; then
    print_success "Python $PYTHON_VERSION found"
else
    print_error "Python 3.11 or higher is required. Found: $PYTHON_VERSION"
    exit 1
fi

if ! command_exists uv; then
    print_warning "uv package manager not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source ~/.bashrc
    if ! command_exists uv; then
        print_error "Failed to install uv. Please install manually from https://github.com/astral-sh/uv"
        exit 1
    fi
    print_success "uv installed successfully"
else
    print_success "uv package manager found"
fi

# Step 2: Create virtual environment
print_step "Setting up virtual environment..."
cd "$PROJECT_DIR"

if [ ! -d "$VENV_PATH" ]; then
    uv venv
    print_success "Virtual environment created"
else
    print_success "Virtual environment already exists"
fi

# Activate virtual environment
source "$VENV_PATH/bin/activate"

# Step 3: Install dependencies
print_step "Installing dependencies..."
uv pip install -e ".[dev]"
print_success "Dependencies installed"

# Step 4: Setup environment variables
print_step "Setting up environment variables..."
if [ ! -f "$PROJECT_DIR/.env" ]; then
    cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
    print_success ".env file created from template"
    print_warning "Please edit .env file with your actual configuration values"
else
    print_success ".env file already exists"
fi

# Step 5: Check PostgreSQL
print_step "Checking PostgreSQL availability..."
if command_exists psql; then
    print_success "PostgreSQL client found"

    # Try to connect to PostgreSQL
    if sudo -u postgres psql -c '\q' 2>/dev/null; then
        print_success "PostgreSQL server is accessible"

        # Check if database exists
        DB_EXISTS=$(sudo -u postgres psql -tAc "SELECT 1 FROM pg_database WHERE datname='pdf_to_markdown_mcp'" 2>/dev/null || echo "0")
        if [ "$DB_EXISTS" = "1" ]; then
            print_success "Database 'pdf_to_markdown_mcp' already exists"
        else
            print_warning "Database 'pdf_to_markdown_mcp' does not exist"
            echo "To create the database, run:"
            echo "  sudo -u postgres createdb pdf_to_markdown_mcp"
            echo "  sudo -u postgres psql -c \"CREATE USER pdf_user WITH PASSWORD 'your_password';\""
            echo "  sudo -u postgres psql -c \"GRANT ALL PRIVILEGES ON DATABASE pdf_to_markdown_mcp TO pdf_user;\""
        fi

        # Check PGVector extension
        PGVECTOR_EXISTS=$(sudo -u postgres psql -d pdf_to_markdown_mcp -tAc "SELECT 1 FROM pg_extension WHERE extname='vector'" 2>/dev/null || echo "0")
        if [ "$PGVECTOR_EXISTS" = "1" ]; then
            print_success "PGVector extension is enabled"
        else
            print_warning "PGVector extension is not enabled"
            echo "To enable PGVector, run:"
            echo "  sudo -u postgres psql -d pdf_to_markdown_mcp -c \"CREATE EXTENSION IF NOT EXISTS vector;\""
        fi
    else
        print_warning "PostgreSQL server is not running or not accessible"
        echo "Please ensure PostgreSQL is installed and running"
    fi
else
    print_warning "PostgreSQL not found"
    echo "Please install PostgreSQL 15+ with PGVector extension"
fi

# Step 6: Check Redis
print_step "Checking Redis availability..."
if command_exists redis-cli; then
    if redis-cli ping >/dev/null 2>&1; then
        print_success "Redis server is running"
    else
        print_warning "Redis server is not running"
        echo "Start Redis with: redis-server"
    fi
else
    print_warning "Redis not found"
    echo "Please install Redis server"
fi

# Step 7: Run database migrations
print_step "Running database migrations..."
if alembic upgrade head 2>/dev/null; then
    print_success "Database migrations completed"
else
    print_warning "Database migrations failed - check database connection"
fi

# Step 8: Install pre-commit hooks
print_step "Installing pre-commit hooks..."
if pre-commit install 2>/dev/null; then
    print_success "Pre-commit hooks installed"
else
    print_warning "Failed to install pre-commit hooks"
fi

# Step 9: Run tests to verify setup
print_step "Running tests to verify setup..."
if pytest tests/ -v --tb=short -x 2>/dev/null; then
    print_success "All tests passed - setup is complete!"
else
    print_warning "Some tests failed - setup may need adjustment"
    echo "Run 'pytest -v' for detailed test results"
fi

echo ""
echo -e "${GREEN}Setup Summary:${NC}"
echo "=============="
echo -e "📁 Project directory: ${PROJECT_DIR}"
echo -e "🐍 Virtual environment: ${VENV_PATH}"
echo -e "⚙️  Environment file: ${PROJECT_DIR}/.env"
echo ""
echo -e "${GREEN}Next Steps:${NC}"
echo "1. Edit .env file with your configuration"
echo "2. Ensure PostgreSQL and Redis are running"
echo "3. Run: source .venv/bin/activate"
echo "4. Start the server: uvicorn pdf_to_markdown_mcp.main:app --reload"
echo ""
echo -e "${BLUE}Useful Commands:${NC}"
echo "• Start services: ./scripts/start_worker_services.sh"
echo "• Stop services: ./scripts/stop_worker_services.sh"
echo "• Run tests: pytest"
echo "• Format code: black src/ tests/"
echo "• Check linting: ruff check src/ tests/"
echo ""
echo -e "${GREEN}Setup completed successfully!${NC}"