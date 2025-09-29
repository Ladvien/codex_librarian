#!/bin/bash
# Database initialization script for PDF to Markdown MCP Server
# This script sets up PostgreSQL database with PGVector extension

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration (can be overridden by .env)
DB_NAME="pdf_to_markdown_mcp"
DB_USER="pdf_user"
DB_PASSWORD="CHANGE_THIS_PASSWORD"
DB_HOST="localhost"
DB_PORT="5432"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo -e "${GREEN}PDF to Markdown MCP Server - Database Initialization${NC}"
echo "====================================================="

# Load environment variables if .env exists
if [ -f "$PROJECT_DIR/.env" ]; then
    echo -e "${BLUE}Loading environment variables from .env${NC}"
    source "$PROJECT_DIR/.env"
fi

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

# Check if PostgreSQL is available
print_step "Checking PostgreSQL availability..."
if ! command_exists psql; then
    print_error "PostgreSQL client (psql) not found. Please install PostgreSQL."
    exit 1
fi

# Check if PostgreSQL server is running
if ! sudo -u postgres psql -c '\q' 2>/dev/null; then
    print_error "Cannot connect to PostgreSQL server. Please ensure PostgreSQL is installed and running."
    exit 1
fi

print_success "PostgreSQL is available"

# Check if database already exists
print_step "Checking if database exists..."
DB_EXISTS=$(sudo -u postgres psql -tAc "SELECT 1 FROM pg_database WHERE datname='$DB_NAME'" 2>/dev/null || echo "0")

if [ "$DB_EXISTS" = "1" ]; then
    print_warning "Database '$DB_NAME' already exists"
    echo -n "Do you want to drop and recreate it? (y/N): "
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        print_step "Dropping existing database..."
        sudo -u postgres dropdb "$DB_NAME" 2>/dev/null || true
        print_success "Database dropped"
    else
        echo "Skipping database creation"
    fi
fi

# Create database if it doesn't exist
DB_EXISTS=$(sudo -u postgres psql -tAc "SELECT 1 FROM pg_database WHERE datname='$DB_NAME'" 2>/dev/null || echo "0")
if [ "$DB_EXISTS" != "1" ]; then
    print_step "Creating database '$DB_NAME'..."
    sudo -u postgres createdb "$DB_NAME"
    print_success "Database '$DB_NAME' created"
fi

# Check if user exists
print_step "Checking database user..."
USER_EXISTS=$(sudo -u postgres psql -tAc "SELECT 1 FROM pg_roles WHERE rolname='$DB_USER'" 2>/dev/null || echo "0")

if [ "$USER_EXISTS" = "1" ]; then
    print_warning "User '$DB_USER' already exists"
else
    print_step "Creating database user '$DB_USER'..."
    sudo -u postgres psql -c "CREATE USER $DB_USER WITH PASSWORD '$DB_PASSWORD';"
    print_success "User '$DB_USER' created"
fi

# Grant privileges
print_step "Granting privileges to user '$DB_USER'..."
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER;"
sudo -u postgres psql -d "$DB_NAME" -c "GRANT ALL ON SCHEMA public TO $DB_USER;"
sudo -u postgres psql -d "$DB_NAME" -c "GRANT CREATE ON SCHEMA public TO $DB_USER;"
print_success "Privileges granted"

# Install PGVector extension
print_step "Installing PGVector extension..."
PGVECTOR_EXISTS=$(sudo -u postgres psql -d "$DB_NAME" -tAc "SELECT 1 FROM pg_extension WHERE extname='vector'" 2>/dev/null || echo "0")

if [ "$PGVECTOR_EXISTS" = "1" ]; then
    print_success "PGVector extension already installed"
else
    # Check if pgvector is available
    if sudo -u postgres psql -d "$DB_NAME" -c "CREATE EXTENSION IF NOT EXISTS vector;" 2>/dev/null; then
        print_success "PGVector extension installed"
    else
        print_error "Failed to install PGVector extension"
        echo "Please install pgvector package:"
        echo "  Ubuntu/Debian: sudo apt install postgresql-15-pgvector"
        echo "  CentOS/RHEL: sudo yum install pgvector"
        echo "  Or compile from source: https://github.com/pgvector/pgvector"
        exit 1
    fi
fi

# Test connection with the new user
print_step "Testing database connection..."
if PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c '\q' 2>/dev/null; then
    print_success "Database connection test successful"
else
    print_error "Cannot connect to database with user '$DB_USER'"
    exit 1
fi

# Run Alembic migrations if available
print_step "Running database migrations..."
cd "$PROJECT_DIR"

# Activate virtual environment if it exists
if [ -f "$PROJECT_DIR/.venv/bin/activate" ]; then
    source "$PROJECT_DIR/.venv/bin/activate"
fi

if command_exists alembic && [ -f "$PROJECT_DIR/alembic.ini" ]; then
    if alembic upgrade head 2>/dev/null; then
        print_success "Database migrations completed"
    else
        print_warning "Database migrations failed or no migrations to run"
    fi
else
    print_warning "Alembic not available - skipping migrations"
fi

# Verify database schema
print_step "Verifying database setup..."
TABLES_COUNT=$(PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -tAc "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public'" 2>/dev/null || echo "0")

echo ""
echo -e "${GREEN}Database Setup Summary:${NC}"
echo "======================"
echo -e "🗄️  Database: ${DB_NAME}"
echo -e "👤 User: ${DB_USER}"
echo -e "🏠 Host: ${DB_HOST}:${DB_PORT}"
echo -e "📊 Tables: ${TABLES_COUNT}"
echo -e "🔌 PGVector: $([ "$PGVECTOR_EXISTS" = "1" ] && echo "Enabled" || echo "Installed")"
echo ""
echo -e "${GREEN}Connection String:${NC}"
echo "postgresql://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME}"
echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo "1. Update your .env file with the database configuration"
echo "2. Run: alembic upgrade head (if not already done)"
echo "3. Start your application"
echo ""
echo -e "${GREEN}Database initialization completed successfully!${NC}"