#!/bin/bash

# Exit on error
set -e

echo "Setting up PostgreSQL database for SAMO-DL..."

# Database configuration - these would typically come from environment variables
DB_NAME=${DB_NAME:-"samodb"}
DB_USER=${DB_USER:-"samouser"}
DB_PASSWORD=${DB_PASSWORD:-"samopassword"} # In production, use a secure password
DB_HOST=${DB_HOST:-"localhost"}
DB_PORT=${DB_PORT:-"5432"}

# Connect to the postgres database first (which always exists)
if psql -d postgres -lqt | cut -d \| -f 1 | grep -qw "${DB_NAME}"; then
    echo "Database ${DB_NAME} already exists."
else
    # Create database and user
    echo "Creating database ${DB_NAME} and user ${DB_USER}..."

    # Create user if not exists
    psql -d postgres -c "DO \$\$
    BEGIN
        IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '${DB_USER}') THEN
            CREATE USER ${DB_USER} WITH PASSWORD '${DB_PASSWORD}';
        END IF;
    END
    \$\$;" || echo "User already exists or error creating user"

    # Create database
    psql -d postgres -c "CREATE DATABASE ${DB_NAME} OWNER ${DB_USER};" || echo "Database already exists or error creating database"

    echo "Database and user created."
fi

# Connect and initialize pgvector extension
echo "Installing pgvector extension..."
psql -d ${DB_NAME} -c "CREATE EXTENSION IF NOT EXISTS vector;" || echo "Failed to create vector extension. Make sure it's installed."

# Apply schema
echo "Applying database schema..."
psql -d ${DB_NAME} -f "$(dirname "$0")/schema.sql"

echo "Database setup complete!"
