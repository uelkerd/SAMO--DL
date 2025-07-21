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

# Check if database already exists
if psql -lqt | cut -d \| -f 1 | grep -qw "${DB_NAME}"; then
    echo "Database ${DB_NAME} already exists."
else
    # Create database and user
    echo "Creating database ${DB_NAME} and user ${DB_USER}..."
    
    # Create user if not exists
    psql -c "CREATE USER ${DB_USER} WITH PASSWORD '${DB_PASSWORD}';" || echo "User already exists"
    
    # Create database
    psql -c "CREATE DATABASE ${DB_NAME} OWNER ${DB_USER};"
    
    echo "Database and user created."
fi

# Connect and initialize pgvector extension
echo "Installing pgvector extension..."
psql -d ${DB_NAME} -c "CREATE EXTENSION IF NOT EXISTS vector;" || echo "Failed to create vector extension. Make sure it's installed."

# Apply schema
echo "Applying database schema..."
psql -d ${DB_NAME} -f "$(dirname "$0")/schema.sql"

echo "Database setup complete!" 