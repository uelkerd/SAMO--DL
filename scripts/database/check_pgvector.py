#!/usr/bin/env python3
"""Script to check if pgvector extension is installed in PostgreSQL."""

import logging
import os
import sys
from urllib.parse import urlparse

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not installed, skip loading
    pass

# Parse DATABASE_URL or fall back to individual env vars
DATABASE_URL = os.environ.get("DATABASE_URL")
if DATABASE_URL:
    parsed = urlparse(DATABASE_URL)
    DB_USER = parsed.username
    DB_PASSWORD = parsed.password
    DB_HOST = parsed.hostname
    DB_PORT = parsed.port or 5432
    DB_NAME = parsed.path.lstrip("/")
else:
    # Fall back to individual environment variables
    DB_USER = os.environ.get("DB_USER")
    DB_PASSWORD = os.environ.get("DB_PASSWORD")
    DB_HOST = os.environ.get("DB_HOST", "localhost")
    DB_PORT = os.environ.get("DB_PORT", "5432")
    DB_NAME = os.environ.get("DB_NAME")

# Validate required environment variables
if not DB_USER:
    raise ValueError("DB_USER environment variable is required")
if not DB_PASSWORD:
    raise ValueError("DB_PASSWORD environment variable is required")
if not DB_NAME:
    raise ValueError("DB_NAME environment variable is required")


def check_pgvector():
    """Check if pgvector extension is installed and available."""
    try:
        # Connect to the database
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT,
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

        # Create a cursor
        cur = conn.cursor()

        # Check if vector extension is available
        cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector';")
        extension_installed = cur.fetchone() is not None

        if extension_installed:
            logging.info("✅ pgvector extension is installed and available.")
        else:
            logging.info("❌ pgvector extension is NOT installed.")
            logging.info("\nTo install pgvector:")
            logging.info("1. Install the extension in your PostgreSQL server:")
            logging.info("   - On Ubuntu/Debian: sudo apt install postgresql-15-pgvector")
            logging.info("   - On macOS with Homebrew: brew install pgvector")
            logging.info("   - From source: https://github.com/pgvector/pgvector#installation")
            logging.info("\n2. Enable the extension in your database:")
            logging.info("   - psql -U postgres")
            logging.info(f"   - \\c {DB_NAME}")
            logging.info("   - CREATE EXTENSION vector;")

        # Close cursor and connection
        cur.close()
        conn.close()

        return extension_installed

    except psycopg2.Error as e:
        logging.info(f"Error connecting to PostgreSQL: {e}")
        return False


if __name__ == "__main__":
    is_installed = check_pgvector()
    sys.exit(0 if is_installed else 1)
