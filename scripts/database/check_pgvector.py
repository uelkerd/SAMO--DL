        # Check if vector extension is available
        # Close cursor and connection
        # Connect to the database
        # Create a cursor
    # Fall back to individual environment variables
    # dotenv not installed, skip loading
    from dotenv import load_dotenv
# Load environment variables from .env file
# Parse DATABASE_URL or fall back to individual env vars
#!/usr/bin/env python3
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from urllib.parse import urlparse
import logging
import os
import psycopg2
import sys






"""Script to check if pgvector extension is installed in PostgreSQL."""

try:
    load_dotenv()
except ImportError:
    pass

DATABASE_URL = os.environ.get("DATABASE_URL")
if DATABASE_URL:
    parsed = urlparse(DATABASE_URL)
    DB_USER = parsed.username
    DB_PASSWORD = parsed.password
    DB_HOST = parsed.hostname
    DB_PORT = parsed.port or 5432
    DB_NAME = parsed.path.lstrip("/")
else:
    DB_USER = os.environ.get("DB_USER", "samouser")
    DB_PASSWORD = os.environ.get("DB_PASSWORD", "samopassword")
    DB_HOST = os.environ.get("DB_HOST", "localhost")
    DB_PORT = os.environ.get("DB_PORT", "5432")
    DB_NAME = os.environ.get("DB_NAME", "samodb")


def check_pgvector():
    """Check if pgvector extension is installed and available."""
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT,
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

        cur = conn.cursor()

        cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector';")
        is_installed = cur.fetchone() is not None

        if is_installed:
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
            logging.info("   - \\c {DB_NAME}")
            logging.info("   - CREATE EXTENSION vector;")

        cur.close()
        conn.close()

        return is_installed

    except psycopg2.Error as e:
        logging.info("Error connecting to PostgreSQL: {e}")
        return False


if __name__ == "__main__":
    is_installed = check_pgvector()
    sys.exit(0 if is_installed else 1)
