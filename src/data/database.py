    # Create tables
    # Import all models here to ensure they're registered with Base.metadata
# Create engine
# Create scoped session for thread safety
# Create sessionmaker
# Create the database URL
# Get database connection details from environment variables
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker
import os
from pathlib import Path
from urllib.parse import quote_plus
from src.common.env import is_truthy



"""Database connection utilities for the SAMO-DL application."""


# Respect DATABASE_URL if provided explicitly (preferred)
_env_database_url = os.environ.get("DATABASE_URL")

DB_USER = os.environ.get("DB_USER")
DB_PASSWORD = os.environ.get("DB_PASSWORD")
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME")

if _env_database_url:
    DATABASE_URL = _env_database_url
elif DB_USER and DB_PASSWORD and DB_NAME:
    # Safely build Postgres URL if all parts are provided
    safe_user = quote_plus(DB_USER)
    safe_password = quote_plus(DB_PASSWORD)
    safe_host = DB_HOST
    safe_port = DB_PORT
    safe_db = DB_NAME
DATABASE_URL =
f"postgresql://{safe_user}:{safe_password}@{safe_host}:{safe_port}/{safe_db}"
else:
    # Fall back to SQLite only when explicitly allowed or in CI/TEST
    allow_sqlite = (
        is_truthy(os.environ.get("ALLOW_SQLITE_FALLBACK"))
        or is_truthy(os.environ.get("TESTING"))
        or is_truthy(os.environ.get("CI"))
    )
    if not allow_sqlite:
        raise RuntimeError(
            "SQLite fallback is disabled. Set DATABASE_URL or all Postgres env vars, "
"or explicitly allow SQLite fallback via ALLOW_SQLITE_FALLBACK=1 in dev/test."
        )
    default_sqlite_path = Path(
                               os.environ.get("SQLITE_PATH",
                               "./samo_local.db")).expanduser().resolve(
                              )
    # Ensure directory for SQLite exists before engine creation
    sqlite_dir = default_sqlite_path.parent
    try:
        sqlite_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        raise RuntimeError(f"Failed to create SQLite directory '{sqlite_dir}': {exc}")
    DATABASE_URL = f"sqlite:///{default_sqlite_path}"

if DATABASE_URL.startswith("sqlite"):
    # SQLite engine options; most pooling params are not applicable
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=NullPool,
    )
else:
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,  # Check connection before using
        pool_size=5,  # Default pool size
        max_overflow=10,  # Allow up to 10 additional connections
        pool_recycle=3600,  # Recycle connections after 1 hour
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

db_session = scoped_session(SessionLocal)

Base = declarative_base()
Base.query = db_session.query_property()


def get_db():
    """Get a database session.

    This function should be used as a dependency in FastAPI endpoints.

    Yields:
        Session: SQLAlchemy database session

    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Initialize the database - create tables if they don't exist.

    This function should be called when the application starts.
    """
    Base.metadata.create_all(bind=engine)
