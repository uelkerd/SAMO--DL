    # Create tables
    # Import all models here to ensure they're registered with Base.metadata
# Create engine
# Create scoped session for thread safety
# Create sessionmaker
# Create the database URL
# Get database connection details from environment variables
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker
import os



"""Database connection utilities for the SAMO-DL application."""


DB_USER = os.environ.get("DB_USER", "samouser")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "samopassword")
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "samodb")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

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
