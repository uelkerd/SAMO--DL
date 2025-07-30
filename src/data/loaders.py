from .database import db_session
from .models import JournalEntry
from .prisma_client import PrismaClient
from typing import Optional
import json
import pandas as pd






def load_entries_from_db(
    limit: Optional[int] = None, user_id: Optional[int] = None
) -> pd.DataFrame:
    """Load journal entries from database.

    Args:
        limit: Maximum number of entries to load
        user_id: Filter entries by user_id

    Returns:
        DataFrame containing journal entries

    """
    query = db_session.query(JournalEntry)

    if user_id is not None:
        query = query.filter(JournalEntry.user_id == user_id)

    if limit is not None:
        query = query.limit(limit)

    entries = query.all()

    data = [
        {
            "id": entry.id,
            "user_id": entry.user_id,
            "title": entry.title,
            "content": entry.content,
            "created_at": entry.created_at,
            "updated_at": entry.updated_at,
            "is_private": entry.is_private,
        }
        for entry in entries
    ]

    return pd.DataFrame(data)


def load_entries_from_prisma(
    limit: Optional[int] = None, user_id: Optional[str] = None
) -> pd.DataFrame:
    """Load journal entries using Prisma client.

    Args:
        limit: Maximum number of entries to load
        user_id: Filter entries by user_id

    Returns:
        DataFrame containing journal entries

    """
    prisma = PrismaClient()

    filters = {}
    if user_id is not None:
        filters["user_id"] = user_id

    entries = (
        prisma.get_journal_entries_by_user(user_id=user_id, limit=limit or 10) if user_id else []
    )

    return pd.DataFrame(entries)


def load_entries_from_json(file_path: str) -> pd.DataFrame:
    """Load journal entries from a JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        DataFrame containing journal entries

    """
    with open(file_path) as f:
        data = json.load(f)

    return pd.DataFrame(data)


def load_entries_from_csv(file_path: str) -> pd.DataFrame:
    """Load journal entries from a CSV file.

    Args:
        file_path: Path to the CSV file

    Returns:
        DataFrame containing journal entries

    """
    return pd.read_csv(file_path)


def save_entries_to_csv(df: pd.DataFrame, output_path: str) -> None:
    """Save journal entries DataFrame to CSV.

    Args:
        df: DataFrame containing journal entries
        output_path: Path to save the CSV file

    """
    df.to_csv(output_path, index=False)


def save_entries_to_json(df: pd.DataFrame, output_path: str) -> None:
    """Save journal entries DataFrame to JSON.

    Args:
        df: DataFrame containing journal entries
        output_path: Path to save the JSON file

    """
    df.to_json(output_path, orient="records")
