# SAMO-DL Project

## Project Overview
SAMO-DL is a deep learning track project focused on journal entries, embeddings, and predictions using PostgreSQL with pgvector extension for vector similarity search.

## Data Pipeline Development

We've successfully developed a comprehensive journal entries analysis pipeline with the following components:

1. **Data Loading**: Support for multiple data sources (JSON, CSV, database)
2. **Validation**: Robust data quality checks and verification
3. **Text Preprocessing**: Configurable text cleaning with stopword removal, lemmatization, etc.
4. **Feature Engineering**: Extraction of sentiment, topics, and readability metrics
5. **Embedding Generation**: CPU-friendly TF-IDF and Word2Vec vector representations
6. **Pipeline Orchestration**: Unified workflow management
7. **Synthetic Data Generation**: Realistic test data for development

### Key Features

- **CPU-Friendly Implementation**: All operations optimized for environments without GPU
- **GoEmotions Classification**: Baseline models for 27-emotion taxonomy
- **PostgreSQL Integration**: Design for pgvector-based similarity search
- **Modular Architecture**: Components can be used independently or as a unified pipeline

### Current Progress

| Component | Status | Completion |
|-----------|--------|------------|
| Data Loading | Complete | 100% |
| Validation | Complete | 100% |
| Preprocessing | Complete | 100% |
| Feature Engineering | Complete | 100% |
| Embedding Generation | Complete | 100% |
| Pipeline Integration | Complete | 100% |
| Database Integration | Designed | 70% |
| Classification Models | Baseline Complete | 75% |
| Testing Framework | Framework Ready | 60% |
| Documentation | Partial | 70% |

### Next Steps

1. **Complete Unit Testing**: Implement comprehensive tests for all components
2. **Implement Database Integration**: Set up PostgreSQL with pgvector extension
3. **Enhance Documentation**: Create API documentation for each module
4. **Improve Classification Models**: Develop ensemble methods for emotion detection
5. **Prepare for GPU Integration**: Design integration path for transformer-based models

A complete demonstration of the pipeline is available in `notebooks/data_pipeline_demo.ipynb`.

## Database Architecture

The database architecture is designed to support the following key features:

1. **Journal Entry Management**: Store and retrieve user journal entries
2. **Vector Embeddings**: Store embeddings of journal entries for semantic search
3. **Voice Transcription**: Process and store voice recordings as text
4. **AI Predictions**: Generate and store predictions based on user data
5. **Privacy Compliance**: GDPR-compliant data handling with user consent

### Schema Overview

The database schema consists of the following tables:

- **users**: Store user information and consent settings
- **journal_entries**: Store journal content with privacy settings
- **embeddings**: Store vector embeddings of journal entries (using pgvector)
- **predictions**: Store AI-generated predictions about user mood, topics, etc.
- **voice_transcriptions**: Store transcriptions from voice recordings
- **tags**: Store categories for journal entries

### Data Access

The project provides multiple ways to access the database:

1. **SQLAlchemy ORM**: Python-based ORM for data access
2. **Prisma ORM**: JavaScript/TypeScript ORM for data access
3. **Raw SQL**: Direct SQL scripts for database setup and management

## Setup Instructions

### Prerequisites

- Python 3.8+
- Node.js 16+
- PostgreSQL 13+ with pgvector extension

### Database Setup

1. Install PostgreSQL and pgvector extension
2. Run the database setup script:
   ```bash
   ./scripts/database/init_db.sh
   ```

### Environment Configuration

Create a `.env` file in the project root with the following variables:

```
# PostgreSQL database connection
DATABASE_URL="postgresql://samouser:samopassword@localhost:5432/samodb?schema=public"

# Application environment
NODE_ENV="development"
```

### Prisma Setup

1. Install Node.js dependencies:
   ```bash
   npm install
   ```

2. Generate Prisma client:
   ```bash
   npm run prisma:generate
   ```

## Development

### Using SQLAlchemy

```python
from src.data.models import User, JournalEntry
from src.data.database import db_session

# Create a user
user = User(email="user@example.com", password_hash="...")
db_session.add(user)
db_session.commit()

# Create a journal entry
entry = JournalEntry(user_id=user.id, title="My Journal", content="Today I...")
db_session.add(entry)
db_session.commit()
```

### Using Prisma

```python
from src.data.prisma_client import PrismaClient

prisma = PrismaClient()

# Create a user
user = prisma.create_user(
    email="user@example.com", 
    password_hash="...", 
    consent_version="1.0"
)

# Create a journal entry
entry = prisma.create_journal_entry(
    user_id=user["id"],
    title="My Journal",
    content="Today I...",
    is_private=True
)
``` 