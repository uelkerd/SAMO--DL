# Prisma Setup for SAMO-DL

This directory contains the Prisma ORM configuration for the SAMO-DL project.

## Setup Instructions

1. Install dependencies:

   ```bash
   npm install
   ```

2. Create a `.env` file in the project root with the following content:

   ```
   # PostgreSQL database connection
   DATABASE_URL="postgresql://samouser:samopassword@localhost:5432/samodb?schema=public"

   # Application environment
   NODE_ENV="development"
   ```

3. Initialize the database:

   ```bash
   # Run the PostgreSQL setup script
   npm run db:setup

   # Generate the Prisma client
   npm run prisma:generate
   ```

4. (Optional) Explore the database with Prisma Studio:

   ```bash
   npm run prisma:studio
   ```

## Schema Information

The Prisma schema (`schema.prisma`) defines the following models:

- `User`: System users with authentication and consent information
- `JournalEntry`: User journal entries with text content
- `Embedding`: Vector embeddings of journal entries (using pgvector)
- `Prediction`: AI-generated predictions about user mood, topics, etc.
- `VoiceTranscription`: Transcribed voice recordings
- `Tag`: Categories for journal entries

## pgvector Extension

This project uses the pgvector extension for PostgreSQL to store and query vector embeddings.
Make sure the extension is installed on your PostgreSQL server before running migrations.
