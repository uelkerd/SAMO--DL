#!/bin/bash

echo "🚀 Triggering CI Pipeline for SAMO Deep Learning"
echo "================================================"

# Check git status
echo "📊 Checking git status..."
git status

# Add all changes
echo "📝 Adding all changes..."
git add .

# Check if there are changes to commit
if git diff --cached --quiet; then
    echo "ℹ️  No changes to commit, checking if we need to push..."
else
    echo "📝 Committing changes..."
    git commit -m "Fix critical CI test failures: BERT mocking and predict_emotions bug"
fi

# Push to trigger CI
echo "🚀 Pushing to trigger CI pipeline..."
git push

echo "✅ Done! Check CircleCI dashboard for the new pipeline run."
