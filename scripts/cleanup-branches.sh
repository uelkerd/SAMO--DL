#!/bin/bash
# Branch cleanup script

echo "🧹 Branch Cleanup"
echo "================="

echo ""
echo "📋 Local branches:"
git branch

echo ""
echo "🗑️  To clean up merged branches:"
echo "git branch --merged | grep -v '\\*\\|main\\|master' | xargs -n 1 git branch -d"

echo ""
echo "🔍 To check for large files in recent commits:"
echo "git log --name-only --pretty=format: | sort | uniq | xargs wc -l 2>/dev/null | sort -nr | head -10"
echo ""
echo "🔍 To check for large files by size:"
echo "git ls-files -z | xargs -0 du -h 2>/dev/null | sort -hr | head -10"
