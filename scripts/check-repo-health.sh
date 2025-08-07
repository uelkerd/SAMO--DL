#!/bin/bash
# Repository health check script

echo "🔍 Repository Health Check"
echo "=========================="

echo ""
echo "📊 Repository Size:"
git count-objects -vH

echo ""
echo "📁 Largest Files:"
git ls-files | xargs wc -l 2>/dev/null | sort -nr | head -10

echo ""
echo "🔍 Potential Model Artifacts:"
find . -name "*.pt" -o -name "*.pth" -o -name "*.bin" -o -name "*.safetensors" -o -name "*.onnx" -o -name "*.arrow" -o -name "merges.txt" 2>/dev/null | head -10

echo ""
echo "📈 Recent Commits:"
git log --oneline -5

echo ""
echo "🌿 Current Branch:"
git branch --show-current
