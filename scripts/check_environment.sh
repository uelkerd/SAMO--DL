#!/bin/bash

echo "🔍 SAMO-DL Environment Check"
echo "============================"

# Check Python version
echo "🐍 Python Environment:"
python3 --version 2>/dev/null || echo "   ❌ Python3 not found"
which python3 || echo "   ❌ Python3 not in PATH"

# Check pip
echo "📦 Package Manager:"
pip3 --version 2>/dev/null || echo "   ❌ pip3 not found"

# Check virtual environment
echo "🔧 Virtual Environment:"
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "   ✅ Virtual environment active: ${VIRTUAL_ENV}"
else
    echo "   ⚠️  No virtual environment active"
fi

# Check key packages
echo "📚 Key Packages:"
python3 -c "import torch; print(f'   ✅ PyTorch: {torch.__version__}')" 2>/dev/null || echo "   ❌ PyTorch not installed"
python3 -c "import transformers; print(f'   ✅ Transformers: {transformers.__version__}')" 2>/dev/null || echo "   ❌ Transformers not installed"
python3 -c "import numpy; print(f'   ✅ NumPy: {numpy.__version__}')" 2>/dev/null || echo "   ❌ NumPy not installed"

# Check project structure
echo "📁 Project Structure:"
[ -f "src/models/emotion_detection/bert_classifier.py" ] && echo "   ✅ BERT classifier exists" || echo "   ❌ BERT classifier missing"
[ -f "scripts/focal_loss_training.py" ] && echo "   ✅ Focal loss script exists" || echo "   ❌ Focal loss script missing"
[ -f "docs/gcp_deployment_guide.md" ] && echo "   ✅ GCP guide exists" || echo "   ❌ GCP guide missing"

# Check git status
echo "📝 Git Status:"
if git status --porcelain 2>/dev/null | grep -q .; then
    echo "   ⚠️  Uncommitted changes detected"
    git status --porcelain | head -5
else
    echo "   ✅ Working directory clean"
fi

echo ""
echo "🎯 RECOMMENDATION:"
echo "=================="

# Check if we have the core components
if [ -f "src/models/emotion_detection/bert_classifier.py" ] && [ -f "scripts/focal_loss_training.py" ]; then
    echo "✅ Core components available"
    echo "🚀 Ready for GCP deployment!"
    echo ""
    echo "📋 Next Steps:"
    echo "   1. Follow docs/gcp_deployment_guide.md"
    echo "   2. Set up GCP project and GPU instance"
    echo "   3. Run focal loss training on GCP"
    echo ""
    echo "💡 Why GCP? Faster, more reliable, and avoids local environment issues"
else
    echo "❌ Core components missing"
    echo "🔧 Need to fix project structure first"
fi

echo ""
echo "📊 Environment Summary:"
echo "======================="
echo "• Python: $(python3 --version 2>/dev/null || echo 'Not available')"
echo "• PyTorch: $(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo 'Not installed')"
echo "• Project Files: $(ls -1 src/models/emotion_detection/*.py 2>/dev/null | wc -l | tr -d ' ') core files"
echo "• Scripts: $(ls -1 scripts/*.py 2>/dev/null | wc -l | tr -d ' ') scripts"
