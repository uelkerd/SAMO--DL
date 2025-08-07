#!/usr/bin/env python3
"""
Fix Imports in Ultimate Notebook
================================

This script adds the missing imports to the ultimate notebook to ensure
all features work properly.
"""

import json

def fix_imports():
    """Add missing imports to the ultimate notebook."""
    
    # Read the existing notebook
    with open('notebooks/ULTIMATE_BULLETPROOF_TRAINING_COLAB.ipynb', 'r') as f:
        notebook = json.load(f)
    
    # Find the imports cell and update it
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code' and 'import torch' in ''.join(cell['source']):
            # Update the imports cell
            cell['source'] = [
                "import torch\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer\n",
                "from datasets import Dataset\n",
                "from sklearn.model_selection import train_test_split\n",
                "from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score\n",
                "from sklearn.utils.class_weight import compute_class_weight\n",
                "import json\n",
                "import warnings\n",
                "warnings.filterwarnings('ignore')\n",
                "\n",
                "print('✅ All packages imported successfully')\n",
                "print(f'PyTorch version: {torch.__version__}')\n",
                "print(f'CUDA available: {torch.cuda.is_available()}')"
            ]
            break
    
    # Save the updated notebook
    with open('notebooks/ULTIMATE_BULLETPROOF_TRAINING_COLAB.ipynb', 'w') as f:
        json.dump(notebook, f, indent=2)
    
    print('✅ Fixed imports in ultimate notebook!')
    print('📋 Added missing imports:')
    print('   ✅ f1_score, accuracy_score, precision_score, recall_score')
    print('   ✅ compute_class_weight')
    print('   ✅ CUDA availability check')

if __name__ == "__main__":
    fix_imports() 