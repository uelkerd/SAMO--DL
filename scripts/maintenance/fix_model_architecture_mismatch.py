#!/usr/bin/env python3
"""
Fix Model Architecture Mismatch
===============================

This script fixes the model architecture mismatch by properly reconfiguring
the model for 12 emotions instead of the original 7.
"""

import json

def fix_model_architecture():
    """Fix the model architecture mismatch in the minimal notebook."""

    # Read the existing notebook
    with open('notebooks/MINIMAL_WORKING_TRAINING_COLAB.ipynb', 'r') as f:
        notebook = json.load(f)

    # Find and replace the model setup cell
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code' and 'model_name =' in ''.join(cell['source']):
            # Replace with fixed model setup
            cell['source'] = [
                "# Load model and tokenizer\n",
                "model_name = 'j-hartmann/emotion-english-distilroberta-base'\n",
                "print(f'🔧 Loading model: {model_name}')\n",
                "\n",
                "tokenizer = AutoTokenizer.from_pretrained(model_name)\n",
                "model = AutoModelForSequenceClassification.from_pretrained(model_name)\n",
                "\n",
                "print(f'Original model labels: {model.config.num_labels}')\n",
                "print(f'Original id2label: {model.config.id2label}')\n",
                "\n",
                "# IMPORTANT: The model was trained for 7 emotions, we need 12\n",
                "# We need to completely reconfigure the classifier layer\n",
                "print('\\n🔧 RECONFIGURING MODEL FOR 12 EMOTIONS')\n",
                "print('=' * 50)\n",
                "\n",
                "# Configure for our emotions\n",
                "model.config.num_labels = len(emotions)\n",
                "model.config.id2label = {i: emotion for i, emotion in enumerate(emotions)}\n",
                "model.config.label2id = {emotion: i for i, emotion in enumerate(emotions)}\n",
                "\n",
                "# CRITICAL: Recreate the classifier layer for 12 emotions\n",
                "from transformers import RobertaClassificationHead\n",
                "model.classifier = RobertaClassificationHead(\n",
                "    config=model.config\n",
                ")\n",
                "\n",
                "# Initialize the new classifier weights\n",
                "model.classifier.dense.weight.data.normal_(mean=0.0, std=0.02)\n",
                "model.classifier.dense.bias.data.zero_()\n",
                "model.classifier.out_proj.weight.data.normal_(mean=0.0, std=0.02)\n",
                "model.classifier.out_proj.bias.data.zero_()\n",
                "\n",
                "print(f'✅ Model reconfigured for {len(emotions)} emotions')\n",
                "print(f'✅ New id2label: {model.config.id2label}')\n",
                "print(f'✅ Classifier layer: {model.classifier.out_proj.out_features} outputs')\n",
                "\n",
                "# Move model to GPU\n",
                "if torch.cuda.is_available():\n",
                "    model = model.to('cuda')\n",
                "    print('✅ Model moved to GPU')\n",
                "else:\n",
                "    print('⚠️ CUDA not available, model will run on CPU')"
            ]
            break

    # Save the updated notebook
    with open('notebooks/MINIMAL_WORKING_TRAINING_COLAB.ipynb', 'w') as f:
        json.dump(notebook, f, indent=2)

    print('✅ Fixed model architecture mismatch!')
    print('📋 Changes made:')
    print('   ✅ Properly reconfigured classifier layer for 12 emotions')
    print('   ✅ Recreated RobertaClassificationHead with correct dimensions')
    print('   ✅ Initialized new classifier weights')
    print('   ✅ Added detailed logging of the reconfiguration process')

if __name__ == "__main__":
    fix_model_architecture()
