#!/usr/bin/env python3
"""
Fix Model Reconfiguration
=========================

This script fixes the model reconfiguration by creating a new model
with the correct architecture from scratch instead of trying to modify
the existing one.
"""

import json


def fix_model_reconfiguration():
    """Fix the model reconfiguration in the minimal notebook."""

    # Read the existing notebook
    with open("notebooks/MINIMAL_WORKING_TRAINING_COLAB.ipynb") as f:
        notebook = json.load(f)

    # Find and replace the model setup cell
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code" and "model_name =" in "".join(cell["source"]):
            # Replace with fixed model setup
            cell["source"] = [
                "# Load model and tokenizer\n",
                "model_name = 'j-hartmann/emotion-english-distilroberta-base'\n",
                "print(f'🔧 Loading model: {model_name}')\n",
                "\n",
                "tokenizer = AutoTokenizer.from_pretrained(model_name)\n",
                "\n",
                "print(f'Original model labels: {AutoModelForSequenceClassification.from_pretrained(model_name).config.num_labels}')\n",
                "print(f'Original id2label: {AutoModelForSequenceClassification.from_pretrained(model_name).config.id2label}')\n",
                "\n",
                "# CRITICAL: Create a NEW model with correct configuration from scratch\n",
                "print('\\n🔧 CREATING NEW MODEL WITH CORRECT ARCHITECTURE')\n",
                "print('=' * 60)\n",
                "\n",
                "# Load the base model without the classification head\n",
                "from transformers import RobertaModel\n",
                "base_model = RobertaModel.from_pretrained(model_name)\n",
                "\n",
                "# Create a new model with the correct number of labels\n",
                "model = AutoModelForSequenceClassification.from_pretrained(\n",
                "    model_name,\n",
                "    num_labels=len(emotions),  # Set to 12 emotions\n",
                "    ignore_mismatched_sizes=True  # Important: ignore size mismatches\n",
                ")\n",
                "\n",
                "# Configure the model properly\n",
                "model.config.num_labels = len(emotions)\n",
                "model.config.id2label = {i: emotion for i, emotion in enumerate(emotions)}\n",
                "model.config.label2id = {emotion: i for i, emotion in enumerate(emotions)}\n",
                "model.config.problem_type = 'single_label_classification'\n",
                "\n",
                "# Verify the configuration\n",
                "print(f'✅ Model created with {model.config.num_labels} labels')\n",
                "print(f'✅ New id2label: {model.config.id2label}')\n",
                "print(f'✅ Classifier output size: {model.classifier.out_proj.out_features}')\n",
                "print(f'✅ Problem type: {model.config.problem_type}')\n",
                "\n",
                "# Test the model with a sample input\n",
                "test_input = tokenizer('I feel happy today', return_tensors='pt', truncation=True, padding=True)\n",
                "with torch.no_grad():\n",
                "    test_output = model(**test_input)\n",
                "    print(f'✅ Test output shape: {test_output.logits.shape}')\n",
                "    print(f'✅ Expected shape: [1, {len(emotions)}]')\n",
                "    assert test_output.logits.shape[1] == len(emotions), f'Output shape mismatch: {test_output.logits.shape[1]} != {len(emotions)}'\n",
                "    print('✅ Model architecture verified!')\n",
                "\n",
                "# Move model to GPU\n",
                "if torch.cuda.is_available():\n",
                "    model = model.to('cuda')\n",
                "    print('✅ Model moved to GPU')\n",
                "else:\n",
                "    print('⚠️ CUDA not available, model will run on CPU')",
            ]
            break

    # Save the updated notebook
    with open("notebooks/MINIMAL_WORKING_TRAINING_COLAB.ipynb", "w") as f:
        json.dump(notebook, f, indent=2)

    print("✅ Fixed model reconfiguration!")
    print("📋 Changes made:")
    print("   ✅ Created new model with correct architecture from scratch")
    print("   ✅ Used ignore_mismatched_sizes=True to handle size differences")
    print("   ✅ Set problem_type to single_label_classification")
    print("   ✅ Added model architecture verification test")
    print("   ✅ Added detailed logging of the configuration process")


if __name__ == "__main__":
    fix_model_reconfiguration()
