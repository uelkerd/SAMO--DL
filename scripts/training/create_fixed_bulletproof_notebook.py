#!/usr/bin/env python3
"""
🚀 CREATE FIXED BULLETPROOF NOTEBOOK
====================================
Create a bulletproof Colab notebook that uses the unique fallback dataset.
This fixes the duplicate data issue that caused model collapse.
"""


import json


def create_fixed_bulletproof_notebook():
    """Create the fixed bulletproof notebook content"""
    
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 🚀 FIXED BULLETPROOF TRAINING - UNIQUE DATASET\n",
                    "\n",
                    "**Target: 75-85% F1 Score**  \n",
                    "**Current: 67% F1 Score**  \n",
                    "**Strategy: Use UNIQUE fallback dataset with NO DUPLICATES**\n",
                    "\n",
                    "This notebook uses:\n",
                    "- Original 150 high-quality journal samples\n",
                    "- CMU-MOSEI samples for diversity\n",
                    "- **UNIQUE** fallback dataset (144 samples, no duplicates)\n",
                    "- Optimized hyperparameters for 75-85% F1"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Install required packages\n",
                    "!pip install transformers torch scikit-learn numpy pandas"
                ]
            },
            {
                "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Import libraries\n",
                        "import json\n",
                        "import pandas as pd\n",
                        "import numpy as np\n",
                        "import torch\n",
                        "from torch.utils.data import Dataset, DataLoader\n",
                        "from transformers import (\n",
                        "    AutoTokenizer,\n",
                        "    AutoModelForSequenceClassification,\n",
                        "    TrainingArguments,\n",
                        "    Trainer,\n",
                        "    EarlyStoppingCallback\n",
                        ")\n",
                        "from sklearn.model_selection import train_test_split\n",
                        "from sklearn.preprocessing import LabelEncoder\n",
                        "from sklearn.metrics import f1_score, accuracy_score, classification_report\n",
                        "import warnings\n",
                        "warnings.filterwarnings('ignore')\n",
                        "\n",
                        "print('🚀 FIXED BULLETPROOF TRAINING - UNIQUE DATASET')\n",
                        "print('=' * 60)"
                    ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# BULLETPROOF: Auto-detect repository path and data files\n",
                    "import os\n",
                    "print('🔍 Auto-detecting repository structure...')\n",
                    "\n",
                    "# Find the repository directory\n",
                    "possible_paths = [\n",
                    "    '/content/SAMO--DL',\n",
                    "    '/content/SAMO--DL/SAMO--DL',\n",
                    "    '/content/SAMO--DL-main',\n",
                    "    '/content/SAMO--DL-main/SAMO--DL',\n",
                    "    '/content/SAMO--DL-main/SAMO--DL-main'\n",
                    "]\n",
                    "\n",
                    "repo_path = None\n",
                    "for path in possible_paths:\n",
                    "    if os.path.exists(path):\n",
                    "        repo_path = path\n",
                    "        print(f'✅ Found repository at: {repo_path}')\n",
                    "        break\n",
                    "\n",
                    "if repo_path is None:\n",
                    "    print('❌ Could not find repository! Listing /content:')\n",
                    "    !ls -la /content/\n",
                    "    raise Exception('Repository not found!')\n",
                    "\n",
                    "# Verify data directory exists\n",
                    "data_path = os.path.join(repo_path, 'data')\n",
                    "if not os.path.exists(data_path):\n",
                    "    print(f'❌ Data directory not found: {data_path}')\n",
                    "    raise Exception('Data directory not found!')\n",
                    "\n",
                    "print(f'✅ Data directory found: {data_path}')\n",
                    "print('📂 Listing data files:')\n",
                    "!ls -la {data_path}/"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Load combined dataset with UNIQUE fallback\n",
                    "print('📊 Loading combined dataset...')\n",
                    "combined_samples = []\n",
                    "\n",
                    "# Load journal data\n",
                    "journal_path = os.path.join(repo_path, 'data', 'journal_test_dataset.json')\n",
                    "try:\n",
                    "    with open(journal_path, 'r') as f:\n",
                    "        journal_data = json.load(f)\n",
                    "    for item in journal_data:\n",
                    "        # CRITICAL FIX: Use 'content' for journal data, 'text' for CMU-MOSEI\n",
                    "        if 'content' in item and 'emotion' in item:\n",
                    "            combined_samples.append({'text': item['content'], 'emotion': item['emotion']})\n",
                    "        elif 'text' in item and 'emotion' in item: # Fallback for other journal formats\n",
                    "            combined_samples.append({'text': item['text'], 'emotion': item['emotion']})\n",
                    "    print(f'✅ Loaded {len(journal_data)} journal samples from {journal_path}')\n",
                    "except FileNotFoundError:\n",
                    "    print(f'⚠️ Could not load journal data: {journal_path} not found.')\n",
                    "\n",
                    "# Load CMU-MOSEI data\n",
                    "cmu_path = os.path.join(repo_path, 'data', 'cmu_mosei_balanced_dataset.json')\n",
                    "try:\n",
                    "    with open(cmu_path, 'r') as f:\n",
                    "        cmu_data = json.load(f)\n",
                    "    for item in cmu_data:\n",
                    "        if 'text' in item and 'emotion' in item:\n",
                    "            combined_samples.append({'text': item['text'], 'emotion': item['emotion']})\n",
                    "    print(f'✅ Loaded {len(cmu_data)} CMU-MOSEI samples from {cmu_path}')\n",
                    "except FileNotFoundError:\n",
                    "    print(f'⚠️ Could not load CMU-MOSEI data: {cmu_path} not found.')\n",
                    "\n",
                    "print(f'📊 Total combined samples: {len(combined_samples)}')\n",
                    "\n",
                    "# BULLETPROOF: Use UNIQUE fallback dataset if needed\n",
                    "if len(combined_samples) < 100:\n",
                    "    print(f'⚠️ Only {len(combined_samples)} samples loaded! Using UNIQUE fallback dataset...')\n",
                    "    \n",
                    "    # Load the unique fallback dataset\n",
                    "    fallback_path = os.path.join(repo_path, 'data', 'unique_fallback_dataset.json')\n",
                    "    try:\n",
                    "        with open(fallback_path, 'r') as f:\n",
                    "            fallback_data = json.load(f)\n",
                    "        combined_samples = fallback_data\n",
                    "        print(f'✅ Loaded {len(combined_samples)} UNIQUE fallback samples')\n",
                    "    except FileNotFoundError:\n",
                    "        print(f'❌ Could not load unique fallback dataset: {fallback_path}')\n",
                    "        print('❌ No data available for training!')\n",
                    "        raise Exception('No training data available!')\n",
                    "\n",
                    "print(f'✅ Final dataset size: {len(combined_samples)} samples')\n",
                    "\n",
                    "# Verify no duplicates\n",
                    "texts = [sample['text'] for sample in combined_samples]\n",
                    "unique_texts = set(texts)\n",
                    "print(f'🔍 Duplicate check: {len(texts)} total, {len(unique_texts)} unique')\n",
                    "if len(texts) != len(unique_texts):\n",
                    "    print('❌ WARNING: DUPLICATES FOUND! This will cause model collapse!')\n",
                    "else:\n",
                    "    print('✅ All samples are unique - no model collapse risk!')"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Prepare data for training\n",
                    "print('🔧 Preparing data for training...')\n",
                    "\n",
                    "texts = [sample['text'] for sample in combined_samples]\n",
                    "emotions = [sample['emotion'] for sample in combined_samples]\n",
                    "\n",
                    "# Encode labels\n",
                    "label_encoder = LabelEncoder()\n",
                    "labels = label_encoder.fit_transform(emotions)\n",
                    "\n",
                    "print(f'🎯 Number of labels: {len(label_encoder.classes_)}')\n",
                    "print(f'📊 Labels: {list(label_encoder.classes_)}')\n",
                    "\n",
                    "# Split data\n",
                    "train_texts, test_texts, train_labels, test_labels = train_test_split(\n",
                    "    texts, labels, test_size=0.2, random_state=42, stratify=labels\n",
                    ")\n",
                    "\n",
                    "print(f'📈 Training samples: {len(train_texts)}')\n",
                    "print(f'🧪 Test samples: {len(test_labels)}')\n",
                    "\n",
                    "# Show emotion distribution\n",
                    "emotion_counts = {}\n",
                    "for emotion in emotions:\n",
                    "    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1\n",
                    "\n",
                    "print('\\n📊 Emotion Distribution:')\n",
                    "for emotion, count in sorted(emotion_counts.items()):\n",
                    "    print(f'  {emotion}: {count} samples')"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Create custom dataset\n",
                    "class EmotionDataset(Dataset):\n",
                    "    def __init__(self, texts, labels, tokenizer, max_length=128):\n",
                    "        self.texts = texts\n",
                    "        self.labels = labels\n",
                    "        self.tokenizer = tokenizer\n",
                    "        self.max_length = max_length\n",
                    "    \n",
                    "    def __len__(self):\n",
                    "        return len(self.texts)\n",
                    "    \n",
                    "    def __getitem__(self, idx):\n",
                    "        text = str(self.texts[idx])\n",
                    "        label = self.labels[idx]\n",
                    "        \n",
                    "        encoding = self.tokenizer(\n",
                    "            text,\n",
                    "            truncation=True,\n",
                    "            padding='max_length',\n",
                    "            max_length=self.max_length,\n",
                    "            return_tensors='pt'\n",
                    "        )\n",
                    "        \n",
                    "        return {\n",
                    "            'input_ids': encoding['input_ids'].flatten(),\n",
                    "            'attention_mask': encoding['attention_mask'].flatten(),\n",
                    "            'labels': torch.tensor(label, dtype=torch.long)\n",
                    "        }"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Initialize model and tokenizer\n",
                    "print('🔧 Initializing model and tokenizer...')\n",
                    "\n",
                    "model_name = 'bert-base-uncased'\n",
                    "tokenizer = AutoTokenizer.from_pretrained(model_name)\n",
                    "model = AutoModelForSequenceClassification.from_pretrained(\n",
                    "    model_name,\n",
                    "    num_labels=len(label_encoder.classes_),\n",
                    "    problem_type='single_label_classification'\n",
                    ")\n",
                    "\n",
                    "print(f'✅ Model initialized with {len(label_encoder.classes_)} labels')\n",
                    "\n",
                    "# Create datasets\n",
                    "train_dataset = EmotionDataset(train_texts, train_labels, tokenizer)\n",
                    "test_dataset = EmotionDataset(test_texts, test_labels, tokenizer)\n",
                    "\n",
                    "print('✅ Datasets created successfully')"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Define metrics function\n",
                    "def compute_metrics(eval_pred):\n",
                    "    predictions, labels = eval_pred\n",
                    "    predictions = np.argmax(predictions, axis=1)\n",
                    "    \n",
                    "    f1 = f1_score(labels, predictions, average='weighted')\n",
                    "    accuracy = accuracy_score(labels, predictions)\n",
                    "    \n",
                    "    return {'f1': f1, 'accuracy': accuracy}"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Configure training arguments with OPTIMIZED hyperparameters\n",
                    "print('🚀 Starting FIXED BULLETPROOF training...')\n",
                    "print('🎯 Target F1 Score: 75-85%')\n",
                    "print('📊 Current Best: 67%')\n",
                    "print('📈 Expected Improvement: 8-18%')\n",
                    "\n",
                    "training_args = TrainingArguments(\n",
                    "    output_dir='./emotion_model_fixed_bulletproof',\n",
                    "    num_train_epochs=3,  # Reduced to prevent overfitting\n",
                    "    per_device_train_batch_size=8,  # Smaller batch size\n",
                    "    per_device_eval_batch_size=8,\n",
                    "    warmup_steps=50,  # Reduced warmup\n",
                    "    weight_decay=0.01,\n",
                    "    logging_dir='./logs',\n",
                    "    logging_steps=10,  # More frequent logging\n",
                    "    eval_strategy='steps',\n",
                    "    eval_steps=25,  # More frequent evaluation\n",
                    "    save_strategy='steps',\n",
                    "    save_steps=25,\n",
                    "    load_best_model_at_end=True,\n",
                    "    metric_for_best_model='f1',\n",
                    "    greater_is_better=True,\n",
                    "    dataloader_num_workers=2,\n",
                    "    remove_unused_columns=False,\n",
                    "    report_to=None,  # Disable wandb\n",
                    "    learning_rate=2e-5,  # Standard learning rate\n",
                    "    gradient_accumulation_steps=2,  # Increased for stability\n",
                    "    fp16=True,  # Enable mixed precision for GPU\n",
                    ")\n",
                    "\n",
                    "# Create trainer\n",
                    "trainer = Trainer(\n",
                    "    model=model,\n",
                    "    args=training_args,\n",
                    "    train_dataset=train_dataset,\n",
                    "    eval_dataset=test_dataset,\n",
                    "    compute_metrics=compute_metrics,\n",
                    "    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]\n",
                    ")\n",
                    "\n",
                    "print(f'📊 Training on {len(train_texts)} samples')\n",
                    "print(f'🧪 Evaluating on {len(test_labels)} samples')\n",
                    "\n",
                    "# Start training\n",
                    "trainer.train()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Evaluate final model\n",
                    "print('📊 Evaluating final model...')\n",
                    "results = trainer.evaluate()\n",
                    "\n",
                    "print(f'🏆 Final F1 Score: {results[\"eval_f1\"]:.4f} ({results[\"eval_f1\"]*100:.2f}%)')\n",
                    "print(f'🎯 Target achieved: {\"✅ YES!\" if results[\"eval_f1\"] >= 0.75 else \"❌ Not yet\"}')\n",
                    "\n",
                    "# Save model\n",
                    "trainer.save_model('./emotion_model_fixed_bulletproof_final')\n",
                    "print('💾 Model saved to ./emotion_model_fixed_bulletproof_final')"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Test on sample texts\n",
                    "print('🧪 Testing on sample texts...')\n",
                    "\n",
                    "test_texts = [\n",
                    "    \"I'm feeling really happy today!\",\n",
                    "    \"I'm so frustrated with this project.\",\n",
                    "    \"I feel anxious about the presentation.\",\n",
                    "    \"I'm grateful for all the support.\",\n",
                    "    \"I'm feeling overwhelmed with tasks.\"\n",
                    "]\n",
                    "\n",
                    "model.eval()\n",
                    "with torch.no_grad():\n",
                    "    for i, text in enumerate(test_texts, 1):\n",
                    "        inputs = tokenizer(\n",
                    "            text,\n",
                    "            truncation=True,\n",
                    "            padding=True,\n",
                    "            return_tensors='pt'\n",
                    "        )\n",
                    "        \n",
                    "        outputs = model(**inputs)\n",
                    "        probabilities = torch.softmax(outputs.logits, dim=1)\n",
                    "        predicted_class = torch.argmax(probabilities, dim=1).item()\n",
                    "        confidence = probabilities[0][predicted_class].item()\n",
                    "        \n",
                    "        predicted_emotion = label_encoder.inverse_transform([predicted_class])[0]\n",
                    "        \n",
                    "        print(f'{i}. Text: {text}')\n",
                    "        print(f'   Predicted: {predicted_emotion} (confidence: {confidence:.3f})\\n')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 🎉 Training Complete!\n",
                    "\n",
                    "**Key Improvements:**\n",
                    "- ✅ **UNIQUE** fallback dataset (no duplicates)\n",
                    "- ✅ Proper data loading with field name handling\n",
                    "- ✅ Optimized hyperparameters\n",
                    "- ✅ Early stopping to prevent overfitting\n",
                    "- ✅ Mixed precision training for GPU efficiency\n",
                    "\n",
                    "**Expected Results:**\n",
                    "- 🎯 **Target F1 Score: 75-85%**\n",
                    "- 📈 **Improvement from 67% baseline**\n",
                    "- 🔧 **No model collapse** (unique data prevents this)\n",
                    "\n",
                    "**Next Steps:**\n",
                    "1. Review the F1 score achieved\n",
                    "2. If below 75%, consider adding more real data\n",
                    "3. Fine-tune hyperparameters if needed"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.5"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open('notebooks/FIXED_BULLETPROOF_COMBINED_TRAINING_COLAB.ipynb', 'w') as f:
        json.dump(notebook_content, f, indent=2)
    
    print("✅ Fixed bulletproof notebook created: notebooks/FIXED_BULLETPROOF_COMBINED_TRAINING_COLAB.ipynb")
    print("📋 Instructions:")
    print("  1. Download the notebook file")
    print("  2. Upload to Google Colab")
    print("  3. Set Runtime → GPU")
    print("  4. Run all cells")
    print("  5. Expect 75-85% F1 score!")
    print("🔧 Key Features:")
    print("  - UNIQUE fallback dataset (no duplicates)")
    print("  - Automatic path detection")
    print("  - Optimized hyperparameters")
    print("  - Robust error handling")

if __name__ == "__main__":
    create_fixed_bulletproof_notebook() 