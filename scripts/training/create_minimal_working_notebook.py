#!/usr/bin/env python3
"""
Create Minimal Working Notebook
==============================

This script creates a minimal working notebook with the most basic
training arguments that should work in any transformers version.
"""


import json


def create_minimal_notebook():
    """Create a minimal working notebook."""
    
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 🚀 MINIMAL WORKING EMOTION DETECTION TRAINING\n",
                    "## Ultra-Simple Version That Should Work\n",
                    "\n",
                    "**FEATURES:**\n",
                    "✅ Basic training (no complex arguments)\n",
                    "✅ Configuration preservation\n",
                    "✅ Simple data processing\n",
                    "✅ Model saving with verification\n",
                    "\n",
                    "**Target**: Get training working first, then optimize"
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
                    "import torch\n",
                    "import numpy as np\n",
                    "import pandas as pd\n",
                    "from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer\n",
                    "from sklearn.model_selection import train_test_split\n",
                    "from sklearn.metrics import classification_report, f1_score, accuracy_score\n",
                    "import json\n",
                    "import warnings\n",
                    "warnings.filterwarnings('ignore')\n",
                    "\n",
                    "print('✅ All packages imported successfully')\n",
                    "print(f'PyTorch version: {torch.__version__}')\n",
                    "print(f'CUDA available: {torch.cuda.is_available()}')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 🎯 SETUP"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Define emotion classes\n",
                    "emotions = ['anxious', 'calm', 'content', 'excited', 'frustrated', 'grateful', 'happy', 'hopeful', 'overwhelmed', 'proud', 'sad', 'tired']\n",
                    "print(f'🎯 Emotion classes: {emotions}')\n",
                    "\n",
                    "# Simple dataset\n",
                    "data = [\n",
                    "    {'text': 'I feel anxious about the presentation.', 'label': 0},\n",
                    "    {'text': 'I am feeling calm today.', 'label': 1},\n",
                    "    {'text': 'I feel content with my life.', 'label': 2},\n",
                    "    {'text': 'I am excited about the opportunity.', 'label': 3},\n",
                    "    {'text': 'I feel frustrated with the issues.', 'label': 4},\n",
                    "    {'text': 'I am grateful for the support.', 'label': 5},\n",
                    "    {'text': 'I feel happy about the success.', 'label': 6},\n",
                    "    {'text': 'I am hopeful for the future.', 'label': 7},\n",
                    "    {'text': 'I feel overwhelmed with tasks.', 'label': 8},\n",
                    "    {'text': 'I am proud of my achievements.', 'label': 9},\n",
                    "    {'text': 'I feel sad about the loss.', 'label': 10},\n",
                    "    {'text': 'I am tired from working.', 'label': 11},\n",
                    "    # Add more samples for each emotion\n",
                    "    {'text': 'I am worried about the results.', 'label': 0},\n",
                    "    {'text': 'I feel peaceful and relaxed.', 'label': 1},\n",
                    "    {'text': 'I am satisfied with the outcome.', 'label': 2},\n",
                    "    {'text': 'I feel thrilled about the news.', 'label': 3},\n",
                    "    {'text': 'I am annoyed with the problems.', 'label': 4},\n",
                    "    {'text': 'I feel thankful for the help.', 'label': 5},\n",
                    "    {'text': 'I am joyful about the completion.', 'label': 6},\n",
                    "    {'text': 'I feel optimistic about tomorrow.', 'label': 7},\n",
                    "    {'text': 'I am stressed with responsibilities.', 'label': 8},\n",
                    "    {'text': 'I feel accomplished and confident.', 'label': 9},\n",
                    "    {'text': 'I am depressed about the situation.', 'label': 10},\n",
                    "    {'text': 'I feel exhausted from the work.', 'label': 11}\n",
                    "]\n",
                    "\n",
                    "print(f'📊 Dataset size: {len(data)} samples')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 🔧 MODEL SETUP"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Load model and tokenizer\n",
                    "model_name = 'j-hartmann/emotion-english-distilroberta-base'\n",
                    "print(f'🔧 Loading model: {model_name}')\n",
                    "\n",
                    "tokenizer = AutoTokenizer.from_pretrained(model_name)\n",
                    "model = AutoModelForSequenceClassification.from_pretrained(model_name)\n",
                    "\n",
                    "# Configure for our emotions\n",
                    "model.config.num_labels = len(emotions)\n",
                    "model.config.id2label = {i: emotion for i, emotion in enumerate(emotions)}\n",
                    "model.config.label2id = {emotion: i for i, emotion in enumerate(emotions)}\n",
                    "\n",
                    "print(f'✅ Model configured for {len(emotions)} emotions')\n",
                    "print(f'✅ id2label: {model.config.id2label}')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 📝 DATA PREPROCESSING"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Prepare data\n",
                    "texts = [item['text'] for item in data]\n",
                    "labels = [item['label'] for item in data]\n",
                    "\n",
                    "# Split data\n",
                    "train_texts, val_texts, train_labels, val_labels = train_test_split(\n",
                    "    texts, labels, test_size=0.2, random_state=42, stratify=labels\n",
                    ")\n",
                    "\n",
                    "print(f'📊 Training samples: {len(train_texts)}')\n",
                    "print(f'📊 Validation samples: {len(val_texts)}')\n",
                    "\n",
                    "# Tokenize\n",
                    "train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors='pt')\n",
                    "val_encodings = tokenizer(val_texts, truncation=True, padding=True, return_tensors='pt')\n",
                    "\n",
                    "# Create dataset class\n",
                    "class SimpleDataset(torch.utils.data.Dataset):\n",
                    "    def __init__(self, encodings, labels):\n",
                    "        self.encodings = encodings\n",
                    "        self.labels = labels\n",
                    "    \n",
                    "    def __getitem__(self, idx):\n",
                    "        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}\n",
                    "        item['labels'] = torch.tensor(self.labels[idx])\n",
                    "        return item\n",
                    "    \n",
                    "    def __len__(self):\n",
                    "        return len(self.labels)\n",
                    "\n",
                    "train_dataset = SimpleDataset(train_encodings, train_labels)\n",
                    "val_dataset = SimpleDataset(val_encodings, val_labels)\n",
                    "\n",
                    "print('✅ Data preprocessing completed')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## ⚙️ MINIMAL TRAINING ARGUMENTS"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Minimal training arguments - only essential parameters\n",
                    "training_args = TrainingArguments(\n",
                    "    output_dir='./minimal_emotion_model',\n",
                    "    num_train_epochs=3,\n",
                    "    per_device_train_batch_size=4,\n",
                    "    per_device_eval_batch_size=4,\n",
                    "    logging_steps=10,\n",
                    "    save_steps=50,\n",
                    "    eval_steps=50\n",
                    ")\n",
                    "\n",
                    "print('✅ Minimal training arguments configured')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 📊 COMPUTE METRICS"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Simple compute metrics\n",
                    "def compute_metrics(eval_pred):\n",
                    "    predictions, labels = eval_pred\n",
                    "    predictions = np.argmax(predictions, axis=1)\n",
                    "    \n",
                    "    return {\n",
                    "        'f1': f1_score(labels, predictions, average='weighted'),\n",
                    "        'accuracy': accuracy_score(labels, predictions)\n",
                    "    }\n",
                    "\n",
                    "print('✅ Compute metrics function ready')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 🚀 TRAINING"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Initialize trainer\n",
                    "trainer = Trainer(\n",
                    "    model=model,\n",
                    "    args=training_args,\n",
                    "    train_dataset=train_dataset,\n",
                    "    eval_dataset=val_dataset,\n",
                    "    tokenizer=tokenizer,\n",
                    "    compute_metrics=compute_metrics\n",
                    ")\n",
                    "\n",
                    "print('✅ Trainer initialized')\n",
                    "\n",
                    "# Start training\n",
                    "print('🚀 STARTING MINIMAL TRAINING')\n",
                    "print('=' * 40)\n",
                    "print(f'📊 Training samples: {len(train_dataset)}')\n",
                    "print(f'🧪 Validation samples: {len(val_dataset)}')\n",
                    "\n",
                    "# Train the model\n",
                    "trainer.train()\n",
                    "\n",
                    "print('✅ Training completed successfully!')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 📈 EVALUATION"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Evaluate the model\n",
                    "print('📈 EVALUATING MODEL')\n",
                    "print('=' * 40)\n",
                    "\n",
                    "results = trainer.evaluate()\n",
                    "print('\\n📊 FINAL RESULTS:')\n",
                    "print(f'F1 Score: {results[\"eval_f1\"]:.4f}')\n",
                    "print(f'Accuracy: {results[\"eval_accuracy\"]:.4f}')\n",
                    "\n",
                    "print('✅ Evaluation completed!')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 💾 MODEL SAVING"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Save model\n",
                    "print('💾 SAVING MODEL')\n",
                    "print('=' * 30)\n",
                    "\n",
                    "model_path = './minimal_emotion_model_final'\n",
                    "trainer.save_model(model_path)\n",
                    "tokenizer.save_pretrained(model_path)\n",
                    "\n",
                    "print(f'✅ Model saved to: {model_path}')\n",
                    "\n",
                    "# Verify configuration\n",
                    "config_path = f'{model_path}/config.json'\n",
                    "with open(config_path, 'r') as f:\n",
                    "    config = json.load(f)\n",
                    "\n",
                    "print(f'\\n🔍 SAVED CONFIGURATION:')\n",
                    "print(f'Model type: {config.get(\"model_type\", \"NOT SET\")}')\n",
                    "print(f'Number of labels: {config.get(\"num_labels\", \"NOT SET\")}')\n",
                    "print(f'id2label: {config.get(\"id2label\", \"NOT SET\")}')\n",
                    "\n",
                    "print('\\n✅ Model saving completed!')"
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
    
    # Save the notebook
    output_path = "notebooks/MINIMAL_WORKING_TRAINING_COLAB.ipynb"
    with open(output_path, 'w') as f:
        json.dump(notebook_content, f, indent=2)
    
    print(f"✅ Created minimal working notebook: {output_path}")
    print("📋 Features:")
    print("   ✅ Ultra-minimal training arguments")
    print("   ✅ No complex parameters")
    print("   ✅ Basic training and evaluation")
    print("   ✅ Model saving with verification")
    print("\\n🚀 This should work in ANY transformers version!")
    
    return output_path

if __name__ == "__main__":
    create_minimal_notebook() 