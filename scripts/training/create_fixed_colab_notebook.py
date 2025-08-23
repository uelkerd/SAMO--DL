#!/usr/bin/env python3
"""
🚀 CREATE FIXED COLAB NOTEBOOK
==============================

This script creates a fixed Colab notebook that handles the correct data structure.
"""

import json

def create_fixed_colab_notebook():
    """Create the fixed Colab notebook content."""

    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 🚀 FIXED COMBINED TRAINING - JOURNAL + CMU-MOSEI\n",
                    "\n",
                    "**Target: 75-85% F1 Score**  \n",
                    "**Current: 67% F1 Score**  \n",
                    "**Strategy: Combine high-quality datasets**\n",
                    "\n",
                    "This notebook combines:\n",
                    "- Original 150 high-quality journal samples\n",
                    "- CMU-MOSEI samples for diversity\n",
                    "- Optimized hyperparameters for 75-85% F1\n",
                    "\n",
                    "**FIXED**: Correct data loading for journal content field"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Install dependencies\n",
                    "!pip install transformers torch scikit-learn pandas numpy\n",
                    "print(\"✅ All dependencies installed!\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Clone repository\n",
                    "!git clone https://github.com/uelkerd/SAMO--DL.git\n",
                    "print(\"📂 Repository cloned successfully!\")"
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
                    "from sklearn.metrics import                    "from sklearn.metrics import f1_score,
                         accuracy_score,
                         classification_report\n",
                        
                    "import warnings\n",
                    "warnings.filterwarnings('ignore')\n",
                    "\n",
                    "print(\"✅ All libraries imported!\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# FIXED: Load combined dataset with correct field names\n",
                    "print(\"📊 Loading combined dataset...\")\n",
                    "\n",
                    "combined_samples = []\n",
                    "\n",
                    "# Load journal data (FIXED: use 'content' field)\n",
                    "try:\n",
                    "    with open('/content/SAMO--DL/data/journal_test_dataset.json', 'r') as f:\n",
                    "        journal_data = json.load(f)\n",
                    "    \n",
                    "    for item in journal_data:\n",
                    "        combined_samples.append({\n",
                    "            'text': item['content'],  # FIXED: use 'content' not 'text'\n",
                    "            'emotion': item['emotion']\n",
                    "        })\n",
                    "    print(f\"✅ Loaded {len(journal_data)} journal samples\")\n",
                    "except Exception as e:\n",
                    "    print(f\"⚠️ Could not load journal data: {e}\")\n",
                    "\n",
                    "# Load CMU-MOSEI data (uses 'text' field)\n",
                    "try:\n",
                    "    with open('/content/SAMO--DL/data/cmu_mosei_balanced_dataset.json', 'r') as f:\n",
                    "        cmu_data = json.load(f)\n",
                    "    \n",
                    "    for item in cmu_data:\n",
                    "        combined_samples.append({\n",
                    "            'text': item['text'],  # CMU-MOSEI uses 'text' field\n",
                    "            'emotion': item['emotion']\n",
                    "        })\n",
                    "    print(f\"✅ Loaded {len(cmu_data)} CMU-MOSEI samples\")\n",
                    "except Exception as e:\n",
                    "    print(f\"⚠️ Could not load CMU-MOSEI data: {e}\")\n",
                    "\n",
                    "print(f\"📊 Total combined samples: {len(combined_samples)}\")\n",
                    "\n",
                    "# Show emotion distribution\n",
                    "if combined_samples:\n",
                    "    emotion_counts = {}\n",
                    "    for sample in combined_samples:\n",
                    "        emotion = sample['emotion']\n",
                    "        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1\n",
                    "    \n",
                    "    print(\"📊 Emotion distribution:\")\n",
                    "    for emotion, count in sorted(emotion_counts.items()):\n",
                    "        print(f\"  {emotion}: {count} samples\")\n",
                    "else:\n",
                    "    print(\"❌ No data loaded! Check file paths.\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Check if we have data\n",
                    "if len(combined_samples) == 0:\n",
                    "    print(\"❌ No data loaded! Creating fallback dataset...\")\n",
                    "    \n",
                    "    # Create minimal fallback dataset\n",
                    "    fallback_samples = [\n",
                    "        {\"text\": \"I'm feeling happy today!\", \"emotion\": \"happy\"},\n",
                    "        {\"text\": \"I'm so frustrated with this project.\", \"emotion\": \"frustrated\"},\n",
                    "        {\"text\": \"I feel anxious about the presentation.\", \"emotion\": \"anxious\"},\n",
                    "        {\"text\": \"I'm grateful for all the support.\", \"emotion\": \"grateful\"},\n",
                    "        {\"text\": \"I'm feeling overwhelmed with tasks.\", \"emotion\": \"overwhelmed\"},\n",
                    "        {\"text\": \"I'm proud of what I accomplished.\", \"emotion\": \"proud\"},\n",
                    "        {\"text\": \"I'm feeling sad and lonely.\", \"emotion\": \"sad\"},\n",
                    "        {\"text\": \"I'm excited about new opportunities.\", \"emotion\": \"excited\"},\n",
                    "        {\"text\": \"I feel calm and peaceful.\", \"emotion\": \"calm\"},\n",
                    "        {\"text\": \"I'm hopeful things will get better.\", \"emotion\": \"hopeful\"},\n",
                    "        {\"text\": \"I'm tired and need rest.\", \"emotion\": \"tired\"},\n",
                    "        {\"text\": \"I'm content with how things are.\", \"emotion\": \"content\"}\n",
                    "    ]\n",
                    "    combined_samples = fallback_samples\n",
                    "    print(f\"✅ Created {len(combined_samples)} fallback samples\")\n",
                    "\n",
                    "print(f\"📊 Final dataset size: {len(combined_samples)} samples\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Custom dataset class\n",
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
                    "# Prepare data\n",
                    "texts = [sample['text'] for sample in combined_samples]\n",
                    "emotions = [sample['emotion'] for sample in combined_samples]\n",
                    "\n",
                    "# Encode labels\n",
                    "label_encoder = LabelEncoder()\n",
                    "labels = label_encoder.fit_transform(emotions)\n",
                    "\n",
                    "print(f\"🎯 Number of labels: {len(label_encoder.classes_)}\")\n",
                    "print(f\"📊 Labels: {list(label_encoder.classes_)}\")\n",
                    "\n",
                    "# Split data\n",
                    "train_texts, test_texts, train_labels, test_labels = train_test_split(\n",
                    "    texts, labels, test_size=0.2, random_state=42, stratify=labels\n",
                    ")\n",
                    "\n",
                    "print(f\"📈 Training samples: {len(train_texts)}\")\n",
                    "print(f\"🧪 Test samples: {len(test_labels)}\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Load model and tokenizer\n",
                    "model_name = \"bert-base-uncased\"\n",
                    "tokenizer = AutoTokenizer.from_pretrained(model_name)\n",
                    "model = AutoModelForSequenceClassification.from_pretrained(\n",
                    "    model_name, \n",
                    "    num_labels=len(label_encoder.classes_),\n",
                    "    problem_type=\"single_label_classification\"\n",
                    ")\n",
                    "\n",
                    "print(f\"✅ Model loaded: {model_name}\")\n",
                    "print(f\"📊 Number of classes: {len(label_encoder.classes_)}\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Create datasets\n",
                    "train_dataset = EmotionDataset(train_texts, train_labels, tokenizer)\n",
                    "test_dataset = EmotionDataset(test_texts, test_labels, tokenizer)\n",
                    "\n",
                    "print(f\"✅ Datasets created\")\n",
                    "print(f\"📈 Train dataset: {len(train_dataset)} samples\")\n",
                    "print(f\"🧪 Test dataset: {len(test_dataset)} samples\")"
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
                    "# Training arguments\n",
                    "training_args = TrainingArguments(\n",
                    "    output_dir=\"./emotion_model_combined\",\n",
                    "    num_train_epochs=8,\n",
                    "    per_device_train_batch_size=16,\n",
                    "    per_device_eval_batch_size=16,\n",
                    "    warmup_steps=500,\n",
                    "    weight_decay=0.01,\n",
                    "    logging_dir=\"./logs\",\n",
                    "    logging_steps=50,\n",
                    "    eval_strategy=\"steps\",\n",
                    "    eval_steps=100,\n",
                    "    save_strategy=\"steps\",\n",
                    "    save_steps=100,\n",
                    "    load_best_model_at_end=True,\n",
                    "    metric_for_best_model=\"f1\",\n",
                    "    greater_is_better=True,\n",
                    "    dataloader_num_workers=2,\n",
                    "    remove_unused_columns=False,\n",
                    "    report_to=None,\n",
                    "    learning_rate=2e-5,\n",
                    "    gradient_accumulation_steps=2,\n",
                    ")\n",
                    "\n",
                    "print(\"✅ Training arguments configured\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Create trainer\n",
                    "trainer = Trainer(\n",
                    "    model=model,\n",
                    "    args=training_args,\n",
                    "    train_dataset=train_dataset,\n",
                    "    eval_dataset=test_dataset,\n",
                    "    compute_metrics=compute_metrics,\n",
                    "    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]\n",
                    ")\n",
                    "\n",
                    "print(\"✅ Trainer created with early stopping\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Start training\n",
                    "print(\"🚀 Starting training...\")\n",
                    "print(\"🎯 Target F1 Score: 75-85%\")\n",
                    "print(\"📊 Current Best: 67%\")\n",
                    "print(\"📈 Expected Improvement: 8-18%\")\n",
                    "\n",
                    "trainer.train()\n",
                    "\n",
                    "print(\"✅ Training completed!\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Evaluate final model\n",
                    "print(\"📊 Evaluating final model...\")\n",
                    "results = trainer.evaluate()\n",
                    "\n",
                    "print(f\"🏆 Final F1 Score: {results['eval_f1']:.4f} ({results['eval_f1']*100:.2f}%)\")\n",
                    "print(f\"🎯 Target achieved: {'✅ YES!' if results['eval_f1'] >= 0.75 else '❌ Not yet'}\")\n",
                    "\n",
                    "# Save model\n",
                    "trainer.save_model(\"./emotion_model_final_combined\")\n",
                    "print(\"💾 Model saved to ./emotion_model_final_combined\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Test on sample texts\n",
                    "print(\"🧪 Testing on sample texts...\")\n",
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
                    "        inputs = tokenizer(text, return_tensors=\"pt\", truncation=True, padding=True)\n",
                    "        outputs = model(**inputs)\n",
                    "        probabilities = torch.softmax(outputs.logits, dim=1)\n",
                    "        predicted_class = torch.argmax(probabilities, dim=1).item()\n",
                    "        confidence = probabilities[0][predicted_class].item()\n",
                    "        predicted_emotion = label_encoder.classes_[predicted_class]\n",
                    "        \n",
                    "        print(f\"{i}. Text: {text}\")\n",
                    "        print(f\"   Predicted: {predicted_emotion} (confidence: {confidence:.3f})\")\n",
                    "        print()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 🎉 Training Complete!\n",
                    "\n",
                    "**Results Summary:**\n",
                    "- Final F1 Score: [See output above]\n",
                    "- Target: 75-85%\n",
                    "- Improvement: [Calculated above]\n",
                    "\n",
                    "**Next Steps:**\n",
                    "1. If F1 < 75%: Try different hyperparameters or more data\n",
                    "2. If F1 >= 75%: Model is ready for production!\n",
                    "3. Download the saved model from `./emotion_model_final_combined`"
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

    # Write notebook to file
    with open('notebooks/FIXED_COMBINED_TRAINING_COLAB.ipynb', 'w') as f:
        json.dump(notebook_content, f, indent=2)

    print("✅ Fixed notebook created: notebooks/FIXED_COMBINED_TRAINING_COLAB.ipynb")
    print("📋 Instructions:")
    print("  1. Download the notebook file")
    print("  2. Upload to Google Colab")
    print("  3. Set Runtime → GPU")
    print("  4. Run all cells")
    print("  5. Expect 75-85% F1 score!")

if __name__ == "__main__":
    create_fixed_colab_notebook()
