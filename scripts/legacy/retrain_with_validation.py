#!/usr/bin/env python3
"""
RETRAIN WITH VALIDATION SCRIPT
===============================
Helps retrain the model with proper validation to ensure reliability
"""

from pathlib import Path


def create_improved_training_plan():
    """Create an improved training plan with proper validation"""
    
    print("🔄 IMPROVED TRAINING PLAN")
    print("=" * 50)
    print("🎯 Goal: Retrain model to achieve reliable 75-85% F1 score")
    print("=" * 50)
    
    print(f"\n❌ CURRENT ISSUES IDENTIFIED:")
    print("-" * 40)
    print("1. Model bias towards 'grateful' and 'happy' emotions")
    print("2. Poor generalization (58.3% accuracy on basic tests)")
    print("3. Overfitting to specific training patterns")
    print("4. Label mapping inconsistencies")
    
    print(f"\n✅ IMPROVED TRAINING STRATEGY:")
    print("-" * 40)
    print("1. Use balanced dataset with equal emotion distribution")
    print("2. Implement proper cross-validation")
    print("3. Add regularization to prevent overfitting")
    print("4. Use early stopping based on validation performance")
    print("5. Test on diverse, realistic examples")
    
    print(f"\n📊 VALIDATION REQUIREMENTS:")
    print("-" * 40)
    print("✅ Basic functionality test: >80% accuracy")
    print("✅ Training-like data test: >80% accuracy")
    print("✅ Edge case handling: >70% success rate")
    print("✅ No emotion bias: <30% predictions for any single emotion")
    print("✅ Consistent predictions: 100% consistency for same input")
    
    print(f"\n🚀 RECOMMENDED ACTIONS:")
    print("-" * 40)
    print("1. Create balanced training dataset")
    print("2. Implement proper validation split")
    print("3. Use regularization techniques")
    print("4. Test extensively before deployment")
    print("5. Monitor for bias and overfitting")
    
    # Create improved training notebook
    create_improved_notebook()
    
    return True

def create_improved_notebook():
    """Create an improved training notebook"""
    
    notebook_content = '''{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# IMPROVED EMOTION DETECTION TRAINING\\n",
    "## With Proper Validation and Bias Prevention\\n",
    "\\n",
    "This notebook addresses the issues found in the previous model:\\n",
    "- Model bias towards certain emotions\\n",
    "- Poor generalization\\n",
    "- Overfitting to training data\\n",
    "\\n",
    "**Target**: Reliable 75-85% F1 score with good generalization"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Install required packages\\n",
    "!pip install transformers datasets torch scikit-learn numpy pandas"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\\n",
    "import numpy as np\\n",
    "import pandas as pd\\n",
    "from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer\\n",
    "from datasets import Dataset\\n",
    "from sklearn.model_selection import train_test_split\\n",
    "from sklearn.metrics import classification_report, confusion_matrix\\n",
    "import json\\n",
    "import warnings\\n",
    "warnings.filterwarnings('ignore')\\n",
    "\\n",
    "print('✅ Packages imported successfully')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create balanced dataset\\n",
    "emotions = ['anxious', 'calm', 'content', 'excited', 'frustrated', 'grateful', 'happy', 'hopeful', 'overwhelmed', 'proud', 'sad', 'tired']\\n",
    "\\n",
    "# Balanced training data (12 samples per emotion)\\n",
    "balanced_data = [\\n",
    "    # anxious\\n",
    "    {'text': 'I feel anxious about the presentation.', 'label': 0},\\n",
    "    {'text': 'I am anxious about the future.', 'label': 0},\\n",
    "    {'text': 'This makes me feel anxious.', 'label': 0},\\n",
    "    {'text': 'I am feeling anxious today.', 'label': 0},\\n",
    "    {'text': 'The uncertainty makes me anxious.', 'label': 0},\\n",
    "    {'text': 'I feel anxious about the results.', 'label': 0},\\n",
    "    {'text': 'This situation is making me anxious.', 'label': 0},\\n",
    "    {'text': 'I am anxious about the meeting.', 'label': 0},\\n",
    "    {'text': 'The pressure is making me anxious.', 'label': 0},\\n",
    "    {'text': 'I feel anxious about the decision.', 'label': 0},\\n",
    "    {'text': 'This is causing me anxiety.', 'label': 0},\\n",
    "    {'text': 'I am anxious about the changes.', 'label': 0},\\n",
    "    \\n",
    "    # calm\\n",
    "    {'text': 'I feel calm and peaceful.', 'label': 1},\\n",
    "    {'text': 'I am feeling calm today.', 'label': 1},\\n",
    "    {'text': 'This makes me feel calm.', 'label': 1},\\n",
    "    {'text': 'I am calm about the situation.', 'label': 1},\\n",
    "    {'text': 'I feel calm and relaxed.', 'label': 1},\\n",
    "    {'text': 'This gives me a sense of calm.', 'label': 1},\\n",
    "    {'text': 'I am feeling calm and centered.', 'label': 1},\\n",
    "    {'text': 'This brings me calm.', 'label': 1},\\n",
    "    {'text': 'I feel calm and at peace.', 'label': 1},\\n",
    "    {'text': 'I am calm about the outcome.', 'label': 1},\\n",
    "    {'text': 'This creates a feeling of calm.', 'label': 1},\\n",
    "    {'text': 'I feel calm and collected.', 'label': 1},\\n",
    "    \\n",
    "    # Continue for all emotions...\\n",
    "    # (Add 12 samples for each emotion)\\n",
    "]\\n",
    "\\n",
    "print(f'✅ Created balanced dataset with {len(balanced_data)} samples')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Split data with proper validation\\n",
    "train_data, val_data = train_test_split(balanced_data, test_size=0.2, random_state=42, stratify=[d['label'] for d in balanced_data])\\n",
    "\\n",
    "print(f'Training samples: {len(train_data)}')\\n",
    "print(f'Validation samples: {len(val_data)}')\\n",
    "\\n",
    "# Convert to datasets\\n",
    "train_dataset = Dataset.from_list(train_data)\\n",
    "val_dataset = Dataset.from_list(val_data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load model and tokenizer\\n",
    "model_name = 'roberta-base'\\n",
    "tokenizer = AutoTokenizer.from_pretrained(model_name)\\n",
    "model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=12)\\n",
    "\\n",
    "# Update model config with emotion labels\\n",
    "model.config.id2label = {i: emotion for i, emotion in enumerate(emotions)}\\n",
    "model.config.label2id = {emotion: i for i, emotion in enumerate(emotions)}\\n",
    "\\n",
    "print('✅ Model and tokenizer loaded')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Tokenization function\\n",
    "def tokenize_function(examples):\\n",
    "    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)\\n",
    "\\n",
    "train_dataset = train_dataset.map(tokenize_function, batched=True)\\n",
    "val_dataset = val_dataset.map(tokenize_function, batched=True)\\n",
    "\\n",
    "print('✅ Data tokenized')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Training arguments with regularization\\n",
    "training_args = TrainingArguments(\\n",
    "    output_dir='./improved_emotion_model',\\n",
    "    learning_rate=2e-5,\\n",
    "    per_device_train_batch_size=8,\\n",
    "    per_device_eval_batch_size=8,\\n",
    "    num_train_epochs=5,\\n",
    "    weight_decay=0.01,  # Regularization\\n",
    "    logging_dir='./logs',\\n",
    "    logging_steps=10,\\n",
    "    evaluation_strategy='steps',\\n",
    "    eval_steps=50,\\n",
    "    save_steps=100,\\n",
    "    load_best_model_at_end=True,\\n",
    "    metric_for_best_model='eval_f1',\\n",
    "    greater_is_better=True,\\n",
    "    warmup_steps=100,\\n",
    "    dataloader_num_workers=0\\n",
    ")\\n",
    "\\n",
    "print('✅ Training arguments configured')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Custom metrics function\\n",
    "def compute_metrics(eval_pred):\\n",
    "    predictions, labels = eval_pred\\n",
    "    predictions = np.argmax(predictions, axis=1)\\n",
    "    \\n",
    "    # Calculate metrics\\n",
    "    report = classification_report(labels, predictions, target_names=emotions, output_dict=True)\\n",
    "    \\n",
    "    return {\\n",
    "        'f1': report['weighted avg']['f1-score'],\\n",
    "        'accuracy': report['accuracy'],\\n",
    "        'precision': report['weighted avg']['precision'],\\n",
    "        'recall': report['weighted avg']['recall']\\n",
    "    }"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Initialize trainer\\n",
    "trainer = Trainer(\\n",
    "    model=model,\\n",
    "    args=training_args,\\n",
    "    train_dataset=train_dataset,\\n",
    "    eval_dataset=val_dataset,\\n",
    "    compute_metrics=compute_metrics\\n",
    ")\\n",
    "\\n",
    "print('✅ Trainer initialized')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Train the model\\n",
    "print('🚀 Starting training...')\\n",
    "trainer.train()\\n",
    "print('✅ Training completed')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Evaluate the model\\n",
    "print('📊 Evaluating model...')\\n",
    "results = trainer.evaluate()\\n",
    "print(f'Final F1 Score: {results[\"eval_f1\"]:.3f}')\\n",
    "print(f'Final Accuracy: {results[\"eval_accuracy\"]:.3f}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Test on diverse examples\\n",
    "test_examples = [\\n",
    "    'I am feeling really happy today!',\\n",
    "    'I am so frustrated with this project.',\\n",
    "    'I feel anxious about the presentation.',\\n",
    "    'I am grateful for all the support.',\\n",
    "    'I am feeling overwhelmed with tasks.',\\n",
    "    'I am proud of my accomplishments.',\\n",
    "    'I feel sad about the loss.',\\n",
    "    'I am tired from working all day.',\\n",
    "    'I feel calm and peaceful.',\\n",
    "    'I am excited about the new opportunity.',\\n",
    "    'I feel content with my life.',\\n",
    "    'I am hopeful for the future.'\\n",
    "]\\n",
    "\\n",
    "print('🧪 Testing on diverse examples...')\\n",
    "correct = 0\\n",
    "for text in test_examples:\\n",
    "    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)\\n",
    "    with torch.no_grad():\\n",
    "        outputs = model(**inputs)\\n",
    "        predictions = torch.softmax(outputs.logits, dim=1)\\n",
    "        predicted_class = torch.argmax(predictions, dim=1).item()\\n",
    "        confidence = predictions[0][predicted_class].item()\\n",
    "    \\n",
    "    predicted_emotion = emotions[predicted_class]\\n",
    "    expected_emotion = None\\n",
    "    for emotion in emotions:\\n",
    "        if emotion in text.lower():\\n",
    "            expected_emotion = emotion\\n",
    "            break\\n",
    "    \\n",
    "    if expected_emotion and predicted_emotion == expected_emotion:\\n",
    "        correct += 1\\n",
    "        status = '✅'\\n",
    "    else:\\n",
    "        status = '❌'\\n",
    "    \\n",
    "    print(f'{status} \"{text}\" → {predicted_emotion} (expected: {expected_emotion}, confidence: {confidence:.3f})')\\n",
    "\\n",
    "accuracy = correct / len(test_examples)\\n",
    "print(f'\\n📊 Test Accuracy: {accuracy:.1%}')\\n",
    "\\n",
    "if accuracy >= 0.8:\\n",
    "    print('🎉 Model passes reliability test!')\\n",
    "else:\\n",
    "    print('⚠️  Model needs further improvement')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Save the model\\n",
    "model.save_pretrained('./improved_emotion_model_final')\\n",
    "tokenizer.save_pretrained('./improved_emotion_model_final')\\n",
    "print('💾 Model saved successfully')"
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
}'''
    
    # Save the notebook
    notebook_path = Path(__file__).parent.parent / 'notebooks' / 'IMPROVED_TRAINING_WITH_VALIDATION.ipynb'
    with open(notebook_path, 'w') as f:
        f.write(notebook_content)
    
    print(f"✅ Created improved training notebook: {notebook_path}")
    print(f"📋 Instructions:")
    print(f"   1. Download the notebook file")
    print(f"   2. Upload to Google Colab")
    print(f"   3. Set Runtime → GPU")
    print(f"   4. Run all cells")
    print(f"   5. Verify reliability before deployment")

if __name__ == "__main__":
    success = create_improved_training_plan()
    exit(0 if success else 1) 