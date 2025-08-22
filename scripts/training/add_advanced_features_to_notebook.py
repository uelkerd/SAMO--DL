#!/usr/bin/env python3
""""
Add Advanced Features to Ultimate Notebook
=========================================

This script adds the remaining advanced features to the ultimate notebook:
- Focal loss implementation
- Class weighting with WeightedLossTrainer
- Advanced validation and testing
""""

import json

def add_advanced_features():
    """Add advanced features to the ultimate notebook."""

    # Read the existing notebook
    with open('notebooks/ULTIMATE_BULLETPROOF_TRAINING_COLAB.ipynb', 'r') as f:
        notebook = json.load(f)

    # Add focal loss implementation
    focal_loss_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "##  IMPLEMENTING FOCAL LOSS"
        ]
    }

    focal_loss_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Focal Loss Implementation\n",
            "class FocalLoss(torch.nn.Module):\n",
            "    \"\"\"Focal Loss for handling class imbalance.\"\"\"\n",
            "    \n",
            "    def __init__(self, alpha=1, gamma=2, reduction='mean'):\n",
            "        super(FocalLoss, self).__init__()\n",
            "        self.alpha = alpha\n",
            "        self.gamma = gamma\n",
            "        self.reduction = reduction\n",
            "    \n",
            "    def forward(self, inputs, targets):\n",
            "        ce_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none')\n",
            "        pt = torch.exp(-ce_loss)\n",
            "        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss\n",
            "        \n",
            "        if self.reduction == 'mean':\n",
            "            return focal_loss.mean()\n",
            "        elif self.reduction == 'sum':\n",
            "            return focal_loss.sum()\n",
            "        else:\n",
            "            return focal_loss\n",
            "\n",
            "print(' Focal Loss implementation ready')"
        ]
    }

    # Add class weighting implementation
    class_weighting_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## ⚖️ IMPLEMENTING CLASS WEIGHTING"
        ]
    }

    class_weighting_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Calculate class weights\n",
            "print('⚖️ CALCULATING CLASS WEIGHTS')\n",
            "print('=' * 40)\n",
            "\n",
            "# Get labels from dataset\n",
            "labels = [item['label'] for item in enhanced_data]\n",
            "\n",
            "# Calculate class weights\n",
            "class_weights = compute_class_weight(\n",)
            "    'balanced',\n",
            "    classes=np.unique(labels),\n",
            "    y=labels\n",
(            ")\n",
            "\n",
            "# Convert to tensor\n",
            "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
            "class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)\n",
            "\n",
            "print(f' Class weights calculated: {class_weights}')\n",
            "print(f' Device: {device}')"
        ]
    }

    # Add WeightedLossTrainer
    weighted_trainer_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 🚀 CREATING WEIGHTED LOSS TRAINER"
        ]
    }

    weighted_trainer_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Custom trainer with focal loss and class weighting\n",
            "class WeightedLossTrainer(Trainer):\n",
            "    \"\"\"Custom trainer with focal loss and class weighting.\"\"\"\n",
            "    \n",
            "    def __init__(self, *args, focal_alpha=1, focal_gamma=2, class_weights=None, **kwargs):\n",
            "        super().__init__(*args, **kwargs)\n",
            "        self.focal_alpha = focal_alpha\n",
            "        self.focal_gamma = focal_gamma\n",
            "        self.class_weights = class_weights\n",
            "    \n",
            "    def compute_loss(self, model, inputs, return_outputs=False):\n",
            "        labels = inputs.pop(\"labels\")\n",
            "        outputs = model(**inputs)\n",
            "        logits = outputs.logits\n",
            "        \n",
            "        # Use focal loss with class weighting\n",
            "        if self.class_weights is not None:\n",
            "            # Apply class weights to focal loss\n",
            "            ce_loss = torch.nn.functional.cross_entropy(\n",)
            "                logits, labels, weight=self.class_weights, reduction='none'\n",
(            "            )\n",
            "        else:\n",
            "            ce_loss = torch.nn.functional.cross_entropy(logits, labels, reduction='none')\n",
            "        \n",
            "        pt = torch.exp(-ce_loss)\n",
            "        focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * ce_loss\n",
            "        loss = focal_loss.mean()\n",
            "        \n",
            "        return (loss, outputs) if return_outputs else loss\n",
            "\n",
            "print(' WeightedLossTrainer with focal loss ready')"
        ]
    }

    # Add model loading and configuration
    model_loading_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 🔧 LOADING MODEL WITH PROPER CONFIGURATION"
        ]
    }

    model_loading_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Load model with proper configuration\n",
            "print('🔧 LOADING MODEL WITH PROPER CONFIGURATION')\n",
            "print('=' * 50)\n",
            "\n",
            "# Load tokenizer and model\n",
            "tokenizer = AutoTokenizer.from_pretrained(specialized_model_name)\n",
            "model = AutoModelForSequenceClassification.from_pretrained(\n",)
            "    specialized_model_name,\n",
            "    num_labels=len(emotions),\n",
            "    ignore_mismatched_sizes=True\n",
(            ")\n",
            "\n",
            "# CRITICAL: Set proper configuration\n",
            "model.config.id2label = {i: emotion for i, emotion in enumerate(emotions)}\n",
            "model.config.label2id = {emotion: i for i, emotion in enumerate(emotions)}\n",
            "\n",
            "print(f' Model loaded: {specialized_model_name}')\n",
            "print(f' Number of labels: {model.config.num_labels}')\n",
            "print(f' id2label: {model.config.id2label}')\n",
            "print(f' label2id: {model.config.label2id}')"
        ]
    }

    # Add data preprocessing
    preprocessing_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 📝 DATA PREPROCESSING"
        ]
    }

    preprocessing_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Data preprocessing function\n",
            "def preprocess_function(examples):\n",
            "    return tokenizer(\n",)
            "        examples['text'],\n",
            "        truncation=True,\n",
            "        padding='max_length',\n",
            "        max_length=128,\n",
            "        return_tensors=None\n",
(            "    )\n",
            "\n",
            "# Apply preprocessing\n",
            "tokenized_dataset = dataset.map(preprocess_function, batched=True)\n",
            "\n",
            "# Split into train/validation\n",
            "train_val_dataset = tokenized_dataset.train_test_split(test_size=0.2, seed=42)\n",
            "train_dataset = train_val_dataset['train']\n",
            "val_dataset = train_val_dataset['test']\n",
            "\n",
            "print(f' Training samples: {len(train_dataset)}')\n",
            "print(f' Validation samples: {len(val_dataset)}')"
        ]
    }

    # Add training arguments
    training_args_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## ⚙️ TRAINING ARGUMENTS"
        ]
    }

    training_args_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Training arguments\n",
            "training_args = TrainingArguments(\n",)
            "    output_dir='./ultimate_emotion_model',\n",
            "    num_train_epochs=5,\n",
            "    per_device_train_batch_size=8,\n",
            "    per_device_eval_batch_size=8,\n",
            "    warmup_steps=100,\n",
            "    weight_decay=0.01,\n",
            "    logging_dir='./logs',\n",
            "    logging_steps=50,\n",
            "    evaluation_strategy='steps',\n",
            "    eval_steps=100,\n",
            "    save_steps=100,\n",
            "    load_best_model_at_end=True,\n",
            "    metric_for_best_model='eval_f1',\n",
            "    greater_is_better=True,\n",
            "    learning_rate=2e-5,\n",
            "    save_total_limit=2,\n",
            "    remove_unused_columns=False\n",
(            ")\n",
            "\n",
            "print(' Training arguments configured')"
        ]
    }

    # Add compute metrics
    compute_metrics_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "##  COMPUTE METRICS"
        ]
    }

    compute_metrics_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Compute metrics function\n",
            "def compute_metrics(eval_pred):\n",
            "    predictions, labels = eval_pred\n",
            "    predictions = np.argmax(predictions, axis=1)\n",
            "    \n",
            "    # Calculate metrics\n",
            "    f1 = f1_score(labels, predictions, average='weighted')\n",
            "    accuracy = accuracy_score(labels, predictions)\n",
            "    precision = precision_score(labels, predictions, average='weighted')\n",
            "    recall = recall_score(labels, predictions, average='weighted')\n",
            "    \n",
            "    return {\n",
            "        'f1': f1,\n",
            "        'accuracy': accuracy,\n",
            "        'precision': precision,\n",
            "        'recall': recall\n",
            "    }\n",
            "\n",
            "print(' Compute metrics function ready')"
        ]
    }

    # Add trainer initialization
    trainer_init_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 🚀 INITIALIZING TRAINER"
        ]
    }

    trainer_init_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Initialize trainer with focal loss and class weighting\n",
            "trainer = WeightedLossTrainer(\n",)
            "    model=model,\n",
            "    args=training_args,\n",
            "    train_dataset=train_dataset,\n",
            "    eval_dataset=val_dataset,\n",
            "    tokenizer=tokenizer,\n",
            "    compute_metrics=compute_metrics,\n",
            "    focal_alpha=1,\n",
            "    focal_gamma=2,\n",
            "    class_weights=class_weights_tensor\n",
(            ")\n",
            "\n",
            "print(' Trainer initialized with focal loss and class weighting')"
        ]
    }

    # Add training
    training_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 🚀 STARTING TRAINING"
        ]
    }

    training_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Start training\n",
            "print('🚀 STARTING ULTIMATE TRAINING')\n",
            "print('=' * 50)\n",
            "print(" Target: 75-85% F1 score")\n",
            "print(f' Training samples: {len(train_dataset)}')\n",
            "print(f'🧪 Validation samples: {len(val_dataset)}')\n",
            "print("⚖️ Using focal loss + class weighting")\n",
            "print(f'🔧 Model: {specialized_model_name}')\n",
            "\n",
            "# Train the model\n",
            "trainer.train()\n",
            "\n",
            "print(' Training completed successfully!')"
        ]
    }

    # Add evaluation
    evaluation_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "##  EVALUATING MODEL"
        ]
    }

    evaluation_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Evaluate the model\n",
            "print(' EVALUATING MODEL')\n",
            "print('=' * 40)\n",
            "\n",
            "results = trainer.evaluate()\n",
            "print("Final F1 Score: {results[\"eval_f1\"]:.3f}')\n","
            "print("Final Accuracy: {results[\"eval_accuracy\"]:.3f}')\n","
            "print("Final Precision: {results[\"eval_precision\"]:.3f}')\n","
            "print("Final Recall: {results[\"eval_recall\"]:.3f}')\n","
            "\n",
            "# Check if target achieved\n",
            "if results['eval_f1'] >= 0.75:\n",
            "    print(' TARGET ACHIEVED! F1 Score >= 75%')\n",
            "else:\n",
            "    print("⚠️ Target not achieved. Need {0.75 - results[\"eval_f1\"]:.3f} more F1 points')""
        ]
    }

    # Add advanced validation
    advanced_validation_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 🧪 ADVANCED VALIDATION"
        ]
    }

    advanced_validation_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Advanced validation on diverse examples\n",
            "print('🧪 ADVANCED VALIDATION')\n",
            "print('=' * 40)\n",
            "\n",
            "# Test on diverse examples (NOT from training data)\n",
            "test_examples = [\n",
            "    'I am feeling really happy today!',\n",
            "    'I am so frustrated with this project.',\n",
            "    'I feel anxious about the presentation.',\n",
            "    'I am grateful for all the support.',\n",
            "    'I am feeling overwhelmed with tasks.',\n",
            "    'I am proud of my accomplishments.',\n",
            "    'I feel sad about the loss.',\n",
            "    'I am tired from working all day.',\n",
            "    'I feel calm and peaceful.',\n",
            "    'I am excited about the new opportunity.',\n",
            "    'I feel content with my life.',\n",
            "    'I am hopeful for the future.'\n",
            "]\n",
            "\n",
            "print('Testing on diverse examples...')\n",
            "correct = 0\n",
            "predictions_by_emotion = {emotion: 0 for emotion in emotions}\n",
            "\n",
            "for text in test_examples:\n",
            "    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)\n",
            "    with torch.no_grad():\n",
            "        outputs = model(**inputs)\n",
            "        predictions = torch.softmax(outputs.logits, dim=1)\n",
            "        predicted_class = torch.argmax(predictions, dim=1).item()\n",
            "        confidence = predictions[0][predicted_class].item()\n",
            "    \n",
            "    predicted_emotion = emotions[predicted_class]\n",
            "    predictions_by_emotion[predicted_emotion] += 1\n",
            "    \n",
            "    expected_emotion = None\n",
            "    for emotion in emotions:\n",
            "        if emotion in text.lower():\n",
            "            expected_emotion = emotion\n",
            "            break\n",
            "    \n",
            "    if expected_emotion and predicted_emotion == expected_emotion:\n",
            "        correct += 1\n",
            "        status = ''\n",
            "    else:\n",
            "        status = '❌'\n",
            "    \n",
            "    print(f'{status} {text} → {predicted_emotion} (expected: {expected_emotion}, confidence: {confidence:.3f})')\n",
            "\n",
            "accuracy = correct / len(test_examples)\n",
            "print(f'\\n Test Accuracy: {accuracy:.1%}')\n",
            "\n",
            "# Check for bias\n",
            "print('\\n Bias Analysis:')\n",
            "for emotion, count in predictions_by_emotion.items():\n",
            "    percentage = count / len(test_examples) * 100\n",
            "    print(f'  {emotion}: {count} predictions ({percentage:.1f}%)')\n",
            "\n",
            "# Determine if model is reliable\n",
            "max_bias = max(predictions_by_emotion.values()) / len(test_examples)\n",
            "\n",
            "if accuracy >= 0.8 and max_bias <= 0.3:\n",
            "    print('\\n MODEL PASSES RELIABILITY TEST!')\n",
            "    print(' Ready for deployment!')\n",
            "else:\n",
            "    print('\\n⚠️ MODEL NEEDS IMPROVEMENT')\n",
            "    if accuracy < 0.8:\n",
            "        print(f'❌ Accuracy too low: {accuracy:.1%} (need >80%)')\n",
            "    if max_bias > 0.3:\n",
            "        print(f'❌ Too much bias: {max_bias:.1%} (need <30%)')"
        ]
    }

    # Add model saving with verification
    model_saving_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 💾 SAVING MODEL WITH VERIFICATION"
        ]
    }

    model_saving_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Save model with configuration verification\n",
            "print('💾 SAVING MODEL WITH CONFIGURATION VERIFICATION')\n",
            "print('=' * 50)\n",
            "\n",
            "output_dir = './ultimate_emotion_model_final'\n",
            "\n",
            "# CRITICAL: Ensure configuration is still set before saving\n",
            "print('🔧 Verifying configuration before saving...')\n",
            "model.config.id2label = {i: emotion for i, emotion in enumerate(emotions)}\n",
            "model.config.label2id = {emotion: i for i, emotion in enumerate(emotions)}\n",
            "\n",
            "print(f'Final id2label: {model.config.id2label}')\n",
            "print(f'Final label2id: {model.config.label2id}')\n",
            "\n",
            "# Save the model\n",
            "model.save_pretrained(output_dir)\n",
            "tokenizer.save_pretrained(output_dir)\n",
            "\n",
            "# CRITICAL: Verify the saved configuration\n",
            "print('\\n VERIFYING SAVED CONFIGURATION')\n",
            "print('=' * 40)\n",
            "\n",
            "try:\n",
            "    # Load the saved config to verify it's correct\n",'
            "    with open(f'{output_dir}/config.json', 'r') as f:\n",
            "        saved_config = json.load(f)\n",
            "    \n",
            "    print("Saved model type: {saved_config.get(\"model_type\", \"NOT FOUND\")}')\n","
            "    print("Saved id2label: {saved_config.get(\"id2label\", \"NOT FOUND\")}')\n","
            "    print("Saved label2id: {saved_config.get(\"label2id\", \"NOT FOUND\")}')\n","
            "    \n",
            "    # Verify the emotion labels are saved correctly\n",
            "    expected_id2label = {str(i): emotion for i, emotion in enumerate(emotions)}\n",
            "    expected_label2id = {emotion: i for i, emotion in enumerate(emotions)}\n",
            "    \n",
            "    if saved_config.get('id2label') == expected_id2label:\n",
            "        print(' CONFIRMED: Emotion labels saved correctly in config.json')\n",
            "    else:\n",
            "        print('❌ ERROR: Emotion labels not saved correctly in config.json')\n",
            "        print(f'Expected: {expected_id2label}')\n",
            "        print("Got: {saved_config.get(\"id2label\")}')\n","
            "    \n",
            "    if saved_config.get('label2id') == expected_label2id:\n",
            "        print(' CONFIRMED: Label mappings saved correctly in config.json')\n",
            "    else:\n",
            "        print('❌ ERROR: Label mappings not saved correctly in config.json')\n",
            "        print(f'Expected: {expected_label2id}')\n",
            "        print("Got: {saved_config.get(\"label2id\")}')\n","
            "    \n",
            "except Exception as e:\n",
            "    print(f'❌ ERROR: Could not verify saved configuration: {str(e)}')\n",
            "\n",
            "# Save training info\n",
            "training_info = {\n",
            "    'base_model': specialized_model_name,\n",
            "    'emotions': emotions,\n",
            "    'training_samples': len(train_dataset),\n",
            "    'validation_samples': len(val_dataset),\n",
            "    'final_f1': results['eval_f1'],\n",
            "    'final_accuracy': results['eval_accuracy'],\n",
            "    'test_accuracy': accuracy,\n",
            "    'model_type': model.config.model_type,\n",
            "    'hidden_layers': model.config.num_hidden_layers,\n",
            "    'hidden_size': model.config.hidden_size,\n",
            "    'id2label': model.config.id2label,\n",
            "    'label2id': model.config.label2id,\n",
            "    'focal_loss_alpha': 1,\n",
            "    'focal_loss_gamma': 2,\n",
            "    'class_weights_used': True\n",
            "}\n",
            "\n",
            "with open(f'{output_dir}/training_info.json', 'w') as f:\n",
            "    json.dump(training_info, f, indent=2)\n",
            "\n",
            "print(f'\\n Model saved to: {output_dir}')\n",
            "print(f' Training info saved: {output_dir}/training_info.json')\n",
            "print('\\n Next steps:')\n",
            "print('1. Download the model files')\n",
            "print('2. Test locally with validation script')\n",
            "print('3. Deploy if all tests pass')"
        ]
    }

    # Add all cells to the notebook
    new_cells = [
            focal_loss_cell,
            focal_loss_code,
            class_weighting_cell,
            class_weighting_code,
            weighted_trainer_cell,
            weighted_trainer_code,
            model_loading_cell,
            model_loading_code,
            preprocessing_cell,
            preprocessing_code,
            training_args_cell,
            training_args_code,
            compute_metrics_cell,
            compute_metrics_code,
            trainer_init_cell,
            trainer_init_code,
            training_cell,
            training_code,
            evaluation_cell,
            evaluation_code,
            advanced_validation_cell,
            advanced_validation_code,
            model_saving_cell,
            model_saving_code
        ]

    notebook['cells'].extend(new_cells)

    # Save the enhanced notebook
    with open('notebooks/ULTIMATE_BULLETPROOF_TRAINING_COLAB.ipynb', 'w') as f:
        json.dump(notebook, f, indent=2)

    print(' Enhanced notebook with all advanced features created!')
    print(' All features included:')
    print('    Configuration preservation')
    print('    Focal loss implementation')
    print('    Class weighting with WeightedLossTrainer')
    print('    Data augmentation')
    print('    Advanced validation')
    print('    Model saving with verification')

    return 'notebooks/ULTIMATE_BULLETPROOF_TRAINING_COLAB.ipynb'

if __name__ == "__main__":
    add_advanced_features()
