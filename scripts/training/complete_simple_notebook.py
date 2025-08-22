#!/usr/bin/env python3
"""
Complete Simple Notebook
========================

This script adds all the missing training, validation, and model saving
components to the simple notebook.
"""

import json

def complete_simple_notebook():
    """Add all missing components to the simple notebook."""

    # Read the existing notebook
    with open('notebooks/SIMPLE_ULTIMATE_BULLETPROOF_TRAINING_COLAB.ipynb', 'r') as f:
        notebook = json.load(f)

    # Add all the missing cells
    new_cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 🎯 FOCAL LOSS IMPLEMENTATION"
            ]
        },
        {
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
                "print('✅ Focal Loss implementation ready')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## ⚖️ CLASS WEIGHTING & WEIGHTED LOSS TRAINER"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Calculate class weights\n",
                "print('⚖️ CALCULATING CLASS WEIGHTS')\n",
                "print('=' * 40)\n",
                "\n",
                "class_weights = compute_class_weight(\n",
                "    'balanced',\n",
                "    classes=np.unique(labels),\n",
                "    y=labels\n",
                ")\n",
                "\n",
                "class_weights_tensor = torch.FloatTensor(class_weights)\n",
                "if torch.cuda.is_available():\n",
                "    class_weights_tensor = class_weights_tensor.cuda()\n",
                "\n",
                "print(f'Class weights: {class_weights}')\n",
                "print(f'Class weights tensor shape: {class_weights_tensor.shape}')\n",
                "print('✅ Class weights calculated')\n",
                "\n",
                "# Weighted Loss Trainer\n",
                "class WeightedLossTrainer(Trainer):\n",
                "    \"\"\"Custom trainer with focal loss and class weighting.\"\"\"\n",
                "    \n",
                "    def __init__(self, focal_alpha=1, focal_gamma=2, class_weights=None, *args, **kwargs):\n",
                "        super().__init__(*args, **kwargs)\n",
                "        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)\n",
                "        self.class_weights = class_weights\n",
                "    \n",
                "    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):\n",
                "        labels = inputs.pop(\"labels\")\n",
                "        outputs = model(**inputs)\n",
                "        logits = outputs.logits\n",
                "        \n",
                "        # Apply focal loss with class weighting\n",
                "        if self.class_weights is not None:\n",
                "            # Apply class weights to focal loss\n",
                "            ce_loss = torch.nn.functional.cross_entropy(logits, labels, reduction='none')\n",
                "            pt = torch.exp(-ce_loss)\n",
                "            focal_loss = (1 - pt) ** self.focal_loss.gamma * ce_loss\n",
                "            \n",
                "            # Apply class weights\n",
                "            for i, weight in enumerate(self.class_weights):\n",
                "                mask = (labels == i)\n",
                "                focal_loss[mask] *= weight\n",
                "            \n",
                "            loss = focal_loss.mean()\n",
                "        else:\n",
                "            loss = self.focal_loss(logits, labels)\n",
                "        \n",
                "        return (loss, outputs) if return_outputs else loss\n",
                "\n",
                "print('✅ WeightedLossTrainer ready')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 🔧 LOADING & CONFIGURING MODEL"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load tokenizer and model\n",
                "print('🔧 LOADING & CONFIGURING MODEL')\n",
                "print('=' * 40)\n",
                "\n",
                "tokenizer = AutoTokenizer.from_pretrained(specialized_model_name)\n",
                "model = AutoModelForSequenceClassification.from_pretrained(specialized_model_name)\n",
                "\n",
                "# Configure model for our emotion classes\n",
                "model.config.num_labels = len(emotions)\n",
                "model.config.id2label = {i: emotion for i, emotion in enumerate(emotions)}\n",
                "model.config.label2id = {emotion: i for i, emotion in enumerate(emotions)}\n",
                "\n",
                "# Verify configuration\n",
                "print(f'✅ Model configured for {len(emotions)} emotions')\n",
                "print(f'✅ id2label: {model.config.id2label}')\n",
                "print(f'✅ label2id: {model.config.label2id}')\n",
                "\n",
                "# Move to GPU if available\n",
                "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
                "model = model.to(device)\n",
                "print(f'✅ Model moved to: {device}')"
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
                "# Simple preprocessing without datasets library\n",
                "print('📝 PREPROCESSING DATA')\n",
                "print('=' * 40)\n",
                "\n",
                "# Split data\n",
                "train_texts, val_texts, train_labels, val_labels = train_test_split(\n",
                "    texts, labels, test_size=0.2, random_state=42, stratify=labels\n",
                ")\n",
                "\n",
                "print(f'📊 Training samples: {len(train_texts)}')\n",
                "print(f'📊 Validation samples: {len(val_texts)}')\n",
                "\n",
                "# Tokenize training data\n",
                "train_encodings = tokenizer(\n",
                "    train_texts,\n",
                "    truncation=True,\n",
                "    padding=True,\n",
                "    max_length=128,\n",
                "    return_tensors='pt'\n",
                ")\n",
                "\n",
                "# Tokenize validation data\n",
                "val_encodings = tokenizer(\n",
                "    val_texts,\n",
                "    truncation=True,\n",
                "    padding=True,\n",
                "    max_length=128,\n",
                "    return_tensors='pt'\n",
                ")\n",
                "\n",
                "# Create simple dataset class\n",
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
                "# Create datasets\n",
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
                "## ⚙️ TRAINING ARGUMENTS"
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
                "    output_dir='./ultimate_emotion_model',\n",
                "    num_train_epochs=5,\n",
                "    per_device_train_batch_size=8,\n",
                "    per_device_eval_batch_size=8,\n",
                "    warmup_steps=100,\n",
                "    weight_decay=0.01,\n",
                "    logging_dir='./logs',\n",
                "    logging_steps=10,\n",
                "    evaluation_strategy='steps',\n",
                "    eval_steps=50,\n",
                "    save_strategy='steps',\n",
                "    save_steps=100,\n",
                "    load_best_model_at_end=True,\n",
                "    metric_for_best_model='f1',\n",
                "    greater_is_better=True,\n",
                "    report_to='wandb',\n",
                "    run_name='ultimate_emotion_model'\n",
                ")\n",
                "\n",
                "print('✅ Training arguments configured')"
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
                "# Compute metrics function\n",
                "def compute_metrics(eval_pred):\n",
                "    \"\"\"Compute evaluation metrics.\"\"\"\n",
                "    predictions, labels = eval_pred\n",
                "    predictions = np.argmax(predictions, axis=1)\n",
                "    \n",
                "    return {\n",
                "        'f1': f1_score(labels, predictions, average='weighted'),\n",
                "        'accuracy': accuracy_score(labels, predictions),\n",
                "        'precision': precision_score(labels, predictions, average='weighted'),\n",
                "        'recall': recall_score(labels, predictions, average='weighted')\n",
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
                "trainer = WeightedLossTrainer(\n",
                "    model=model,\n",
                "    args=training_args,\n",
                "    train_dataset=train_dataset,\n",
                "    eval_dataset=val_dataset,\n",
                "    tokenizer=tokenizer,\n",
                "    compute_metrics=compute_metrics,\n",
                "    focal_alpha=1,\n",
                "    focal_gamma=2,\n",
                "    class_weights=class_weights_tensor\n",
                ")\n",
                "\n",
                "print('✅ Trainer initialized with focal loss and class weighting')\n",
                "\n",
                "# Start training\n",
                "print('🚀 STARTING ULTIMATE TRAINING')\n",
                "print('=' * 50)\n",
                "print(f'🎯 Target: 75-85% F1 score')\n",
                "print(f'📊 Training samples: {len(train_dataset)}')\n",
                "print(f'🧪 Validation samples: {len(val_dataset)}')\n",
                "print(f'⚖️ Using focal loss + class weighting')\n",
                "print(f'🔧 Model: {specialized_model_name}')\n",
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
                "print(f'Precision: {results[\"eval_precision\"]:.4f}')\n",
                "print(f'Recall: {results[\"eval_recall\"]:.4f}')\n",
                "\n",
                "print('✅ Evaluation completed!')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 🧪 ADVANCED VALIDATION"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Advanced validation on diverse examples\n",
                "print('🧪 ADVANCED VALIDATION')\n",
                "print('=' * 40)\n",
                "\n",
                "# Test examples\n",
                "test_examples = [\n",
                "    'I am feeling anxious about the presentation tomorrow.',\n",
                "    'I feel calm and peaceful after meditation.',\n",
                "    'I am excited about the new job opportunity!',\n",
                "    'I feel frustrated with the technical issues.',\n",
                "    'I am grateful for all the support I received.',\n",
                "    'I feel happy about the successful completion.',\n",
                "    'I am hopeful for a better future.',\n",
                "    'I feel overwhelmed with all the responsibilities.',\n",
                "    'I am proud of my achievements.',\n",
                "    'I feel sad about the recent loss.',\n",
                "    'I am tired from working long hours.',\n",
                "    'I feel content with my current situation.'\n",
                "]\n",
                "\n",
                "print('🔍 Testing on diverse examples:')\n",
                "for i, example in enumerate(test_examples):\n",
                "    inputs = tokenizer(example, return_tensors='pt', truncation=True, padding=True)\n",
                "    inputs = {k: v.to(device) for k, v in inputs.items()}\n",
                "    \n",
                "    with torch.no_grad():\n",
                "        outputs = model(**inputs)\n",
                "        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)\n",
                "        predicted_class = torch.argmax(predictions, dim=-1).item()\n",
                "        confidence = predictions[0][predicted_class].item()\n",
                "    \n",
                "    print(f'{i+1:2d}. \"{example}\" → {emotions[predicted_class]} ({confidence:.3f})')\n",
                "\n",
                "print('✅ Advanced validation completed!')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 💾 MODEL SAVING WITH VERIFICATION"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Save model with verification\n",
                "print('💾 SAVING MODEL WITH VERIFICATION')\n",
                "print('=' * 50)\n",
                "\n",
                "# Save the model\n",
                "model_path = './ultimate_emotion_model_final'\n",
                "trainer.save_model(model_path)\n",
                "tokenizer.save_pretrained(model_path)\n",
                "\n",
                "print(f'✅ Model saved to: {model_path}')\n",
                "\n",
                "# Verify the saved configuration\n",
                "print('\\n🔍 VERIFYING SAVED CONFIGURATION:')\n",
                "config_path = f'{model_path}/config.json'\n",
                "with open(config_path, 'r') as f:\n",
                "    config = json.load(f)\n",
                "\n",
                "print(f'Model type: {config.get(\"model_type\", \"NOT SET\")}')\n",
                "print(f'Number of labels: {config.get(\"num_labels\", \"NOT SET\")}')\n",
                "print(f'id2label: {config.get(\"id2label\", \"NOT SET\")}')\n",
                "print(f'label2id: {config.get(\"label2id\", \"NOT SET\")}')\n",
                "\n",
                "# Test loading the saved model\n",
                "print('\\n🧪 TESTING SAVED MODEL:')\n",
                "test_tokenizer = AutoTokenizer.from_pretrained(model_path)\n",
                "test_model = AutoModelForSequenceClassification.from_pretrained(model_path)\n",
                "\n",
                "test_input = 'I feel happy about the results!'\n",
                "test_encoding = test_tokenizer(test_input, return_tensors='pt', truncation=True, padding=True)\n",
                "test_encoding = {k: v.to(device) for k, v in test_encoding.items()}\n",
                "\n",
                "with torch.no_grad():\n",
                "    test_outputs = test_model(**test_encoding)\n",
                "    test_predictions = torch.nn.functional.softmax(test_outputs.logits, dim=-1)\n",
                "    test_predicted_class = torch.argmax(test_predictions, dim=-1).item()\n",
                "    test_confidence = test_predictions[0][test_predicted_class].item()\n",
                "\n",
                "print(f'Test input: \"{test_input}\"')\n",
                "print(f'Predicted emotion: {test_model.config.id2label[test_predicted_class]}')\n",
                "print(f'Confidence: {test_confidence:.3f}')\n",
                "\n",
                "print('\\n✅ Model saving and verification completed!')"
            ]
        }
    ]

    # Add all new cells
    notebook['cells'].extend(new_cells)

    # Save the completed notebook
    with open('notebooks/SIMPLE_ULTIMATE_BULLETPROOF_TRAINING_COLAB.ipynb', 'w') as f:
        json.dump(notebook, f, indent=2)

    print('✅ Completed simple notebook with ALL components!')
    print('📋 Added components:')
    print('   ✅ Focal Loss implementation')
    print('   ✅ Class weighting & WeightedLossTrainer')
    print('   ✅ Model loading & configuration')
    print('   ✅ Data preprocessing (simple approach)')
    print('   ✅ Training arguments')
    print('   ✅ Compute metrics')
    print('   ✅ Training execution')
    print('   ✅ Evaluation')
    print('   ✅ Advanced validation')
    print('   ✅ Model saving with verification')
    print('\\n🚀 The notebook is now COMPLETE and ready to use!')

if __name__ == "__main__":
    complete_simple_notebook()
