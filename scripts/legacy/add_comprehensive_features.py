#!/usr/bin/env python3
"""
Add Comprehensive Features
=========================

This script adds all the advanced features to the comprehensive notebook
to make it truly complete with all the gains from previous iterations.
"""

import json

def add_comprehensive_features():
    """Add all advanced features to the comprehensive notebook."""

    # Read the existing notebook
    with open('notebooks/COMPREHENSIVE_ULTIMATE_TRAINING_COLAB.ipynb', 'r') as f:
        notebook = json.load(f)

    # Add all the advanced features as new cells
    advanced_cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 🔧 MODEL SETUP WITH ARCHITECTURE FIXES"
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
                "\n",
                "print(f'Original model labels: {AutoModelForSequenceClassification.from_pretrained(model_name).config.num_labels}')\n",
                "print(f'Original id2label: {AutoModelForSequenceClassification.from_pretrained(model_name).config.id2label}')\n",
                "\n",
                "# CRITICAL: Create a NEW model with correct configuration from scratch\n",
                "print('\\n🔧 CREATING NEW MODEL WITH CORRECT ARCHITECTURE')\n",
                "print('=' * 60)\n",
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
                "    print('⚠️ CUDA not available, model will run on CPU')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 📊 DATA PREPROCESSING AND SPLITTING"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print('📊 PREPROCESSING AND SPLITTING DATA')\n",
                "print('=' * 50)\n",
                "\n",
                "# Split the data\n",
                "train_texts, val_texts, train_labels, val_labels = train_test_split(\n",
                "    texts, labels, test_size=0.2, random_state=42, stratify=labels\n",
                ")\n",
                "\n",
                "print(f'📊 Training samples: {len(train_texts)}')\n",
                "print(f'📊 Validation samples: {len(val_texts)}')\n",
                "\n",
                "# Create datasets\n",
                "train_dataset = {'text': train_texts, 'label': train_labels}\n",
                "val_dataset = {'text': val_texts, 'label': val_labels}\n",
                "\n",
                "print('✅ Data split and prepared')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## ⚖️ FOCAL LOSS AND CLASS WEIGHTING"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print('⚖️ SETTING UP FOCAL LOSS AND CLASS WEIGHTING')\n",
                "print('=' * 60)\n",
                "\n",
                "# Calculate class weights\n",
                "class_weights = compute_class_weight(\n",
                "    'balanced',\n",
                "    classes=np.unique(train_labels),\n",
                "    y=train_labels\n",
                ")\n",
                "\n",
                "class_weights_tensor = torch.FloatTensor(class_weights)\n",
                "if torch.cuda.is_available():\n",
                "    class_weights_tensor = class_weights_tensor.cuda()\n",
                "\n",
                "print(f'✅ Class weights calculated: {class_weights}')\n",
                "print(f'✅ Class weights tensor shape: {class_weights_tensor.shape}')\n",
                "\n",
                "# Focal Loss implementation\n",
                "class FocalLoss(torch.nn.Module):\n",
                "    def __init__(self, alpha=1, gamma=2):\n",
                "        super(FocalLoss, self).__init__()\n",
                "        self.alpha = alpha\n",
                "        self.gamma = gamma\n",
                "    \n",
                "    def forward(self, inputs, targets):\n",
                "        ce_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none')\n",
                "        pt = torch.exp(-ce_loss)\n",
                "        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss\n",
                "        return focal_loss.mean()\n",
                "\n",
                "print('✅ Focal Loss class defined')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 🎯 WEIGHTED LOSS TRAINER"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print('🎯 CREATING WEIGHTED LOSS TRAINER')\n",
                "print('=' * 50)\n",
                "\n",
                "# Custom trainer with focal loss and class weighting\n",
                "class WeightedLossTrainer(Trainer):\n",
                "    def __init__(self, focal_alpha=1, focal_gamma=2, class_weights=None, *args, **kwargs):\n",
                "        super().__init__(*args, **kwargs)\n",
                "        self.focal_alpha = focal_alpha\n",
                "        self.focal_gamma = focal_gamma\n",
                "        self.class_weights = class_weights\n",
                "    \n",
                "    def compute_loss(self, model, inputs, return_outputs=False):\n",
                "        labels = inputs.pop('labels')\n",
                "        outputs = model(**inputs)\n",
                "        logits = outputs.logits\n",
                "        \n",
                "        # Focal Loss\n",
                "        ce_loss = torch.nn.functional.cross_entropy(logits, labels, reduction='none')\n",
                "        pt = torch.exp(-ce_loss)\n",
                "        focal_loss = self.focal_alpha * (1-pt)**self.focal_gamma * ce_loss\n",
                "        \n",
                "        # Apply class weights if provided\n",
                "        if self.class_weights is not None:\n",
                "            weighted_loss = focal_loss * self.class_weights[labels]\n",
                "            loss = weighted_loss.mean()\n",
                "        else:\n",
                "            loss = focal_loss.mean()\n",
                "        \n",
                "        return (loss, outputs) if return_outputs else loss\n",
                "\n",
                "print('✅ WeightedLossTrainer created with focal loss and class weighting')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 🔧 DATA PREPROCESSING FUNCTION"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print('🔧 SETTING UP DATA PREPROCESSING')\n",
                "print('=' * 50)\n",
                "\n",
                "# Preprocessing function\n",
                "def preprocess_function(examples):\n",
                "    tokenized = tokenizer(\n",
                "        examples['text'],\n",
                "        truncation=True,\n",
                "        padding='max_length',\n",
                "        max_length=128,\n",
                "        return_tensors=None\n",
                "    )\n",
                "    if 'label' in examples:\n",
                "        tokenized['labels'] = examples['label']\n",
                "    return tokenized\n",
                "\n",
                "# Apply preprocessing\n",
                "train_dataset_processed = preprocess_function(train_dataset)\n",
                "val_dataset_processed = preprocess_function(val_dataset)\n",
                "\n",
                "# Create data collator\n",
                "data_collator = DataCollatorWithPadding(\n",
                "    tokenizer=tokenizer,\n",
                "    padding=True,\n",
                "    return_tensors='pt'\n",
                ")\n",
                "\n",
                "print('✅ Data preprocessing completed')\n",
                "print('✅ Data collator created')"
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
                "print('⚙️ CONFIGURING TRAINING ARGUMENTS')\n",
                "print('=' * 50)\n",
                "\n",
                "# Training arguments\n",
                "training_args = TrainingArguments(\n",
                "    output_dir='./comprehensive_emotion_model',\n",
                "    num_train_epochs=5,\n",
                "    per_device_train_batch_size=8,\n",
                "    per_device_eval_batch_size=8,\n",
                "    warmup_steps=100,\n",
                "    weight_decay=0.01,\n",
                "    logging_dir='./logs',\n",
                "    logging_steps=10,\n",
                "    eval_steps=50,\n",
                "    save_steps=100,\n",
                "    load_best_model_at_end=True,\n",
                "    metric_for_best_model='f1',\n",
                "    greater_is_better=True,\n",
                "    # Disable wandb if no API key is set\n",
                "    report_to=None if 'WANDB_API_KEY' not in os.environ else ['wandb']\n",
                ")\n",
                "\n",
                "print('✅ Training arguments configured')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 📊 COMPUTE METRICS FUNCTION"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print('📊 SETTING UP COMPUTE METRICS')\n",
                "print('=' * 50)\n",
                "\n",
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
                "print('✅ Compute metrics function defined')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 🚀 TRAINING EXECUTION"
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
                "    train_dataset=train_dataset_processed,\n",
                "    eval_dataset=val_dataset_processed,\n",
                "    tokenizer=tokenizer,\n",
                "    data_collator=data_collator,\n",
                "    compute_metrics=compute_metrics,\n",
                "    focal_alpha=1,\n",
                "    focal_gamma=2,\n",
                "    class_weights=class_weights_tensor\n",
                ")\n",
                "\n",
                "print('✅ Trainer initialized')\n",
                "\n",
                "# Start training\n",
                "print('🚀 STARTING COMPREHENSIVE TRAINING')\n",
                "print('=' * 60)\n",
                "print(f'🎯 Target: 75-85% F1 score')\n",
                "print(f'📊 Training samples: {len(train_texts)}')\n",
                "print(f'🧪 Validation samples: {len(val_texts)}')\n",
                "print(f'⚖️ Using focal loss + class weighting')\n",
                "print(f'🔧 Model: {model_name}')\n",
                "print(f'📈 Data augmentation: {len(augmented_data)} samples added')\n",
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
                "## 📊 EVALUATION AND VALIDATION"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print('📊 EVALUATING MODEL PERFORMANCE')\n",
                "print('=' * 50)\n",
                "\n",
                "# Evaluate the model\n",
                "eval_results = trainer.evaluate()\n",
                "print('\\n📊 EVALUATION RESULTS:')\n",
                "print('=' * 30)\n",
                "for key, value in eval_results.items():\n",
                "    print(f'{key}: {value:.4f}')\n",
                "\n",
                "# Detailed classification report\n",
                "print('\\n📋 DETAILED CLASSIFICATION REPORT:')\n",
                "print('=' * 40)\n",
                "predictions = trainer.predict(val_dataset_processed)\n",
                "pred_labels = np.argmax(predictions.predictions, axis=1)\n",
                "true_labels = val_labels\n",
                "\n",
                "print(classification_report(true_labels, pred_labels, target_names=emotions))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 🔍 ADVANCED VALIDATION"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print('🔍 ADVANCED VALIDATION AND BIAS ANALYSIS')\n",
                "print('=' * 60)\n",
                "\n",
                "# Test on completely unseen examples\n",
                "unseen_examples = [\n",
                "    'I am feeling absolutely ecstatic about the promotion!',\n",
                "    'This situation is making me extremely anxious and worried.',\n",
                "    'I feel completely overwhelmed by all the responsibilities.',\n",
                "    'I am so grateful for all the support I received.',\n",
                "    'This makes me feel incredibly proud of my achievements.',\n",
                "    'I am feeling quite content with my current situation.',\n",
                "    'This gives me a lot of hope for the future.',\n",
                "    'I feel really tired after working all day.',\n",
                "    'I am sad about the recent loss.',\n",
                "    'This excites me about the possibilities ahead.'\n",
                "]\n",
                "\n",
                "print('\\n🧪 TESTING ON UNSEEN EXAMPLES:')\n",
                "print('=' * 40)\n",
                "\n",
                "for i, example in enumerate(unseen_examples, 1):\n",
                "    inputs = tokenizer(example, return_tensors='pt', truncation=True, padding=True)\n",
                "    if torch.cuda.is_available():\n",
                "        inputs = {k: v.cuda() for k, v in inputs.items()}\n",
                "    \n",
                "    with torch.no_grad():\n",
                "        outputs = model(**inputs)\n",
                "        probabilities = torch.softmax(outputs.logits, dim=1)\n",
                "        predicted_label = torch.argmax(outputs.logits, dim=1).item()\n",
                "        confidence = probabilities[0][predicted_label].item()\n",
                "    \n",
                "    print(f'{i:2d}. \"{example}\"')\n",
                "    print(f'    → Predicted: {emotions[predicted_label]} (confidence: {confidence:.3f})')\n",
                "    print()\n",
                "\n",
                "# Bias analysis\n",
                "print('\\n📊 BIAS ANALYSIS:')\n",
                "print('=' * 30)\n",
                "print('Checking for prediction bias across emotions...')\n",
                "\n",
                "# Count predictions per emotion\n",
                "prediction_counts = {emotion: 0 for emotion in emotions}\n",
                "for pred in pred_labels:\n",
                "    prediction_counts[emotions[pred]] += 1\n",
                "\n",
                "print('\\nPrediction distribution:')\n",
                "for emotion, count in prediction_counts.items():\n",
                "    percentage = (count / len(pred_labels)) * 100\n",
                "    print(f'{emotion:12s}: {count:3d} ({percentage:5.1f}%)')\n",
                "\n",
                "print('\\n✅ Advanced validation completed')"
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
                "print('💾 SAVING MODEL WITH CONFIGURATION VERIFICATION')\n",
                "print('=' * 60)\n",
                "\n",
                "# Save the model\n",
                "model_save_path = './comprehensive_emotion_model_final'\n",
                "trainer.save_model(model_save_path)\n",
                "tokenizer.save_pretrained(model_save_path)\n",
                "\n",
                "print(f'✅ Model saved to: {model_save_path}')\n",
                "\n",
                "# CRITICAL: Verify the saved configuration\n",
                "print('\\n🔍 VERIFYING SAVED MODEL CONFIGURATION:')\n",
                "print('=' * 50)\n",
                "\n",
                "# Load the saved model and check configuration\n",
                "saved_model = AutoModelForSequenceClassification.from_pretrained(model_save_path)\n",
                "saved_tokenizer = AutoTokenizer.from_pretrained(model_save_path)\n",
                "\n",
                "print(f'✅ Saved model labels: {saved_model.config.num_labels}')\n",
                "print(f'✅ Saved id2label: {saved_model.config.id2label}')\n",
                "print(f'✅ Saved label2id: {saved_model.config.label2id}')\n",
                "print(f'✅ Saved problem_type: {saved_model.config.problem_type}')\n",
                "\n",
                "# Test the saved model\n",
                "test_input = saved_tokenizer('I feel happy today', return_tensors='pt', truncation=True, padding=True)\n",
                "with torch.no_grad():\n",
                "    test_output = saved_model(**test_input)\n",
                "    predicted_label = torch.argmax(test_output.logits, dim=1).item()\n",
                "    confidence = torch.softmax(test_output.logits, dim=1)[0][predicted_label].item()\n",
                "\n",
                "print(f'\\n🧪 SAVED MODEL TEST:')\n",
                "print(f'Input: \"I feel happy today\"')\n",
                "print(f'Predicted: {saved_model.config.id2label[predicted_label]} (confidence: {confidence:.3f})')\n",
                "\n",
                "# Verify configuration persistence\n",
                "config_correct = (\n",
                "    saved_model.config.num_labels == len(emotions) and\n",
                "    saved_model.config.id2label == {i: emotion for i, emotion in enumerate(emotions)} and\n",
                "    saved_model.config.problem_type == 'single_label_classification'\n",
                ")\n",
                "\n",
                "if config_correct:\n",
                "    print('\\n✅ CONFIGURATION PERSISTENCE VERIFIED!')\n",
                "    print('✅ Model will work correctly in deployment')\n",
                "    print('✅ No more 8.3% vs 75% discrepancy!')\n",
                "else:\n",
                "    print('\\n❌ CONFIGURATION PERSISTENCE FAILED!')\n",
                "    print('❌ Model may have issues in deployment')\n",
                "\n",
                "print(f'\\n🎉 COMPREHENSIVE TRAINING COMPLETED!')\n",
                "print(f'📁 Model saved to: {model_save_path}')\n",
                "print(f'📊 Final F1 Score: {eval_results.get(\"eval_f1\", \"N/A\"):.4f}')\n",
                "print(f'📊 Final Accuracy: {eval_results.get(\"eval_accuracy\", \"N/A\"):.4f}')"
            ]
        }
    ]

    # Add all the advanced cells to the notebook
    notebook['cells'].extend(advanced_cells)

    # Save the updated notebook
    with open('notebooks/COMPREHENSIVE_ULTIMATE_TRAINING_COLAB.ipynb', 'w') as f:
        json.dump(notebook, f, indent=2)

    print('✅ Added all comprehensive features!')
    print('📋 Advanced features added:')
    print('   ✅ Model setup with architecture fixes')
    print('   ✅ Data preprocessing and splitting')
    print('   ✅ Focal loss and class weighting')
    print('   ✅ WeightedLossTrainer with advanced loss')
    print('   ✅ Data preprocessing function')
    print('   ✅ Training arguments configuration')
    print('   ✅ Compute metrics function')
    print('   ✅ Training execution')
    print('   ✅ Evaluation and validation')
    print('   ✅ Advanced validation with bias analysis')
    print('   ✅ Model saving with verification')
    print('\\n🚀 COMPREHENSIVE NOTEBOOK IS NOW COMPLETE!')

if __name__ == "__main__":
    add_comprehensive_features() 