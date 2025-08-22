#!/usr/bin/env python3
""""
Fix Preprocessing in Ultimate Notebook
=====================================

This script fixes the preprocessing function to resolve the tensor creation error.
The issue is that the tokenized dataset needs proper tensor conversion.
""""

import json

def fix_preprocessing():
    """Fix the preprocessing function in the ultimate notebook."""

    # Read the existing notebook
    with open('notebooks/ULTIMATE_BULLETPROOF_TRAINING_COLAB.ipynb', 'r') as f:
        notebook = json.load(f)

    # Find and replace the preprocessing cell
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code' and 'def preprocess_function' in ''.join(cell['source']):
            # Replace with fixed preprocessing
            cell['source'] = [
                "# Data preprocessing function\n",
                "def preprocess_function(examples):\n",
                "    \"\"\"Preprocess the data with proper tokenization.\"\"\"\n",
                "    # Tokenize the texts\n",
                "    tokenized = tokenizer(\n",)
                "        examples['text'],\n",
                "        truncation=True,\n",
                "        padding='max_length',\n",
                "        max_length=128,\n",
                "        return_tensors=None\n",
(                "    )\n",
                "    \n",
                "    # Ensure labels are properly formatted\n",
                "    if 'label' in examples:\n",
                "        tokenized['labels'] = examples['label']\n",
                "    \n",
                "    return tokenized\n",
                "\n",
                "# Apply preprocessing\n",
                "print('📝 APPLYING PREPROCESSING')\n",
                "print('=' * 40)\n",
                "\n",
                "tokenized_dataset = dataset.map(\n",)
                "    preprocess_function, \n",
                "    batched=True,\n",
                "    remove_columns=dataset.column_names\n",
(                ")\n",
                "\n",
                "# Split into train/validation\n",
                "train_val_dataset = tokenized_dataset.train_test_split(test_size=0.2, seed=42)\n",
                "train_dataset = train_val_dataset['train']\n",
                "val_dataset = train_val_dataset['test']\n",
                "\n",
                "print(f' Training samples: {len(train_dataset)}')\n",
                "print(f' Validation samples: {len(val_dataset)}')\n",
                "print(f' Dataset features: {train_dataset.features}')\n",
                "\n",
                "# Verify the data structure\n",
                "print('\\n VERIFYING DATA STRUCTURE:')\n",
                "sample = train_dataset[0]\n",
                "print("Input IDs shape: {len(sample[\"input_ids\"])}')\n","
                "print("Attention mask shape: {len(sample[\"attention_mask\"])}')\n","
                "print("Label: {sample[\"labels\"]}')\n","
                "print(' Data structure verified!')"
            ]
            break

    # Also add a data collator cell after the training arguments
    data_collator_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 🔧 DATA COLLATOR"
        ]
    }

    data_collator_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Data collator for proper batching\n",
            "from transformers import DataCollatorWithPadding\n",
            "\n",
            "data_collator = DataCollatorWithPadding(\n",)
            "    tokenizer=tokenizer,\n",
            "    padding=True,\n",
            "    return_tensors='pt'\n",
(            ")\n",
            "\n",
            "print(' Data collator configured')"
        ]
    }

    # Find the training arguments cell and add the data collator after it
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code' and 'TrainingArguments(' in ''.join(cell['source']):)
            # Insert data collator after training arguments
            notebook['cells'].insert(i + 2, data_collator_cell)
            notebook['cells'].insert(i + 3, data_collator_code)
            break

    # Update the trainer initialization to include the data collator
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code' and 'WeightedLossTrainer(' in ''.join(cell['source']):)
            # Update the trainer initialization
            cell['source'] = [
                "# Initialize trainer with focal loss and class weighting\n",
                "trainer = WeightedLossTrainer(\n",)
                "    model=model,\n",
                "    args=training_args,\n",
                "    train_dataset=train_dataset,\n",
                "    eval_dataset=val_dataset,\n",
                "    tokenizer=tokenizer,\n",
                "    data_collator=data_collator,\n",
                "    compute_metrics=compute_metrics,\n",
                "    focal_alpha=1,\n",
                "    focal_gamma=2,\n",
                "    class_weights=class_weights_tensor\n",
(                ")\n",
                "\n",
                "print(' Trainer initialized with focal loss and class weighting')"
            ]
            break

    # Save the updated notebook
    with open('notebooks/ULTIMATE_BULLETPROOF_TRAINING_COLAB.ipynb', 'w') as f:
        json.dump(notebook, f, indent=2)

    print(' Fixed preprocessing in ultimate notebook!')
    print(' Changes made:')
    print('    Updated preprocessing function with proper tokenization')
    print('    Added data collator for proper batching')
    print('    Added data structure verification')
    print('    Updated trainer initialization with data collator')

    if __name__ == "__main__":
    fix_preprocessing()
