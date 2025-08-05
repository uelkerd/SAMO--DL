#!/usr/bin/env python3
"""
Fix Training Arguments
=====================

This script fixes the training arguments in the simple notebook to remove
unsupported parameters like evaluation_strategy.
"""

import json

def fix_training_arguments():
    """Fix the training arguments in the simple notebook."""
    
    # Read the existing notebook
    with open('notebooks/SIMPLE_ULTIMATE_BULLETPROOF_TRAINING_COLAB.ipynb', 'r') as f:
        notebook = json.load(f)
    
    # Find and replace the training arguments cell
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code' and 'TrainingArguments(' in ''.join(cell['source']):
            # Replace with fixed training arguments
            cell['source'] = [
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
                "    eval_steps=50,\n",
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
            break
    
    # Save the updated notebook
    with open('notebooks/SIMPLE_ULTIMATE_BULLETPROOF_TRAINING_COLAB.ipynb', 'w') as f:
        json.dump(notebook, f, indent=2)
    
    print('✅ Fixed training arguments in simple notebook!')
    print('📋 Changes made:')
    print('   ✅ Removed evaluation_strategy parameter')
    print('   ✅ Removed save_strategy parameter')
    print('   ✅ Kept all other parameters intact')

if __name__ == "__main__":
    fix_training_arguments() 