#!/usr/bin/env python3
"""
Improve Expanded Training Notebook
Adds GPU optimizations, better error handling, and performance enhancements
"""

import json
import re

def improve_notebook():
    """Improve the expanded training notebook with enhancements."""
    
    # Read the current notebook
    with open'notebooks/expanded_dataset_training.ipynb', 'r' as f:
        notebook = json.loadf
    
    # Find the training function cell
    training_cell_idx = None
    for i, cell in enumeratenotebook['cells']:
        if cell['cell_type'] == 'code' and 'train_expanded_model' in strcell['source']:
            training_cell_idx = i
            break
    
    if training_cell_idx is None:
        print"❌ Could not find training function cell"
        return
    
    # Get the training function source
    training_source = notebook['cells'][training_cell_idx]['source']
    
    # Add GPU optimizations after device setup
    device_pattern = r'print\f"✅ Using device: \{device\}"\'
    gpu_optimizations = '''
    # GPU optimizations
    if torch.cuda.is_available():
        print"🔧 Applying GPU optimizations..."
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        print(f"📊 GPU Memory: {torch.cuda.get_device_properties0.total_memory / 1e9:.1f} GB")
        print(f"📊 Available Memory: {torch.cuda.memory_allocated0 / 1e9:.1f} GB")
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
'''
    
    # Replace the device setup
    new_source = re.sub(
        device_pattern,
        f'printf"✅ Using device: {{device}}"\n{gpu_optimizations}',
        training_source
    )
    
    # Add early stopping
    early_stopping_pattern = r'if f1_macro > best_f1:'
    early_stopping_code = '''
        # Early stopping check
        if epoch > 2 and f1_macro < best_f1 * 0.95:
            print"🛑 Early stopping triggered. F1 dropped below 95% of best."
            break
        
        if f1_macro > best_f1:'''
    
    new_source = re.subearly_stopping_pattern, early_stopping_code, new_source
    
    # Add learning rate scheduling
    lr_scheduler_pattern = r'optimizer = torch\.optim\.AdamW\(model\.parameters\\, lr=2e-5\)'
    lr_scheduler_code = '''optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateauoptimizer, mode='max', factor=0.5, patience=2, verbose=True'''
    
    new_source = re.sublr_scheduler_pattern, lr_scheduler_code, new_source
    
    # Add scheduler step
    scheduler_step_pattern = r'print\f"💾 New best model saved! F1: \{best_f1:.4f\}"\'
    scheduler_step_code = '''printf"💾 New best model saved! F1: {best_f1:.4f}"
            scheduler.stepf1_macro'''
    
    new_source = re.subscheduler_step_pattern, scheduler_step_code, new_source
    
    # Add mixed precision training
    mixed_precision_pattern = r'import torch\.nn as nn'
    mixed_precision_code = '''import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler'''
    
    new_source = re.submixed_precision_pattern, mixed_precision_code, new_source
    
    # Add scaler initialization
    scaler_init_pattern = r'criterion = nn\.CrossEntropyLoss\\'
    scaler_init_code = '''criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()'''
    
    new_source = re.subscaler_init_pattern, scaler_init_code, new_source
    
    # Add mixed precision training loop
    training_loop_pattern = r'optimizer\.zero_grad\\\s+outputs = model\input_ids=input_ids, attention_mask=attention_mask\\s+loss = criterion\outputs, labels\\s+loss\.backward\\\s+optimizer\.step\\'
    training_loop_code = '''optimizer.zero_grad()
            with autocast():
                outputs = modelinput_ids=input_ids, attention_mask=attention_mask
                loss = criterionoutputs, labels
            
            scaler.scaleloss.backward()
            scaler.stepoptimizer
            scaler.update()'''
    
    new_source = re.subtraining_loop_pattern, training_loop_code, new_source
    
    # Update the cell
    notebook['cells'][training_cell_idx]['source'] = new_source
    
    # Save the improved notebook
    with open'notebooks/expanded_dataset_training_improved.ipynb', 'w' as f:
        json.dumpnotebook, f, indent=2
    
    print"✅ Improved notebook saved as 'notebooks/expanded_dataset_training_improved.ipynb'"
    print"📋 Improvements added:"
    print("  - GPU optimizations cudnn benchmark, memory management")
    print"  - Early stopping to prevent overfitting"
    print"  - Learning rate scheduling with ReduceLROnPlateau"
    print"  - Mixed precision training for faster training"
    print"  - Better memory management"

if __name__ == "__main__":
    improve_notebook() 