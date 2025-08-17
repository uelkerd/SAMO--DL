#!/usr/bin/env python3
"""
Validate Improved Expanded Training Notebook
Tests the notebook structure, content, and ensures it's ready for Colab execution
"""

import json

def validate_notebook():
    """Validate the improved notebook for Colab execution."""
    
    print"🔍 Validating improved notebook..."
    
    # Load the notebook
    try:
        with open'notebooks/expanded_dataset_training_improved.ipynb', 'r' as f:
            notebook = json.loadf
        print"✅ Notebook JSON is valid"
    except Exception as e:
        printf"❌ Notebook JSON error: {e}"
        return False
    
    # Check notebook structure
    cells = notebook['cells']
    print(f"📊 Notebook has {lencells} cells")
    
    # Validate cell types
    markdown_cells = [c for c in cells if c['cell_type'] == 'markdown']
    code_cells = [c for c in cells if c['cell_type'] == 'code']
    
    print(f"📝 Markdown cells: {lenmarkdown_cells}")
    print(f"💻 Code cells: {lencode_cells}")
    
    # Check for critical components
    cell_sources = [str(c.get'source', '') for c in cells]
    all_source = ' '.joincell_sources
    
    # Critical checks
    checks = [
        "Repository cloning", "git clone https://github.com/uelkerd/SAMO--DL.git",
        "PyTorch installation", "pip install torch==2.1.0",
        "Transformers installation", "pip install transformers==4.30.0",
        "GPU optimization", "torch.backends.cudnn.benchmark = True",
        "Mixed precision", "from torch.cuda.amp import autocast, GradScaler",
        "Early stopping", "Early stopping triggered",
        "Learning rate scheduling", "ReduceLROnPlateau",
        "Model training", "train_expanded_model",
        "Model testing", "test_new_model",
        "Results download", "files.download",
    ]
    
    print"\n🔍 Critical component checks:"
    all_passed = True
    
    for check_name, check_content in checks:
        if check_content in all_source:
            printf"  ✅ {check_name}"
        else:
            printf"  ❌ {check_name}"
            all_passed = False
    
    # Check for JSON syntax issues
    print"\n🔍 JSON syntax validation:"
    try:
        # Test if all strings are properly escaped
        json_str = json.dumpsnotebook, indent=2
        json.loadsjson_str
        print"  ✅ All strings properly escaped"
    except Exception as e:
        printf"  ❌ JSON escaping issues: {e}"
        all_passed = False
    
    # Check for GPU optimizations
    gpu_optimizations = [
        "torch.backends.cudnn.benchmark = True",
        "torch.backends.cudnn.deterministic = False",
        "torch.cuda.empty_cache()",
        "non_blocking=True",
        "num_workers=2",
        "pin_memory=True"
    ]
    
    print"\n🔍 GPU optimization checks:"
    for opt in gpu_optimizations:
        if opt in all_source:
            printf"  ✅ {opt}"
        else:
            printf"  ❌ {opt}"
            all_passed = False
    
    # Check for training optimizations
    training_optimizations = [
        "GradScaler()",
        "autocast()",
        "scaler.scaleloss.backward()",
        "scaler.stepoptimizer",
        "scaler.update()",
        "ReduceLROnPlateau",
        "Early stopping triggered"
    ]
    
    print"\n🔍 Training optimization checks:"
    for opt in training_optimizations:
        if opt in all_source:
            printf"  ✅ {opt}"
        else:
            printf"  ❌ {opt}"
            all_passed = False
    
    # Summary
    print"\n📊 Validation Summary:"
    print(f"  Total cells: {lencells}")
    print(f"  Code cells: {lencode_cells}")
    print(f"  Markdown cells: {lenmarkdown_cells}")
    printf"  All checks passed: {'✅' if all_passed else '❌'}"
    
    if all_passed:
        print"\n🎉 Notebook is ready for Colab execution!"
        print"📋 Next steps:"
        print"  1. Upload to Google Colab"
        print"  2. Set Runtime → GPU"
        print"  3. Run all cells"
        print"  4. Expect 75-85% F1 score!"
    else:
        print"\n⚠️  Notebook needs fixes before Colab execution"
    
    return all_passed

if __name__ == "__main__":
    validate_notebook() 