#!/usr/bin/env python3
"""
SAMO Deep Learning - Colab Compatibility Debug Script

This script helps identify and fix PyTorch/Transformers compatibility issues
that commonly occur in Google Colab environments.

Usage:
    python scripts/debug_colab_compatibility.py
"""

import sys
import subprocess
import warnings

warnings.filterwarnings('ignore')

def run_command(command, description):
    """Run a command and return success status."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} successful")
            return True, result.stdout
        else:
            print(f"❌ {description} failed: {result.stderr}")
            return False, result.stderr
    except Exception as e:
        print(f"❌ {description} failed: {e}")
        return False, str(e)

def check_python_version():
    """Check Python version compatibility."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print("✅ Python version is compatible")
        return True
    else:
        print("❌ Python version may be incompatible (recommend 3.8+)")
        return False

def check_gpu_availability():
    """Check GPU availability and CUDA compatibility."""
    print("🖥️ Checking GPU availability...")
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA available")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(
                  f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
                 )
            print(f"CUDA version: {torch.version.cuda}")
            return True
        else:
            print("⚠️ CUDA not available - will use CPU")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def check_pytorch_installation():
    """Check PyTorch installation and compatibility."""
    print("🔍 Checking PyTorch installation...")
    
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        
        # Test basic operations
        x = torch.randn(2, 2)
        y = torch.randn(2, 2)
        z = torch.mm(x, y)
        print("✅ Basic PyTorch operations work")
        
        # Test CUDA operations if available
        if torch.cuda.is_available():
            x_cuda = x.cuda()
            y_cuda = y.cuda()
            z_cuda = torch.mm(x_cuda, y_cuda)
            print("✅ CUDA operations work")
        
        return True
    except Exception as e:
        print(f"❌ PyTorch test failed: {e}")
        return False

def check_transformers_installation():
    """Check Transformers installation and compatibility."""
    print("🤗 Checking Transformers installation...")
    
    try:
        import transformers
        print(f"Transformers: {transformers.__version__}")
        
        # Test basic imports
        from transformers import AutoModel, AutoTokenizer
        print("✅ Transformers imports successful")
        
        # Test model loading
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModel.from_pretrained("bert-base-uncased")
        print("✅ Model loading successful")
        
        return True
    except Exception as e:
        print(f"❌ Transformers test failed: {e}")
        return False

def check_triton_compatibility():
    """Check Triton compatibility (common source of errors)."""
    print("🔧 Checking Triton compatibility...")
    
    try:
        import torch
        
        # Check if Triton is available
        if hasattr(torch, 'sparse') and hasattr(torch.sparse, '_triton_ops_meta'):
            print("✅ Triton ops available")
            return True
        else:
            print("⚠️ Triton ops not available - this may cause issues")
            
            # Try to import triton directly
            try:
                import triton
                print(f"Triton version: {triton.__version__}")
                return True
            except ImportError:
                print("❌ Triton not installed")
                return False
    except Exception as e:
        print(f"❌ Triton check failed: {e}")
        return False

def fix_pytorch_installation():
    """Fix PyTorch installation issues."""
    print("🔧 Fixing PyTorch installation...")
    
    # Uninstall existing PyTorch
    success, _ = run_command(
        "pip uninstall torch torchvision torchaudio -y",
        "Uninstalling existing PyTorch"
    )
    
    if not success:
        print("⚠️ Failed to uninstall PyTorch")
    
    # Install compatible PyTorch
    success, _ = run_command(
"pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url
https://download.pytorch.org/whl/cu118",
        "Installing compatible PyTorch"
    )
    
    if success:
        print("✅ PyTorch installation fixed")
        return True
    else:
        print("❌ PyTorch installation failed")
        return False

def fix_transformers_installation():
    """Fix Transformers installation issues."""
    print("🔧 Fixing Transformers installation...")
    
    # Uninstall existing Transformers
    success, _ = run_command(
        "pip uninstall transformers -y",
        "Uninstalling existing Transformers"
    )
    
    if not success:
        print("⚠️ Failed to uninstall Transformers")
    
    # Install compatible Transformers
    success, _ = run_command(
        "pip install transformers==4.30.0",
        "Installing compatible Transformers"
    )
    
    if success:
        print("✅ Transformers installation fixed")
        return True
    else:
        print("❌ Transformers installation failed")
        return False

def test_model_initialization():
    """Test model initialization to catch common errors."""
    print("🧪 Testing model initialization...")
    
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
        
        # Test tokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        print("✅ Tokenizer loaded")
        
        # Test model
        model = AutoModel.from_pretrained("bert-base-uncased")
        print("✅ Model loaded")
        
        # Test forward pass
        inputs = tokenizer("Hello world", return_tensors="pt")
        outputs = model(**inputs)
        print("✅ Forward pass successful")
        
        # Test GPU if available
        if torch.cuda.is_available():
            model = model.cuda()
            inputs = {k: v.cuda() for k, v in inputs.items()}
            outputs = model(**inputs)
            print("✅ GPU forward pass successful")
        
        return True
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_dataset_loading():
    """Check dataset loading capabilities."""
    print("📊 Checking dataset loading...")
    
    try:
        from datasets import load_dataset
        
        # Test loading GoEmotions
        dataset = load_dataset("go_emotions", "simplified")
        print(f"✅ GoEmotions dataset loaded: {len(dataset['train'])} samples")
        
        # Test journal dataset
        import json
        with open('data/journal_test_dataset.json', 'r') as f:
            journal_data = json.load(f)
        print(f"✅ Journal dataset loaded: {len(journal_data)} samples")
        
        return True
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return False

def generate_compatibility_report():
    """Generate a comprehensive compatibility report."""
    print("📋 Generating compatibility report...")
    
    report = {
        "python_version": check_python_version(),
        "gpu_available": check_gpu_availability(),
        "pytorch_working": check_pytorch_installation(),
        "transformers_working": check_transformers_installation(),
        "triton_compatible": check_triton_compatibility(),
        "model_initialization": test_model_initialization(),
        "dataset_loading": check_dataset_loading()
    }
    
    print("\n" + "="*50)
    print("COMPATIBILITY REPORT")
    print("="*50)
    
    for test, result in report.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test.replace('_', ' ').title()}: {status}")
    
    all_passed = all(report.values())
    print(f"\nOverall Status: {'✅ READY' if all_passed else '❌ NEEDS FIXES'}")
    
    if not all_passed:
        print("\n🔧 Recommended fixes:")
        if not report["pytorch_working"]:
            print("- Run: fix_pytorch_installation()")
        if not report["transformers_working"]:
            print("- Run: fix_transformers_installation()")
        if not report["triton_compatible"]:
            print("- Consider reinstalling PyTorch with Triton support")
    
    return report

def main():
    """Main debugging function."""
    print("🚀 SAMO Deep Learning - Colab Compatibility Debug")
    print("="*50)
    
    # Check if we're in Colab
    try:
        import google.colab
        print("✅ Running in Google Colab")
    except ImportError:
        print("⚠️ Not running in Google Colab")
    
    # Generate report
    report = generate_compatibility_report()
    
    # Offer fixes
    if not report["pytorch_working"]:
        print("\n🔧 Would you like to fix PyTorch installation? (y/n)")
        response = input().lower()
        if response == 'y':
            fix_pytorch_installation()
    
    if not report["transformers_working"]:
        print("\n🔧 Would you like to fix Transformers installation? (y/n)")
        response = input().lower()
        if response == 'y':
            fix_transformers_installation()
    
    print("\n🎯 Debug complete!")
    print("📋 If issues persist, try:")
    print("  1. Restart Colab runtime")
    print("  2. Use the fixed notebook: domain_adaptation_gpu_training_fixed.ipynb")
    print("  3. Check the Colab GPU development guide")

if __name__ == "__main__":
    main() 