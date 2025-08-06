#!/usr/bin/env python3
"""
Convert PyTorch Model to ONNX for Deployment
Quick conversion script to eliminate PyTorch dependencies
"""

import os
import sys
import torch
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

from models.emotion_detection.bert_classifier import create_bert_emotion_classifier
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_model_to_onnx():
    """Convert PyTorch model to ONNX format."""
    try:
        # Paths
        model_path = "models/best_simple_model.pth"
        onnx_output_path = "deployment/cloud-run/model/bert_emotion_classifier.onnx"
        tokenizer_output_path = "deployment/cloud-run/model/tokenizer.json"
        
        # Create output directory
        os.makedirs("deployment/cloud-run/model", exist_ok=True)
        
        # Load model
        logger.info("🔄 Loading PyTorch model...")
        device = torch.device("cpu")
        checkpoint = torch.load(model_path, map_location=device)
        
        model, _ = create_bert_emotion_classifier()
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
        else:
            # Direct model state dict
            model.load_state_dict(checkpoint)
        
        model.eval()
        
        # Load tokenizer
        logger.info("🔄 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        # Save tokenizer
        tokenizer.save_pretrained("deployment/cloud-run/model/")
        logger.info(f"✅ Tokenizer saved to {tokenizer_output_path}")
        
        # Create dummy input
        logger.info("🔄 Creating dummy input for ONNX export...")
        dummy_text = "This is a test sentence for ONNX conversion."
        inputs = tokenizer(
            dummy_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        
        # Export to ONNX
        logger.info("🔄 Converting to ONNX format...")
        torch.onnx.export(
            model,
            (
                inputs["input_ids"],
                inputs["attention_mask"],
                inputs.get("token_type_ids", torch.zeros_like(inputs["input_ids"]))
            ),
            onnx_output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size"},
                "attention_mask": {0: "batch_size"},
                "token_type_ids": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
        )
        
        logger.info(f"✅ ONNX model saved to {onnx_output_path}")
        
        # Verify ONNX model
        try:
            import onnx
            onnx_model = onnx.load(onnx_output_path)
            onnx.checker.check_model(onnx_model)
            logger.info("✅ ONNX model verification passed")
        except ImportError:
            logger.warning("⚠️ ONNX not available for verification")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ ONNX conversion failed: {e}")
        return False


if __name__ == "__main__":
    logger.info("🚀 Starting ONNX conversion...")
    success = convert_model_to_onnx()
    
    if success:
        logger.info("✅ ONNX conversion completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ ONNX conversion failed!")
        sys.exit(1) 