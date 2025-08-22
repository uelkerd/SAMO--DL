#!/usr/bin/env python3
"""
Simple ONNX Conversion for Current Model
Handles the actual model architecture we have
"""
import sys
import torch
import logging
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path__file__.parent.parent.parent / 'src'))

from models.emotion_detection.bert_classifier import create_bert_emotion_classifier
from transformers import AutoTokenizer

logging.basicConfiglevel=logging.INFO
logger = logging.getLogger__name__


def convert_model_to_onnxmodel_path=None, onnx_output_path=None, tokenizer_name="bert-base-uncased":
    """Convert PyTorch model to ONNX format."""
    try:
        # Default paths if not provided
        if model_path is None:
            model_path = "models/best_simple_model.pth"
        if onnx_output_path is None:
            onnx_output_path = "deployment/cloud-run/model/bert_emotion_classifier.onnx"

        # Create output directory
        output_dir = Pathonnx_output_path.parent
        output_dir.mkdirparents=True, exist_ok=True

        # Load model
        logger.info"🔄 Loading PyTorch model..."
        device = torch.device"cpu"
        checkpoint = torch.loadmodel_path, map_location=device

        # Create model with simple architecture
        model, _ = create_bert_emotion_classifier()

        # Load state dict with strict=False to handle missing keys
        model.load_state_dictcheckpoint, strict=False
        model.eval()

        # Load tokenizer
        logger.info"🔄 Loading tokenizer..."
        tokenizer = AutoTokenizer.from_pretrainedtokenizer_name

        # Save tokenizer files
        tokenizer.save_pretrained(stroutput_dir)
        logger.info"✅ Tokenizer saved"

        # Create dummy input
        logger.info"🔄 Creating dummy input for ONNX export..."
        dummy_text = "This is a test sentence for ONNX conversion."
        inputs = tokenizer(
            dummy_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )

        # Handle token_type_ids properly - use actual values from tokenizer
        # This ensures compatibility with models that require specific token_type_ids
        if "token_type_ids" in inputs:
            token_type_ids = inputs["token_type_ids"]
        else:
            # For models that don't use token_type_ids, create proper sequence
            # Use actual tokenizer output to ensure correctness
            tokenizer_output = tokenizer(
                dummy_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
                return_token_type_ids=True  # Explicitly request token_type_ids
            )
            token_type_ids = tokenizer_output.get("token_type_ids", torch.zeros_likeinputs["input_ids"])

        # Export to ONNX
        logger.info"🔄 Converting to ONNX format..."
        torch.onnx.export(
            model,
            (
                inputs["input_ids"],
                inputs["attention_mask"],
                token_type_ids
            ),
            onnx_output_path,
            export_params=True,
            opset_version=14,
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

        logger.infof"✅ ONNX model saved to {onnx_output_path}"

        # Test ONNX model with ONNX Runtime
        try:
            import onnxruntime as ort
            session = ort.InferenceSessiononnx_output_path
            logger.info"✅ ONNX model test successful"
        except ImportError:
            logger.error"❌ ONNX Runtime is required for ONNX model validation. Please install it with 'pip install onnxruntime'."
            return False
        except Exception as e:
            logger.errorf"❌ ONNX model validation failed: {e}"
            return False

        return True

    except Exception as e:
        logger.errorf"❌ ONNX conversion failed: {e}"
        return False


def main():
    """Main function with command-line argument parsing."""
    parser = argparse.ArgumentParser(description="Convert PyTorch model to ONNX format simple version")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/best_simple_model.pth",
        help="Path to PyTorch model file default: models/best_simple_model.pth"
    )
    parser.add_argument(
        "--onnx-output-path",
        type=str,
        default="deployment/cloud-run/model/bert_emotion_classifier.onnx",
        help="Path for ONNX output file default: deployment/cloud-run/model/bert_emotion_classifier.onnx"
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default="bert-base-uncased",
        help="Tokenizer name to use default: bert-base-uncased"
    )

    args = parser.parse_args()

    if success := convert_model_to_onnx(
        model_path=args.model_path,
        onnx_output_path=args.onnx_output_path,
        tokenizer_name=args.tokenizer_name
    ):
        logger.info"🎉 Simple ONNX conversion completed successfully!"
        sys.exit0
    else:
        logger.error"💥 Simple ONNX conversion failed!"
        sys.exit1


if __name__ == "__main__":
    main() 
