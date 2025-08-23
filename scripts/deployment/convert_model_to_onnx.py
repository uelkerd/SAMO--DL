#!/usr/bin/env python3
"""Convert PyTorch Model to ONNX for Deployment Quick conversion script to eliminate
PyTorch dependencies."""
import argparse
import logging
import sys
import torch
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

from models.emotion_detection.bert_classifier import create_bert_emotion_classifier
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_model_to_onnx(model_path=None, onnx_output_path=None, tokenizer_name="bert-base-uncased"):
    """Convert PyTorch model to ONNX format."""
    try:
        # Default paths if not provided
        if model_path is None:
            model_path = "models/best_simple_model.pth"
        if onnx_output_path is None:
            onnx_output_path = "deployment/cloud-run/model/bert_emotion_classifier.onnx"

        # Create output directory
        output_dir = Path(onnx_output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

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
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Save tokenizer
        tokenizer_dir = output_dir / "tokenizer"
        tokenizer.save_pretrained(str(tokenizer_dir))
        logger.info(f"✅ Tokenizer saved to {tokenizer_dir}")

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

        # Handle token_type_ids properly - use actual values if available, otherwise zeros
        if "token_type_ids" in inputs:
            token_type_ids = inputs["token_type_ids"]
        else:
            # For models that don't use token_type_ids, create zeros
            token_type_ids = torch.zeros_like(inputs["input_ids"])

        # Export to ONNX
        logger.info("🔄 Converting to ONNX format...")
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

        logger.info(f"✅ ONNX model saved to {onnx_output_path}")

        # Verify ONNX model
        try:
            import onnx
            onnx_model = onnx.load(onnx_output_path)
            onnx.checker.check_model(onnx_model)
            logger.info("✅ ONNX model validation successful")
        except ImportError:
            logger.warning("⚠️ ONNX not available for validation")
        except Exception as e:
            logger.error(f"❌ ONNX model validation failed: {e}")
            return False

        # Test ONNX model with ONNX Runtime
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(onnx_output_path)
            logger.info("✅ ONNX Runtime test successful")
        except ImportError:
            logger.error("❌ ONNX Runtime is required for ONNX model validation. Please install it with 'pip install onnxruntime'.")
            return False
        except Exception as e:
            logger.error(f"❌ ONNX Runtime test failed: {e}")
            return False

        return True

    except Exception as e:
        logger.error(f"❌ ONNX conversion failed: {e}")
        return False


def main():
    """Main function with command-line argument parsing."""
    parser = argparse.ArgumentParser(description="Convert PyTorch model to ONNX format")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/best_simple_model.pth",
        help="Path to PyTorch model file (default: models/best_simple_model.pth)"
    )
    parser.add_argument(
        "--onnx-output-path",
        type=str,
        default="deployment/cloud-run/model/bert_emotion_classifier.onnx",
        help="Path for ONNX output file (default: deployment/cloud-run/model/bert_emotion_classifier.onnx)"
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default="bert-base-uncased",
        help="Tokenizer name to use (default: bert-base-uncased)"
    )

    args = parser.parse_args()

    if success := convert_model_to_onnx(
        model_path=args.model_path,
        onnx_output_path=args.onnx_output_path,
        tokenizer_name=args.tokenizer_name
    ):
        logger.info("🎉 ONNX conversion completed successfully!")
        sys.exit(0)
    else:
        logger.error("💥 ONNX conversion failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
