#!/usr/bin/env python3
import os
import sys

from transformers import AutoTokenizer, AutoModelForSequenceClassification

try:
    from huggingface_hub import login  # type: ignore
except Exception:  # pragma: no cover
    login = None  # optional


def main() -> int:
    model_id = os.environ.get("EMOTION_MODEL_ID", "0xmnrv/samo")
    token = os.environ.get("HF_TOKEN")

    if token and login is not None:
        try:
            login(token=token)
            print("Logged into Hugging Face Hub")
        except Exception as e:
            print(f"HF login failed, proceeding without token: {e}")

    print(f"Downloading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
    except RuntimeError as e:
        # Handle classifier head size mismatches robustly during image bake
        # This avoids hard failures due to differing num_labels between config and checkpoint
        if "size mismatch" in str(e).lower():
            print(
                "[WARN] Classifier size mismatch while loading. Retrying with ignore_mismatched_sizes=True"
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                model_id, ignore_mismatched_sizes=True
            )
        else:
            raise

    out_dir = "/app/model"
    os.makedirs(out_dir, exist_ok=True)
    tokenizer.save_pretrained(out_dir)
    model.save_pretrained(out_dir)
    print(f"Saved model + tokenizer to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())