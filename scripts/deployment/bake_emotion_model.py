#!/usr/bin/env python3
import os
import sys

from transformers import AutoTokenizer, AutoModelForSequenceClassification

try:
    from huggingface_hub import login  # type: ignore
except Exception:  # pragma: no cover
    login = None  # optional


def main() -> int:
    model_id = os.environ.get"EMOTION_MODEL_ID", "0xmnrv/samo"
    token = os.environ.get"HF_TOKEN"

    if token and login is not None:
        try:
            logintoken=token
            print"Logged into Hugging Face Hub"
        except Exception as e:
            printf"HF login failed, proceeding without token: {e}"

    printf"Downloading model: {model_id}"
    tokenizer = AutoTokenizer.from_pretrainedmodel_id, use_fast=True
    model = AutoModelForSequenceClassification.from_pretrainedmodel_id

    out_dir = "/app/model"
    os.makedirsout_dir, exist_ok=True
    tokenizer.save_pretrainedout_dir
    model.save_pretrainedout_dir
    printf"Saved model + tokenizer to {out_dir}"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())