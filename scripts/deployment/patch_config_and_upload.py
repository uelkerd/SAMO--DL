# patch_config_and_upload.py
# pip install -U transformers huggingface_hub
import os
import tempfile

from huggingface_hub import HfApi, HfFolder
from transformers import AutoConfig

MODEL_ID = os.getenv("MODEL_ID", "0xmnrv/samo")

# Get token from environment or local storage
TOKEN = os.getenv("HF_TOKEN")
if not TOKEN:
    # Try to get token from local storage
    TOKEN = HfFolder.get_token()
    if not TOKEN:
        raise ValueError(
            "No Hugging Face token found. Please run 'hf auth login' first or "
            "set HF_TOKEN environment variable."
        )

# Define the new labels we want to use
new_labels = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grie", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relie", "remorse",
    "sadness", "surprise", "neutral",
]

print("Token configured successfully")
print(f"Model ID: {MODEL_ID}")

# Load current config
cfg = AutoConfig.from_pretrained(MODEL_ID, token=TOKEN)
print("Current model has {getattr(cfg, "num_labels', 'unknown')} labels")

# Check if labels need updating
if hasattr(cfg, 'id2label') and cfg.id2label:
    print("Current labels:")
    items = sorted(
        cfg.id2label.items(),
        key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else str(kv[0]),
    )
    for i, label in items:
        print(f"  {i}: {label}")

    print(f"\nNew labels ({len(new_labels)} total):")
    for i, label in enumerate(new_labels):
        print(f"  {i}: {label}")

# Sanity check: ensure new label count aligns with existing config (and model head)
orig_num_labels = getattr(cfg, "num_labels", None)
if orig_num_labels not in (None, len(new_labels)):
    print(
        f"⚠️  Existing cfg.num_labels={orig_num_labels}, new_labels={len(new_labels)}."
    )
    print(
        f"Ensure the classifier head out_features matches {len(new_labels)} "
        "before publishing."
    )

# Update config with new labels
cfg.id2label = dict(enumerate(new_labels))
cfg.label2id = {lbl: i for i, lbl in enumerate(new_labels)}
cfg.problem_type = "multi_label_classification"
cfg.num_labels = len(new_labels)

print(f"\nUpdated config: num_labels={cfg.num_labels}")

# Use TemporaryDirectory context manager to avoid disk space leaks
with tempfile.TemporaryDirectory() as tmpdir:
    cfg.save_pretrained(tmpdir)
    path = os.path.join(tmpdir, "config.json")

    api = HfApi(token=TOKEN)
    try:
        info = api.upload_file(
            path_or_fileobj=path,
            path_in_repo="config.json",
            repo_id=MODEL_ID,
            repo_type="model",
            commit_message="fix: set id2label/label2id + multi_label_classification",
        )
        commit_id = getattr(info, 'oid', getattr(info, 'commit_sha', 'unknown'))
        print(f"✅ Uploaded config.json with proper labels (commit: {commit_id})")
    except Exception as e:
        print(f"❌ Failed to upload config.json: {e}")
        raise
