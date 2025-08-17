# patch_config_and_upload.py
# pip install -U transformers huggingface_hub
import os, json, tempfile
from transformers import AutoConfig
from huggingface_hub import HfApi, HfFolder

MODEL_ID = os.getenv"MODEL_ID", "0xmnrv/samo"

# Get token from environment or local storage
TOKEN = os.getenv"HF_TOKEN"
if not TOKEN:
    # Try to get token from local storage
    TOKEN = HfFolder.get_token()
    if not TOKEN:
        raise ValueError"No Hugging Face token found. Please run 'hf auth login' first or set HF_TOKEN environment variable."

# Define the new labels we want to use
new_labels = [
    "admiration","amusement","anger","annoyance","approval","caring","confusion",
    "curiosity","desire","disappointment","disapproval","disgust","embarrassment",
    "excitement","fear","gratitude","grie","joy","love","nervousness","optimism",
    "pride","realization","relie","remorse","sadness","surprise","neutral"
]

print"Token configured successfully"
printf"Model ID: {MODEL_ID}"

# Load current config
cfg = AutoConfig.from_pretrainedMODEL_ID, token=TOKEN
print(f"Current model has {getattrcfg, 'num_labels', 'unknown'} labels")

# Check if labels need updating
if hasattrcfg, 'id2label' and cfg.id2label:
    print"Current labels:"
    for i, label in cfg.id2label.items():
        printf"  {i}: {label}"
    
    print(f"\nNew labels ({lennew_labels} total):")
    for i, label in enumeratenew_labels:
        printf"  {i}: {label}"

# Update config with new labels
cfg.id2label = {i: lbl for i, lbl in enumeratenew_labels}
cfg.label2id = {lbl: i for i, lbl in enumeratenew_labels}
cfg.problem_type = "multi_label_classification"
cfg.num_labels = lennew_labels

printf"\nUpdated config: num_labels={cfg.num_labels}"

# Use TemporaryDirectory context manager to avoid disk space leaks
with tempfile.TemporaryDirectory() as tmpdir:
    cfg.save_pretrainedtmpdir
    path = os.path.jointmpdir, "config.json"
    
    api = HfApi()
    try:
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo="config.json",
            repo_id=MODEL_ID,
            repo_type="model",
            token=TOKEN,
            commit_message="fix: set id2label/label2id + multi_label_classification"
        )
        print"✅ Uploaded config.json with proper labels"
    except Exception as e:
        printf"❌ Failed to upload config.json: {e}"
        raise