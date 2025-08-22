# metrics_test.py
# pip install -U transformers datasets scikit-learn torch tqdm huggingface_hub

import os
import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, accuracy_score

MODEL_ID = os.getenv("MODEL_ID", "0xmnrv/samo")
TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = int(os.getenv("BATCH_SIZE", "32"))
SPLIT = os.getenv("SPLIT", "validation")  # validation | test | train


def norm(s: str) -> str:
    return str(s).strip().lower().replace(" ", "_").replace("-", "_")


# 1) Load model + tokenizer (private repos require token)
tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, token=TOKEN)
mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, token=TOKEN).to(DEVICE).eval()
cfg = mdl.config
num_labels = int(getattr(cfg, "num_labels", len(getattr(cfg, "id2label", {})) or 28))

# 2) Build robust label maps from config
# Prefer id2label → label2id (robust even if label2id has weird strings)
cfg_id2label = {}
id2label_raw = getattr(cfg, "id2label", {}) or {}
for k, v in id2label_raw.items():
    try:
        key = int(k) if not isinstance(k, int) else k
        cfg_id2label[key] = str(v)
    except Exception:
        if isinstance(k, int):
            cfg_id2label[k] = str(v)

cfg_label2id = {norm(v): k for k, v in cfg_id2label.items()}

# Fallback: use label2id only if values are numeric
if not cfg_label2id:
    l2i = getattr(cfg, "label2id", {}) or {}
    tmp = {}
    for k, v in l2i.items():
        try:
            tmp[norm(k)] = int(v)
        except Exception:
            pass
    if tmp:
        cfg_label2id = tmp

# Detect generic LABEL_# names (likely unhelpful)
generic_names = [v for v in cfg_id2label.values() if norm(v).startswith("label_")]
if len(generic_names) > 0 and len(generic_names) == len(cfg_id2label):
    print("Warning: Model config labels look generic (LABEL_#). Mapping may be poor.")

# 3) Load GoEmotions split and build dataset label list
ds = load_dataset("go_emotions")
val = ds[SPLIT]
ds_names = [norm(n) for n in val.features["labels"].feature.names]

# 4) Map dataset label index -> model label index; keep only mapped labels
ds_to_model = {}
for i, name in enumerate(ds_names):
    j = cfg_label2id.get(name)
    if j is not None and 0 <= j < num_labels:
        ds_to_model[i] = j

mapped_count = len(ds_to_model)
print(f"Mapped labels: {mapped_count}/{len(ds_names)}")

# Fallbacks if mapping is empty/too small
kept_ds_indices = []
kept_model_indices = []
if mapped_count >= 5:
    kept_ds_indices = [i for i in range(len(ds_names)) if i in ds_to_model]
    kept_model_indices = [ds_to_model[i] for i in kept_ds_indices]
else:
    if num_labels == len(ds_names):
        print("Low mapping coverage; falling back to identity mapping (assumes same order).")
        kept_ds_indices = list(range(num_labels))
        kept_model_indices = list(range(num_labels))
    else:
        m = min(num_labels, len(ds_names))
        print(f"Low mapping coverage; evaluating on min-dim identity mapping ({m} labels).")
        kept_ds_indices = list(range(m))
        kept_model_indices = list(range(m))

D = len(kept_ds_indices)
kept_ds_pos = {ds_idx: pos for pos, ds_idx in enumerate(kept_ds_indices)}


# 5) Build multi-hot ground truth in model-space order (kept labels only)
def to_multihot(example):
    y = np.zeros(D, dtype=np.int64)
    for ds_idx in example["labels"]:
        pos = kept_ds_pos.get(ds_idx)
        if pos is not None:
            y[pos] = 1
    example["y"] = y
    return example


val = val.map(to_multihot)


# 6) Batched inference (full probs), then slice to kept_model_indices
def predict_probs(batch_texts):
    enc = tok(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        batch_logits = mdl(**enc).logits
        batch_probs = torch.sigmoid(batch_logits).cpu().numpy()   # (B, num_labels)
    return batch_probs


all_probs_full, all_true = [], []
for i in tqdm(range(0, len(val), BATCH)):
    batch = val[i:i + BATCH]
    batch_probs = predict_probs(batch["text"])
    all_probs_full.append(batch_probs)
    all_true.append(np.stack(batch["y"]))
all_probs_full = np.concatenate(all_probs_full, axis=0)
all_true = np.concatenate(all_true, axis=0)

# Slice predictions to kept labels
all_probs = all_probs_full[:, kept_model_indices]  # shape (N, D)


def evaluate(th):
    pred = (all_probs >= th).astype(int)
    macro = f1_score(all_true, pred, average="macro", zero_division=0)
    micro = f1_score(all_true, pred, average="micro", zero_division=0)
    subset_acc = accuracy_score(all_true, pred)
    return macro, micro, subset_acc


# 7) Report default and tuned thresholds
m05, mi05, a05 = evaluate(0.50)
ths = np.linspace(0.05, 0.6, 12)
scores = [evaluate(t) for t in ths]
best_idx = int(np.argmax([m for (m, _, _) in scores]))
best_th = float(ths[best_idx])
mb, mib, ab = scores[best_idx]

print("\nDefault threshold 0.50")
print(f"- Macro F1: {m05:.4f}")
print(f"- Micro F1: {mi05:.4f}")
print(f"- Accuracy (subset): {a05:.4f}")

print(f"\nBest threshold {best_th:.2f}")
print(f"- Macro F1: {mb:.4f}")
print(f"- Micro F1: {mib:.4f}")
print(f"- Accuracy (subset): {ab:.4f}")
