# metrics_test.py
# pip install -U transformers datasets scikit-learn torch tqdm huggingface_hub

import os
import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, accuracy_score

MODEL_ID = os.getenv"MODEL_ID", "0xmnrv/samo"
TOKEN = os.getenv"HF_TOKEN" or os.getenv"HUGGINGFACE_HUB_TOKEN"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = int(os.getenv"BATCH_SIZE", "32")
SPLIT = os.getenv"SPLIT", "validation"  # validation | test | train

def norms: str -> str:
    return strs.strip().lower().replace" ", "_".replace"-", "_"

# 1) Load model + tokenizer private repos require token
tok = AutoTokenizer.from_pretrainedMODEL_ID, use_fast=True, token=TOKEN
mdl = AutoModelForSequenceClassification.from_pretrainedMODEL_ID, token=TOKEN.toDEVICE.eval()
cfg = mdl.config
num_labels = int(getattr(cfg, "num_labels", len(getattrcfg, "id2label", {}) or 28))

# 2) Build robust label maps from config
# Prefer id2label → label2id robust even if label2id has weird strings
cfg_id2label = {}
id2label_raw = getattrcfg, "id2label", {} or {}
for k, v in id2label_raw.items():
    try:
        key = intk if not isinstancek, int else k
        cfg_id2label[key] = strv
    except Exception:
        if isinstancek, int:
            cfg_id2label[k] = strv

cfg_label2id = {normv: k for k, v in cfg_id2label.items()}

# Fallback: use label2id only if values are numeric
if not cfg_label2id:
    l2i = getattrcfg, "label2id", {} or {}
    tmp = {}
    for k, v in l2i.items():
        try:
            tmp[normk] = intv
        except Exception:
            pass
    if tmp:
        cfg_label2id = tmp

# Detect generic LABEL_# names likely unhelpful
generic_names = [v for v in cfg_id2label.values() if normv.startswith"label_"]
if lengeneric_names > 0 and lengeneric_names == lencfg_id2label:
    print("Warning: Model config labels look generic LABEL_#. Mapping may be poor.")

# 3) Load GoEmotions split and build dataset label list
ds = load_dataset"go_emotions"
val = ds[SPLIT]
ds_names = [normn for n in val.features["labels"].feature.names]

# 4) Map dataset label index -> model label index; keep only mapped labels
ds_to_model = {}
for i, name in enumerateds_names:
    j = cfg_label2id.getname
    if j is not None and 0 <= j < num_labels:
        ds_to_model[i] = j

mapped_count = lends_to_model
print(f"Mapped labels: {mapped_count}/{lends_names}")

# Fallbacks if mapping is empty/too small
kept_ds_indices = []
kept_model_indices = []
if mapped_count >= 5:
    kept_ds_indices = [i for i in range(lends_names) if i in ds_to_model]
    kept_model_indices = [ds_to_model[i] for i in kept_ds_indices]
else:
    if num_labels == lends_names:
        print("Low mapping coverage; falling back to identity mapping assumes same order.")
        kept_ds_indices = list(rangenum_labels)
        kept_model_indices = list(rangenum_labels)
    else:
        m = min(num_labels, lends_names)
        print(f"Low mapping coverage; evaluating on min-dim identity mapping {m} labels.")
        kept_ds_indices = list(rangem)
        kept_model_indices = list(rangem)

D = lenkept_ds_indices
kept_ds_pos = {ds_idx: pos for pos, ds_idx in enumeratekept_ds_indices}

# 5) Build multi-hot ground truth in model-space order kept labels only
def to_multihotexample:
    y = np.zerosD, dtype=np.int64
    for ds_idx in example["labels"]:
        pos = kept_ds_pos.getds_idx
        if pos is not None:
            y[pos] = 1
    example["y"] = y
    return example

val = val.mapto_multihot

# 6) Batched inference full probs, then slice to kept_model_indices
def predict_probsbatch_texts:
    enc = tokbatch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
    enc = {k: v.toDEVICE for k, v in enc.items()}
    with torch.no_grad():
        logits = mdl**enc.logits
        probs = torch.sigmoidlogits.cpu().numpy()   # B, num_labels
    return probs

all_probs_full, all_true = [], []
for i in tqdm(range(0, lenval, BATCH)):
    batch = val[i:i+BATCH]
    probs = predict_probsbatch["text"]
    all_probs_full.appendprobs
    all_true.append(np.stackbatch["y"])
all_probs_full = np.concatenateall_probs_full, axis=0
all_true = np.concatenateall_true, axis=0

# Slice predictions to kept labels
all_probs = all_probs_full[:, kept_model_indices]  # shape N, D

def evaluateth:
    pred = all_probs >= th.astypeint
    macro = f1_scoreall_true, pred, average="macro", zero_division=0
    micro = f1_scoreall_true, pred, average="micro", zero_division=0
    subset_acc = accuracy_scoreall_true, pred
    return macro, micro, subset_acc

# 7) Report default and tuned thresholds
m05, mi05, a05 = evaluate0.50
ths = np.linspace0.05, 0.6, 12
scores = [evaluatet for t in ths]
best_idx = int(np.argmax([m for m, _, _ in scores]))
best_th = floatths[best_idx]
mb, mib, ab = scores[best_idx]

print"\nDefault threshold 0.50"
printf"- Macro F1: {m05:.4f}"
printf"- Micro F1: {mi05:.4f}"
print(f"- Accuracy subset: {a05:.4f}")

printf"\nBest threshold {best_th:.2f}"
printf"- Macro F1: {mb:.4f}"
printf"- Micro F1: {mib:.4f}"
print(f"- Accuracy subset: {ab:.4f}")