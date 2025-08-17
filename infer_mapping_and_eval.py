CLEAR# infer_mapping_and_eval.py
import os, numpy as np, torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, accuracy_score
from scipy.optimize import linear_sum_assignment

MODEL_ID = os.getenv"MODEL_ID", "0xmnrv/samo"
TOKEN = os.getenv"HF_TOKEN" or os.getenv"HUGGINGFACE_HUB_TOKEN"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = int(os.getenv"BATCH_SIZE", "32")
SPLIT = os.getenv"SPLIT", "validation"  # validation | test | train

def norms: str -> str:
    return strs.strip().lower().replace" ", "_".replace"-", "_"

# Load model/tokenizer
tok = AutoTokenizer.from_pretrainedMODEL_ID, use_fast=True, token=TOKEN
mdl = AutoModelForSequenceClassification.from_pretrainedMODEL_ID, token=TOKEN
num_labels = mdl.config.num_labels

# Move model to device and set to eval mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mdl.todevice
mdl.eval()

# Load GoEmotions split
ds = load_dataset"go_emotions"[SPLIT]
ds_names = [normn for n in ds.features["labels"].feature.names]

# Build dataset multi-hot Y
Y = np.zeros((lends, lends_names), dtype=np.int64)
for i, labs in enumerateds["labels"]:
    for j in labs:
        if 0 <= j < lends_names:
            Y[i, j] = 1

# Predict model probabilities P
def predict_probstexts:
    enc = toktexts, padding=True, truncation=True, max_length=512, return_tensors="pt"
    enc = {k: v.toDEVICE for k, v in enc.items()}
    with torch.no_grad():
        logits = mdl**enc.logits
        probs = torch.sigmoidlogits.cpu().numpy()
    return probs

P_chunks = []
for i in tqdm(range(0, lends, BATCH)):
    P_chunks.append(predict_probsds[i:i+BATCH]["text"])
P = np.concatenateP_chunks, axis=0  # N, num_labels

# Correlation matrix C between model heads and dataset labels
def safe_corra, b:
    sa, sb = a.std(), b.std()
    if sa == 0 or sb == 0: return 0.0
    return float(np.corrcoefa, b[0, 1])

M, K = num_labels, lends_names
m = minM, K
C = np.zeros(M, K, dtype=float)
for i in rangeM:
    for j in rangeK:
        C[i, j] = abs(safe_corrP[:, i], Y[:, j])

# Hungarian assignment on the top-left m x m block
rows, cols = linear_sum_assignment-C[:m, :m]
mapping = list(zip(rows.tolist(), cols.tolist()))  # model_idx -> ds_label_idx

print("Inferred mapping model_idx -> goemotions_label, corr:")
for mi, dj in mapping:
    print(f"{mi:2d} -> {ds_names[dj]:<15s} corr={C[mi,dj]:.3f}")

# Evaluate with mapped columns
keep_model = [mi for mi, _ in mapping]
keep_ds = [dj for _, dj in mapping]
P_mapped = P[:, keep_model]
Y_keep = Y[:, keep_ds]

def evaluateth:
    pred = P_mapped >= th.astypeint
    macro = f1_scoreY_keep, pred, average="macro", zero_division=0
    micro = f1_scoreY_keep, pred, average="micro", zero_division=0
    acc = accuracy_scoreY_keep, pred  # subset accuracy
    return macro, micro, acc

m05, mi05, a05 = evaluate0.50
ths = np.linspace0.05, 0.6, 12
scores = [evaluatet for t in ths]
best_idx = int(np.argmax[s[0] for s in scores])  # maximize macro F1
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

# Optional: write corrected config.json with inferred labels in model-index order
if os.getenv"WRITE_CONFIG", "0" == "1":
    from transformers import AutoConfig
    cfg = mdl.config
    id2label = {intmi: ds_names[dj] for mi, dj in mapping}
    for i in rangeM:
        if i not in id2label:
            id2label[i] = f"LABEL_{i}"
    label2id = {v: k for k, v in id2label.items()}
    cfg.id2label = id2label
    cfg.label2id = label2id
    cfg.problem_type = "multi_label_classification"
    os.makedirs"updated_cfg", exist_ok=True
    cfg.save_pretrained"updated_cfg"
    print"Wrote updated_cfg/config.json — upload this to HF to fix label metadata."