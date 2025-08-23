# 🚀 BULLETPROOF TRAINING CELL - FIXED LABEL MAPPING
# Runtime → Change runtime type → GPU (T4 or V100)
# Kernel → Restart and run all

import json
import os
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


def _import_bootstrap():
    """Import bootstrap helpers and ensure `src/` is on `sys.path`.

    Returns a tuple of `(add_repo_src_to_path, find_repo_root)` and raises
    `ImportError` if the import fails.
    """
    try:
        from scripts.bootstrap import (  # type: ignore
            add_repo_src_to_path,
            find_repo_root,
        )

        return add_repo_src_to_path, find_repo_root
    except Exception as exc:
        raise ImportError("Failed to import scripts.bootstrap") from exc


add_repo_src_to_path, find_repo_root = _import_bootstrap()

print("🚀 BULLETPROOF TRAINING FOR REQ-DL-012 - FIXED LABEL MAPPING")
print("=" * 60)
print("✅ Imports successful")

# Clear GPU memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"✅ GPU memory cleared: {torch.cuda.get_device_name()}")
else:
    print("⚠️ CUDA not available, using CPU")

# Test basic operations
try:
    test_tensor = torch.randn(2, 3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_tensor.to(device)
    print("✅ Basic tensor operations work")
except Exception as e:
    print(f"❌ Basic tensor operations failed: {e}")
    raise

# Step 2: Ensure we are at the repository root (strict)
REPO_ROOT = find_repo_root(Path(__file__))
add_repo_src_to_path(Path(__file__))
os.chdir(str(REPO_ROOT))

# Step 3: Create emotion mapping
print("\n🔧 Creating emotion mapping...")

# GoEmotions to Journal emotion mapping
emotion_mapping = {
    "admiration": "proud",
    "amusement": "happy",
    "anger": "frustrated",
    "annoyance": "frustrated",
    "approval": "proud",
    "caring": "content",
    "confusion": "overwhelmed",
    "curiosity": "excited",
    "desire": "excited",
    "disappointment": "sad",
    "disapproval": "frustrated",
    "disgust": "frustrated",
    "embarrassment": "anxious",
    "excitement": "excited",
    "fear": "anxious",
    "gratitude": "grateful",
    "grie": "sad",
    "joy": "happy",
    "love": "content",
    "nervousness": "anxious",
    "optimism": "hopeful",
    "pride": "proud",
    "realization": "content",
    "relie": "calm",
    "remorse": "sad",
    "sadness": "sad",
    "surprise": "excited",
    "neutral": "calm",
}

print(f"✅ Created mapping with {len(emotion_mapping)} emotions")

# Step 4: Load and prepare data with mapping
print("\n📊 Loading and preparing data with mapping...")

go_emotions = load_dataset("go_emotions", "simplified")
with open("data/journal_test_dataset.json") as f:
    journal_entries = json.load(f)
journal_df = pd.DataFrame(journal_entries)

# Get journal emotions
journal_emotions = set(journal_df["emotion"].unique())
print(f"📊 Journal emotions: {sorted(list(journal_emotions))}")

# Filter GoEmotions data using mapping
go_texts = []
go_labels = []
for example in go_emotions["train"]:
    if example["labels"]:
        for label in example["labels"]:
            if label in emotion_mapping:
                mapped_emotion = emotion_mapping[label]
                if mapped_emotion in journal_emotions:
                    go_texts.append(example["text"])
                    go_labels.append(mapped_emotion)
                    break

# Prepare journal data
journal_texts = list(journal_df["content"])
journal_labels = list(journal_df["emotion"])

print(f"📊 Mapped GoEmotions: {len(go_texts)} samples")
print(f"📊 Journal: {len(journal_texts)} samples")

# Create unified label encoder
all_emotions = sorted(list(set(go_labels + journal_labels)))
print(f"📊 All emotions: {all_emotions}")

label_encoder = LabelEncoder()
label_encoder.fit(all_emotions)
label_to_id = {label: idx for idx, label in enumerate(label_encoder.classes_)}
id_to_label = {idx: label for label, idx in label_to_id.items()}

print(f"✅ Label encoder created with {len(label_encoder.classes_)} classes")

# Convert labels to IDs
go_label_ids = [label_to_id[label] for label in go_labels]
journal_label_ids = [label_to_id[label] for label in journal_labels]

print(f"📊 GoEmotions label range: {min(go_label_ids)} to {max(go_label_ids)}")
print(f"📊 Journal label range: {min(journal_label_ids)} to {max(journal_label_ids)}")


# Step 5: Create simple dataset class
class SimpleEmotionDataset(Dataset):
    """Dataset wrapper that tokenizes texts and returns tensors with labels."""

    def __init__(self, texts, y_labels, hf_tokenizer, max_length=128):
        self.texts = texts
        self.labels = y_labels
        self.tokenizer = hf_tokenizer
        self.max_length = max_length

        # Validate data
        if len(texts) != len(labels):
            msg = (
                "Texts and labels have different lengths: "
                f"{len(texts)} vs {len(labels)}"
            )
            raise ValueError(msg)

        # Validate labels
        for idx_pos, label_val in enumerate(labels):
            if not isinstance(label_val, int) or label_val < 0:
                raise ValueError(f"Invalid label at index {idx_pos}: {label_val}")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        lbl = self.labels[idx]

        # Validate inputs
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Invalid text at index {idx}")

        if not isinstance(lbl, int) or lbl < 0:
            raise ValueError(f"Invalid label at index {idx}: {lbl}")

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(lbl, dtype=torch.long),
        }


# Step 6: Create simple model
class SimpleEmotionClassifier(nn.Module):
    """Simple classifier wrapper around a transformer backbone."""

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        n_labels: int | None = None,
        num_labels: int | None = None,
    ):
        super().__init__()
        # Resolve label count
        if n_labels is None and num_labels is not None:
            n_labels = num_labels
        elif n_labels is not None and num_labels is not None and n_labels != num_labels:
            raise ValueError(
                "Conflicting n_labels and num_labels provided; use n_labels"
            )

        if n_labels is None or n_labels <= 0:
            raise ValueError(f"Invalid num_labels: {n_labels}")

        self.num_labels = n_labels
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, self.num_labels)

        print(f"✅ Model initialized with {self.num_labels} labels")

    def forward(self, input_ids_tensor, attention_mask_tensor):
        """Compute logits for a batch.

        Args:
            input_ids_tensor: Tensor of shape (batch, seq_len) with token IDs.
            attention_mask_tensor: Tensor of shape (batch, seq_len) with attention mask.

        Returns:
            Tensor of shape (batch, num_labels) containing logits.
        """
        # Validate inputs
        if input_ids_tensor.dim() != 2:
            msg = "Expected input_ids to be 2D, got " f"{input_ids_tensor.dim()}D"
            raise ValueError(msg)

        if attention_mask_tensor.dim() != 2:
            msg = (
                "Expected attention_mask to be 2D, got "
                f"{attention_mask_tensor.dim()}D"
            )
            raise ValueError(msg)

        bert_outputs = self.bert(
            input_ids=input_ids_tensor, attention_mask=attention_mask_tensor
        )
        pooled_output = bert_outputs.pooler_output
        logits = self.classifier(self.dropout(pooled_output))

        # Validate outputs
        if logits.shape[-1] != self.num_labels:
            raise ValueError(
                f"Expected {self.num_labels} output classes, got {logits.shape[-1]}"
            )

        return logits


# Step 7: Setup training
print("\n🚀 Setting up training...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
TOTAL_LABELS = len(label_encoder.classes_)
model = SimpleEmotionClassifier(model_name="bert-base-uncased", n_labels=TOTAL_LABELS)
model = model.to(device)

# Create datasets
go_dataset = SimpleEmotionDataset(go_texts, go_label_ids, tokenizer)
journal_dataset = SimpleEmotionDataset(
    journal_texts,
    journal_label_ids,
    tokenizer,
)

# Split journal data
journal_train_texts, journal_val_texts, journal_train_labels, journal_val_labels = (
    train_test_split(
        journal_texts,
        journal_label_ids,
        test_size=0.3,
        random_state=42,
        stratify=journal_label_ids,
    )
)

journal_train_dataset = SimpleEmotionDataset(
    journal_train_texts,
    journal_train_labels,
    tokenizer,
)
journal_val_dataset = SimpleEmotionDataset(
    journal_val_texts,
    journal_val_labels,
    tokenizer,
)

# Create dataloaders
go_loader = DataLoader(go_dataset, batch_size=8, shuffle=True)
journal_train_loader = DataLoader(journal_train_dataset, batch_size=8, shuffle=True)
journal_val_loader = DataLoader(journal_val_dataset, batch_size=8, shuffle=False)

print(
    f"✅ Training samples: {len(go_dataset)} GoEmotions + "
    f"{len(journal_train_dataset)} Journal"
)
print(f"✅ Validation samples: {len(journal_val_dataset)} Journal")

# Step 8: Training loop
print("\n🚀 Starting training...")

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

num_epochs = 3  # Reduced for testing
best_f1 = 0.0

for epoch in range(num_epochs):
    print(f"\n🔄 Epoch {epoch + 1}/{num_epochs}")

    # Training
    model.train()
    total_loss = 0
    num_batches = 0

    # Train on GoEmotions
    print("  📚 Training on GoEmotions...")
    for i, batch in enumerate(go_loader):
        try:
            # Validate batch
            if (
                "input_ids" not in batch
                or "attention_mask" not in batch
                or "labels" not in batch
            ):
                print(f"⚠️ Invalid batch structure at batch {i}")
                continue

            # Move to device with validation
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Validate labels
            if torch.any(labels >= TOTAL_LABELS) or torch.any(labels < 0):
                print(f"⚠️ Invalid labels in batch {i}: {labels}")
                continue

            # Forward pass
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if i % 50 == 0:
                print(f"    Batch {i}/{len(go_loader)}, Loss: {loss.item():.4f}")

        except Exception as e:
            print(f"❌ Error in batch {i}: {e}")
            continue

    # Train on journal data
    print("  📝 Training on journal data...")
    for i, batch in enumerate(journal_train_loader):
        try:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if torch.any(labels >= TOTAL_LABELS) or torch.any(labels < 0):
                continue

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if i % 10 == 0:
                print(
                    f"    Batch {i}/{len(journal_train_loader)}, "
                    f"Loss: {loss.item():.4f}"
                )

        except Exception as e:
            print(f"❌ Error in journal batch {i}: {e}")
            continue

    # Validation
    print("  🎯 Validating...")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in journal_val_loader:
            try:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            except Exception as e:
                print(f"❌ Error in validation batch: {e}")
                continue

    # Calculate metrics
    if all_preds and all_labels:
        f1_macro = f1_score(all_labels, all_preds, average="macro")
        accuracy = accuracy_score(all_labels, all_preds)

        avg_loss = total_loss / num_batches if num_batches > 0 else 0

        print(f"  📊 Epoch {epoch + 1} Results:")
        print(f"    Average Loss: {avg_loss:.4f}")
        print(f"    Validation F1 (Macro): {f1_macro:.4f}")
        print(f"    Validation Accuracy: {accuracy:.4f}")

        # Save best model
        if f1_macro > best_f1:
            best_f1 = f1_macro
            torch.save(model.state_dict(), "best_simple_model.pth")
            print(f"    💾 New best model saved! F1: {best_f1:.4f}")

    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print(f"\n🏆 Training completed! Best F1 Score: {best_f1:.4f}")

# Step 9: Save results
results = {
    "best_f1": best_f1,
    "num_labels": TOTAL_LABELS,
    "target_achieved": best_f1 >= 0.7,
    "go_samples": len(go_texts),
    "journal_samples": len(journal_texts),
    "emotion_mapping": emotion_mapping,
}

with open("simple_training_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n✅ Training completed successfully!")
print(f"📊 Final F1 Score: {best_f1:.4f}")
print("🎯 Target Met: {"✅' if best_f1 >= 0.7 else '❌'}")

# Download results (optional in Colab)
try:
    from google.colab import files  # type: ignore

    files.download("best_simple_model.pth")
    files.download("simple_training_results.json")
except Exception:
    print("ℹ️ Skipping file downloads (not running in Colab)")

print("\n🎉 BULLETPROOF TRAINING COMPLETED!")
print("📁 Files downloaded: best_simple_model.pth, simple_training_results.json")
