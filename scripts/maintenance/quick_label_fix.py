#!/usr/bin/env python3
"""
Quick fix for CUDA device-side assert errors caused by label mismatches.
Run this before your training to fix the label encoding issues.
"""

import json
import pandas as pd
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder
import pickle

def quick_label_fix():
    """Quick fix for label mismatch issues."""
    print("🔧 Applying quick label fix...")

    # Load datasets
    go_emotions = load_dataset("go_emotions", "simplified")

    with open('data/journal_test_dataset.json', 'r') as f:
        journal_entries = json.load(f)
    journal_df = pd.DataFrame(journal_entries)

    # Get all unique labels
    go_labels = set()
    for example in go_emotions['train']:
        if example['labels']:
            go_labels.update(example['labels'])

    journal_labels = set(journal_df['emotion'].unique())

    # Use only common labels to avoid mismatches
    common_labels = sorted(list(go_labels.intersection(journal_labels)))

    if not common_labels:
        print("⚠️ No common labels found! Using all labels...")
        common_labels = sorted(list(go_labels.union(journal_labels)))

    print(f"📊 Using {len(common_labels)} labels: {common_labels}")

    # Create label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(common_labels)

    # Create mappings
    label_to_id = {label: idx for idx, label in enumerate(label_encoder.classes_)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    # Save fixed encoder
    with open('fixed_label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

    # Save mappings
    with open('label_mappings.json', 'w') as f:
        json.dump({
            'label_to_id': label_to_id,
            'id_to_label': id_to_label,
            'num_labels': len(label_encoder.classes_),
            'classes': label_encoder.classes_.tolist()
        }, f, indent=2)

    print(f"✅ Fixed label encoder saved!")
    print(f"📊 Use num_labels={len(label_encoder.classes_)} in your model")
    print(f"📊 Label encoder: fixed_label_encoder.pkl")
    print(f"📊 Mappings: label_mappings.json")

    return len(label_encoder.classes_)

if __name__ == "__main__":
    num_labels = quick_label_fix()
    print(f"\n🎉 Quick fix completed! Use num_labels={num_labels}") 