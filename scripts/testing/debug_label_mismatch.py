#!/usr/bin/env python3
"""
Debug script to identify and fix CUDA device-side assert errors caused by label mismatches.
"""

import json
import pandas as pd
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_label_mismatch():
    """Debug the label mismatch causing CUDA errors."""
    logger.info("🔍 Debugging label mismatch issue...")

    try:
        # Step 1: Load datasets
        logger.info("📊 Loading datasets...")

        # Load GoEmotions dataset
        go_emotions = load_dataset("go_emotions", "simplified")
        logger.info(f"✅ GoEmotions loaded: {len(go_emotions['train'])} training examples")

        # Load journal dataset
        with open('data/journal_test_dataset.json', 'r') as f:
            journal_entries = json.load(f)
        journal_df = pd.DataFrame(journal_entries)
        logger.info(f"✅ Journal dataset loaded: {len(journal_df)} entries")

        # Step 2: Analyze GoEmotions labels
        logger.info("🔍 Analyzing GoEmotions labels...")
        go_labels = set()
        go_label_counts = {}

        for example in go_emotions['train']:
            if example['labels']:
                for label in example['labels']:
                    go_labels.add(label)
                    go_label_counts[label] = go_label_counts.get(label, 0) + 1

        logger.info(f"📊 GoEmotions unique labels: {len(go_labels)}")
        logger.info(f"📊 GoEmotions labels: {sorted(list(go_labels))}")
        logger.info(f"📊 GoEmotions label counts: {dict(sorted(go_label_counts.items(), key=lambda x: x[1], reverse=True)[:10])}")

        # Step 3: Analyze journal labels
        logger.info("🔍 Analyzing journal labels...")
        journal_labels = set(journal_df['emotion'].unique())
        journal_label_counts = journal_df['emotion'].value_counts().to_dict()

        logger.info(f"📊 Journal unique labels: {len(journal_labels)}")
        logger.info(f"📊 Journal labels: {sorted(list(journal_labels))}")
        logger.info(f"📊 Journal label counts: {journal_label_counts}")

        # Step 4: Check for label mismatches
        logger.info("🔍 Checking for label mismatches...")

        # Find labels that exist in one dataset but not the other
        go_only = go_labels - journal_labels
        journal_only = journal_labels - go_labels
        common_labels = go_labels.intersection(journal_labels)

        logger.info(f"📊 Labels only in GoEmotions: {sorted(list(go_only))}")
        logger.info(f"📊 Labels only in Journal: {sorted(list(journal_only))}")
        logger.info(f"📊 Common labels: {sorted(list(common_labels))}")

        if go_only:
            logger.warning(f"⚠️ {len(go_only)} labels only in GoEmotions - may cause issues")
        if journal_only:
            logger.warning(f"⚠️ {len(journal_only)} labels only in Journal - may cause issues")

        # Step 5: Create unified label encoder
        logger.info("🧬 Creating unified label encoder...")

        # Option 1: Use only common labels (safer)
        if len(common_labels) > 0:
            all_labels = sorted(list(common_labels))
            logger.info(f"📊 Using only common labels: {len(all_labels)} labels")
        else:
            # Option 2: Use all labels (may cause issues)
            all_labels = sorted(list(go_labels.union(journal_labels)))
            logger.warning(f"⚠️ No common labels found! Using all labels: {len(all_labels)}")

        label_encoder = LabelEncoder()
        label_encoder.fit(all_labels)
        num_labels = len(label_encoder.classes_)

        logger.info(f"📊 Final num_labels: {num_labels}")
        logger.info(f"📊 Encoded classes: {label_encoder.classes_}")

        # Step 6: Test label encoding
        logger.info("🧪 Testing label encoding...")

        # Test GoEmotions encoding
        go_encoded = []
        go_encoding_errors = []

        for i, example in enumerate(go_emotions['train'][:100]):  # Test first 100
            if example['labels']:
                try:
                    # Take first label for simplicity
                    label = example['labels'][0]
                    if label in label_encoder.classes_:
                        encoded = label_encoder.transform([label])[0]
                        go_encoded.append(encoded)
                    else:
                        go_encoding_errors.append(f"Label '{label}' not in encoder classes")
                except Exception as e:
                    go_encoding_errors.append(f"Error encoding label '{label}': {e}")

        # Test journal encoding
        journal_encoded = []
        journal_encoding_errors = []

        for i, emotion in enumerate(journal_df['emotion'][:100]):  # Test first 100
            try:
                if emotion in label_encoder.classes_:
                    encoded = label_encoder.transform([emotion])[0]
                    journal_encoded.append(encoded)
                else:
                    journal_encoding_errors.append(f"Label '{emotion}' not in encoder classes")
            except Exception as e:
                journal_encoding_errors.append(f"Error encoding label '{emotion}': {e}")

        # Report encoding results
        if go_encoded:
            logger.info(f"✅ GoEmotions encoding successful: {len(go_encoded)} samples")
            logger.info(f"📊 GoEmotions label range: {min(go_encoded)} to {max(go_encoded)}")
        if go_encoding_errors:
            logger.error(f"❌ GoEmotions encoding errors: {len(go_encoding_errors)}")
            for error in go_encoding_errors[:5]:  # Show first 5 errors
                logger.error(f"  - {error}")

        if journal_encoded:
            logger.info(f"✅ Journal encoding successful: {len(journal_encoded)} samples")
            logger.info(f"📊 Journal label range: {min(journal_encoded)} to {max(journal_encoded)}")
        if journal_encoding_errors:
            logger.error(f"❌ Journal encoding errors: {len(journal_encoding_errors)}")
            for error in journal_encoding_errors[:5]:  # Show first 5 errors
                logger.error(f"  - {error}")

        # Step 7: Validate label ranges
        logger.info("🔍 Validating label ranges...")

        expected_range = list(range(num_labels))
        go_range = list(range(min(go_encoded), max(go_encoded) + 1)) if go_encoded else []
        journal_range = list(range(min(journal_encoded), max(journal_encoded) + 1)) if journal_encoded else []

        logger.info(f"📊 Expected range: {expected_range}")
        logger.info(f"📊 GoEmotions range: {go_range}")
        logger.info(f"📊 Journal range: {journal_range}")

        # Check for out-of-bounds labels
        go_out_of_bounds = [label for label in go_encoded if label < 0 or label >= num_labels]
        journal_out_of_bounds = [label for label in journal_encoded if label < 0 or label >= num_labels]

        if go_out_of_bounds:
            logger.error(f"❌ GoEmotions has {len(go_out_of_bounds)} out-of-bounds labels")
        if journal_out_of_bounds:
            logger.error(f"❌ Journal has {len(journal_out_of_bounds)} out-of-bounds labels")

        # Step 8: Provide recommendations
        logger.info("💡 Recommendations:")

        if go_encoding_errors or journal_encoding_errors:
            logger.info("1. 🔧 Use only common labels between datasets")
            logger.info("2. 🔧 Filter out samples with non-common labels")
            logger.info("3. 🔧 Create a more robust label mapping")
        else:
            logger.info("1. ✅ Label encoding looks good!")
            logger.info("2. ✅ Proceed with training using the unified label encoder")

        # Step 9: Create fixed label encoder
        logger.info("🔧 Creating fixed label encoder...")

        # Save the working label encoder
        import pickle
        with open('fixed_label_encoder.pkl', 'wb') as f:
            pickle.dump(label_encoder, f)

        # Create label mappings
        label_to_id = {label: idx for idx, label in enumerate(label_encoder.classes_)}
        id_to_label = {idx: label for label, idx in label_to_id.items()}

        # Save mappings
        with open('label_mappings.json', 'w') as f:
            json.dump({
                'label_to_id': label_to_id,
                'id_to_label': id_to_label,
                'num_labels': num_labels,
                'classes': label_encoder.classes_.tolist()
            }, f, indent=2)

        logger.info("✅ Fixed label encoder saved:")
        logger.info("  - fixed_label_encoder.pkl")
        logger.info("  - label_mappings.json")

        return {
            'num_labels': num_labels,
            'label_encoder': label_encoder,
            'label_to_id': label_to_id,
            'id_to_label': id_to_label,
            'go_encoding_errors': len(go_encoding_errors),
            'journal_encoding_errors': len(journal_encoding_errors)
        }

    except Exception as e:
        logger.error(f"❌ Debugging failed: {e}")
        return None

if __name__ == "__main__":
    result = debug_label_mismatch()
    if result:
        print(f"\n🎉 Debugging completed successfully!")
        print(f"📊 Use num_labels={result['num_labels']} in your model")
        print(f"📊 Label encoder saved as 'fixed_label_encoder.pkl'")
    else:
        print(f"\n❌ Debugging failed!") 