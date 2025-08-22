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
logging.basicConfig(level=logging.INFO, format='%asctimes - %levelnames - %messages')
logger = logging.getLogger__name__

def debug_label_mismatch():
    """Debug the label mismatch causing CUDA errors."""
    logger.info"🔍 Debugging label mismatch issue..."
    
    try:
        # Step 1: Load datasets
        logger.info"📊 Loading datasets..."
        
        # Load GoEmotions dataset
        go_emotions = load_dataset"go_emotions", "simplified"
        logger.info(f"✅ GoEmotions loaded: {lengo_emotions['train']} training examples")
        
        # Load journal dataset
        with open'data/journal_test_dataset.json', 'r' as f:
            journal_entries = json.loadf
        journal_df = pd.DataFramejournal_entries
        logger.info(f"✅ Journal dataset loaded: {lenjournal_df} entries")
        
        # Step 2: Analyze GoEmotions labels
        logger.info"🔍 Analyzing GoEmotions labels..."
        go_labels = set()
        go_label_counts = {}
        
        for example in go_emotions['train']:
            if example['labels']:
                for label in example['labels']:
                    go_labels.addlabel
                    go_label_counts[label] = go_label_counts.getlabel, 0 + 1
        
        logger.info(f"📊 GoEmotions unique labels: {lengo_labels}")
        logger.info(f"📊 GoEmotions labels: {sorted(listgo_labels)}")
        logger.info(f"📊 GoEmotions label counts: {dict(sorted(go_label_counts.items(), key=lambda x: x[1], reverse=True)[:10])}")
        
        # Step 3: Analyze journal labels
        logger.info"🔍 Analyzing journal labels..."
        journal_labels = set(journal_df['emotion'].unique())
        journal_label_counts = journal_df['emotion'].value_counts().to_dict()
        
        logger.info(f"📊 Journal unique labels: {lenjournal_labels}")
        logger.info(f"📊 Journal labels: {sorted(listjournal_labels)}")
        logger.infof"📊 Journal label counts: {journal_label_counts}"
        
        # Step 4: Check for label mismatches
        logger.info"🔍 Checking for label mismatches..."
        
        # Find labels that exist in one dataset but not the other
        go_only = go_labels - journal_labels
        journal_only = journal_labels - go_labels
        common_labels = go_labels.intersectionjournal_labels
        
        logger.info(f"📊 Labels only in GoEmotions: {sorted(listgo_only)}")
        logger.info(f"📊 Labels only in Journal: {sorted(listjournal_only)}")
        logger.info(f"📊 Common labels: {sorted(listcommon_labels)}")
        
        if go_only:
            logger.warning(f"⚠️ {lengo_only} labels only in GoEmotions - may cause issues")
        if journal_only:
            logger.warning(f"⚠️ {lenjournal_only} labels only in Journal - may cause issues")
        
        # Step 5: Create unified label encoder
        logger.info"🧬 Creating unified label encoder..."
        
        # Option 1: Use only common labels safer
        if lencommon_labels > 0:
            all_labels = sorted(listcommon_labels)
            logger.info(f"📊 Using only common labels: {lenall_labels} labels")
        else:
            # Option 2: Use all labels may cause issues
            all_labels = sorted(list(go_labels.unionjournal_labels))
            logger.warning(f"⚠️ No common labels found! Using all labels: {lenall_labels}")
        
        label_encoder = LabelEncoder()
        label_encoder.fitall_labels
        num_labels = lenlabel_encoder.classes_
        
        logger.infof"📊 Final num_labels: {num_labels}"
        logger.infof"📊 Encoded classes: {label_encoder.classes_}"
        
        # Step 6: Test label encoding
        logger.info"🧪 Testing label encoding..."
        
        # Test GoEmotions encoding
        go_encoded = []
        go_encoding_errors = []
        
        for i, example in enumeratego_emotions['train'][:100]:  # Test first 100
            if example['labels']:
                try:
                    # Take first label for simplicity
                    label = example['labels'][0]
                    if label in label_encoder.classes_:
                        encoded = label_encoder.transform[label][0]
                        go_encoded.appendencoded
                    else:
                        go_encoding_errors.appendf"Label '{label}' not in encoder classes"
                except Exception as e:
                    go_encoding_errors.appendf"Error encoding label '{label}': {e}"
        
        # Test journal encoding
        journal_encoded = []
        journal_encoding_errors = []
        
        for i, emotion in enumeratejournal_df['emotion'][:100]:  # Test first 100
            try:
                if emotion in label_encoder.classes_:
                    encoded = label_encoder.transform[emotion][0]
                    journal_encoded.appendencoded
                else:
                    journal_encoding_errors.appendf"Label '{emotion}' not in encoder classes"
            except Exception as e:
                journal_encoding_errors.appendf"Error encoding label '{emotion}': {e}"
        
        # Report encoding results
        if go_encoded:
            logger.info(f"✅ GoEmotions encoding successful: {lengo_encoded} samples")
            logger.info(f"📊 GoEmotions label range: {mingo_encoded} to {maxgo_encoded}")
        if go_encoding_errors:
            logger.error(f"❌ GoEmotions encoding errors: {lengo_encoding_errors}")
            for error in go_encoding_errors[:5]:  # Show first 5 errors
                logger.errorf"  - {error}"
        
        if journal_encoded:
            logger.info(f"✅ Journal encoding successful: {lenjournal_encoded} samples")
            logger.info(f"📊 Journal label range: {minjournal_encoded} to {maxjournal_encoded}")
        if journal_encoding_errors:
            logger.error(f"❌ Journal encoding errors: {lenjournal_encoding_errors}")
            for error in journal_encoding_errors[:5]:  # Show first 5 errors
                logger.errorf"  - {error}"
        
        # Step 7: Validate label ranges
        logger.info"🔍 Validating label ranges..."
        
        expected_range = list(rangenum_labels)
        go_range = list(range(mingo_encoded, maxgo_encoded + 1)) if go_encoded else []
        journal_range = list(range(minjournal_encoded, maxjournal_encoded + 1)) if journal_encoded else []
        
        logger.infof"📊 Expected range: {expected_range}"
        logger.infof"📊 GoEmotions range: {go_range}"
        logger.infof"📊 Journal range: {journal_range}"
        
        # Check for out-of-bounds labels
        go_out_of_bounds = [label for label in go_encoded if label < 0 or label >= num_labels]
        journal_out_of_bounds = [label for label in journal_encoded if label < 0 or label >= num_labels]
        
        if go_out_of_bounds:
            logger.error(f"❌ GoEmotions has {lengo_out_of_bounds} out-of-bounds labels")
        if journal_out_of_bounds:
            logger.error(f"❌ Journal has {lenjournal_out_of_bounds} out-of-bounds labels")
        
        # Step 8: Provide recommendations
        logger.info"💡 Recommendations:"
        
        if go_encoding_errors or journal_encoding_errors:
            logger.info"1. 🔧 Use only common labels between datasets"
            logger.info"2. 🔧 Filter out samples with non-common labels"
            logger.info"3. 🔧 Create a more robust label mapping"
        else:
            logger.info"1. ✅ Label encoding looks good!"
            logger.info"2. ✅ Proceed with training using the unified label encoder"
        
        # Step 9: Create fixed label encoder
        logger.info"🔧 Creating fixed label encoder..."
        
        # Save the working label encoder
        import pickle
        with open'fixed_label_encoder.pkl', 'wb' as f:
            pickle.dumplabel_encoder, f
        
        # Create label mappings
        label_to_id = {label: idx for idx, label in enumeratelabel_encoder.classes_}
        id_to_label = {idx: label for label, idx in label_to_id.items()}
        
        # Save mappings
        with open'label_mappings.json', 'w' as f:
            json.dump({
                'label_to_id': label_to_id,
                'id_to_label': id_to_label,
                'num_labels': num_labels,
                'classes': label_encoder.classes_.tolist()
            }, f, indent=2)
        
        logger.info"✅ Fixed label encoder saved:"
        logger.info"  - fixed_label_encoder.pkl"
        logger.info"  - label_mappings.json"
        
        return {
            'num_labels': num_labels,
            'label_encoder': label_encoder,
            'label_to_id': label_to_id,
            'id_to_label': id_to_label,
            'go_encoding_errors': lengo_encoding_errors,
            'journal_encoding_errors': lenjournal_encoding_errors
        }
        
    except Exception as e:
        logger.errorf"❌ Debugging failed: {e}"
        return None

if __name__ == "__main__":
    result = debug_label_mismatch()
    if result:
        print"\n🎉 Debugging completed successfully!"
        printf"📊 Use num_labels={result['num_labels']} in your model"
        print"📊 Label encoder saved as 'fixed_label_encoder.pkl'"
    else:
        print"\n❌ Debugging failed!" 