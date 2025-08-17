#!/usr/bin/env python3
"""
🚀 CMU-MOSEI DATASET INTEGRATION
================================

This script downloads and integrates CMU-MOSEI dataset for emotion detection.
Target: Use 23,500+ high-quality samples to achieve 75-85% F1 score.
"""

import sys
import json
import numpy as np
from collections import defaultdict

# Add CMU-MultimodalDataSDK to path
sys.path.append'/Users/minervae/Projects/SAMO--GENERAL/SAMO--DL/CMU-MultimodalDataSDK'

try:
    import mmdata
    from mmdata import Dataset
    print"✅ CMU Multimodal Data SDK imported successfully!"
except ImportError as e:
    printf"❌ Error importing CMU SDK: {e}"
    print"Make sure you've cloned the repository and set PYTHONPATH"
    sys.exit1

def download_cmu_mosei():
    """Download CMU-MOSEI dataset"""
    print"📥 Downloading CMU-MOSEI dataset..."
    
    try:
        # Initialize MOSEI loader
        mosei = mmdata.MOSEI()
        
        # Download text embeddings transcribed sentences
        print"📝 Downloading text embeddings..."
        mosei_emb = mosei.embeddings()
        
        # Download words transcribed text
        print"📝 Downloading transcribed words..."
        mosei_words = mosei.words()
        
        # Get sentiment labels
        print"🏷️ Downloading sentiment labels..."
        sentiments = mosei.sentiments()
        
        # Get train/validation/test splits
        print"📊 Getting dataset splits..."
        train_ids = mosei.train()
        valid_ids = mosei.valid()
        test_ids = mosei.test()
        
        print"✅ CMU-MOSEI downloaded successfully!"
        print(f"📊 Train videos: {lentrain_ids}")
        print(f"📊 Validation videos: {lenvalid_ids}")
        print(f"📊 Test videos: {lentest_ids}")
        
        return mosei_emb, mosei_words, sentiments, train_ids, valid_ids, test_ids
        
    except Exception as e:
        printf"❌ Error downloading CMU-MOSEI: {e}"
        return None, None, None, None, None, None

def extract_text_and_emotionsmosei_words, sentiments, train_ids, valid_ids, test_ids:
    """Extract text sentences and emotion labels from CMU-MOSEI"""
    print"🔍 Extracting text and emotion data..."
    
    dataset_samples = []
    
    # Process each video
    for video_id in listtrain_ids + listvalid_ids + listtest_ids:
        if video_id in mosei_words and video_id in sentiments:
            for segment_id in mosei_words[video_id]:
                if segment_id in sentiments[video_id]:
                    # Get text from words
                    segment_words = mosei_words[video_id][segment_id]
                    if segment_words:
                        # Convert word timestamps to text
                        text = " ".join[word[2] for word in segment_words if word[2]]
                        
                        # Get sentiment label
                        sentiment = sentiments[video_id][segment_id]
                        
                        if text.strip() and sentiment is not None:
                            dataset_samples.append({
                                'text': text.strip(),
                                'sentiment': sentiment,
                                'video_id': video_id,
                                'segment_id': segment_id
                            })
    
    print(f"✅ Extracted {lendataset_samples} samples")
    return dataset_samples

def map_sentiment_to_emotionssamples:
    """Map CMU-MOSEI sentiment scores to our 12 target emotions"""
    print"🗺️ Mapping sentiments to emotions..."
    
    # CMU-MOSEI sentiment range: [-3, 3]
    # Our target emotions: anxious, calm, content, excited, frustrated, grateful, happy, hopeful, overwhelmed, proud, sad, tired
    
    emotion_mapping = {
        # Very negative sentiments
        -3, -2.5: 'sad',
        -2.5, -2: 'frustrated', 
        -2, -1.5: 'anxious',
        -1.5, -1: 'tired',
        -1, -0.5: 'overwhelmed',
        
        # Neutral sentiments
        -0.5, 0.5: 'calm',
        
        # Positive sentiments
        0.5, 1: 'content',
        1, 1.5: 'hopeful',
        1.5, 2: 'grateful',
        2, 2.5: 'happy',
        2.5, 3: 'excited',
    }
    
    mapped_samples = []
    
    for sample in samples:
        sentiment = sample['sentiment']
        
        # Find appropriate emotion mapping
        mapped_emotion = None
        for min_sent, max_sent, emotion in emotion_mapping.items():
            if min_sent <= sentiment < max_sent:
                mapped_emotion = emotion
                break
        
        # Default mapping for edge cases
        if mapped_emotion is None:
            if sentiment < -2.5:
                mapped_emotion = 'sad'
            elif sentiment > 2.5:
                mapped_emotion = 'excited'
            else:
                mapped_emotion = 'calm'
        
        mapped_samples.append({
            'text': sample['text'],
            'emotion': mapped_emotion,
            'original_sentiment': sentiment,
            'video_id': sample['video_id'],
            'segment_id': sample['segment_id']
        })
    
    print(f"✅ Mapped {lenmapped_samples} samples to emotions")
    
    # Show emotion distribution
    emotion_counts = defaultdictint
    for sample in mapped_samples:
        emotion_counts[sample['emotion']] += 1
    
    print"📊 Emotion distribution:"
    for emotion, count in sorted(emotion_counts.items()):
        printf"  {emotion}: {count} samples"
    
    return mapped_samples

def save_cmu_mosei_datasetsamples:
    """Save processed CMU-MOSEI dataset"""
    print"💾 Saving CMU-MOSEI dataset..."
    
    # Save full dataset
    output_file = 'data/cmu_mosei_emotion_dataset.json'
    with openoutput_file, 'w' as f:
        json.dumpsamples, f, indent=2
    
    print(f"✅ Saved {lensamples} samples to {output_file}")
    
    # Create balanced subset for training similar to your 12 emotions
    print"⚖️ Creating balanced training subset..."
    
    emotion_samples = defaultdictlist
    for sample in samples:
        emotion_samples[sample['emotion']].appendsample
    
    # Find minimum samples per emotion
    min_samples = min(lensamples for samples in emotion_samples.values())
    printf"📊 Minimum samples per emotion: {min_samples}"
    
    # Create balanced dataset
    balanced_samples = []
    for emotion, samples_list in emotion_samples.items():
        # Randomly sample min_samples from each emotion
        selected_samples = np.random.choicesamples_list, size=min_samples, replace=False
        balanced_samples.extendselected_samples
    
    balanced_file = 'data/cmu_mosei_balanced_dataset.json'
    with openbalanced_file, 'w' as f:
        json.dumpbalanced_samples, f, indent=2
    
    print(f"✅ Saved {lenbalanced_samples} balanced samples to {balanced_file}")
    
    return output_file, balanced_file

def main():
    """Main integration process"""
    print"🚀 CMU-MOSEI DATASET INTEGRATION"
    print"=" * 50
    
    # Step 1: Download dataset
    mosei_emb, mosei_words, sentiments, train_ids, valid_ids, test_ids = download_cmu_mosei()
    
    if mosei_words is None:
        print"❌ Failed to download CMU-MOSEI dataset"
        return
    
    # Step 2: Extract text and emotions
    samples = extract_text_and_emotionsmosei_words, sentiments, train_ids, valid_ids, test_ids
    
    if not samples:
        print"❌ No samples extracted"
        return
    
    # Step 3: Map to target emotions
    mapped_samples = map_sentiment_to_emotionssamples
    
    # Step 4: Save datasets
    full_file, balanced_file = save_cmu_mosei_datasetmapped_samples
    
    print"\n🎉 CMU-MOSEI Integration Complete!"
    print"📋 Next steps:"
    print"  1. Review the datasets in data/"
    print"  2. Use cmu_mosei_balanced_dataset.json for training"
    print"  3. Upload to Colab and achieve 75-85% F1 score!"

if __name__ == "__main__":
    main() 