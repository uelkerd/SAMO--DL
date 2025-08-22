#!/usr/bin/env python3
"""
Simple F1 Score Evaluation Script

This script evaluates the current F1 score of the emotion detection model.
"""

import logging
import sys
from pathlib import Path

import torch
from sklearn.metrics import f1_score, precision_score, recall_score

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.models.emotion_detection.bert_classifier import create_bert_emotion_classifier
from src.models.emotion_detection.dataset_loader import GoEmotionsDataLoader
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def evaluate_current_f1():
    """Evaluate the current F1 score of the emotion detection model."""
    logger.info("🎯 Evaluating Current F1 Score")
    logger.info("=" * 50)

    try:
        # Load dataset
        logger.info("📊 Loading GoEmotions dataset...")
        data_loader = GoEmotionsDataLoader()
        data_loader.download_dataset()
        datasets = data_loader.prepare_datasets()

        # Load model
        logger.info("🤖 Loading emotion detection model...")
        model, loss_fn = create_bert_emotion_classifier()
        
        # Check for existing checkpoint
        checkpoint_paths = [
            "models/checkpoints/bert_emotion_classifier_final.pt",
            "test_checkpoints/best_model.pt",
            "test_checkpoints_dev/best_model.pt",
        ]
        
        checkpoint_loaded = False
        for checkpoint_path in checkpoint_paths:
            if Path(checkpoint_path).exists():
                try:
                    logger.info(f"📁 Loading checkpoint: {checkpoint_path}")
                    checkpoint = torch.load(checkpoint_path, map_location="cpu")
                    if "model_state_dict" in checkpoint:
                        model.load_state_dict(checkpoint["model_state_dict"])
                        checkpoint_loaded = True
                        logger.info("✅ Checkpoint loaded successfully")
                        break
                except Exception as e:
                    logger.warning(
                                   f"⚠️ Failed to load checkpoint {checkpoint_path}: {e}"
                                  )
                    continue
        
        if not checkpoint_loaded:
            logger.warning("⚠️ No valid checkpoint found, using untrained model")

        # Create tokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        # Evaluate on test set
        logger.info("🧪 Evaluating on test set...")
        
        test_data = datasets["test_data"]
        all_predictions = []
        all_labels = []
        
        batch_size = 16
        num_classes = 28  # GoEmotions has 28 emotion classes
        
        with torch.no_grad():
            for i in range(0, len(test_data), batch_size):
                end_idx = min(i + batch_size, len(test_data))
                batch_data = test_data.select(range(i, end_idx))
                
                texts = batch_data["text"]
                labels = batch_data["labels"]
                
                # Convert labels to one-hot format
                batch_labels = []
                for label_list in labels:
                    label_vector = [0] * num_classes
                    for label_idx in label_list:
                        if 0 <= label_idx < num_classes:
                            label_vector[label_idx] = 1
                    batch_labels.append(label_vector)
                
                # Tokenize
                inputs = tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
                
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)
                
                # Get predictions
                outputs = model(input_ids, attention_mask)
                predictions = torch.sigmoid(outputs) > 0.5
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch_labels)
                
                if (i // batch_size + 1) % 10 == 0:
                    logger.info(f"   Processed {end_idx}/{len(test_data)} samples")

        # Calculate metrics
        logger.info("📈 Calculating metrics...")
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        # Calculate F1 scores
        micro_f1 = f1_score(
                            all_labels,
                            all_predictions,
                            average='micro',
                            zero_division=0
                           )
        macro_f1 = f1_score(
                            all_labels,
                            all_predictions,
                            average='macro',
                            zero_division=0
                           )
        weighted_f1 = f1_score(
                               all_labels,
                               all_predictions,
                               average='weighted',
                               zero_division=0
                              )
        
        # Calculate precision and recall
        micro_precision = precision_score(
                                          all_labels,
                                          all_predictions,
                                          average='micro',
                                          zero_division=0
                                         )
        micro_recall = recall_score(
                                    all_labels,
                                    all_predictions,
                                    average='micro',
                                    zero_division=0
                                   )
        
        # Display results
        logger.info("📊 EVALUATION RESULTS:")
        logger.info("=" * 50)
        logger.info(f"Micro F1 Score:     {micro_f1:.4f} ({micro_f1*100:.2f}%)")
        logger.info(f"Macro F1 Score:     {macro_f1:.4f} ({macro_f1*100:.2f}%)")
        logger.info(f"Weighted F1 Score:  {weighted_f1:.4f} ({weighted_f1*100:.2f}%)")
        logger.info(
                    f"Micro Precision:    {micro_precision:.4f} ({micro_precision*100:.2f}%)"
                   )
        logger.info(f"Micro Recall:       {micro_recall:.4f} ({micro_recall*100:.2f}%)")
        logger.info("=" * 50)
        
        # Assessment
        target_f1 = 0.80  # 80% target
        progress = (micro_f1 / target_f1) * 100
        
        logger.info(f"🎯 TARGET F1: {target_f1*100:.0f}%")
        logger.info(f"📊 CURRENT F1: {micro_f1*100:.2f}%")
        logger.info(f"📈 PROGRESS: {progress:.1f}% of target")
        
        if micro_f1 >= target_f1:
            logger.info("🎉 TARGET ACHIEVED!")
        else:
            gap = target_f1 - micro_f1
            logger.info(f"📉 GAP: {gap*100:.2f} percentage points needed")
            
        return {
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "target_f1": target_f1,
            "progress_percent": progress
        }

    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    import numpy as np
    results = evaluate_current_f1()
    if results:
        logger.info("✅ Evaluation completed successfully")
    else:
        logger.error("❌ Evaluation failed")
        sys.exit(1) 