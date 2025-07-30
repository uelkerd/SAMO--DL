                    # Backward pass
                    # Calculate loss
                    # Create one-hot encoded labels
                    # Forward pass
                    # Move batch to device
                    # Tokenize
                # Add to augmented dataset
                # Apply sigmoid and threshold
                # Average predictions
                # Back-translate
                # Create one-hot encoded labels
                # English to German
                # Forward pass
                # German to English
                # Get predictions from all models
                # Move batch to device
                # Tokenize
                # Update ensemble temperature
                # Update temperature for all models
            # Augment a subset of the training data
            # Combine original and augmented data
            # Create augmented dataset
            # Create custom datasets
            # Create data loaders
            # Create model and tokenizer
            # Create optimizer
            # Function for back-translation
            # Get a small subset for augmentation (to save time)
            # Move models to device
            # Save model
            # Train for a few epochs
            # Train model on augmented data
            from torch.utils.data import Dataset, DataLoader
            from transformers import AutoTokenizer
            from transformers import MarianMTModel, MarianTokenizer
        # Apply class weights if provided
        # Calculate binary cross entropy loss
        # Calculate class weights
        # Calculate focal loss
        # Calculate focal weight
        # Calculate metrics
        # Check if transformers is available
        # Convert logits to probabilities
        # Create base model
        # Create data loader for validation set
        # Create dataset
        # Create ensemble model
        # Create ensemble of models with different configurations
        # Create ensemble wrapper class
        # Create model
        # Create model with fewer frozen layers
        # Create model with more frozen layers
        # Create tokenizer
        # Create trainer
        # Create trainer with Focal Loss
        # Create validation dataset and loader
        # Evaluate
        # Evaluate ensemble model
        # Evaluate model
        # Load checkpoint for base model
        # Load dataset
        # Load dataset
        # Load dataset
        # Load state dict for all models
        # Load translation models (English -> German -> English)
        # Perform back-translation augmentation on training data
        # Process validation data
        # Save ensemble model
        # Save model
        # Set models to evaluation mode
        # Set optimal temperature and threshold
        # Train model
        from torch.utils.data import DataLoader
        from torch.utils.data import Dataset
        from transformers import AutoTokenizer
    # Apply selected technique
    # Set output model path
# Add src to path
# Configure logging
# Constants
#!/usr/bin/env python3
from pathlib import Path
from sklearn.metrics import f1_score
from src.models.emotion_detection.bert_classifier import create_bert_emotion_classifier
from src.models.emotion_detection.dataset_loader import GoEmotionsDataLoader
from src.models.emotion_detection.training_pipeline import EmotionDetectionTrainer
from torch import nn
from typing import Optional
import argparse
import logging
import sys
import torch
import torch.nn.functional as F







"""
Improve Model F1 Score

This script implements advanced techniques for improving the BERT emotion classifier's F1 score:
1. Data augmentation using back-translation
2. Focal loss for handling class imbalance
3. Advanced class weighting
4. Ensemble prediction

Usage:
    python scripts/improve_model_f1.py [--technique TECHNIQUE] [--output_model PATH]

Arguments:
    --technique: Improvement technique to apply (augmentation, focal_loss, weighting, ensemble)
    --output_model: Path to save improved model (default: models/checkpoints/bert_emotion_classifier_improved.pt)
"""

sys.path.append(str(Path(__file__).parent.parent.resolve()))
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_MODEL = "models/checkpoints/bert_emotion_classifier_improved.pt"
CHECKPOINT_PATH = "test_checkpoints/best_model.pt"
OPTIMAL_TEMPERATURE = 1.0
OPTIMAL_THRESHOLD = 0.6


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance.

    Focal Loss reduces the relative loss for well-classified examples,
    focusing more on hard, misclassified examples.

    Reference: https://arxiv.org/abs/1708.02002
    """

    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None):
        """Initialize Focal Loss.

        Args:
            gamma: Focusing parameter (>= 0). Higher values focus more on hard examples.
            alpha: Optional class weights. If provided, should be a tensor of shape (num_classes,).
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            inputs: Model predictions (logits) of shape (batch_size, num_classes)
            targets: Ground truth labels of shape (batch_size, num_classes)

        Returns:
            Focal loss value
        """
        probs = torch.sigmoid(inputs)

        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_weight = alpha_t * focal_weight

        focal_loss = focal_weight * bce_loss

        return focal_loss.mean()


def improve_with_focal_loss() -> bool:
    """Improve model F1 score using Focal Loss.

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info("Improving model with Focal Loss...")

        data_loader = GoEmotionsDataLoader()
        datasets = data_loader.prepare_datasets()

        class_weights = data_loader.compute_class_weights()
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

        model, _ = create_bert_emotion_classifier()

        focal_loss = FocalLoss(gamma=2.0, alpha=class_weights_tensor)

        trainer = EmotionDetectionTrainer(
            model=model,
            loss_fn=focal_loss,
            learning_rate=2e-5,
            batch_size=32,
            num_epochs=3,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            checkpoint_dir=Path("models/checkpoints"),
            early_stopping_patience=3,
            dev_mode=True,  # Use smaller dataset for faster training
        )

        logger.info("Training model with Focal Loss...")
        trainer.train(datasets["train"], datasets["validation"])

        logger.info("Evaluating model...")
        metrics = trainer.evaluate(datasets["test"])

        logger.info("Micro F1: {metrics['micro_f1']:.4f}")
        logger.info("Macro F1: {metrics['macro_f1']:.4f}")

        output_path = Path(DEFAULT_OUTPUT_MODEL)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Saving improved model to {output_path}...")
        trainer.save_checkpoint(output_path, metrics)

        logger.info("✅ Model improvement with Focal Loss complete!")
        return True

    except Exception as e:
        logger.error("Error improving model with Focal Loss: {e}")
        return False


def improve_with_data_augmentation() -> bool:
    """Improve model F1 score using data augmentation.

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info("Improving model with data augmentation...")

        try:
        except ImportError:
            logger.error(
                "transformers library not found. Please install it with: pip install transformers"
            )
            return False

        data_loader = GoEmotionsDataLoader()
        datasets = data_loader.prepare_datasets()

        logger.info("Performing back-translation augmentation...")

        logger.info("Loading translation models...")
        try:
            en_de_model_name = "Helsinki-NLP/opus-mt-en-de"
            de_en_model_name = "Helsinki-NLP/opus-mt-de-en"

            en_de_tokenizer = MarianTokenizer.from_pretrained(en_de_model_name)
            en_de_model = MarianMTModel.from_pretrained(en_de_model_name)

            de_en_tokenizer = MarianTokenizer.from_pretrained(de_en_model_name)
            de_en_model = MarianMTModel.from_pretrained(de_en_model_name)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            en_de_model.to(device)
            de_en_model.to(device)

            def back_translate(text: str) -> str:
                en_de_inputs = en_de_tokenizer(text, return_tensors="pt").to(device)
                en_de_outputs = en_de_model.generate(**en_de_inputs)
                de_text = en_de_tokenizer.decode(en_de_outputs[0], skip_special_tokens=True)

                de_en_inputs = de_en_tokenizer(de_text, return_tensors="pt").to(device)
                de_en_outputs = de_en_model.generate(**de_en_inputs)
                en_text = de_en_tokenizer.decode(de_en_outputs[0], skip_special_tokens=True)

                return en_text

            logger.info("Augmenting training data...")

            train_texts = datasets["train"]["text"][:100]  # Adjust size as needed
            train_labels = datasets["train"]["labels"][:100]

            augmented_texts = []
            augmented_labels = []

            for i, (text, label) in enumerate(zip(train_texts, train_labels)):
                if i % 10 == 0:
                    logger.info("Augmenting example {i}/{len(train_texts)}...")

                augmented_text = back_translate(text)

                augmented_texts.append(augmented_text)
                augmented_labels.append(label)

            combined_texts = list(train_texts) + augmented_texts
            combined_labels = list(train_labels) + augmented_labels

            logger.info("Augmented dataset size: {len(combined_texts)} examples")

            class AugmentedDataset(Dataset):
                def __init__(self, texts, labels, tokenizer, max_length=512):
                    self.texts = texts
                    self.labels = labels
                    self.tokenizer = tokenizer
                    self.max_length = max_length

                def __len__(self):
                    return len(self.texts)

                def __getitem__(self, idx):
                    text = self.texts[idx]
                    label = self.labels[idx]

                    encoding = self.tokenizer(
                        text,
                        max_length=self.max_length,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    )

                    num_labels = 28  # GoEmotions has 28 emotions
                    one_hot_labels = torch.zeros(num_labels)
                    for label_idx in label:
                        one_hot_labels[label_idx] = 1.0

                    return {
                        "input_ids": encoding["input_ids"].squeeze(),
                        "attention_mask": encoding["attention_mask"].squeeze(),
                        "labels": one_hot_labels,
                    }

            model, _ = create_bert_emotion_classifier()
            tokenizer = AutoTokenizer.from_pretrained(model.model_name)

            augmented_dataset = AugmentedDataset(combined_texts, combined_labels, tokenizer)

            augmented_loader = DataLoader(augmented_dataset, batch_size=8, shuffle=True)

            logger.info("Training model on augmented data...")

            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

            model.train()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

            for epoch in range(3):
                logger.info("Epoch {epoch+1}/3")
                for _batch_idx, batch in enumerate(augmented_loader):
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)

                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                    loss = F.binary_cross_entropy_with_logits(outputs, labels)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if batch_idx % 10 == 0:
                        logger.info("Batch {batch_idx}, Loss: {loss.item():.4f}")

            output_path = Path(DEFAULT_OUTPUT_MODEL)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            logger.info("Saving improved model to {output_path}...")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "augmentation": "back_translation",
                    "temperature": OPTIMAL_TEMPERATURE,
                    "threshold": OPTIMAL_THRESHOLD,
                },
                output_path,
            )

            logger.info("✅ Model improvement with data augmentation complete!")
            return True

        except Exception as e:
            logger.error("Error in back-translation: {e}")
            return False

    except Exception as e:
        logger.error("Error improving model with data augmentation: {e}")
        return False


def improve_with_ensemble() -> bool:
    """Improve model F1 score using ensemble prediction.

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info("Improving model with ensemble prediction...")

        data_loader = GoEmotionsDataLoader()
        datasets = data_loader.prepare_datasets()

        logger.info("Creating ensemble of models...")

        base_model, _ = create_bert_emotion_classifier()

        frozen_model, _ = create_bert_emotion_classifier()
        frozen_model.freeze_bert_layers = 8  # Freeze more layers

        unfrozen_model, _ = create_bert_emotion_classifier()
        unfrozen_model.freeze_bert_layers = 4  # Freeze fewer layers

        checkpoint_path = Path(CHECKPOINT_PATH)
        if not checkpoint_path.exists():
            logger.error("Checkpoint not found: {checkpoint_path}")
            return False

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            base_model.load_state_dict(checkpoint["model_state_dict"])
            frozen_model.load_state_dict(checkpoint["model_state_dict"])
            unfrozen_model.load_state_dict(checkpoint["model_state_dict"])
        else:
            logger.error("Unexpected checkpoint format: {type(checkpoint)}")
            return False

        base_model.set_temperature(OPTIMAL_TEMPERATURE)
        base_model.prediction_threshold = OPTIMAL_THRESHOLD

        frozen_model.set_temperature(OPTIMAL_TEMPERATURE * 1.2)  # Slightly different temperature
        frozen_model.prediction_threshold = OPTIMAL_THRESHOLD * 0.9  # Slightly different threshold

        unfrozen_model.set_temperature(OPTIMAL_TEMPERATURE * 0.8)  # Slightly different temperature
        unfrozen_model.prediction_threshold = (
            OPTIMAL_THRESHOLD * 1.1
        )  # Slightly different threshold

        class EnsembleModel(nn.Module):
            def __init__(self, models: list[nn.Module], weights: Optional[list[float]] = None):
                super().__init__()
                self.models = nn.ModuleList(models)
                self.weights = weights if weights is not None else [1.0] * len(models)
                self.prediction_threshold = OPTIMAL_THRESHOLD
                self.temperature = nn.Parameter(torch.ones(1))
                self.model_name = "bert-base-uncased"  # For compatibility

            def forward(self, **kwargs):
                outputs = []
                for i, model in enumerate(self.models):
                    with torch.no_grad():
                        output = model(**kwargs)
                        outputs.append(output * self.weights[i])

                ensemble_output = torch.stack(outputs).mean(dim=0)
                return ensemble_output

            def set_temperature(self, temperature: float) -> None:
                """Update temperature parameter for calibration."""
                if temperature <= 0:
                    raise ValueError("Temperature must be positive")

                for model in self.models:
                    model.set_temperature(temperature)

                with torch.no_grad():
                    self.temperature.fill_(temperature)

        ensemble = EnsembleModel(
            models=[base_model, frozen_model, unfrozen_model],
            weights=[0.5, 0.25, 0.25],  # Base model has higher weight
        )

        logger.info("Evaluating ensemble model...")

        ensemble.eval()

        tokenizer = AutoTokenizer.from_pretrained(base_model.model_name)

        val_texts = datasets["validation"]["text"]
        val_labels = datasets["validation"]["labels"]

        class ValidationDataset(Dataset):
            def __init__(self, texts, labels, tokenizer, max_length=512):
                self.texts = texts
                self.labels = labels
                self.tokenizer = tokenizer
                self.max_length = max_length

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                text = self.texts[idx]
                label = self.labels[idx]

                encoding = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )

                num_labels = 28  # GoEmotions has 28 emotions
                one_hot_labels = torch.zeros(num_labels)
                for label_idx in label:
                    one_hot_labels[label_idx] = 1.0

                return {
                    "input_ids": encoding["input_ids"].squeeze(),
                    "attention_mask": encoding["attention_mask"].squeeze(),
                    "labels": one_hot_labels,
                }

        val_dataset = ValidationDataset(val_texts, val_labels, tokenizer)
        val_loader = DataLoader(val_dataset, batch_size=32)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ensemble.to(device)

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"]

                outputs = ensemble(input_ids=input_ids, attention_mask=attention_mask)

                probs = torch.sigmoid(outputs / ensemble.temperature)
                preds = (probs > ensemble.prediction_threshold).float().cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        micro_f1 = f1_score(all_labels, all_preds, average="micro")
        macro_f1 = f1_score(all_labels, all_preds, average="macro")

        logger.info("Ensemble Micro F1: {micro_f1:.4f}")
        logger.info("Ensemble Macro F1: {macro_f1:.4f}")

        output_path = Path(DEFAULT_OUTPUT_MODEL)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Saving ensemble model to {output_path}...")
        torch.save(
            {
                "model_type": "ensemble",
                "base_model_state_dict": base_model.state_dict(),
                "frozen_model_state_dict": frozen_model.state_dict(),
                "unfrozen_model_state_dict": unfrozen_model.state_dict(),
                "weights": ensemble.weights,
                "temperature": OPTIMAL_TEMPERATURE,
                "threshold": OPTIMAL_THRESHOLD,
                "micro_f1": micro_f1,
                "macro_f1": macro_f1,
            },
            output_path,
        )

        logger.info("✅ Model improvement with ensemble prediction complete!")
        return True

    except Exception as e:
        logger.error("Error improving model with ensemble prediction: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Improve BERT emotion classifier F1 score")
    parser.add_argument(
        "--technique",
        type=str,
        choices=["focal_loss", "augmentation", "ensemble"],
        default="ensemble",
        help="Improvement technique to apply",
    )
    parser.add_argument(
        "--output_model",
        type=str,
        default=DEFAULT_OUTPUT_MODEL,
        help="Path to save improved model (default: {DEFAULT_OUTPUT_MODEL})",
    )

    args = parser.parse_args()

    DEFAULT_OUTPUT_MODEL = args.output_model

    if args.technique == "focal_loss":
        success = improve_with_focal_loss()
    elif args.technique == "augmentation":
        success = improve_with_data_augmentation()
    elif args.technique == "ensemble":
        success = improve_with_ensemble()
    else:
        logger.error("Unknown technique: {args.technique}")
        success = False

    sys.exit(0 if success else 1)
