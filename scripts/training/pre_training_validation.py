                # Test write permissions
            # Backward pass
            # Check CUDA availability
            # Check available disk space
            # Check class weights
            # Check data shapes and types
            # Check dataset structure
            # Check for all-zero or all-one labels
            # Check gradients
            # Check output directories
            # Create model
            # Create trainer
            # Forward pass
            # Load data in dev mode
            # Move to device if available
            # Optimizer step
            # Prepare data
            # Test forward pass
            # Test forward pass with dummy data
            # Test learning rate
            # Test loss function
            # Test one training step
            # Test optimizer
            # Test scheduler
            # Validate first batch
            # Validate labels
            # Validate outputs
import logging
            import shutil
import sys
# Add src to path
# Configure logging
        # Critical issues
    # Exit with appropriate code
        # Final recommendation
    # Generate report
    # Run all validations
        # Summary
        # Warnings
#!/usr/bin/env python3
import numpy as np
            import pandas as pd
            import torch
import torch
            import transformers
# Import torch early for validation
from pathlib import Path
            from src.models.emotion_detection.bert_classifier import create_bert_emotion_classifier
            from src.models.emotion_detection.dataset_loader import create_goemotions_loader
            from src.models.emotion_detection.training_pipeline import EmotionDetectionTrainer
            from torch.optim import AdamW









""""
Pre-Training Validation Script for SAMO Deep Learning.

This script performs comprehensive validation BEFORE training starts to prevent
issues like 0.0000 loss, data problems, model issues, etc.
""""

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class PreTrainingValidator:
    """Comprehensive pre-training validation system."""

    def __init__(self):
        self.validation_results = {}
        self.critical_issues = []
        self.warnings = []

    def validate_environment(self) -> bool:
        """Validate Python environment and dependencies."""
        logger.info(" Validating environment...")

        try:
            logger.info(" PyTorch version: {torch.__version__}")
            logger.info(" Transformers version: {transformers.__version__}")
            logger.info(" NumPy version: {np.__version__}")
            logger.info(" Pandas version: {pd.__version__}")

            if torch.cuda.is_available():
                logger.info(" CUDA available: {torch.cuda.get_device_name(0)}")
                logger.info()
                    " CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
(                )
            else:
                logger.warning("⚠️  CUDA not available, using CPU")

            self.validation_results["environment"] = True
            return True

        except ImportError as _:
            logger.error("❌ Missing dependency: {e}")
            self.critical_issues.append("Missing dependency: {e}")
            self.validation_results["environment"] = False
            return False

        def validate_data_loading(self) -> bool:
        """Validate data loading and preprocessing."""
        logger.info(" Validating data loading...")

        try:
            datasets = create_goemotions_loader(dev_mode=True)

            required_keys = ["train_dataloader", "val_dataloader", "class_weights"]
            for key in required_keys:
                if key not in datasets:
                    logger.error("❌ Missing dataset key: {key}")
                    self.critical_issues.append("Missing dataset key: {key}")
                    return False

            train_dataloader = datasets["train_dataloader"]
            val_dataloader = datasets["val_dataloader"]
            class_weights = datasets["class_weights"]

            logger.info(" Train batches: {len(train_dataloader)}")
            logger.info(" Val batches: {len(val_dataloader)}")

            first_batch = next(iter(train_dataloader))
            required_batch_keys = ["input_ids", "attention_mask", "labels"]

                for key in required_batch_keys:
                if key not in first_batch:
                    logger.error("❌ Missing batch key: {key}")
                    self.critical_issues.append("Missing batch key: {key}")
                    return False

            input_ids = first_batch["input_ids"]
            attention_mask = first_batch["attention_mask"]
            labels = first_batch["labels"]

            logger.info(" Input shape: {input_ids.shape}")
            logger.info(" Attention shape: {attention_mask.shape}")
            logger.info(" Labels shape: {labels.shape}")

                if labels.dtype not in (torch.float32, torch.float64):
                logger.error("❌ Labels should be float, got: {labels.dtype}")
                self.critical_issues.append("Invalid labels dtype: {labels.dtype}")
                return False

            labels_sum = labels.sum().item()
            labels_total = labels.numel()

            logger.info(" Labels sum: {labels_sum}")
            logger.info(" Labels total: {labels_total}")
            logger.info(" Labels mean: {labels.float().mean().item():.6f}")

                if labels_sum == 0:
                logger.error("❌ CRITICAL: All labels are zero!")
                self.critical_issues.append("All labels are zero")
                return False

                if labels_sum == labels_total:
                logger.error("❌ CRITICAL: All labels are one!")
                self.critical_issues.append("All labels are one")
                return False

                if class_weights is not None:
                logger.info(" Class weights shape: {class_weights.shape}")
                logger.info(" Class weights min: {class_weights.min():.6f}")
                logger.info(" Class weights max: {class_weights.max():.6f}")

                if class_weights.min() <= 0:
                    logger.error("❌ CRITICAL: Class weights contain zero or negative values!")
                    self.critical_issues.append("Invalid class weights")
                    return False

                if class_weights.max() > 100:
                    logger.warning("⚠️  Class weights contain very large values")
                    self.warnings.append("Large class weights detected")

            self.validation_results["data_loading"] = True
            return True

        except Exception as e:
            logger.error("❌ Data loading validation failed: {e}")
            self.critical_issues.append("Data loading error: {e}")
            self.validation_results["data_loading"] = False
            return False

                def validate_model_architecture(self) -> bool:
        """Validate model architecture and initialization."""
        logger.info(" Validating model architecture...")

        try:
            model, loss_fn = create_bert_emotion_classifier()
                model_name="bert-base-uncased",
                class_weights=None,  # Test without weights first
                freeze_bert_layers=6,
(            )

            logger.info(" Model created successfully")
            logger.info(" Model parameters: {model.count_parameters():,}")
            logger.info(" Loss function: {type(loss_fn).__name__}")

            batch_size = 4
            seq_length = 128
            num_classes = 28

            dummy_input_ids = torch.randint(0, 1000, (batch_size, seq_length))
            dummy_attention_mask = torch.ones(batch_size, seq_length)
            dummy_labels = torch.randint(0, 2, (batch_size, num_classes)).float()

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            dummy_input_ids = dummy_input_ids.to(device)
            dummy_attention_mask = dummy_attention_mask.to(device)
            dummy_labels = dummy_labels.to(device)

            model.eval()
            with torch.no_grad():
                logits = model(dummy_input_ids, dummy_attention_mask)
                loss = loss_fn(logits, dummy_labels)

            logger.info(" Forward pass successful")
            logger.info(" Logits shape: {logits.shape}")
            logger.info(" Loss value: {loss.item():.6f}")

                if logits.shape != (batch_size, num_classes):
                logger.error("❌ Wrong logits shape: {logits.shape}")
                self.critical_issues.append("Wrong logits shape: {logits.shape}")
                return False

                if torch.isnan(logits).any():
                logger.error("❌ CRITICAL: NaN values in model outputs!")
                self.critical_issues.append("NaN in model outputs")
                return False

                if torch.isinf(logits).any():
                logger.error("❌ CRITICAL: Inf values in model outputs!")
                self.critical_issues.append("Inf in model outputs")
                return False

                if loss.item() <= 0:
                logger.error("❌ CRITICAL: Loss is zero or negative: {loss.item()}")
                self.critical_issues.append("Invalid loss value: {loss.item()}")
                return False

                if torch.isnan(loss).any():
                logger.error("❌ CRITICAL: NaN loss!")
                self.critical_issues.append("NaN loss")
                return False

            self.validation_results["model_architecture"] = True
            return True

        except Exception as e:
            logger.error("❌ Model architecture validation failed: {e}")
            self.critical_issues.append("Model architecture error: {e}")
            self.validation_results["model_architecture"] = False
            return False

                def validate_training_components(self) -> bool:
        """Validate training components (optimizer, scheduler, etc.)."""
        logger.info(" Validating training components...")

        try:
            trainer = EmotionDetectionTrainer()
                model_name="bert-base-uncased",
                batch_size=8,
                learning_rate=2e-6,
                num_epochs=1,
                dev_mode=True,
(            )

            trainer.prepare_data(dev_mode=True)
            trainer.initialize_model()

            logger.info(" Trainer created successfully")
            logger.info(" Optimizer: {type(trainer.optimizer).__name__}")
            logger.info(" Scheduler: {type(trainer.scheduler).__name__}")
            logger.info(" Learning rate: {trainer.learning_rate}")

                if not isinstance(trainer.optimizer, AdamW):
                logger.warning("⚠️  Optimizer is not AdamW")
                self.warnings.append("Non-standard optimizer")

                if trainer.scheduler is None:
                logger.error("❌ Scheduler is None!")
                self.critical_issues.append("Missing scheduler")
                return False

                if trainer.learning_rate <= 0:
                logger.error("❌ Invalid learning rate: {trainer.learning_rate}")
                self.critical_issues.append("Invalid learning rate: {trainer.learning_rate}")
                return False

                if trainer.learning_rate > 1e-3:
                logger.warning("⚠️  Learning rate might be too high: {trainer.learning_rate}")
                self.warnings.append("High learning rate: {trainer.learning_rate}")

            batch = next(iter(trainer.train_dataloader))
            input_ids = batch["input_ids"].to(trainer.device)
            attention_mask = batch["attention_mask"].to(trainer.device)
            labels = batch["labels"].to(trainer.device)

            trainer.model.train()
            logits = trainer.model(input_ids, attention_mask)
            loss = trainer.loss_fn(logits, labels)

            trainer.optimizer.zero_grad()
            loss.backward()

            total_norm = 0
            param_count = 0
                for p in trainer.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1

                if param_count > 0:
                total_norm = total_norm ** (1.0 / 2)
                logger.info(" Gradient norm: {total_norm:.6f}")

                if total_norm > 100:
                    logger.warning("⚠️  Large gradient norm detected")
                    self.warnings.append("Large gradient norm: {total_norm}")

                if total_norm < 1e-8:
                    logger.warning("⚠️  Very small gradient norm detected")
                    self.warnings.append("Small gradient norm: {total_norm}")

            trainer.optimizer.step()
            trainer.scheduler.step()

            logger.info(" Training step completed successfully")
            logger.info(" Loss after step: {loss.item():.6f}")

            self.validation_results["training_components"] = True
            return True

        except Exception as e:
            logger.error("❌ Training components validation failed: {e}")
            self.critical_issues.append("Training components error: {e}")
            self.validation_results["training_components"] = False
            return False

                def validate_file_system(self) -> bool:
        """Validate file system and permissions."""
        logger.info(" Validating file system...")

        try:
            output_dirs = ["./models/emotion_detection", "./data/cache", "./logs"]

                for dir_path in output_dirs:
                path = Path(dir_path)
                if not path.exists():
                    path.mkdir(parents=True, exist_ok=True)
                    logger.info(" Created directory: {dir_path}")

                test_file = path / "test_write.tmp"
                try:
                    test_file.write_text("test")
                    test_file.unlink()
                    logger.info(" Write permission: {dir_path}")
                except Exception:
                    logger.error("❌ No write permission: {dir_path}")
                    self.critical_issues.append("No write permission: {dir_path}")
                    return False

            total, used, free = shutil.disk_usage(".")
            free_gb = free / (1024**3)

            logger.info(" Available disk space: {free_gb:.1f} GB")

                if free_gb < 10:
                logger.warning("⚠️  Low disk space (< 10 GB)")
                self.warnings.append("Low disk space: {free_gb:.1f} GB")

            self.validation_results["file_system"] = True
            return True

        except Exception as e:
            logger.error("❌ File system validation failed: {e}")
            self.critical_issues.append("File system error: {e}")
            self.validation_results["file_system"] = False
            return False

                def run_all_validations(self) -> bool:
        """Run all validation checks."""
        logger.info("🚀 Starting comprehensive pre-training validation...")

        validations = [
            ("Environment", self.validate_environment),
            ("File System", self.validate_file_system),
            ("Data Loading", self.validate_data_loading),
            ("Model Architecture", self.validate_model_architecture),
            ("Training Components", self.validate_training_components),
        ]

        all_passed = True

                for name, validation_func in validations:
            logger.info("\n{'='*60}")
            logger.info("Running: {name} Validation")
            logger.info("{'='*60}")

            try:
                if not validation_func():
                    all_passed = False
                    logger.error("❌ {name} validation FAILED")
                else:
                    logger.info(" {name} validation PASSED")
            except Exception as e:
                logger.error("❌ {name} validation ERROR: {e}")
                self.critical_issues.append("{name} validation error: {e}")
                all_passed = False

        return all_passed

                def generate_report(self) -> None:
        """Generate comprehensive validation report."""
        logger.info("\n{'='*80}")
        logger.info(" PRE-TRAINING VALIDATION REPORT")
        logger.info("{'='*80}")

        total_checks = len(self.validation_results)
        passed_checks = sum(self.validation_results.values())

        logger.info(" Validation Summary:")
        logger.info("   Total checks: {total_checks}")
        logger.info("   Passed: {passed_checks}")
        logger.info("   Failed: {total_checks - passed_checks}")

                if self.critical_issues:
            logger.error("\n❌ CRITICAL ISSUES ({len(self.critical_issues)}):")
                for i, issue in enumerate(self.critical_issues, 1):
                logger.error("   {i}. {issue}")

                if self.warnings:
            logger.warning("\n⚠️  WARNINGS ({len(self.warnings)}):")
                for i, warning in enumerate(self.warnings, 1):
                logger.warning("   {i}. {warning}")

                if self.critical_issues:
            logger.error()
                "\n🚫 TRAINING BLOCKED: {len(self.critical_issues)} critical issues found!"
(            )
            logger.error("   Please fix all critical issues before starting training.")
        elif self.warnings:
            logger.warning("\n⚠️  TRAINING ALLOWED with {len(self.warnings)} warnings.")
            logger.warning("   Consider addressing warnings before training.")
        else:
            logger.info("\n TRAINING READY: All validations passed!")
            logger.info("   You can safely start training.")


                def main():
    """Main validation function."""
    validator = PreTrainingValidator()

    validator.run_all_validations()

    validator.generate_report()

                if validator.critical_issues:
        logger.error("❌ Validation failed - training blocked!")
        return False
    else:
        logger.info(" Validation passed - training can proceed!")
        return True


                if __name__ == "__main__":
    success = main()
                if not success:
        sys.exit(1)
