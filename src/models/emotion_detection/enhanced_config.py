"""
Enhanced Configuration Manager for SAMO Emotion Detection.

This module provides robust configuration loading, validation, and management
for the emotion detection system with comprehensive error handling and fallbacks.
"""

import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class DeviceType(Enum):
    """Device types."""
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


@dataclass
class ModelConfig:
    """Model configuration parameters."""
    name: str = "bert-base-uncased"
    device: Optional[str] = None
    use_mixed_precision: bool = True
    cache_embeddings: bool = False
    max_sequence_length: int = 512


@dataclass
class EmotionDetectionConfig:
    """Emotion detection configuration parameters."""
    num_emotions: int = 28
    prediction_threshold: float = 0.6
    temperature: float = 1.0
    top_k: int = 5


@dataclass
class ArchitectureConfig:
    """Model architecture configuration."""
    hidden_dropout_prob: float = 0.3
    attention_probs_dropout_prob: float = 0.3
    classifier_dropout_prob: float = 0.5
    freeze_bert_layers: int = 6
    use_class_weights: bool = True


@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    train_batch_size: int = 16
    eval_batch_size: int = 32
    bert_learning_rate: float = 2e-5
    classifier_learning_rate: float = 5e-4
    num_epochs: int = 10
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.01


@dataclass
class DataConfig:
    """Data processing configuration."""
    max_length: int = 512
    truncation: bool = True
    padding: str = "max_length"
    enable_augmentation: bool = False
    validation_split: float = 0.2
    test_split: float = 0.1


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    metrics: List[str] = field(default_factory=lambda: [
        "precision", "recall", "f1_micro", "f1_macro", "accuracy"
    ])
    threshold: float = 0.2
    top_k_evaluation: bool = True
    top_k_values: List[int] = field(default_factory=lambda: [1, 3, 5])


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    log_interval: int = 100
    save_interval: int = 1000
    enable_tensorboard: bool = True
    log_dir: str = "logs/emotion_detection"


@dataclass
class ModelSavingConfig:
    """Model saving configuration."""
    save_dir: str = "models/emotion_detection"
    save_best_metric: str = "f1_macro"
    save_checkpoints: bool = True
    checkpoint_interval: int = 1


@dataclass
class PerformanceConfig:
    """Performance optimization configuration."""
    use_amp: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    gradient_checkpointing: bool = False
    use_torchscript: bool = False


@dataclass
class SAMOOptimizationsConfig:
    """SAMO-specific optimizations."""
    journal_entry_mode: bool = True
    context_awareness: bool = True
    multi_label_mode: bool = True
    calibration_enabled: bool = True
    intensity_scaling: bool = True


@dataclass
class ErrorHandlingConfig:
    """Error handling configuration."""
    max_retries: int = 3
    retry_delay: float = 1.0
    fallback_to_cpu: bool = True
    graceful_degradation: bool = True
    log_errors: bool = True
    error_log_file: str = "logs/emotion_detection_errors.log"


@dataclass
class SecurityConfig:
    """Security and privacy configuration."""
    sanitize_input: bool = True
    filter_sensitive_emotions: bool = False
    rate_limit_requests: int = 1000
    anonymize_predictions: bool = False


@dataclass
class DevelopmentConfig:
    """Development and debugging configuration."""
    debug_mode: bool = False
    verbose: bool = False
    test_mode: bool = False
    enable_profiling: bool = False
    profile_steps: int = 100


@dataclass
class EnhancedEmotionDetectionConfig:
    """Enhanced configuration container for emotion detection."""
    model: ModelConfig = field(default_factory=ModelConfig)
    emotion_detection: EmotionDetectionConfig = field(
        default_factory=EmotionDetectionConfig
    )
    architecture: ArchitectureConfig = field(
        default_factory=ArchitectureConfig
    )
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    model_saving: ModelSavingConfig = field(
        default_factory=ModelSavingConfig
    )
    performance: PerformanceConfig = field(
        default_factory=PerformanceConfig
    )
    samo_optimizations: SAMOOptimizationsConfig = field(
        default_factory=SAMOOptimizationsConfig
    )
    error_handling: ErrorHandlingConfig = field(
        default_factory=ErrorHandlingConfig
    )
    security: SecurityConfig = field(default_factory=SecurityConfig)
    development: DevelopmentConfig = field(
        default_factory=DevelopmentConfig
    )


class EnhancedConfigManager:
    """Enhanced configuration manager with validation and fallbacks."""

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize configuration manager.

        Args:
            config_path: Path to configuration file. If None, uses default.
        """
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> EnhancedEmotionDetectionConfig:
        """Load configuration with comprehensive error handling."""
        if self.config_path is None:
            self.config_path = self._find_default_config()

        if self.config_path is None or not Path(self.config_path).exists():
            logger.warning("No configuration file found, using defaults")
            return self._create_default_config()

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f) or {}

            logger.info("Configuration loaded from: %s", self.config_path)
            return self._parse_config(config_data)

        except yaml.YAMLError as e:
            logger.error("YAML parsing error: %s", e)
            logger.warning("Using default configuration due to YAML error")
            return self._create_default_config()
        except Exception as e:
            logger.error("Configuration loading failed: %s", e)
            logger.warning("Using default configuration due to loading error")
            return self._create_default_config()

    @staticmethod
    def _find_default_config() -> Optional[Path]:
        """Find default configuration file."""
        possible_paths = [
            Path("configs/samo_emotion_detection_config.yaml"),
            Path("configs/emotion_detection_config.yaml"),
            Path("emotion_detection_config.yaml"),
        ]

        for path in possible_paths:
            if path.exists():
                return path

        return None

    @staticmethod
    def _create_default_config() -> EnhancedEmotionDetectionConfig:
        """Create default configuration."""
        logger.info("Creating default configuration")
        return EnhancedEmotionDetectionConfig()

    def _parse_config(self, config_data: Dict[str, Any]) -> EnhancedEmotionDetectionConfig:
        """Parse configuration data into structured format."""
        try:
            # Parse each section with validation
            model_config = self._parse_model_config(
                config_data.get("model", {})
            )
            emotion_config = self._parse_emotion_detection_config(
                config_data.get("emotion_detection", {})
            )
            architecture_config = self._parse_architecture_config(
                config_data.get("architecture", {})
            )
            training_config = self._parse_training_config(
                config_data.get("training", {})
            )
            data_config = self._parse_data_config(
                config_data.get("data", {})
            )
            evaluation_config = self._parse_evaluation_config(
                config_data.get("evaluation", {})
            )
            logging_config = self._parse_logging_config(
                config_data.get("logging", {})
            )
            model_saving_config = self._parse_model_saving_config(
                config_data.get("model_saving", {})
            )
            performance_config = self._parse_performance_config(
                config_data.get("performance", {})
            )
            samo_optimizations = self._parse_samo_optimizations(
                config_data.get("samo_optimizations", {})
            )
            error_handling = self._parse_error_handling(
                config_data.get("error_handling", {})
            )
            security_config = self._parse_security_config(
                config_data.get("security", {})
            )
            development_config = self._parse_development_config(
                config_data.get("development", {})
            )

            return EnhancedEmotionDetectionConfig(
                model=model_config,
                emotion_detection=emotion_config,
                architecture=architecture_config,
                training=training_config,
                data=data_config,
                evaluation=evaluation_config,
                logging=logging_config,
                model_saving=model_saving_config,
                performance=performance_config,
                samo_optimizations=samo_optimizations,
                error_handling=error_handling,
                security=security_config,
                development=development_config,
            )
        except Exception as e:
            logger.error("Configuration parsing failed: %s", e)
            logger.warning("Using default configuration due to parsing error")
            return self._create_default_config()

    def _parse_model_config(self, data: Dict[str, Any]) -> ModelConfig:
        """Parse model configuration with validation."""
        return ModelConfig(
            name=self._validate_string(
                data.get("name", "bert-base-uncased"), "model.name"
            ),
            device=self._validate_device(data.get("device")),
            use_mixed_precision=self._validate_bool(
                data.get("use_mixed_precision", True), "model.use_mixed_precision"
            ),
            cache_embeddings=self._validate_bool(
                data.get("cache_embeddings", False), "model.cache_embeddings"
            ),
            max_sequence_length=self._validate_positive_int(
                data.get("max_sequence_length", 512), "model.max_sequence_length"
            ),
        )

    def _parse_emotion_detection_config(self, data: Dict[str, Any]) -> EmotionDetectionConfig:
        """Parse emotion detection configuration with validation."""
        return EmotionDetectionConfig(
            num_emotions=self._validate_positive_int(
                data.get("num_emotions", 28), "emotion_detection.num_emotions"
            ),
            prediction_threshold=self._validate_float_range(
                data.get("prediction_threshold", 0.6), 0.0, 1.0, 
                "emotion_detection.prediction_threshold"
            ),
            temperature=self._validate_positive_float(
                data.get("temperature", 1.0), "emotion_detection.temperature"
            ),
            top_k=self._validate_positive_int(
                data.get("top_k", 5), "emotion_detection.top_k"
            ),
        )

    def _parse_architecture_config(self, data: Dict[str, Any]) -> ArchitectureConfig:
        """Parse architecture configuration with validation."""
        return ArchitectureConfig(
            hidden_dropout_prob=self._validate_float_range(data.get("hidden_dropout_prob", 0.3), 0.0, 1.0, "architecture.hidden_dropout_prob"),
            attention_probs_dropout_prob=self._validate_float_range(data.get("attention_probs_dropout_prob", 0.3), 0.0, 1.0, "architecture.attention_probs_dropout_prob"),
            classifier_dropout_prob=self._validate_float_range(data.get("classifier_dropout_prob", 0.5), 0.0, 1.0, "architecture.classifier_dropout_prob"),
            freeze_bert_layers=self._validate_non_negative_int(data.get("freeze_bert_layers", 6), "architecture.freeze_bert_layers"),
            use_class_weights=self._validate_bool(data.get("use_class_weights", True), "architecture.use_class_weights"),
        )

    def _parse_training_config(self, data: Dict[str, Any]) -> TrainingConfig:
        """Parse training configuration with validation."""
        return TrainingConfig(
            train_batch_size=self._validate_positive_int(data.get("train_batch_size", 16), "training.train_batch_size"),
            eval_batch_size=self._validate_positive_int(data.get("eval_batch_size", 32), "training.eval_batch_size"),
            bert_learning_rate=self._validate_positive_float(data.get("bert_learning_rate", 2e-5), "training.bert_learning_rate"),
            classifier_learning_rate=self._validate_positive_float(data.get("classifier_learning_rate", 5e-4), "training.classifier_learning_rate"),
            num_epochs=self._validate_positive_int(data.get("num_epochs", 10), "training.num_epochs"),
            warmup_steps=self._validate_non_negative_int(data.get("warmup_steps", 100), "training.warmup_steps"),
            max_grad_norm=self._validate_positive_float(data.get("max_grad_norm", 1.0), "training.max_grad_norm"),
            gradient_accumulation_steps=self._validate_positive_int(data.get("gradient_accumulation_steps", 1), "training.gradient_accumulation_steps"),
            early_stopping_patience=self._validate_positive_int(data.get("early_stopping_patience", 3), "training.early_stopping_patience"),
            early_stopping_threshold=self._validate_positive_float(data.get("early_stopping_threshold", 0.01), "training.early_stopping_threshold"),
        )

    def _parse_data_config(self, data: Dict[str, Any]) -> DataConfig:
        """Parse data configuration with validation."""
        return DataConfig(
            max_length=self._validate_positive_int(data.get("max_length", 512), "data.max_length"),
            truncation=self._validate_bool(data.get("truncation", True), "data.truncation"),
            padding=self._validate_string(data.get("padding", "max_length"), "data.padding"),
            enable_augmentation=self._validate_bool(data.get("enable_augmentation", False), "data.enable_augmentation"),
            validation_split=self._validate_float_range(data.get("validation_split", 0.2), 0.0, 1.0, "data.validation_split"),
            test_split=self._validate_float_range(data.get("test_split", 0.1), 0.0, 1.0, "data.test_split"),
        )

    def _parse_evaluation_config(self, data: Dict[str, Any]) -> EvaluationConfig:
        """Parse evaluation configuration with validation."""
        return EvaluationConfig(
            metrics=self._validate_list(data.get("metrics", ["precision", "recall", "f1_micro", "f1_macro", "accuracy"]), "evaluation.metrics"),
            threshold=self._validate_float_range(data.get("threshold", 0.2), 0.0, 1.0, "evaluation.threshold"),
            top_k_evaluation=self._validate_bool(data.get("top_k_evaluation", True), "evaluation.top_k_evaluation"),
            top_k_values=self._validate_list(data.get("top_k_values", [1, 3, 5]), "evaluation.top_k_values"),
        )

    def _parse_logging_config(self, data: Dict[str, Any]) -> LoggingConfig:
        """Parse logging configuration with validation."""
        return LoggingConfig(
            level=self._validate_log_level(data.get("level", "INFO"), "logging.level"),
            log_interval=self._validate_positive_int(data.get("log_interval", 100), "logging.log_interval"),
            save_interval=self._validate_positive_int(data.get("save_interval", 1000), "logging.save_interval"),
            enable_tensorboard=self._validate_bool(data.get("enable_tensorboard", True), "logging.enable_tensorboard"),
            log_dir=self._validate_string(data.get("log_dir", "logs/emotion_detection"), "logging.log_dir"),
        )

    def _parse_model_saving_config(self, data: Dict[str, Any]) -> ModelSavingConfig:
        """Parse model saving configuration with validation."""
        return ModelSavingConfig(
            save_dir=self._validate_string(data.get("save_dir", "models/emotion_detection"), "model_saving.save_dir"),
            save_best_metric=self._validate_string(data.get("save_best_metric", "f1_macro"), "model_saving.save_best_metric"),
            save_checkpoints=self._validate_bool(data.get("save_checkpoints", True), "model_saving.save_checkpoints"),
            checkpoint_interval=self._validate_positive_int(data.get("checkpoint_interval", 1), "model_saving.checkpoint_interval"),
        )

    def _parse_performance_config(self, data: Dict[str, Any]) -> PerformanceConfig:
        """Parse performance configuration with validation."""
        return PerformanceConfig(
            use_amp=self._validate_bool(data.get("use_amp", True), "performance.use_amp"),
            num_workers=self._validate_non_negative_int(data.get("num_workers", 4), "performance.num_workers"),
            pin_memory=self._validate_bool(data.get("pin_memory", True), "performance.pin_memory"),
            gradient_checkpointing=self._validate_bool(data.get("gradient_checkpointing", False), "performance.gradient_checkpointing"),
            use_torchscript=self._validate_bool(data.get("use_torchscript", False), "performance.use_torchscript"),
        )

    def _parse_samo_optimizations(self, data: Dict[str, Any]) -> SAMOOptimizationsConfig:
        """Parse SAMO optimizations configuration with validation."""
        return SAMOOptimizationsConfig(
            journal_entry_mode=self._validate_bool(data.get("journal_entry_mode", True), "samo_optimizations.journal_entry_mode"),
            context_awareness=self._validate_bool(data.get("context_awareness", True), "samo_optimizations.context_awareness"),
            multi_label_mode=self._validate_bool(data.get("multi_label_mode", True), "samo_optimizations.multi_label_mode"),
            calibration_enabled=self._validate_bool(data.get("calibration_enabled", True), "samo_optimizations.calibration_enabled"),
            intensity_scaling=self._validate_bool(data.get("intensity_scaling", True), "samo_optimizations.intensity_scaling"),
        )

    def _parse_error_handling(self, data: Dict[str, Any]) -> ErrorHandlingConfig:
        """Parse error handling configuration with validation."""
        return ErrorHandlingConfig(
            max_retries=self._validate_non_negative_int(
                data.get("max_retries", 3), "error_handling.max_retries"
            ),
            retry_delay=self._validate_positive_float(
                data.get("retry_delay", 1.0), "error_handling.retry_delay"
            ),
            fallback_to_cpu=self._validate_bool(
                data.get("fallback_to_cpu", True), "error_handling.fallback_to_cpu"
            ),
            graceful_degradation=self._validate_bool(
                data.get("graceful_degradation", True), "error_handling.graceful_degradation"
            ),
            log_errors=self._validate_bool(
                data.get("log_errors", True), "error_handling.log_errors"
            ),
            error_log_file=self._validate_string(
                data.get("error_log_file", "logs/emotion_detection_errors.log"),
                "error_handling.error_log_file"
            ),
        )

    def _parse_security_config(self, data: Dict[str, Any]) -> SecurityConfig:
        """Parse security configuration with validation."""
        return SecurityConfig(
            sanitize_input=self._validate_bool(
                data.get("sanitize_input", True), "security.sanitize_input"
            ),
            filter_sensitive_emotions=self._validate_bool(
                data.get("filter_sensitive_emotions", False),
                "security.filter_sensitive_emotions"
            ),
            rate_limit_requests=self._validate_positive_int(
                data.get("rate_limit_requests", 1000), "security.rate_limit_requests"
            ),
            anonymize_predictions=self._validate_bool(
                data.get("anonymize_predictions", False), "security.anonymize_predictions"
            ),
        )

    def _parse_development_config(self, data: Dict[str, Any]) -> DevelopmentConfig:
        """Parse development configuration with validation."""
        return DevelopmentConfig(
            debug_mode=self._validate_bool(
                data.get("debug_mode", False), "development.debug_mode"
            ),
            verbose=self._validate_bool(
                data.get("verbose", False), "development.verbose"
            ),
            test_mode=self._validate_bool(
                data.get("test_mode", False), "development.test_mode"
            ),
            enable_profiling=self._validate_bool(
                data.get("enable_profiling", False), "development.enable_profiling"
            ),
            profile_steps=self._validate_positive_int(
                data.get("profile_steps", 100), "development.profile_steps"
            ),
        )

    # Validation methods
    @staticmethod
    def _validate_string(value: Any, field_name: str) -> str:
        """Validate string value."""
        if not isinstance(value, str):
            logger.warning(
                "Invalid string value for %s: %s, using default", field_name, value
            )
            return ""
        return value

    @staticmethod
    def _validate_bool(value: Any, field_name: str) -> bool:
        """Validate boolean value."""
        if not isinstance(value, bool):
            logger.warning(
                "Invalid boolean value for %s: %s, using default", field_name, value
            )
            return False
        return value

    @staticmethod
    def _validate_positive_int(value: Any, field_name: str) -> int:
        """Validate positive integer value."""
        try:
            int_val = int(value)
            if int_val <= 0:
                logger.warning(
                    "Non-positive integer for %s: %s, using default", field_name, value
                )
                return 1
            return int_val
        except (ValueError, TypeError):
            logger.warning(
                "Invalid integer for %s: %s, using default", field_name, value
            )
            return 1

    @staticmethod
    def _validate_non_negative_int(value: Any, field_name: str) -> int:
        """Validate non-negative integer value."""
        try:
            int_val = int(value)
            if int_val < 0:
                logger.warning(
                    "Negative integer for %s: %s, using default", field_name, value
                )
                return 0
            return int_val
        except (ValueError, TypeError):
            logger.warning(
                "Invalid integer for %s: %s, using default", field_name, value
            )
            return 0

    @staticmethod
    def _validate_positive_float(value: Any, field_name: str) -> float:
        """Validate positive float value."""
        try:
            float_val = float(value)
            if float_val <= 0:
                logger.warning(
                    "Non-positive float for %s: %s, using default", field_name, value
                )
                return 1.0
            return float_val
        except (ValueError, TypeError):
            logger.warning(
                "Invalid float for %s: %s, using default", field_name, value
            )
            return 1.0

    @staticmethod
    def _validate_float_range(
        value: Any, min_val: float, max_val: float, field_name: str
    ) -> float:
        """Validate float value within range."""
        try:
            float_val = float(value)
            if not min_val <= float_val <= max_val:
                logger.warning(
                    "Float out of range for %s: %s, using default", field_name, value
                )
                return (min_val + max_val) / 2
            return float_val
        except (ValueError, TypeError):
            logger.warning(
                "Invalid float for %s: %s, using default", field_name, value
            )
            return (min_val + max_val) / 2

    @staticmethod
    def _validate_device(value: Any) -> Optional[str]:
        """Validate device value."""
        if value is None:
            return None
        if not isinstance(value, str):
            logger.warning("Invalid device value: %s, using auto", value)
            return None
        valid_devices = ["auto", "cpu", "cuda", "mps"]
        if value.lower() not in valid_devices:
            logger.warning("Invalid device: %s, using auto", value)
            return None
        return value.lower()

    @staticmethod
    def _validate_log_level(value: Any, field_name: str) -> str:
        """Validate log level value."""
        if not isinstance(value, str):
            logger.warning(
                "Invalid log level for %s: %s, using default", field_name, value
            )
            return "INFO"
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if value.upper() not in valid_levels:
            logger.warning(
                "Invalid log level for %s: %s, using default", field_name, value
            )
            return "INFO"
        return value.upper()

    @staticmethod
    def _validate_list(value: Any, field_name: str) -> List:
        """Validate list value."""
        if not isinstance(value, list):
            logger.warning("Invalid list for %s: %s, using default", field_name, value)
            return []
        return value

    def get_config(self) -> EnhancedEmotionDetectionConfig:
        """Get the current configuration."""
        return self.config

    @staticmethod
    def update_config(updates: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        try:
            # This would need more sophisticated merging logic
            # For now, just log the attempt
            logger.info("Configuration update requested: %s", updates)
        except Exception as e:
            logger.error("Configuration update failed: %s", e)

    def save_config(self, path: Optional[Union[str, Path]] = None) -> None:
        """Save current configuration to file."""
        if path is None:
            path = self.config_path or "configs/samo_emotion_detection_config.yaml"

        try:
            # Convert config to dictionary and save as YAML
            config_dict = self._config_to_dict()
            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            logger.info("Configuration saved to: %s", path)
        except Exception as e:
            logger.error("Failed to save configuration: %s", e)

    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        # This would need proper serialization logic
        # For now, return a basic structure
        return {
            "model": {
                "name": self.config.model.name,
                "device": self.config.model.device,
                "use_mixed_precision": self.config.model.use_mixed_precision,
                "cache_embeddings": self.config.model.cache_embeddings,
                "max_sequence_length": self.config.model.max_sequence_length,
            },
            "emotion_detection": {
                "num_emotions": self.config.emotion_detection.num_emotions,
                "prediction_threshold": (
                    self.config.emotion_detection.prediction_threshold
                ),
                "temperature": self.config.emotion_detection.temperature,
                "top_k": self.config.emotion_detection.top_k,
            },
            # Add other sections as needed
        }


def create_enhanced_config_manager(
    config_path: Optional[Union[str, Path]] = None
) -> EnhancedConfigManager:
    """Create an enhanced configuration manager.

    Args:
        config_path: Path to configuration file

    Returns:
        Enhanced configuration manager instance
    """
    return EnhancedConfigManager(config_path)
