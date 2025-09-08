"""Model Validator for Secure Model Loading.

This module provides model validation capabilities including:
- Model structure validation
- Version compatibility checks
- Configuration validation
- Performance validation
"""
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

logger = logging.getLogger(__name__)


class ModelValidator:
    """Model validator for secure model loading.

    Provides comprehensive model validation including:
    - Model structure validation
    - Version compatibility checks
    - Configuration validation
    - Performance validation
    """

    def __init__(self,
                 allowed_model_types: Optional[List[str]] = None,
                 max_model_size_mb: int = 2048,
                 required_config_keys: Optional[List[str]] = None):
        """Initialize model validator.

        Args:
            allowed_model_types: List of allowed model types
            max_model_size_mb: Maximum model size in MB
            required_config_keys: Required configuration keys
        """
        self.allowed_model_types = allowed_model_types or [
            'BERTEmotionClassifier', 'T5Summarizer', 'WhisperTranscriber'
        ]
        self.max_model_size_mb = max_model_size_mb
        self.required_config_keys = required_config_keys or [
            'model_name', 'num_emotions', 'hidden_dropout_prob'
        ]

        # Version compatibility matrix
        self.version_compatibility = {
            'torch': '>=1.9.0',
            'transformers': '>=4.20.0',
            'tokenizers': '>=0.12.0'
        }

    def validate_model_structure(self, model: nn.Module) -> Tuple[bool, Dict]:
        """Validate model structure.

        Args:
            model: PyTorch model to validate

        Returns:
            Tuple of (is_valid, validation_info)
        """
        validation_info = {
            'model_type': type(model).__name__,
            'parameter_count': 0,
            'layers': [],
            'issues': []
        }

        try:
            # Check model type
            if type(model).__name__ not in self.allowed_model_types:
                validation_info['issues'].append(f"Model type {type(model).__name__} not allowed")

            # Count parameters
            param_count = sum(p.numel() for p in model.parameters())
            validation_info['parameter_count'] = param_count

            # Check for reasonable parameter count
            if param_count > 500_000_000:  # 500M parameters
                validation_info['issues'].append("Model has too many parameters")

            # Analyze model layers
            for name, module in model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d, nn.LSTM, nn.Transformer)):
                    validation_info['layers'].append({
                        'name': name,
                        'type': type(module).__name__,
                        'parameters': sum(p.numel() for p in module.parameters())
                    })

            # Check for required methods
            required_methods = ['forward', 'eval', 'train']
            for method in required_methods:
                if not hasattr(model, method):
                    validation_info['issues'].append(f"Missing required method: {method}")

            is_valid = len(validation_info['issues']) == 0
            return is_valid, validation_info

        except Exception as e:
            validation_info['issues'].append(f"Validation error: {e}")
            return False, validation_info

    def validate_model_config(self, config: Dict[str, Any]) -> Tuple[bool, Dict]:
        """Validate model configuration.

        Args:
            config: Model configuration dictionary

        Returns:
            Tuple of (is_valid, validation_info)
        """
        validation_info = {
            'config_keys': list(config.keys()),
            'missing_keys': [],
            'invalid_values': [],
            'issues': []
        }

        try:
            # Check required keys
            for key in self.required_config_keys:
                if key not in config:
                    validation_info['missing_keys'].append(key)

            # Validate specific config values
            if 'num_emotions' in config:
                num_emotions = config['num_emotions']
                if not isinstance(num_emotions, int) or num_emotions <= 0:
                    validation_info['invalid_values'].append(f"num_emotions: {num_emotions}")

            if 'hidden_dropout_prob' in config:
                dropout = config['hidden_dropout_prob']
                if not isinstance(dropout, (int, float)) or dropout < 0 or dropout > 1:
                    validation_info['invalid_values'].append(f"hidden_dropout_prob: {dropout}")

            # Check for issues
            if validation_info['missing_keys']:
                validation_info['issues'].append(f"Missing required keys: {validation_info['missing_keys']}")

            if validation_info['invalid_values']:
                validation_info['issues'].append(f"Invalid values: {validation_info['invalid_values']}")

            is_valid = len(validation_info['issues']) == 0
            return is_valid, validation_info

        except Exception as e:
            validation_info['issues'].append(f"Config validation error: {e}")
            return False, validation_info

    def validate_model_file(self, model_path: str) -> Tuple[bool, Dict]:
        """Validate model file.

        Args:
            model_path: Path to the model file

        Returns:
            Tuple of (is_valid, validation_info)
        """
        validation_info = {
            'file_path': model_path,
            'file_size_mb': 0,
            'file_exists': False,
            'is_readable': False,
            'loadable': False,
            'issues': []
        }

        try:
            # Check file existence
            if not os.path.exists(model_path):
                validation_info['issues'].append("Model file does not exist")
                return False, validation_info

            validation_info['file_exists'] = True

            # Check file size
            file_size = os.path.getsize(model_path)
            file_size_mb = file_size / (1024 * 1024)
            validation_info['file_size_mb'] = file_size_mb

            if file_size_mb > self.max_model_size_mb:
                validation_info['issues'].append(f"Model file too large: {file_size_mb:.2f}MB")

            # Check if file is readable
            if not os.access(model_path, os.R_OK):
                validation_info['issues'].append("Model file is not readable")
                return False, validation_info

            validation_info['is_readable'] = True

            # Try to load the model
            try:
                model_data = torch.load(model_path, map_location='cpu', weights_only=True)
                validation_info['loadable'] = True

                # Validate model data structure
                if not isinstance(model_data, dict):
                    validation_info['issues'].append("Model file is not a valid state dict")
                else:
                    # Check for required keys
                    if 'state_dict' not in model_data:
                        validation_info['issues'].append("Model file missing state_dict")

                    if 'config' not in model_data:
                        validation_info['issues'].append("Model file missing config")

            except Exception as e:
                validation_info['issues'].append(f"Failed to load model: {e}")

            is_valid = len(validation_info['issues']) == 0
            return is_valid, validation_info

        except Exception as e:
            validation_info['issues'].append(f"File validation error: {e}")
            return False, validation_info

    def validate_version_compatibility(self, model_config: Dict[str, Any]) -> Tuple[bool, Dict]:
        """Validate version compatibility.

        Args:
            model_config: Model configuration

        Returns:
            Tuple of (is_valid, validation_info)
        """
        validation_info = {
            'current_versions': {},
            'required_versions': self.version_compatibility,
            'compatibility_issues': [],
            'issues': []
        }

        try:
            # Get current versions
            import torch
            import transformers

            validation_info['current_versions'] = {
                'torch': torch.__version__,
                'transformers': transformers.__version__
            }

            # Check version compatibility
            for package, _required_version in self.version_compatibility.items():
                if package in validation_info['current_versions']:
                    current_version = validation_info['current_versions'][package]
                    # Enhanced version check that supports PyTorch 2.x
                    if package == 'torch':
                        # Allow PyTorch 1.x and 2.x versions
                        if not (current_version.startswith('1.') or current_version.startswith('2.')):
                            validation_info['compatibility_issues'].append(f"PyTorch version {current_version} may not be compatible")
                    elif package == 'transformers' and not current_version.startswith('4.'):
                        validation_info['compatibility_issues'].append(f"Transformers version {current_version} may not be compatible")

            # Check for issues
            if validation_info['compatibility_issues']:
                validation_info['issues'].extend(validation_info['compatibility_issues'])

            is_valid = len(validation_info['issues']) == 0
            return is_valid, validation_info

        except Exception as e:
            validation_info['issues'].append(f"Version validation error: {e}")
            return False, validation_info

    def validate_model_performance(self, model: nn.Module, test_input: torch.Tensor) -> Tuple[bool, Dict]:
        """Validate model performance with test input.

        Args:
            model: PyTorch model
            test_input: Test input tensor

        Returns:
            Tuple of (is_valid, validation_info)
        """
        validation_info = {
            'forward_pass_time': 0,
            'memory_usage_mb': 0,
            'output_shape': None,
            'issues': []
        }

        try:
            import time

            # Set model to eval mode
            model.eval()

            # Measure forward pass time
            start_time = time.time()
            with torch.no_grad():
                output = model(test_input)
            end_time = time.time()

            validation_info['forward_pass_time'] = end_time - start_time
            validation_info['output_shape'] = list(output.shape)

            # Check performance constraints
            if validation_info['forward_pass_time'] > 5.0:  # 5 seconds
                validation_info['issues'].append("Forward pass too slow")

            # Check output shape
            if output.dim() != 2:  # Expected 2D output for classification
                validation_info['issues'].append("Unexpected output shape")

            # Measure memory usage
            if hasattr(torch.cuda, 'memory_allocated'):
                memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                validation_info['memory_usage_mb'] = memory_mb

                if memory_mb > 2048:  # 2GB
                    validation_info['issues'].append("Memory usage too high")

            is_valid = len(validation_info['issues']) == 0
            return is_valid, validation_info

        except Exception as e:
            validation_info['issues'].append(f"Performance validation error: {e}")
            return False, validation_info

    def comprehensive_validation(self,
                                model_path: str,
                                model_class: type,
                                model_config: Dict[str, Any],
                                test_input: Optional[torch.Tensor] = None) -> Tuple[bool, Dict]:
        """Perform comprehensive model validation.

        Args:
            model_path: Path to the model file
            model_class: Model class
            model_config: Model configuration
            test_input: Optional test input for performance validation

        Returns:
            Tuple of (is_valid, comprehensive_validation_info)
        """
        comprehensive_info = {
            'file_validation': {},
            'config_validation': {},
            'version_validation': {},
            'structure_validation': {},
            'performance_validation': {},
            'overall_valid': False,
            'issues': []
        }

        try:
            # 1. File validation
            file_valid, file_info = self.validate_model_file(model_path)
            comprehensive_info['file_validation'] = file_info
            if not file_valid:
                comprehensive_info['issues'].extend(file_info['issues'])

            # 2. Config validation
            config_valid, config_info = self.validate_model_config(model_config)
            comprehensive_info['config_validation'] = config_info
            if not config_valid:
                comprehensive_info['issues'].extend(config_info['issues'])

            # 3. Version validation
            version_valid, version_info = self.validate_version_compatibility(model_config)
            comprehensive_info['version_validation'] = version_info
            if not version_valid:
                comprehensive_info['issues'].extend(version_info['issues'])

            # 4. Structure validation (if file is valid)
            if file_valid:
                try:
                    model_data = torch.load(model_path, map_location='cpu', weights_only=True)

                    # Filter model_config to only include valid constructor parameters
                    import inspect
                    constructor_params = inspect.signature(model_class.__init__).parameters
                    valid_params = {k: v for k, v in model_config.items() if k in constructor_params}
                    model = model_class(**valid_params)

                    if 'state_dict' in model_data:
                        model.load_state_dict(model_data['state_dict'])

                    structure_valid, structure_info = self.validate_model_structure(model)
                    comprehensive_info['structure_validation'] = structure_info
                    if not structure_valid:
                        comprehensive_info['issues'].extend(structure_info['issues'])

                    # 5. Performance validation (if structure is valid and test input provided)
                    if structure_valid and test_input is not None:
                        perf_valid, perf_info = self.validate_model_performance(model, test_input)
                        comprehensive_info['performance_validation'] = perf_info
                        if not perf_valid:
                            comprehensive_info['issues'].extend(perf_info['issues'])

                except Exception as e:
                    comprehensive_info['issues'].append(f"Model loading error: {e}")

            # Overall validation result
            comprehensive_info['overall_valid'] = len(comprehensive_info['issues']) == 0

            return comprehensive_info['overall_valid'], comprehensive_info

        except Exception as e:
            comprehensive_info['issues'].append(f"Comprehensive validation error: {e}")
            return False, comprehensive_info
