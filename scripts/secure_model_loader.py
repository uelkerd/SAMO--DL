#!/usr/bin/env python3
"""
Secure Model Loader
==================

This module provides secure model loading utilities to address PyTorch RCE vulnerabilities.
It implements multiple layers of security to prevent remote code execution attacks.
"""

import os
import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import safetensors.torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecureModelLoader:
    """
    Secure model loader that prevents RCE vulnerabilities in PyTorch model loading.
    """
    
    def __init__(self, model_path: str, allowed_model_types: Optional[list] = None):
        """
        Initialize secure model loader.
        
        Args:
            model_path: Path to the model directory
            allowed_model_types: List of allowed model types (e.g., ['roberta', 'distilroberta'])
        """
        self.model_path = Path(model_path)
        self.allowed_model_types = allowed_model_types or ['roberta', 'distilroberta', 'bert']
        self.expected_files = ['config.json', 'tokenizer.json', 'vocab.json']
        self.security_checks_passed = False
        
    def validate_model_path(self) -> bool:
        """Validate that the model path exists and contains expected files."""
        try:
            if not self.model_path.exists():
                logger.error(f"Model path does not exist: {self.model_path}")
                return False
                
            if not self.model_path.is_dir():
                logger.error(f"Model path is not a directory: {self.model_path}")
                return False
                
            # Check for required files
            missing_files = []
            for file in self.expected_files:
                if not (self.model_path / file).exists():
                    missing_files.append(file)
                    
            if missing_files:
                logger.error(f"Missing required files: {missing_files}")
                return False
                
            logger.info("✅ Model path validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Model path validation failed: {e}")
            return False
    
    def validate_config_security(self) -> bool:
        """Validate model configuration for security."""
        try:
            config_path = self.model_path / 'config.json'
            
            # Read config safely
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Check model type
            model_type = config.get('model_type', '').lower()
            if model_type not in self.allowed_model_types:
                logger.error(f"Model type '{model_type}' not in allowed types: {self.allowed_model_types}")
                return False
            
            # Check for suspicious configurations
            suspicious_keys = ['_name_or_path', 'custom_pipelines', 'trust_remote_code']
            for key in suspicious_keys:
                if key in config:
                    logger.warning(f"Suspicious config key found: {key}")
            
            # Validate architecture
            architectures = config.get('architectures', [])
            if not architectures:
                logger.error("No architectures specified in config")
                return False
                
            allowed_architectures = [
                'RobertaForSequenceClassification',
                'DistilRobertaForSequenceClassification', 
                'BertForSequenceClassification'
            ]
            
            for arch in architectures:
                if arch not in allowed_architectures:
                    logger.error(f"Architecture '{arch}' not in allowed list: {allowed_architectures}")
                    return False
            
            logger.info("✅ Config security validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Config security validation failed: {e}")
            return False
    
    def calculate_model_hash(self) -> str:
        """Calculate SHA256 hash of model files for integrity checking."""
        try:
            hasher = hashlib.sha256()
            
            # Hash config file
            config_path = self.model_path / 'config.json'
            with open(config_path, 'rb') as f:
                hasher.update(f.read())
            
            # Hash model weights (prefer safetensors)
            model_files = ['model.safetensors', 'pytorch_model.bin']
            for model_file in model_files:
                model_path = self.model_path / model_file
                if model_path.exists():
                    with open(model_path, 'rb') as f:
                        hasher.update(f.read())
                    break
            
            # Hash tokenizer files
            tokenizer_files = ['tokenizer.json', 'vocab.json', 'merges.txt']
            for tokenizer_file in tokenizer_files:
                tokenizer_path = self.model_path / tokenizer_file
                if tokenizer_path.exists():
                    with open(tokenizer_path, 'rb') as f:
                        hasher.update(f.read())
            
            return hasher.hexdigest()
            
        except Exception as e:
            logger.error(f"Hash calculation failed: {e}")
            return ""
    
    def load_model_safely(self) -> tuple[AutoTokenizer, AutoModelForSequenceClassification]:
        """
        Load model with multiple security layers.
        
        Returns:
            Tuple of (tokenizer, model)
        """
        try:
            # Step 1: Validate model path
            if not self.validate_model_path():
                raise SecurityError("Model path validation failed")
            
            # Step 2: Validate config security
            if not self.validate_config_security():
                raise SecurityError("Config security validation failed")
            
            # Step 3: Calculate and log model hash
            model_hash = self.calculate_model_hash()
            logger.info(f"Model hash: {model_hash}")
            
            # Step 4: Load config safely
            logger.info("Loading model configuration...")
            config = AutoConfig.from_pretrained(
                str(self.model_path),
                trust_remote_code=False,  # CRITICAL: Disable remote code
                local_files_only=True     # CRITICAL: Only load local files
            )
            
            # Step 5: Load tokenizer safely
            logger.info("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path),
                trust_remote_code=False,  # CRITICAL: Disable remote code
                local_files_only=True     # CRITICAL: Only load local files
            )
            
            # Step 6: Load model safely
            logger.info("Loading model...")
            model = AutoModelForSequenceClassification.from_pretrained(
                str(self.model_path),
                config=config,
                trust_remote_code=False,  # CRITICAL: Disable remote code
                local_files_only=True,    # CRITICAL: Only load local files
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            
            # Step 7: Verify model integrity
            if not self.verify_model_integrity(model, config):
                raise SecurityError("Model integrity verification failed")
            
            self.security_checks_passed = True
            logger.info("✅ Model loaded securely")
            
            return tokenizer, model
            
        except Exception as e:
            logger.error(f"Secure model loading failed: {e}")
            raise SecurityError(f"Model loading failed: {e}")
    
    def verify_model_integrity(self, model: AutoModelForSequenceClassification, config: AutoConfig) -> bool:
        """Verify model integrity after loading."""
        try:
            # Check model type
            if not hasattr(model, 'config'):
                logger.error("Model has no config attribute")
                return False
            
            # Check expected attributes
            expected_attrs = ['classifier', 'roberta', 'distilbert']
            has_expected_attr = any(hasattr(model, attr) for attr in expected_attrs)
            if not has_expected_attr:
                logger.error("Model missing expected attributes")
                return False
            
            # Check classifier output size
            if hasattr(model, 'classifier'):
                if hasattr(model.classifier, 'out_proj'):
                    output_size = model.classifier.out_proj.out_features
                    expected_size = config.num_labels
                    if output_size != expected_size:
                        logger.error(f"Classifier output size mismatch: {output_size} != {expected_size}")
                        return False
            
            logger.info("✅ Model integrity verification passed")
            return True
            
        except Exception as e:
            logger.error(f"Model integrity verification failed: {e}")
            return False

class SecurityError(Exception):
    """Custom exception for security-related errors."""
    pass

def load_emotion_model_securely(model_path: str) -> tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    """
    Secure wrapper for loading emotion detection model.
    
    Args:
        model_path: Path to the model directory
        
    Returns:
        Tuple of (tokenizer, model)
        
    Raises:
        SecurityError: If security checks fail
    """
    loader = SecureModelLoader(model_path)
    return loader.load_model_safely()

# Security utilities
def sanitize_input(text: str) -> str:
    """Sanitize user input to prevent injection attacks."""
    if not isinstance(text, str):
        raise ValueError("Input must be a string")
    
    # Remove potentially dangerous characters
    dangerous_chars = ['<', '>', '"', "'", '&', ';', '|', '`', '$', '(', ')', '{', '}']
    for char in dangerous_chars:
        text = text.replace(char, '')
    
    # Limit length
    if len(text) > 1000:
        text = text[:1000]
    
    return text.strip()

def validate_prediction_output(prediction: Dict[str, Any]) -> bool:
    """Validate prediction output for security."""
    try:
        required_keys = ['predicted_emotion', 'confidence']
        for key in required_keys:
            if key not in prediction:
                return False
        
        # Validate confidence is a number between 0 and 1
        confidence = prediction.get('confidence', 0)
        if not isinstance(confidence, (int, float)) or not (0 <= confidence <= 1):
            return False
        
        # Validate emotion is a string
        emotion = prediction.get('predicted_emotion', '')
        if not isinstance(emotion, str):
            return False
        
        return True
        
    except Exception:
        return False

if __name__ == "__main__":
    # Test the secure loader
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python secure_model_loader.py <model_path>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    try:
        tokenizer, model = load_emotion_model_securely(model_path)
        print("✅ Model loaded securely!")
        print(f"Model type: {type(model).__name__}")
        print(f"Tokenizer type: {type(tokenizer).__name__}")
        
    except SecurityError as e:
        print(f"❌ Security error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1) 