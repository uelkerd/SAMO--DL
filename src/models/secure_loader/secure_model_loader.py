"""
Secure Model Loader for SAMO Deep Learning.

This module provides the main secure model loading interface that integrates
all security components: integrity checking, sandboxed execution, and validation.
"""

import logging
import os
import time
from typing import Any, Dict, Optional, Tuple, Type, Union

import torch
import torch.nn as nn

from .integrity_checker import IntegrityChecker
from .sandbox_executor import SandboxExecutor
from .model_validator import ModelValidator

logger = logging.getLogger(__name__)


class SecureModelLoader:
    """Secure model loader with defense-in-depth security.
    
    Provides comprehensive secure model loading with:
    - Integrity verification (checksums, file validation)
    - Sandboxed execution (resource limits, isolation)
    - Model validation (structure, configuration, performance)
    - Caching for performance
    - Audit logging
    """

    def __init__(self,
                 trusted_checksums_file: Optional[str] = None,
                 enable_sandbox: bool = True,
                 enable_caching: bool = True,
                 cache_dir: Optional[str] = None,
                 max_cache_size_mb: int = 1024,
                 audit_log_file: Optional[str] = None):
        """Initialize secure model loader.
        
        Args:
            trusted_checksums_file: Path to trusted checksums file
            enable_sandbox: Whether to enable sandboxed execution
            enable_caching: Whether to enable model caching
            cache_dir: Directory for model cache
            max_cache_size_mb: Maximum cache size in MB
            audit_log_file: Path to audit log file
        """
        self.enable_sandbox = enable_sandbox
        self.enable_caching = enable_caching
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), '.model_cache')
        self.max_cache_size_mb = max_cache_size_mb
        self.audit_log_file = audit_log_file
        
        # Initialize security components
        self.integrity_checker = IntegrityChecker(trusted_checksums_file)
        self.sandbox_executor = SandboxExecutor() if enable_sandbox else None
        self.model_validator = ModelValidator()
        
        # Model cache
        self.model_cache = {}
        self.cache_metadata = {}
        
        # Audit log
        self.audit_logger = self._setup_audit_logger()
        
        # Create cache directory
        if enable_caching:
            os.makedirs(self.cache_dir, exist_ok=True)

    def _setup_audit_logger(self) -> logging.Logger:
        """Set up audit logger.
        
        Returns:
            Configured audit logger
        """
        audit_logger = logging.getLogger('secure_model_loader.audit')
        audit_logger.setLevel(logging.INFO)
        
        if self.audit_log_file:
            handler = logging.FileHandler(self.audit_log_file)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            audit_logger.addHandler(handler)
        
        return audit_logger

    def _log_audit_event(self, event_type: str, details: Dict[str, Any]):
        """Log audit event.
        
        Args:
            event_type: Type of audit event
            details: Event details
        """
        audit_entry = {
            'timestamp': time.time(),
            'event_type': event_type,
            'details': details
        }
        
        self.audit_logger.info(f"AUDIT: {audit_entry}")
        logger.info(f"Audit event: {event_type} - {details}")

    def _get_cache_key(self, model_path: str, model_class: type, **kwargs) -> str:
        """Generate cache key for model.
        
        Args:
            model_path: Path to model file
            model_class: Model class
            **kwargs: Model parameters
            
        Returns:
            Cache key string
        """
        import hashlib
        
        # Create cache key from model path, class, and parameters
        key_data = f"{model_path}:{model_class.__name__}:{sorted(kwargs.items())}"
        return hashlib.sha256(key_data.encode()).hexdigest()

    def _is_cached(self, cache_key: str) -> bool:
        """Check if model is cached.
        
        Args:
            cache_key: Cache key
            
        Returns:
            True if model is cached
        """
        if not self.enable_caching:
            return False
        
        return cache_key in self.model_cache

    def _load_from_cache(self, cache_key: str) -> Optional[nn.Module]:
        """Load model from cache.
        
        Args:
            cache_key: Cache key
            
        Returns:
            Cached model or None
        """
        if not self.enable_caching or cache_key not in self.model_cache:
            return None
        
        self._log_audit_event('cache_hit', {'cache_key': cache_key})
        logger.info(f"Loading model from cache: {cache_key}")
        return self.model_cache[cache_key]

    def _save_to_cache(self, cache_key: str, model: nn.Module):
        """Save model to cache.
        
        Args:
            cache_key: Cache key
            model: Model to cache
        """
        if not self.enable_caching:
            return
        
        # Check cache size
        current_size = sum(
            os.path.getsize(os.path.join(self.cache_dir, f)) 
            for f in os.listdir(self.cache_dir) 
            if os.path.isfile(os.path.join(self.cache_dir, f))
        ) / (1024 * 1024)  # Convert to MB
        
        if current_size > self.max_cache_size_mb:
            logger.warning("Cache size limit exceeded, clearing old entries")
            self._clear_cache()
        
        # Save model to cache
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pt")
        torch.save(model.state_dict(), cache_file)
        
        self.model_cache[cache_key] = model
        self.cache_metadata[cache_key] = {
            'timestamp': time.time(),
            'file_path': cache_file
        }
        
        self._log_audit_event('cache_save', {
            'cache_key': cache_key,
            'cache_file': cache_file
        })

    def _clear_cache(self):
        """Clear model cache."""
        if not self.enable_caching:
            return
        
        # Remove cache files
        for cache_key, metadata in self.cache_metadata.items():
            if os.path.exists(metadata['file_path']):
                os.remove(metadata['file_path'])
        
        # Clear memory cache
        self.model_cache.clear()
        self.cache_metadata.clear()
        
        self._log_audit_event('cache_clear', {})

    def load_model(self,
                  model_path: str,
                  model_class: Type[nn.Module],
                  expected_checksum: Optional[str] = None,
                  test_input: Optional[torch.Tensor] = None,
                  **kwargs) -> Tuple[nn.Module, Dict[str, Any]]:
        """Load model securely.
        
        Args:
            model_path: Path to model file
            model_class: Model class to instantiate
            expected_checksum: Expected checksum for integrity verification
            test_input: Optional test input for performance validation
            **kwargs: Additional arguments for model class
            
        Returns:
            Tuple of (loaded_model, loading_info)
        """
        start_time = time.time()
        loading_info = {
            'model_path': model_path,
            'model_class': model_class.__name__,
            'loading_time': 0,
            'cache_used': False,
            'integrity_check': {},
            'validation': {},
            'sandbox_execution': {},
            'issues': []
        }
        
        try:
            # Generate cache key
            cache_key = self._get_cache_key(model_path, model_class, **kwargs)
            
            # Check cache first
            if self._is_cached(cache_key):
                model = self._load_from_cache(cache_key)
                if model is not None:
                    loading_info['cache_used'] = True
                    loading_info['loading_time'] = time.time() - start_time
                    self._log_audit_event('model_loaded', {
                        'model_path': model_path,
                        'cache_used': True,
                        'loading_time': loading_info['loading_time']
                    })
                    return model, loading_info
            
            # 1. Integrity check
            logger.info(f"Performing integrity check for {model_path}")
            integrity_valid, integrity_info = self.integrity_checker.comprehensive_validation(
                model_path, expected_checksum
            )
            loading_info['integrity_check'] = integrity_info
            
            if not integrity_valid:
                loading_info['issues'].extend(integrity_info['findings'])
                raise ValueError(f"Integrity check failed: {integrity_info['findings']}")
            
            # 2. Model validation
            logger.info(f"Validating model {model_path}")
            # Filter out non-model-config parameters
            model_config = {k: v for k, v in kwargs.items() if k not in ['expected_checksum']}
            validation_valid, validation_info = self.model_validator.comprehensive_validation(
                model_path, model_class, model_config, test_input
            )
            loading_info['validation'] = validation_info
            
            if not validation_valid:
                loading_info['issues'].extend(validation_info['issues'])
                raise ValueError(f"Model validation failed: {validation_info['issues']}")
            
            # 3. Load model (with or without sandbox)
            logger.info(f"Loading model {model_path}")
            if self.enable_sandbox and self.sandbox_executor:
                model, sandbox_info = self.sandbox_executor.load_model_safely(
                    model_path, model_class, **kwargs
                )
                loading_info['sandbox_execution'] = sandbox_info
            else:
                # Load without sandbox (less secure but faster)
                model_data = torch.load(model_path, map_location='cpu', weights_only=True)
                
                # Filter kwargs to only include valid constructor parameters
                import inspect
                constructor_params = inspect.signature(model_class.__init__).parameters
                valid_params = {k: v for k, v in kwargs.items() if k in constructor_params}
                model = model_class(**valid_params)
                
                if 'state_dict' in model_data:
                    model.load_state_dict(model_data['state_dict'])
            
            # 4. Cache model
            if self.enable_caching:
                self._save_to_cache(cache_key, model)
            
            # 5. Final validation
            model.eval()
            
            loading_info['loading_time'] = time.time() - start_time
            
            self._log_audit_event('model_loaded', {
                'model_path': model_path,
                'cache_used': False,
                'loading_time': loading_info['loading_time'],
                'model_type': type(model).__name__
            })
            
            logger.info(f"Model loaded successfully in {loading_info['loading_time']:.2f}s")
            return model, loading_info
            
        except Exception as e:
            loading_info['loading_time'] = time.time() - start_time
            loading_info['issues'].append(f"Loading failed: {e}")
            
            self._log_audit_event('model_load_failed', {
                'model_path': model_path,
                'error': str(e),
                'loading_time': loading_info['loading_time']
            })
            
            logger.error(f"Failed to load model {model_path}: {e}")
            raise

    def validate_model(self,
                      model_path: str,
                      model_class: Type[nn.Module],
                      expected_checksum: Optional[str] = None,
                      test_input: Optional[torch.Tensor] = None,
                      **kwargs) -> Tuple[bool, Dict[str, Any]]:
        """Validate model without loading it.
        
        Args:
            model_path: Path to model file
            model_class: Model class
            test_input: Optional test input
            **kwargs: Model parameters
            
        Returns:
            Tuple of (is_valid, validation_info)
        """
        validation_info = {
            'model_path': model_path,
            'integrity_check': {},
            'validation': {},
            'overall_valid': False,
            'issues': []
        }
        
        try:
            # Integrity check
            integrity_valid, integrity_info = self.integrity_checker.comprehensive_validation(
                model_path, expected_checksum
            )
            validation_info['integrity_check'] = integrity_info
            
            if not integrity_valid:
                validation_info['issues'].extend(integrity_info['findings'])
            
            # Model validation - filter out non-model-config parameters
            model_config = {k: v for k, v in kwargs.items() if k not in ['expected_checksum']}
            validation_valid, model_validation_info = self.model_validator.comprehensive_validation(
                model_path, model_class, model_config, test_input
            )
            validation_info['validation'] = model_validation_info
            
            if not validation_valid:
                validation_info['issues'].extend(model_validation_info['issues'])
            
            # Overall validation result
            validation_info['overall_valid'] = integrity_valid and validation_valid
            
            self._log_audit_event('model_validated', {
                'model_path': model_path,
                'is_valid': validation_info['overall_valid'],
                'issues': validation_info['issues']
            })
            
            return validation_info['overall_valid'], validation_info
            
        except Exception as e:
            validation_info['issues'].append(f"Validation error: {e}")
            validation_info['overall_valid'] = False
            
            self._log_audit_event('model_validation_failed', {
                'model_path': model_path,
                'error': str(e)
            })
            
            return False, validation_info

    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information.
        
        Returns:
            Cache information dictionary
        """
        if not self.enable_caching:
            return {'enabled': False}
        
        cache_size = 0
        if os.path.exists(self.cache_dir):
            cache_size = sum(
                os.path.getsize(os.path.join(self.cache_dir, f)) 
                for f in os.listdir(self.cache_dir) 
                if os.path.isfile(os.path.join(self.cache_dir, f))
            ) / (1024 * 1024)  # Convert to MB
        
        return {
            'enabled': True,
            'cache_dir': self.cache_dir,
            'cache_size_mb': cache_size,
            'max_cache_size_mb': self.max_cache_size_mb,
            'cached_models': len(self.model_cache),
            'cache_entries': list(self.cache_metadata.keys())
        }

    def clear_cache(self):
        """Clear the model cache."""
        self._clear_cache()
        self._log_audit_event('cache_cleared', {})

    def cleanup(self):
        """Clean up resources."""
        if self.sandbox_executor:
            self.sandbox_executor.cleanup()
        
        self._log_audit_event('cleanup', {})
        logger.info("Secure model loader cleanup completed") 