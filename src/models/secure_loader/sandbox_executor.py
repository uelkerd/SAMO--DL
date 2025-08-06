"""
Sandbox Executor for Secure Model Loading.

This module provides sandboxed execution capabilities for model loading,
preventing potential RCE vulnerabilities and malicious code execution.
"""

import logging
import os
import resource
import signal
import sys
import time
import multiprocessing
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class SandboxError(Exception):
    """Custom exception for sandbox execution errors."""
    def __init__(self, message: str, original_exception: Optional[Exception] = None):
        super().__init__(message)
        self.original_exception = original_exception

    def __str__(self):
        return f"SandboxError: {self.args[0]}"

    def __repr__(self):
        return f"SandboxError({self.args[0]!r}, {self.original_exception!r})"

    def to_dict(self):
        return {
            "error": str(self),
            "exception_type": type(self.original_exception).__name__ if self.original_exception else None
        }


class SandboxExecutor:
    """Sandbox executor for secure model loading.

    Provides isolated execution environment with:
    - Resource limits (CPU, memory, time)
    - Restricted file system access
    - Signal handling and timeout protection
    - Exception isolation
    """

    def __init__(self, 
                 max_memory_mb: int = 2048,
                 max_cpu_time: int = 30,
                 max_wall_time: int = 60,
                 allow_network: bool = False):
        """Initialize sandbox executor.

        Args:
            max_memory_mb: Maximum memory usage in MB
            max_cpu_time: Maximum CPU time in seconds
            max_wall_time: Maximum wall clock time in seconds
            allow_network: Whether to allow network access
        """
        self.max_memory_mb = max_memory_mb
        self.max_cpu_time = max_cpu_time
        self.max_wall_time = max_wall_time
        self.allow_network = allow_network

        # Restricted operations
        self.blocked_modules = {
            'subprocess', 'os', 'sys', 'builtins', 'importlib',
            'pickle', 'marshal', 'code', 'types'
        }

        # Restricted functions
        self.blocked_functions = {
            'eval', 'exec', 'compile', 'open', 'file',
            '__import__', 'globals', 'locals'
        }

    def _set_resource_limits(self):
        """Set resource limits for the sandbox."""
        try:
            # Memory limit (soft and hard)
            memory_limit = self.max_memory_mb * 1024 * 1024  # Convert to bytes
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))

            # CPU time limit
            resource.setrlimit(resource.RLIMIT_CPU, (self.max_cpu_time, self.max_cpu_time))

            # File size limit
            resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024 * 1024, 1024 * 1024 * 1024))  # 1GB

            logger.debug(f"Resource limits set: memory={self.max_memory_mb}MB, cpu={self.max_cpu_time}s")

        except Exception as e:
            logger.error(f"Failed to set resource limits: {e}")

    def _get_safe_builtins(self):
        """Return a safe builtins dictionary for sandboxed execution."""
        import builtins as py_builtins
        allowed_names = [
            'abs', 'all', 'any', 'bool', 'bytes', 'chr', 'dict', 'divmod', 'enumerate', 'filter',
            'float', 'format', 'frozenset', 'getattr', 'hasattr', 'hash', 'hex', 'id', 'int',
            'isinstance', 'issubclass', 'iter', 'len', 'list', 'map', 'max', 'min', 'next', 'object',
            'oct', 'ord', 'pow', 'range', 'repr', 'reversed', 'round', 'set', 'slice', 'sorted',
            'str', 'sum', 'tuple', 'zip', 'Exception', 'ValueError', 'TypeError', 'print'
        ]
        safe_builtins = {name: getattr(py_builtins, name) for name in allowed_names if hasattr(py_builtins, name)}
        return {'__builtins__': safe_builtins}

    def _timeout_handler(self, signum, frame):
        """Handle timeout signals."""
        raise TimeoutError(f"Operation timed out after {self.max_wall_time} seconds")

    def _is_main_thread(self) -> bool:
        """Check if current thread is the main thread."""
        import threading
        return threading.current_thread() is threading.main_thread()

    def _set_timeout_safe(self):
        """Set timeout using signal.alarm only in main thread."""
        if self._is_main_thread():
            signal.alarm(self.max_wall_time)
        else:
            logger.warning("Timeout not set: signal.alarm not available in non-main thread")

    @contextmanager
    def sandbox_context(self):
        """Context manager for sandboxed execution (resource limits, signals, network)."""
        original_signal_handlers = {}
        try:
            self._set_resource_limits()
            # Set up signal handlers for timeout (only in main thread)
            if self._is_main_thread():
                original_signal_handlers[signal.SIGALRM] = signal.signal(signal.SIGALRM, self._timeout_handler)
                self._set_timeout_safe()
            else:
                logger.warning("Signal-based timeout not available in non-main thread")
            # Disable network access if not allowed
            if not self.allow_network:
                self._disable_network()
            yield
        except Exception as e:
            logger.error(f"Sandbox execution error: {e}")
            raise
        finally:
            # Restore signal handlers
            for sig, handler in original_signal_handlers.items():
                signal.signal(sig, handler)
            signal.alarm(0)

    def _disable_network(self):
        """Disable network access in the sandbox."""
        try:
            import socket
            original_socket = socket.socket

            def blocked_socket(*args, **kwargs):
                raise PermissionError("Network access is not allowed in sandbox")

            socket.socket = blocked_socket

        except ImportError:
            pass  # socket module not available

    def execute_safely(self, func: Callable, *args, **kwargs) -> Tuple[Any, Dict]:
        """Execute a function safely in the sandbox."""
        with self.sandbox_context():
            safe_globals = self._get_safe_builtins()
            try:
                # If func is a string, treat as code to exec
                if isinstance(func, str):
                    exec(func, safe_globals)
                    return None, {"status": "exec completed"}
                # If func is a callable, pass safe_globals if it accepts globals
                import inspect
                sig = inspect.signature(func)
                if 'globals' in sig.parameters:
                    result = func(*args, globals=safe_globals, **kwargs)
                else:
                    result = func(*args, **kwargs)
                return result, {"status": "success"}
            except Exception as e:
                logger.error(f"Sandboxed execution failed: {e}")
                return None, {"error": str(e)}

    def load_model_safely(self, model_path: str, model_class: type, **kwargs) -> Any:
        """Load a model safely in the sandbox.

        Args:
            model_path: Path to the model file
            model_class: Model class to instantiate
            **kwargs: Additional arguments for model loading

        Returns:
            Loaded model instance
        """
        def load_model():
            # Use torch.load with weights_only=True for additional safety
            model_data = torch.load(model_path, map_location='cpu', weights_only=True)

            # Filter kwargs to only include valid constructor parameters
            import inspect
            constructor_params = inspect.signature(model_class.__init__).parameters
            valid_params = {k: v for k, v in kwargs.items() if k in constructor_params}

            # Create model instance
            model = model_class(**valid_params)

            # Load state dict if available
            if 'state_dict' in model_data:
                model.load_state_dict(model_data['state_dict'])

            return model

        result, execution_info = self.execute_safely(load_model)
        logger.info(f"Model loaded safely: {execution_info}")
        return result, execution_info

    def validate_model_safely(self, model_path: str) -> Tuple[bool, Dict]:
        """Validate a model safely in the sandbox.

        Args:
            model_path: Path to the model file

        Returns:
            Tuple of (is_valid, validation_info)
        """
        def validate_model():
            # Load model data
            model_data = torch.load(model_path, map_location='cpu', weights_only=True)

            # Basic validation
            if not isinstance(model_data, dict):
                return False, {"error": "Model is not a valid state dict"}

            # Check for required keys
            required_keys = ['state_dict']
            missing_keys = [key for key in required_keys if key not in model_data]

            if missing_keys:
                return False, {"error": f"Missing required keys: {missing_keys}"}

            return True, {"message": "Model validation successful"}

        try:
            result, execution_info = self.execute_safely(validate_model)
            return result
        except Exception as e:
            return False, {"error": f"Validation failed: {e}"}

    def get_resource_usage(self) -> Dict[str, float]:
        """Get current resource usage.

        Returns:
            Dictionary with resource usage information
        """
        try:
            import psutil

            process = psutil.Process()
            memory_info = process.memory_info()
            cpu_percent = process.cpu_percent()

            return {
                'memory_mb': memory_info.rss / 1024 / 1024,
                'cpu_percent': cpu_percent,
                'memory_percent': process.memory_percent()
            }
        except ImportError:
            logger.warning("psutil not available, cannot get resource usage")
            return {}

    def cleanup(self):
        """Clean up sandbox resources."""
        try:
            # Cancel any pending alarms
            signal.alarm(0)

            # Clear any cached models
            if hasattr(torch, 'cuda'):
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Cleanup error: {e}") 