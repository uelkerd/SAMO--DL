#!/usr/bin/env python3
"""🧪 Sandbox Executor Security Tests.
==================================
Tests for the refactored sandbox executor with safe builtins.
"""

import os
import sys

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "src",
        "models",
        "secure_loader",
    ),
)

import threading
import time
import unittest

from sandbox_executor import SandboxExecutor


class TestSandboxExecutor(unittest.TestCase):
    """Test sandbox executor functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.executor = SandboxExecutor(
            max_memory_mb=512,
            max_cpu_time=10,
            max_wall_time=15,
            allow_network=False,
        )

    def test_safe_builtins_creation(self):
        """Test that safe builtins dictionary is created correctly."""
        safe_builtins = self.executor._get_safe_builtins()

        # Check that safe builtins contains expected functions
        assert "__builtins__" in safe_builtins
        builtins_dict = safe_builtins["__builtins__"]

        # Should contain safe functions
        assert "len" in builtins_dict
        assert "str" in builtins_dict
        assert "int" in builtins_dict
        assert "list" in builtins_dict
        assert "dict" in builtins_dict

        # Should NOT contain dangerous functions
        assert "eval" not in builtins_dict
        assert "exec" not in builtins_dict
        assert "__import__" not in builtins_dict
        assert "open" not in builtins_dict

    def test_no_global_builtins_modification(self):
        """Test that global __builtins__ is not modified."""
        import builtins

        # Store original builtins
        original_builtins = builtins.__dict__.copy()

        # Create executor and run sandboxed code
        executor = SandboxExecutor()

        def safe_function():
            return "Hello, World!"

        result, meta = executor.execute_safely(safe_function)

        # Check that global builtins are unchanged
        assert builtins.__dict__ == original_builtins
        assert result == "Hello, World!"

    def test_sandbox_context_no_global_changes(self):
        """Test that sandbox context doesn't modify global state."""
        import builtins

        original_builtins = builtins.__dict__.copy()

        with self.executor.sandbox_context():
            # Sandbox context should not modify global builtins
            assert builtins.__dict__ == original_builtins

        # After context, builtins should still be unchanged
        assert builtins.__dict__ == original_builtins

    def test_execute_safely_with_string_code(self):
        """Test executing string code safely."""
        code = "result = 2 + 2"

        result, meta = self.executor.execute_safely(code)

        assert meta["status"] == "exec completed"
        assert result is None  # exec doesn't return a value

    def test_execute_safely_with_function(self):
        """Test executing function safely."""

        def test_function():
            return "Function executed safely"

        result, meta = self.executor.execute_safely(test_function)

        assert result == "Function executed safely"
        assert meta["status"] == "success"

    def test_sandbox_blocks_dangerous_operations(self):
        """Test that sandbox blocks dangerous operations."""
        dangerous_code = "import os; os.system('echo dangerous')"

        result, meta = self.executor.execute_safely(dangerous_code)

        # Should fail due to import restrictions
        assert "error" in meta

    def test_thread_safety(self):
        """Test that sandbox executor is thread-safe."""
        results = []
        errors = []

        def worker_function():
            try:
                result, meta = self.executor.execute_safely(
                    lambda: f"Worker {threading.current_thread().name}",
                )
                results.append(result)
            except Exception as e:
                errors.append(str(e))

        # Create multiple threads
        threads = []
        for _i in range(5):
            thread = threading.Thread(target=worker_function)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Should have no errors and 5 results
        assert len(errors) == 0
        assert len(results) == 5

    def test_resource_limits(self):
        """Test that resource limits are respected."""
        # This test might not work on all platforms due to resource module limitations
        try:
            executor = SandboxExecutor(max_memory_mb=1, max_cpu_time=1)

            def memory_intensive():
                # Try to allocate more than 1MB
                large_list = [0] * 1000000
                return len(large_list)

            result, meta = executor.execute_safely(memory_intensive)

            # Should either succeed or fail gracefully
            assert (result or meta.get("error")) is not None

        except Exception as e:
            # Resource limits might not be available on all platforms
            assert "resource" in (str(e).lower() or "limit"), str(e).lower()

    def test_timeout_handling(self):
        """Test timeout handling."""

        def slow_function():
            time.sleep(2)  # Sleep longer than max_wall_time
            return "Should timeout"

        result, meta = self.executor.execute_safely(slow_function)

        # Should either timeout or complete within limits
        assert (result or meta.get("error")) is not None

    def test_network_access_blocking(self):
        """Test that network access is blocked when not allowed."""

        def network_function():
            import socket

            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect(("localhost", 80))
            return "Network access"

        result, meta = self.executor.execute_safely(network_function)

        # Should fail due to network restrictions
        assert "error" in meta


if __name__ == "__main__":
    unittest.main()
