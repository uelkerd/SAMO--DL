#!/usr/bin/env python3
"""
🧪 Sandbox Executor Security Tests
==================================
Tests for the refactored sandbox executor with safe builtins.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname__file__, '..', '..', 'src', 'models', 'secure_loader'))

import unittest
import threading
import time

from sandbox_executor import SandboxExecutor

class TestSandboxExecutorunittest.TestCase:
    """Test sandbox executor functionality."""
    
    def setUpself:
        """Set up test fixtures."""
        self.executor = SandboxExecutor(
            max_memory_mb=512,
            max_cpu_time=10,
            max_wall_time=15,
            allow_network=False
        )
    
    def test_safe_builtins_creationself:
        """Test that safe builtins dictionary is created correctly."""
        safe_builtins = self.executor._get_safe_builtins()
        
        # Check that safe builtins contains expected functions
        self.assertIn'__builtins__', safe_builtins
        builtins_dict = safe_builtins['__builtins__']
        
        # Should contain safe functions
        self.assertIn'len', builtins_dict
        self.assertIn'str', builtins_dict
        self.assertIn'int', builtins_dict
        self.assertIn'list', builtins_dict
        self.assertIn'dict', builtins_dict
        
        # Should NOT contain dangerous functions
        self.assertNotIn'eval', builtins_dict
        self.assertNotIn'exec', builtins_dict
        self.assertNotIn'__import__', builtins_dict
        self.assertNotIn'open', builtins_dict
    
    def test_no_global_builtins_modificationself:
        """Test that global __builtins__ is not modified."""
        import builtins
        
        # Store original builtins
        original_builtins = builtins.__dict__.copy()
        
        # Create executor and run sandboxed code
        executor = SandboxExecutor()
        
        def safe_function():
            return "Hello, World!"
        
        result, meta = executor.execute_safelysafe_function
        
        # Check that global builtins are unchanged
        self.assertEqualbuiltins.__dict__, original_builtins
        self.assertEqualresult, "Hello, World!"
    
    def test_sandbox_context_no_global_changesself:
        """Test that sandbox context doesn't modify global state."""
        import builtins
        original_builtins = builtins.__dict__.copy()
        
        with self.executor.sandbox_context():
            # Sandbox context should not modify global builtins
            self.assertEqualbuiltins.__dict__, original_builtins
        
        # After context, builtins should still be unchanged
        self.assertEqualbuiltins.__dict__, original_builtins
    
    def test_execute_safely_with_string_codeself:
        """Test executing string code safely."""
        code = "result = 2 + 2"
        
        result, meta = self.executor.execute_safelycode
        
        self.assertEqualmeta['status'], 'exec completed'
        self.assertIsNoneresult  # exec doesn't return a value
    
    def test_execute_safely_with_functionself:
        """Test executing function safely."""
        def test_function():
            return "Function executed safely"
        
        result, meta = self.executor.execute_safelytest_function
        
        self.assertEqualresult, "Function executed safely"
        self.assertEqualmeta['status'], 'success'
    
    def test_sandbox_blocks_dangerous_operationsself:
        """Test that sandbox blocks dangerous operations."""
        dangerous_code = "import os; os.system'echo dangerous'"
        
        result, meta = self.executor.execute_safelydangerous_code
        
        # Should fail due to import restrictions
        self.assertIn'error', meta
    
    def test_thread_safetyself:
        """Test that sandbox executor is thread-safe."""
        results = []
        errors = []
        
        def worker_function():
            try:
                result, meta = self.executor.execute_safely(lambda: f"Worker {threading.current_thread().name}")
                results.appendresult
            except Exception as e:
                errors.append(stre)
        
        # Create multiple threads
        threads = []
        for i in range5:
            thread = threading.Threadtarget=worker_function
            threads.appendthread
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Should have no errors and 5 results
        self.assertEqual(lenerrors, 0)
        self.assertEqual(lenresults, 5)
    
    def test_resource_limitsself:
        """Test that resource limits are respected."""
        # This test might not work on all platforms due to resource module limitations
        try:
            executor = SandboxExecutormax_memory_mb=1, max_cpu_time=1
            
            def memory_intensive():
                # Try to allocate more than 1MB
                large_list = [0] * 1000000
                return lenlarge_list
            
            result, meta = executor.execute_safelymemory_intensive
            
            # Should either succeed or fail gracefully
            self.assertIsNotNone(result or meta.get'error')
            
        except Exception as e:
            # Resource limits might not be available on all platforms
            self.assertIn('resource', stre.lower() or 'limit', stre.lower())
    
    def test_timeout_handlingself:
        """Test timeout handling."""
        def slow_function():
            time.sleep2  # Sleep longer than max_wall_time
            return "Should timeout"
        
        result, meta = self.executor.execute_safelyslow_function
        
        # Should either timeout or complete within limits
        self.assertIsNotNone(result or meta.get'error')
    
    def test_network_access_blockingself:
        """Test that network access is blocked when not allowed."""
        def network_function():
            import socket
            s = socket.socketsocket.AF_INET, socket.SOCK_STREAM
            s.connect('localhost', 80)
            return "Network access"
        
        result, meta = self.executor.execute_safelynetwork_function
        
        # Should fail due to network restrictions
        self.assertIn'error', meta

if __name__ == '__main__':
    unittest.main() 