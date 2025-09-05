#!/usr/bin/env python3
"""
API Routing Fixes Verification
==================================
Simple test to verify Flask-RESTX routing fixes without heavy dependencies.
"""
import unittest
import re
from pathlib import Path

# Base path for project files
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

class TestRoutingFixes(unittest.TestCase):
    """Test that routing fixes have been applied correctly."""

    def test_secure_api_server_namespaces_no_leading_slash(self):
        """Test that secure_api_server.py has namespaces without leading slashes."""
        server_file = PROJECT_ROOT / 'deployment' / 'cloud-run' / 'secure_api_server.py'
        self.assertTrue(server_file.exists(), f"Server file not found: {server_file}")
        content = server_file.read_text()

        # Use regex to check namespace declarations are quote- and whitespace-agnostic
        # Should match: main_ns = Namespace('api'  or main_ns=Namespace("api" etc.
        main_ns_pattern = re.compile(r'main_ns\s*=\s*Namespace\s*\(\s*[\'"]([^\'"]*)[\'"]', re.IGNORECASE)
        admin_ns_pattern = re.compile(r'admin_ns\s*=\s*Namespace\s*\(\s*[\'"]([^\'"]*)[\'"]', re.IGNORECASE)

        main_match = main_ns_pattern.search(content)
        admin_match = admin_ns_pattern.search(content)

        self.assertIsNotNone(main_match, "main_ns namespace declaration not found")
        self.assertIsNotNone(admin_match, "admin_ns namespace declaration not found")

        main_ns_value = main_match.group(1)
        admin_ns_value = admin_match.group(1)

        self.assertEqual(main_ns_value, 'api', f"main_ns should be 'api', got '{main_ns_value}'")
        self.assertEqual(admin_ns_value, 'admin', f"admin_ns should be 'admin', got '{admin_ns_value}'")

        # Ensure no leading slashes in namespace values
        self.assertFalse(main_ns_value.startswith('/'), f"main_ns should not start with '/', got '{main_ns_value}'")
        self.assertFalse(admin_ns_value.startswith('/'), f"admin_ns should not start with '/', got '{admin_ns_value}'")

    def test_root_endpoint_registered_before_flask_restx(self):
        """Test that root endpoint is registered before Flask-RESTX initialization."""
        server_file = PROJECT_ROOT / 'deployment' / 'cloud-run' / 'secure_api_server.py'

        with open(server_file) as f:
            content = f.read()

        # Find the positions of root endpoint registration and Flask-RESTX initialization
        # More flexible regex to handle different formatting (quotes, whitespace, methods)
        root_route_match = re.search(r"@app\.route\s*\(\s*['\"]/['\"]\s*(?:,\s*methods\s*=\s*\[.*?\])?\s*\)", content)
        api_init_match = re.search(r"api\s*=\s*Api\s*\(", content)

        # Explicit assertions to ensure patterns are found
        self.assertIsNotNone(root_route_match, "Root route pattern not found in source code")
        self.assertIsNotNone(api_init_match, "API initialization pattern not found in source code")

        root_pos = root_route_match.start()
        api_pos = api_init_match.start()
        self.assertLess(root_pos, api_pos, "Root endpoint should be registered before Flask-RESTX initialization")

    def test_test_files_fixed(self):
        """Test that test files have been fixed with correct namespace definitions."""
        # Test each file individually to avoid loops in tests
        test_file = PROJECT_ROOT / 'deployment' / 'cloud-run' / 'test_swagger_debug.py'
        self.assertTrue(test_file.exists(), f"Expected file not found: {test_file}")
        with open(test_file) as f:
            content = f.read()
        namespace_matches = re.findall(r"Namespace\('([^']*)'", content)
        for match in namespace_matches:
            self.assertFalse(match.startswith('/'), f"Found leading slash in namespace '{match}' in {test_file}")

    def test_root_endpoints_before_api_init_in_test_files(self):
        """Test that test files have root endpoints registered before Flask-RESTX init."""
        # Test one file at a time to avoid loops in tests
        test_file = PROJECT_ROOT / 'deployment' / 'cloud-run' / 'test_swagger_debug.py'
        self.assertTrue(test_file.exists(), f"Test file not found: {test_file}")
        content = test_file.read_text()
        root_route_match = re.search(r"@app\.route\s*\(\s*['\"]/['\"]\s*(?:,\s*methods\s*=\s*\[.*?\])?\s*\)", content)
        api_init_match = re.search(r"api\s*=\s*Api\s*\(", content)

        # Explicit assertions to ensure patterns are found
        self.assertIsNotNone(root_route_match, f"Root route pattern not found in {test_file}")
        self.assertIsNotNone(api_init_match, f"API initialization pattern not found in {test_file}")

        root_pos = root_route_match.start()
        api_pos = api_init_match.start()
        self.assertLess(root_pos, api_pos, f"Root endpoint should be before API init in {test_file}")

    def test_no_double_slashes_in_routes(self):
        """Test that there are no double slashes in route definitions."""
        server_file = PROJECT_ROOT / 'deployment' / 'cloud-run' / 'secure_api_server.py'

        with open(server_file) as f:
            content = f.read()

        # Check for any double slashes in route definitions (test one route at a time)
        route_matches = re.findall(r"@[^)]*\.route\('([^']*)'", content)
        self.assertGreater(len(route_matches), 0, "No routes found in secure_api_server.py")
        for route in route_matches:
            self.assertNotIn('//', route, f"Found double slash in route: {route}")

if __name__ == '__main__':
    unittest.main()