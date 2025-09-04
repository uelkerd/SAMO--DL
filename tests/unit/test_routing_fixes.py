#!/usr/bin/env python3
"""
API Routing Fixes Verification
==================================
Simple test to verify Flask-RESTX routing fixes without heavy dependencies.
"""
import os
import unittest
import re

class TestRoutingFixes(unittest.TestCase):
    """Test that routing fixes have been applied correctly."""

    def test_secure_api_server_namespaces_no_leading_slash(self):
        """Test that secure_api_server.py has namespaces without leading slashes."""
        server_file = os.path.join(os.path.dirname(__file__), '..', '..', 'deployment', 'cloud-run', 'secure_api_server.py')

        with open(server_file, 'r') as f:
            content = f.read()

        # Check that main_ns is defined without leading slash
        self.assertIn("main_ns = Namespace('api'", content)
        self.assertNotIn("main_ns = Namespace('/api'", content)

        # Check that admin_ns is defined without leading slash
        self.assertIn("admin_ns = Namespace('admin'", content)
        self.assertNotIn("admin_ns = Namespace('/admin'", content)

    def test_root_endpoint_registered_before_flask_restx(self):
        """Test that root endpoint is registered before Flask-RESTX initialization."""
        server_file = os.path.join(os.path.dirname(__file__), '..', '..', 'deployment', 'cloud-run', 'secure_api_server.py')

        with open(server_file, 'r') as f:
            content = f.read()

        # Find the positions of root endpoint registration and Flask-RESTX initialization
        root_route_match = re.search(r"@app\.route\('/', methods=\['GET'\]\)", content)
        api_init_match = re.search(r"api = Api\(.*?\)", content, re.DOTALL)

        if root_route_match and api_init_match:
            root_pos = root_route_match.start()
            api_pos = api_init_match.start()
            self.assertLess(root_pos, api_pos, "Root endpoint should be registered before Flask-RESTX initialization")

    def test_test_files_fixed(self):
        """Test that test files have been fixed with correct namespace definitions."""
        test_files = [
            'deployment/cloud-run/test_swagger_debug.py',
            'deployment/cloud-run/test_routing_debug.py',
            'deployment/cloud-run/test_debug_server.py',
            'deployment/cloud-run/test_routing_minimal.py'
        ]

        for test_file in test_files:
            file_path = os.path.join(os.path.dirname(__file__), '..', '..', test_file)
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    content = f.read()

                # Check for Namespace definitions without leading slashes
                namespace_matches = re.findall(r"Namespace\('([^']*)'", content)
                for match in namespace_matches:
                    self.assertFalse(match.startswith('/'), f"Found leading slash in namespace '{match}' in {test_file}")

    def test_root_endpoints_before_api_init_in_test_files(self):
        """Test that test files have root endpoints registered before Flask-RESTX init."""
        test_files = [
            'deployment/cloud-run/test_swagger_debug.py',
            'deployment/cloud-run/test_routing_debug.py',
            'deployment/cloud-run/test_debug_server.py',
            'deployment/cloud-run/test_routing_minimal.py'
        ]

        for test_file in test_files:
            file_path = os.path.join(os.path.dirname(__file__), '..', '..', test_file)
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    content = f.read()

                # Find root route and API initialization
                root_route_match = re.search(r"@app\.route\('/', methods=\['GET'\]\)|@app\.route\('/', methods=\[\"GET\"\]\)|@app\.route\('/'\)", content)
                api_init_match = re.search(r"api = Api\(.*?\)", content, re.DOTALL)

                if root_route_match and api_init_match:
                    root_pos = root_route_match.start()
                    api_pos = api_init_match.start()
                    self.assertLess(root_pos, api_pos, f"Root endpoint should be before API init in {test_file}")

    def test_no_double_slashes_in_routes(self):
        """Test that there are no double slashes in route definitions."""
        server_file = os.path.join(os.path.dirname(__file__), '..', '..', 'deployment', 'cloud-run', 'secure_api_server.py')

        with open(server_file, 'r') as f:
            content = f.read()

        # Check for any double slashes in route definitions
        route_matches = re.findall(r"@[^)]*\.route\('([^']*)'", content)
        for route in route_matches:
            self.assertNotIn('//', route, f"Found double slash in route: {route}")

if __name__ == '__main__':
    unittest.main()