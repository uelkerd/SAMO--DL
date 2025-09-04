#!/usr/bin/env python3
"""
Debug script to understand Flask-RESTX routing behavior
"""

from flask import Flask, jsonify
from flask_restx import Api, Resource, Namespace
import unittest
from unittest.mock import patch

class TestAPIRouting(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures and mock objects for API routing tests."""
        # Set env vars before import if needed
        # Patch functions to avoid actual initialization
        with patch('flask_restx.Api') as mock_api, \
             patch('flask_restx.Namespace') as mock_ns:
            self.mock_api = mock_api
            self.mock_ns = mock_ns

            # Create Flask app
            self.app = Flask(__name__)

            # Register root endpoint BEFORE Flask-RESTX initialization
            try:
                @self.app.route('/')
                def root():
                    """Return the root endpoint message."""
                    return jsonify({'message': 'Root endpoint'})
            except Exception as e:
                raise

            # Initialize Flask-RESTX API
            self.api = Api(
                self.app,
                version='1.0.0',
                title='Test API',
                description='Minimal test to isolate routing issues',
                doc='/docs'
            )

            # Create namespace
            main_ns = Namespace('api', description='Main operations')
            self.api.add_namespace(main_ns)

            # Test endpoint in namespace
            @main_ns.route('/health')
            class _Health(Resource):
                @staticmethod
                def get():
                    """Return health status of the service."""
                    return {'status': 'healthy'}

            # Test direct Flask route
            @self.app.route('/test')
            def test():
                """Test route that returns a simple JSON response."""
                return jsonify({'message': 'Test route'})

    def test_routing_58(self):
        """Test routing configuration and check for endpoint conflicts."""
        print("\n=== Final state ===")
        print("App routes:", [rule.rule for rule in self.app.url_map.iter_rules()])

        # Check for endpoint name conflicts
        endpoints = {}
        for rule in self.app.url_map.iter_rules():
            if rule.endpoint in endpoints:
                print(f"⚠️  CONFLICT: Endpoint '{rule.endpoint}' appears multiple times:")
                print(f"   - {endpoints[rule.endpoint]} -> {rule.rule}")
                print(f"   - {rule.endpoint} -> {rule.rule}")
            else:
                endpoints[rule.endpoint] = rule.rule

        print("\n=== All endpoints ===")
        for endpoint, rule in endpoints.items():
            print(f"{endpoint} -> {rule}")

        # Check what Flask-RESTX created for the root route
        print("\n=== Flask-RESTX root route details ===")
        for rule in self.app.url_map.iter_rules():
            if rule.rule == '/':
                print(f"Root route: {rule.rule} -> {rule.endpoint}")
                print(f"  Methods: {rule.methods}")
                print(f"  View function: {rule.endpoint}")

    def test_routing_71(self):
        """Test routing behavior for line 71."""
        raise NotImplementedError()

    def test_routing_82(self):
        """Test routing behavior for line 82."""
        raise NotImplementedError()

    def test_routing_94(self):
        """Test routing behavior for line 94."""
        raise NotImplementedError()

    def test_routing_111(self):
        """Test routing behavior for line 111."""
        raise NotImplementedError()

    def test_routing_123(self):
        """Test routing behavior for line 123."""
        raise NotImplementedError()

    def test_routing_139(self):
        """Test routing behavior for line 139."""
        raise NotImplementedError()

    def test_routing_151(self):
        """Test routing behavior for line 151."""
        raise NotImplementedError()

    def test_routing_161(self):
        """Test routing behavior for line 161."""
        raise NotImplementedError()

    def test_routing_174(self):
        """Test routing behavior for line 174."""
        raise NotImplementedError()

    def test_routing_187(self):
        """Test routing behavior for line 187."""
        raise NotImplementedError()

    def test_routing_200(self):
        """Test routing behavior for line 200."""
        raise NotImplementedError()

if __name__ == '__main__':
    unittest.main()