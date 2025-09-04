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
        # Set env vars before import if needed
        # Patch functions to avoid actual initialization
        with patch('flask_restx.Api') as mock_api, \
             patch('flask_restx.Namespace') as mock_ns:
            self.mock_api = mock_api
            self.mock_ns = mock_ns

            # Create Flask app
            self.app = Flask(__name__)

            print("=== After Flask app creation ===")
            print("App routes:", [rule.rule for rule in self.app.url_map.iter_rules()])

            # Register root endpoint BEFORE Flask-RESTX initialization
            print("\n=== Registering root endpoint BEFORE Flask-RESTX ===")
            try:
                @self.app.route('/')
                def root():
                    """Return the root endpoint message."""
                    return jsonify({'message': 'Root endpoint'})
                print("✅ Root endpoint added successfully")
            except Exception as e:
                print(f"❌ Failed to add root endpoint: {e}")

            # Initialize Flask-RESTX API
            self.api = Api(
                self.app,
                version='1.0.0',
                title='Test API',
                description='Minimal test to isolate routing issues',
                doc='/docs'
            )

            print("\n=== After API creation ===")
            print("App routes:", [rule.rule for rule in self.app.url_map.iter_rules()])

            # Create namespace
            main_ns = Namespace('api', description='Main operations')
            self.api.add_namespace(main_ns)

            print("\n=== After adding namespace ===")
            print("App routes:", [rule.rule for rule in self.app.url_map.iter_rules()])

            # Test endpoint in namespace
            @main_ns.route('/health')
            class _Health(Resource):
                @staticmethod
                def get():
                    return {'status': 'healthy'}

            print("\n=== After adding namespace route ===")
            print("App routes:", [rule.rule for rule in self.app.url_map.iter_rules()])

            # Test direct Flask route
            @self.app.route('/test')
            def test():
                return jsonify({'message': 'Test route'})

            print("\n=== After adding Flask route ===")
            print("App routes:", [rule.rule for rule in self.app.url_map.iter_rules()])

    def test_routing_58(self):
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
        pass

    def test_routing_82(self):
        pass

    def test_routing_94(self):
        pass

    def test_routing_111(self):
        pass

    def test_routing_123(self):
        pass

    def test_routing_139(self):
        pass

    def test_routing_151(self):
        pass

    def test_routing_161(self):
        pass

    def test_routing_174(self):
        pass

    def test_routing_187(self):
        pass

    def test_routing_200(self):
        pass

if __name__ == '__main__':
    unittest.main()