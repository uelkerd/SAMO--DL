#!/usr/bin/env python3
"""
Debug script to understand Flask-RESTX routing behavior
"""

from flask import Flask, jsonify
from flask_restx import Api, Resource, Namespace

# Create Flask app
app = Flask(__name__)

print("=== After Flask app creation ===")
print("App routes:", [rule.rule for rule in app.url_map.iter_rules()])

# Register root endpoint BEFORE Flask-RESTX initialization
print("\n=== Registering root endpoint BEFORE Flask-RESTX ===")
try:
    @app.route('/')
    def root():
        return jsonify({'message': 'Root endpoint'})
    print("✅ Root endpoint added successfully")
except Exception as e:
    print(f"❌ Failed to add root endpoint: {e}")

# Initialize Flask-RESTX API
api = Api(
    app,
    version='1.0.0',
    title='Test API',
    description='Minimal test to isolate routing issues',
    doc='/docs'
)

print("\n=== After API creation ===")
print("App routes:", [rule.rule for rule in app.url_map.iter_rules()])

# Create namespace
main_ns = Namespace('api', description='Main operations')
api.add_namespace(main_ns)

print("\n=== After adding namespace ===")
print("App routes:", [rule.rule for rule in app.url_map.iter_rules()])

# Test endpoint in namespace
@main_ns.route('/health')
class Health(Resource):
    def get(self):
        return {'status': 'healthy'}

print("\n=== After adding namespace route ===")
print("App routes:", [rule.rule for rule in app.url_map.iter_rules()])

# Test direct Flask route
@app.route('/test')
def test():
    return jsonify({'message': 'Test route'})

print("\n=== After adding Flask route ===")
print("App routes:", [rule.rule for rule in app.url_map.iter_rules()])

print("\n=== Final state ===")
print("App routes:", [rule.rule for rule in app.url_map.iter_rules()])

# Check for endpoint name conflicts
endpoints = {}
for rule in app.url_map.iter_rules():
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
for rule in app.url_map.iter_rules():
    if rule.rule == '/':
        print(f"Root route: {rule.rule} -> {rule.endpoint}")
        print(f"  Methods: {rule.methods}")
        print(f"  View function: {rule.endpoint}") 