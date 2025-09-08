#!/usr/bin/env python3
"""Debug script to understand Flask-RESTX routing behavior."""

from flask import Flask, jsonify
from flask_restx import Api, Resource, Namespace

# Create Flask app
app = Flask(__name__)


# Initialize Flask-RESTX API
api = Api(
    app,
    version='1.0.0',
    title='Test API',
    description='Minimal test to isolate routing issues',
    doc='/docs'
)


# Create namespace
main_ns = Namespace('/api', description='Main operations')
api.add_namespace(main_ns)


# Test endpoint in namespace
@main_ns.route('/health')
class Health(Resource):
    def get(self):
        return {'status': 'healthy'}


# Test direct Flask route
@app.route('/test')
def test():
    return jsonify({'message': 'Test route'})


# Now try to add root endpoint
try:
    @app.route('/')
    def root():
        return jsonify({'message': 'Root endpoint'})
except Exception:
    pass


# Check for endpoint name conflicts
endpoints = {}
for rule in app.url_map.iter_rules():
    if rule.endpoint in endpoints:
        pass
    else:
        endpoints[rule.endpoint] = rule.rule

for _endpoint, rule in endpoints.items():
    pass

# Check what Flask-RESTX created for the root route
for rule in app.url_map.iter_rules():
    if rule.rule == '/':
        pass
