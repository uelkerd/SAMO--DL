from functools import wraps
from flask import request, jsonify

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if api_key != 'REPLACE_WITH_ACTUAL_API_KEY':  # Replace with actual key or env var
            return jsonify({'error': 'API key required'}), 401
        return f(*args, **kwargs)
    return decorated_function
