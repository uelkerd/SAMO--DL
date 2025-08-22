#!/usr/bin/env python3
"""Rate Limiter for Flask API"""


import time
import threading
from collections import defaultdict, deque
from flask import request, jsonify
from functools import wraps


class RateLimiter:
    def __init__(self, requests_per_minute: int = 100):
        self.requests_per_minute = requests_per_minute
        self.requests = defaultdict(lambda: deque(maxlen=requests_per_minute))
        self.lock = threading.Lock()

    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed"""
        current_time = time.time()

        with self.lock:
            # Clean old requests (older than 1 minute)
            while (self.requests[client_id] and
                   current_time - self.requests[client_id][0] > 60):
                self.requests[client_id].popleft()

            # Check if under limit
            if len(self.requests[client_id]) < self.requests_per_minute:
                self.requests[client_id].append(current_time)
                return True

            return False

    @staticmethod
    def get_client_id(request) -> str:
        """Get client identifier"""
        # Try API key first
        api_key = request.headers.get('X-API-Key')
        if api_key:
            return f"api_key:{api_key}"

        # Fall back to IP address
        return f"ip:{request.remote_addr}"

def rate_limit(requests_per_minute: int = 100):
    """Rate limiting decorator"""
    limiter = RateLimiter(requests_per_minute)

    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            client_id = limiter.get_client_id(request)

            if not limiter.is_allowed(client_id):
                return jsonify({
                    'error': 'Rate limit exceeded',
                    'retry_after': 60
                }), 429

            return f(*args, **kwargs)
        return decorated_function
    return decorator
