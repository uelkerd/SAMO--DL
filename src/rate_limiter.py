from collections import defaultdict
from datetime import datetime, timedelta
from functools import wraps
from flask import abort, request

# Simple rate limiter using memory (use Redis for production)
_rate_limit_storage = defaultdict(list)

def rate_limit(max_requests=100, window_minutes=1):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            client_ip = request.remote_addr
            now = datetime.utcnow()
            window_start = now - timedelta(minutes=window_minutes)
            _rate_limit_storage[client_ip] = [req_time for req_time in _rate_limit_storage[client_ip] if req_time > window_start]
            if len(_rate_limit_storage[client_ip]) >= max_requests:
                abort(429, description="Rate limit exceeded")
            _rate_limit_storage[client_ip].append(now)
            return f(*args, **kwargs)
        return decorated_function
    return decorator
