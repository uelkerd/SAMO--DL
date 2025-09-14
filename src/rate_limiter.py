from collections import defaultdict
from datetime import datetime, timedelta
from flask import abort

# Simple rate limiter using memory (use Redis for production)
rate_limit = defaultdict(list)

def rate_limit(max_requests=100, window_minutes=1):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            client_ip = request.remote_addr
            now = datetime.utcnow()
            window_start = now - timedelta(minutes=window_minutes)
            rate_limit[client_ip] = [req_time for req_time in rate_limit[client_ip] if req_time > window_start]
            if len(rate_limit[client_ip]) >= max_requests:
                abort(429, description="Rate limit exceeded")
            rate_limit[client_ip].append(now)
            return f(*args, **kwargs)
        return decorated_function
    return decorator
