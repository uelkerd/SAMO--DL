from collections import defaultdict
from typing import Optional
import time

class RateLimiter:
    def __init__(self, max_requests: int = 100, window_seconds: int = 3600):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)

    def is_allowed(self, identifier: str) -> bool:
        now = time.time()
        window_start = now - self.window_seconds
        self.requests[identifier] = [
            timestamp for timestamp in self.requests[identifier]
            if timestamp > window_start
        ]
        if len(self.requests[identifier]) < self.max_requests:
            self.requests[identifier].append(now)
            return True
        return False

    def get_remaining_requests(self, identifier: str) -> int:
        now = time.time()
        window_start = now - self.window_seconds
        self.requests[identifier] = [
            timestamp for timestamp in self.requests[identifier]
            if timestamp > window_start
        ]
        return max(0, self.max_requests - len(self.requests[identifier]))
