"""
Rate limiting implementation
"""
from collections import defaultdict
from datetime import datetime, timedelta
from fastapi import HTTPException, status
from config.settings import settings


class RateLimiter:
    """Simple in-memory rate limiter"""
    
    def __init__(self):
        self.requests = defaultdict(list)
    
    def is_allowed(self, client_ip: str) -> bool:
        """Check if request is allowed for given IP"""
        now = datetime.utcnow()
        # Remove old requests outside the time window
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip]
            if req_time > now - timedelta(seconds=settings.RATE_LIMIT_PERIOD)
        ]
        
        if len(self.requests[client_ip]) >= settings.RATE_LIMIT_REQUESTS:
            return False
        
        self.requests[client_ip].append(now)
        return True


rate_limiter = RateLimiter()


def check_rate_limit(client_ip: str):
    """Check rate limit and raise exception if exceeded"""
    if not rate_limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: {settings.RATE_LIMIT_REQUESTS} requests per {settings.RATE_LIMIT_PERIOD} seconds"
        )
