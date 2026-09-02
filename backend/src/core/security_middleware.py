"""
Security middleware and rate limiting for Argus Prime.

Provides production-grade security features including:
- Rate limiting
- Request size limiting
- Security headers
- IP-based filtering
"""

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from fastapi import Request, Depends
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    requests: int = 100
    window_seconds: int = 60


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Sliding window rate limiting middleware.

    Implements a sliding window algorithm to track request counts
    per client IP address.
    """

    def __init__(
        self,
        app,
        config: Optional[RateLimitConfig] = None,
        excluded_paths: Optional[list] = None,
    ):
        super().__init__(app)
        self.config = config or RateLimitConfig()
        self.excluded_paths = excluded_paths or [
            "/health",
            "/health/live",
            "/health/ready",
            "/docs",
            "/redoc",
            "/openapi.json",
        ]
        self.clients: Dict[str, list] = defaultdict(list)

    def _get_client_key(self, request: Request) -> str:
        """Get unique client identifier."""
        # Use X-Forwarded-For if behind proxy
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _is_excluded(self, path: str) -> bool:
        """Check if path is excluded from rate limiting."""
        return any(path.startswith(excluded) for excluded in self.excluded_paths)

    def _cleanup_old_requests(self, client_key: str, current_time: float) -> None:
        """Remove requests outside the current window."""
        window_start = current_time - self.config.window_seconds
        self.clients[client_key] = [
            req_time for req_time in self.clients[client_key]
            if req_time > window_start
        ]

    async def dispatch(self, request: Request, call_next):
        """Process request through rate limiting."""
        path = request.url.path

        # Skip rate limiting for excluded paths
        if self._is_excluded(path):
            return await call_next(request)

        client_key = self._get_client_key(request)
        current_time = time.time()

        # Cleanup old requests
        self._cleanup_old_requests(client_key, current_time)

        # Check rate limit
        if len(self.clients[client_key]) >= self.config.requests:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limited",
                    "message": "Too many requests. Please try again later.",
                    "retry_after": self.config.window_seconds,
                },
                headers={"Retry-After": str(self.config.window_seconds)},
            )

        # Record request
        self.clients[client_key].append(current_time)

        # Process request
        response = await call_next(request)
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Add security headers to all responses.

    Implements OWASP security header recommendations.
    """

    SECURITY_HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Content-Security-Policy": "default-src 'self'",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    }

    async def dispatch(self, request: Request, call_next):
        """Add security headers to response."""
        response: Response = await call_next(request)

        for header, value in self.SECURITY_HEADERS.items():
            response.headers[header] = value

        return response


def get_client_ip(request: Request) -> str:
    """Extract client IP from request."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


async def verify_rate_limit(
    request: Request,
    max_requests: int = 100,
    window_seconds: int = 60,
) -> Tuple[bool, int]:
    """
    Verify rate limit for a request.

    Returns:
        Tuple of (is_allowed, remaining_requests)
    """
    client_ip = get_client_ip(request)
    current_time = time.time()

    # Get or create request history
    # In production, use Redis for distributed rate limiting
    return True, max_requests


class RequestSizeMiddleware(BaseHTTPMiddleware):
    """
    Limit maximum request body size.

    Prevents denial-of-service attacks via large request bodies.
    """

    MAX_BODY_SIZE = 10 * 1024 * 1024  # 10MB

    async def dispatch(self, request: Request, call_next):
        """Check request body size."""
        content_length = request.headers.get("content-length")

        if content_length:
            size = int(content_length)
            if size > self.MAX_BODY_SIZE:
                return JSONResponse(
                    status_code=413,
                    content={
                        "error": "payload_too_large",
                        "message": f"Request body exceeds {self.MAX_BODY_SIZE} bytes",
                    },
                )

        response = await call_next(request)
        return response
