"""
Middleware to normalize URL paths by removing trailing slashes.

Allows API endpoints to work with or without trailing slashes.
Routes should be defined WITHOUT trailing slashes.
"""

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class TrailingSlashMiddleware(BaseHTTPMiddleware):
    """Removes trailing slashes from request paths (except root)."""

    async def dispatch(self, request: Request, call_next) -> Response:
        path = request.url.path

        if path != "/" and path.endswith("/"):
            normalized_path = path.rstrip("/")
            request.scope["path"] = normalized_path
            request.scope["raw_path"] = normalized_path.encode("utf-8")

        return await call_next(request)
