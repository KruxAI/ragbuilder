"""Security boundaries for the local RAGBuilder application."""

import base64
import hmac
import ipaddress
import os
from pathlib import Path
from urllib.parse import urlsplit

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def validate_bind_address(host: str) -> None:
    """Require authentication before exposing either server to the network."""
    token = os.getenv("RAGBUILDER_API_TOKEN", "")
    if token and len(token) < 32:
        raise ValueError("RAGBUILDER_API_TOKEN must contain at least 32 characters")
    if not is_loopback(host) and not token:
        raise ValueError("Network access requires RAGBUILDER_API_TOKEN (at least 32 characters)")


class LocalAccessMiddleware(BaseHTTPMiddleware):
    """Authenticate remote use and reject cross-origin browser requests."""

    async def dispatch(self, request, call_next):
        token = os.getenv("RAGBUILDER_API_TOKEN", "")
        origin = request.headers.get("origin")
        if origin:
            expected = f"{request.url.scheme}://{request.url.netloc}"
            if origin != expected:
                return JSONResponse({"detail": "Cross-origin requests are disabled"}, status_code=403)
        if token:
            if len(token) < 32:
                return JSONResponse({"detail": "Invalid server authentication configuration"}, status_code=503)
            authorization = request.headers.get("authorization", "")
            supplied = ""
            if authorization.startswith("Bearer "):
                supplied = authorization[7:]
            elif authorization.startswith("Basic "):
                try:
                    user, supplied = base64.b64decode(authorization[6:], validate=True).decode().split(":", 1)
                    if user != "ragbuilder":
                        supplied = ""
                except (ValueError, UnicodeError):
                    supplied = ""
            if not hmac.compare_digest(supplied.encode(), token.encode()):
                return JSONResponse(
                    {"detail": "Authentication required"}, status_code=401,
                    headers={"WWW-Authenticate": 'Basic realm="RAGBuilder"'},
                )
        else:
            # Checking Host as well as the peer prevents DNS rebinding against localhost.
            peer = request.client.host if request.client else ""
            if not is_loopback(peer) or not is_loopback(request.url.hostname or ""):
                return JSONResponse({"detail": "Local access only"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Cache-Control"] = "no-store"
        return response


def validate_source_path(source: str) -> str:
    """Limit UI file access to its configured data directory, including symlinks."""
    if urlsplit(source).scheme in {"http", "https"}:
        return source
    root = Path(os.getenv("RAGBUILDER_DATA_ROOT", os.getcwd())).expanduser().resolve()
    path = Path(source).expanduser()
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(root):
        raise ValueError("Source must be inside RAGBUILDER_DATA_ROOT")
    if any(part.startswith(".") for part in path.relative_to(root).parts):
        raise ValueError("Hidden files and directories cannot be used as sources")
    if path.is_dir():
        for child in path.rglob("*"):
            if child.is_symlink() and not child.resolve().is_relative_to(root):
                raise ValueError("Source directory contains a symlink outside RAGBUILDER_DATA_ROOT")
    return str(path)
