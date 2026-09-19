"""The web pages -- the session dashboard and the connect page -- and the JSON they poll.

The page is plain HTML, CSS and JavaScript shipped inside the package, so it has
no build step and loads nothing from the internet -- it works on an isolated
network.

Watching must not cost the router anything that matters:

- The snapshot is built and serialised at most once per SNAPSHOT_TTL_S, however
  many dashboards are open. Every viewer in that window is sent the same bytes.
- The page stops polling while its tab is hidden.
- Its requests are kept out of the access log, which would otherwise gain a line
  every couple of seconds for every open dashboard.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from importlib import resources
from typing import TYPE_CHECKING, Any

from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

if TYPE_CHECKING:
    from .proxy import Router

SNAPSHOT_TTL_S = 1.0

# Served from memory once read: name -> media type.
ASSETS = {
    "index.html": "text/html; charset=utf-8",
    "connect.html": "text/html; charset=utf-8",
    "app.js": "text/javascript; charset=utf-8",
    "connect.js": "text/javascript; charset=utf-8",
    "app.css": "text/css; charset=utf-8",
}
# Served at their own paths rather than as assets.
PAGES = {"index.html", "connect.html"}

# Everything the page shows comes from client-supplied headers (user names, session
# ids), so it is rendered as text, never markup. This is the backstop if that ever
# slips: no inline script, nothing from another origin.
CONTENT_SECURITY_POLICY = (
    "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; "
    "img-src 'self' data:; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
)

# Paths whose polling is kept out of the access log.
QUIET_PATHS = ("/sessions", "/dashboard")

_assets: dict[str, bytes] = {}


def _asset(name: str) -> bytes:
    body = _assets.get(name)
    if body is None:
        body = resources.files(__package__).joinpath("static", name).read_bytes()
        _assets[name] = body
    return body


class SnapshotCache:
    """Builds a JSON body at most once per `ttl_s`, and hands every caller in
    between the same bytes."""

    def __init__(self, build: Callable[[], Any], ttl_s: float = SNAPSHOT_TTL_S) -> None:
        self._build = build
        self._ttl_s = ttl_s
        self._body: bytes | None = None
        self._built_at = 0.0

    def get(self) -> bytes:
        now = time.monotonic()
        if self._body is None or now - self._built_at >= self._ttl_s:
            self._body = json.dumps(self._build(), separators=(",", ":")).encode()
            self._built_at = now
        return self._body


class QuietAccessLog(logging.Filter):
    """Drops uvicorn access-log lines for dashboard polling."""

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        # uvicorn logs (client, method, path, http version, status).
        if isinstance(args, tuple) and len(args) >= 3 and isinstance(args[2], str):
            return not args[2].startswith(QUIET_PATHS)
        return True


def routes(router: Router) -> list[Route]:
    cache = SnapshotCache(router.tracking_snapshot)

    def static(name: str) -> Response:
        return Response(
            _asset(name),
            media_type=ASSETS[name],
            headers={
                "content-security-policy": CONTENT_SECURITY_POLICY,
                "x-content-type-options": "nosniff",
                "cache-control": "no-cache",
            },
        )

    async def page(request: Request) -> Response:
        return static("index.html")

    async def connect(request: Request) -> Response:
        return static("connect.html")

    async def asset(request: Request) -> Response:
        name = request.path_params["name"]
        if name not in ASSETS or name in PAGES:
            return Response(status_code=404)
        return static(name)

    async def sessions(request: Request) -> Response:
        return Response(
            cache.get(),
            media_type="application/json",
            headers={"cache-control": "no-store"},
        )

    async def session_detail(request: Request) -> Response:
        # Uncached, but bounded: one session and its last few requests.
        detail = router.tracker.session_detail(request.path_params["n"])
        if detail is None:
            return JSONResponse({"error": "no such session"}, status_code=404)
        return JSONResponse(detail, headers={"cache-control": "no-store"})

    return [
        Route("/dashboard", page, methods=["GET"]),
        Route("/connect", connect, methods=["GET"]),
        Route("/dashboard/{name}", asset, methods=["GET"]),
        Route("/sessions", sessions, methods=["GET"]),
        Route("/sessions/{n:int}", session_detail, methods=["GET"]),
    ]
