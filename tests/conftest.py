"""Shared fixtures: real servers on ephemeral ports, so HTTP behaviour is genuine."""

from __future__ import annotations

import asyncio
import contextlib
import sys
from pathlib import Path

import pytest
import uvicorn

sys.path.insert(0, str(Path(__file__).parent))


@contextlib.asynccontextmanager
async def running_app(app):
    """Serve an ASGI app on an ephemeral port, running its lifespan."""
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    try:
        for _ in range(250):
            if server.started and server.servers and server.servers[0].sockets:
                break
            await asyncio.sleep(0.02)
        else:
            raise RuntimeError("server failed to start")
        port = server.servers[0].sockets[0].getsockname()[1]
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        with contextlib.suppress(asyncio.TimeoutError, Exception):
            await asyncio.wait_for(task, timeout=5)


@pytest.fixture
def anyio_backend():
    return "asyncio"
