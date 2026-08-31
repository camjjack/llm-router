"""Command line entry point."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

import uvicorn

from .config import ConfigError, load_config
from .proxy import Router, create_app


def _configure_logging(level: str, log_file: str | None) -> None:
    handlers: list[logging.Handler] = []
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    else:
        handlers.append(logging.StreamHandler(sys.stderr))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


async def _serve(config, tui: bool, log_level: str) -> None:
    router = Router(config)
    app = create_app(config, router)

    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host=config.host,
            port=config.port,
            log_level=log_level.lower(),
            access_log=not tui,
            # Streaming responses must not be buffered or timed out by the server.
            timeout_keep_alive=75,
        )
    )

    if not tui:
        await server.serve()
        return

    # With the dashboard on, logs would corrupt the display: send them to a file.
    from .tui import run_dashboard

    serve_task = asyncio.create_task(server.serve())
    # Give uvicorn a moment to bind so a startup failure surfaces as an error
    # rather than an empty dashboard.
    await asyncio.sleep(0.5)
    if serve_task.done():
        await serve_task
        return

    dashboard = asyncio.create_task(run_dashboard(router.snapshot))
    try:
        await serve_task
    finally:
        dashboard.cancel()
        try:
            await dashboard
        except asyncio.CancelledError:
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="llm-router",
        description="Capacity-aware, session-sticky load balancer for local LLM backends.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the proxy")
    serve.add_argument("-c", "--config", default="config.yaml", help="path to config YAML")
    serve.add_argument("--host", help="override listen host")
    serve.add_argument("--port", type=int, help="override listen port")
    serve.add_argument("--tui", action="store_true", help="show the live dashboard")
    serve.add_argument(
        "--log-file", help="write logs here instead of stderr (implied by --tui)"
    )
    serve.add_argument("--log-level", default="info")

    top = sub.add_parser("top", help="attach the dashboard to a running router")
    top.add_argument(
        "--url", default="http://127.0.0.1:8080", help="base URL of the running router"
    )

    check = sub.add_parser("check", help="validate a config file and exit")
    check.add_argument("-c", "--config", default="config.yaml")

    args = parser.parse_args(argv)

    if args.command == "top":
        from .tui import run_remote_dashboard

        try:
            asyncio.run(run_remote_dashboard(args.url))
        except KeyboardInterrupt:
            pass
        return 0

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    if args.command == "check":
        print(f"config OK: {len(config.backends)} backend(s), "
              f"models: {', '.join(config.all_models)}")
        for backend in config.backends:
            print(f"  {backend.name:<16} {backend.url:<32} "
                  f"capacity={backend.capacity} models={','.join(backend.models)}")
        return 0

    if args.host:
        config = replace_listen(config, host=args.host)
    if args.port:
        config = replace_listen(config, port=args.port)

    log_file = args.log_file or config.log_file
    if args.tui and not log_file:
        log_file = "llm-router.log"
        print(f"dashboard mode: logs -> {log_file}", file=sys.stderr)
    _configure_logging(args.log_level, log_file)

    try:
        asyncio.run(_serve(config, tui=args.tui, log_level=args.log_level))
    except KeyboardInterrupt:
        pass
    return 0


def replace_listen(config, host: str | None = None, port: int | None = None):
    import dataclasses

    changes = {}
    if host is not None:
        changes["host"] = host
    if port is not None:
        changes["port"] = port
    return dataclasses.replace(config, **changes)


if __name__ == "__main__":
    raise SystemExit(main())
