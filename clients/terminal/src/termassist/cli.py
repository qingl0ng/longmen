"""Entry point — arg parsing, startup, shutdown."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from . import __version__ as _version

if TYPE_CHECKING:
    from .config import ClientConfig


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="longmen",
        description="Longmen — terminal client for the Gateway coding assistant",
    )
    parser.add_argument("--version", action="version", version=f"longmen {_version}")
    parser.add_argument("--gateway", metavar="URL", help="Gateway WebSocket URL")
    parser.add_argument("--project", metavar="ID", help="Auto-select project")
    parser.add_argument("--config", metavar="PATH", help="Config file path")
    parser.add_argument("--theme", choices=["default", "light", "minimal"], help="Color theme")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Log all raw gateway messages to ~/.longmen/terminal/messages.jsonl",
    )

    subparsers = parser.add_subparsers(dest="subcommand")
    pair_parser = subparsers.add_parser("pair", help="Initiate device pairing")
    pair_parser.add_argument("--code", metavar="CODE", help="Pairing code")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    from .config import ClientConfig

    config_path = Path(args.config) if args.config else None
    try:
        config = ClientConfig.load(config_path)
    except Exception as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        sys.exit(1)

    # Apply CLI overrides
    if args.gateway:
        config.gateway.url = args.gateway
    if args.project:
        config.project.id = args.project
    if args.theme:
        config.display.theme = args.theme
    if args.debug:
        config.logging.message_log = str(
            Path.home() / ".longmen" / "terminal" / "messages.jsonl"
        )
        config.display.show_thinking = True  # also enable finish_reason display

    if args.subcommand == "pair":
        asyncio.run(_run_pair(config, args.code))
    else:
        from .app import App
        with contextlib.suppress(KeyboardInterrupt):
            asyncio.run(App(config).run())


async def _run_pair(config: ClientConfig, code: str | None) -> None:
    import platform

    from .connection import Connection
    from .protocol import make_pair_request

    print("Connecting to gateway for pairing...")
    try:
        conn = await Connection.connect(config.gateway.url)
    except Exception as e:
        print(f"Connection failed: {e}", file=sys.stderr)
        return

    if code is None:
        code = input("Enter pairing code: ").strip()

    device_info = {
        "hostname": platform.node(),
        "os": platform.system(),
        "client_type": "terminal",
        "client_version": _version,
    }
    await conn.send(make_pair_request(code, device_info))

    print("Waiting for gateway approval...")
    msg = await conn.recv()
    if msg["type"] == "pair_result":
        payload = msg["payload"]
        if payload.get("success"):
            token = payload.get("token", "")
            config.gateway.auth.mode = "paired"
            config.gateway.auth.token = token
            config.save()
            print("Pairing successful. Token saved to config.")
        else:
            print(f"Pairing failed: {payload.get('error', 'Unknown error')}")
    else:
        print(f"Unexpected response: {msg['type']}")

    await conn.close()
