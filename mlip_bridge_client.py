#!/usr/bin/env python3

"""Persistent stdin/stdout bridge to MLIPClient transports."""

import argparse
import json
import sys
from typing import Any, Dict

from mlip_client import build_client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bridge JSONL requests to MLIP worker transport")
    parser.add_argument("--transport", default="http", choices=["http", "zmq"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    return parser.parse_args()


def emit(payload: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def main() -> int:
    args = parse_args()
    client = build_client(args.transport, host=args.host, port=args.port)

    try:
        for line in sys.stdin:
            raw = line.strip()
            if raw == "":
                continue

            payload: Dict[str, Any]
            shutdown_requested = False
            try:
                payload = json.loads(raw)
                shutdown_requested = payload.get("cmd") == "shutdown"
                result = client.request(payload)
                emit({"status": "ok", "result": result})
            except Exception as exc:
                emit({"status": "error", "error": str(exc)})

            if shutdown_requested:
                break
    finally:
        client.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
