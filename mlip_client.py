#!/usr/bin/env python3

"""Unified MLIP clients for HTTP and ZeroMQ transports."""

import abc
import argparse
import json
import time
import urllib.request
from typing import Any, Dict, Optional


class MLIPClient(abc.ABC):
    """Abstract client interface used by managers/benchmarkers."""

    @abc.abstractmethod
    def request(self, payload: Dict[str, Any]) -> Any:
        """Send one protocol request and return protocol result."""

    def ping(self) -> str:
        return self.request({"cmd": "ping"})

    def calculate(self, xyz: str, gradients: bool = True) -> Dict[str, Any]:
        return self.request({"cmd": "calculate", "xyz": xyz, "gradients": gradients})

    def shutdown(self) -> str:
        return self.request({"cmd": "shutdown"})

    def close(self) -> None:
        pass


class HTTPMLIPClient(MLIPClient):
    def __init__(self, host: str, port: int, timeout: float = 900.0) -> None:
        self.url = f"http://{host}:{port}"
        self.timeout = timeout

    def request(self, payload: Dict[str, Any]) -> Any:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            body = response.read().decode("utf-8")

        decoded = json.loads(body)
        if decoded.get("status") != "ok":
            raise RuntimeError(decoded.get("error", "Unknown worker error"))
        return decoded.get("result")


class ZeroMQMLIPClient(MLIPClient):
    def __init__(self, host: str, port: int, timeout_ms: int = 900000) -> None:
        import zmq

        self._zmq = zmq
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.connect(f"tcp://{host}:{port}")
        self._socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, timeout_ms)

    def request(self, payload: Dict[str, Any]) -> Any:
        self._socket.send_json(payload)
        decoded = self._socket.recv_json()
        if decoded.get("status") != "ok":
            raise RuntimeError(decoded.get("error", "Unknown worker error"))
        return decoded.get("result")

    def close(self) -> None:
        self._socket.close(linger=0)
        self._context.term()


def build_client(transport: str, host: str, port: int) -> MLIPClient:
    if transport == "http":
        return HTTPMLIPClient(host=host, port=port)
    if transport == "zmq":
        return ZeroMQMLIPClient(host=host, port=port)
    raise ValueError(f"Unsupported transport: {transport}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send calculation requests to a persistent MLIP worker")
    parser.add_argument("--transport", default="http", choices=["http", "zmq"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--xyz", default=None, help="Path to XYZ file. If omitted, XYZ is read from stdin.")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--energy-only", action="store_true", help="Skip force calculation")
    parser.add_argument("--shutdown", action="store_true", help="Send shutdown command after benchmark")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.xyz:
        with open(args.xyz, "r", encoding="utf-8") as handle:
            xyz_text = handle.read()
    else:
        import sys

        xyz_text = sys.stdin.read()

    client = build_client(args.transport, host=args.host, port=args.port)

    start = time.perf_counter()
    last_result: Optional[Dict[str, Any]] = None
    for _ in range(args.iterations):
        last_result = client.calculate(xyz=xyz_text, gradients=not args.energy_only)
    elapsed_s = time.perf_counter() - start

    output = {
        "iterations": args.iterations,
        "elapsed_s": elapsed_s,
        "avg_ms": (elapsed_s / max(args.iterations, 1)) * 1000.0,
        "last_result": last_result,
    }
    print(json.dumps(output))

    if args.shutdown:
        client.shutdown()

    client.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
