#!/usr/bin/env python3

"""Unified persistent MLIP worker with pluggable transport and backend."""

import abc
import argparse
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional

from mlip_workers import (
    MLIPWorker,
    TorchMDNetWorker,
    FairchemWorker,
    FennolWorker,
    MACEWorker,
    OrbitalWorker,
    MlatomWorker,
    So3lrWorker,
    NequipWorker,
    AimnetWorker,
)

WORKER_CLASSES = {
    "torchmdnet": TorchMDNetWorker,
    "fairchem": FairchemWorker,
    "fennol": FennolWorker,
    "mace": MACEWorker,
    "orbital": OrbitalWorker,
    "mlatom": MlatomWorker,
    "so3lr": So3lrWorker,
    "nequip": NequipWorker,
    "aimnet": AimnetWorker,
}


class WorkerService:
    def __init__(self, worker: MLIPWorker) -> None:
        self._worker = worker
        self._lock = threading.Lock()

    def handle(self, request: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        cmd = request.get("cmd")

        if cmd == "ping":
            return {"status": "ok", "result": "OK"}, False

        if cmd == "calculate":
            xyz = request.get("xyz")
            if not isinstance(xyz, str):
                return {"status": "error", "error": "Missing 'xyz' payload"}, False
            gradients = request.get("gradients", True)
            if not isinstance(gradients, bool):
                return {"status": "error", "error": "Invalid 'gradients' value"}, False
            charge = request.get("charge")
            if charge is not None:
                try:
                    charge = int(charge)
                except (TypeError, ValueError):
                    return {"status": "error", "error": "Invalid 'charge' value"}, False
            with self._lock:
                result = self._worker.calculate(xyz=xyz, gradients=gradients, charge=charge)
            return {"status": "ok", "result": result}, False

        if cmd == "shutdown":
            return {"status": "ok", "result": "bye"}, True

        return {"status": "error", "error": f"Unknown command: {cmd}"}, False


class TransportServer(abc.ABC):
    def __init__(self, host: str, port: int, service: WorkerService) -> None:
        self.host = host
        self.port = port
        self.service = service

    @abc.abstractmethod
    def serve_forever(self) -> None:
        """Start serving requests until shutdown command is received."""


class HTTPTransportServer(TransportServer):
    def serve_forever(self) -> None:
        service = self.service

        class RequestHandler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                response = {"status": "error", "error": "Malformed request"}
                should_stop = False
                try:
                    length = int(self.headers.get("Content-Length", 0))
                    body = self.rfile.read(length)
                    payload = json.loads(body.decode("utf-8"))
                    response, should_stop = service.handle(payload)
                except Exception as exc:
                    response = {"status": "error", "error": str(exc)}

                data = json.dumps(response).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

                if should_stop:
                    threading.Thread(target=self.server.shutdown, daemon=True).start()

            def log_message(self, format: str, *args: Any) -> None:
                return

        server = ThreadingHTTPServer((self.host, self.port), RequestHandler)
        server.serve_forever()


class ZeroMQTransportServer(TransportServer):
    def serve_forever(self) -> None:
        import zmq

        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(f"tcp://{self.host}:{self.port}")

        try:
            stop = False
            while not stop:
                payload = socket.recv_json()
                response, stop = self.service.handle(payload)
                socket.send_json(response)
        finally:
            socket.close(linger=0)
            context.term()


def build_transport(transport: str, host: str, port: int, service: WorkerService) -> TransportServer:
    transport_name = transport.lower()
    if transport_name == "http":
        return HTTPTransportServer(host=host, port=port, service=service)
    if transport_name == "zmq":
        return ZeroMQTransportServer(host=host, port=port, service=service)
    raise ValueError(f"Unsupported transport: {transport}")


def build_worker(
    backend: str,
    model_path: str,
    device: str,
    sp_only: bool,
    cpu_threads: int,
    cuda_memory_fraction: Optional[float],
) -> MLIPWorker:
    backend_name = backend.lower()
    
    return WORKER_CLASSES[backend_name](
        model_path=model_path,
        device=device,
        sp_only=sp_only,
        cpu_threads=cpu_threads,
        cuda_memory_fraction=cuda_memory_fraction,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent MLIP worker server")
    parser.add_argument("--backend", required=True, choices=sorted(WORKER_CLASSES))
    parser.add_argument("--model", required=True, help="Path to model file")
    parser.add_argument("--transport", default="http", choices=["http", "zmq"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cpu-threads", type=int, default=0)
    parser.add_argument("--cuda-memory-fraction", type=float, default=None)
    parser.add_argument("--sp-only", action="store_true", help="Load model without gradients when supported")
    return parser.parse_args()


def validate_and_prepare(args: argparse.Namespace) -> None:
    if args.cpu_threads > 0:
        thread_count = str(args.cpu_threads)
        os.environ["OMP_NUM_THREADS"] = thread_count
        os.environ["MKL_NUM_THREADS"] = thread_count
        os.environ["OPENBLAS_NUM_THREADS"] = thread_count
        os.environ["VECLIB_MAXIMUM_THREADS"] = thread_count
        os.environ["NUMEXPR_NUM_THREADS"] = thread_count

    if args.cuda_memory_fraction is not None:
        if args.cuda_memory_fraction <= 0.0 or args.cuda_memory_fraction > 1.0:
            raise ValueError("--cuda-memory-fraction must be within (0, 1]")


def main() -> int:
    args = parse_args()
    validate_and_prepare(args)

    worker = build_worker(
        backend=args.backend,
        model_path=args.model,
        device=args.device,
        sp_only=args.sp_only,
        cpu_threads=args.cpu_threads,
        cuda_memory_fraction=args.cuda_memory_fraction,
    )
    worker.load()

    service = WorkerService(worker)
    server = build_transport(args.transport, host=args.host, port=args.port, service=service)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
