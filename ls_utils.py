#!/usr/bin/env python3
"""
Common helper functions for starting, stopping, and communicating
with llama-server.
"""

import requests
import subprocess
import sys
import time
from pathlib import Path


# Server config
SERVER_STARTUP_TIMEOUT = 120  # seconds
REQUEST_TIMEOUT = 60  # seconds per completion


def wait_for_server(port: int, timeout: int = SERVER_STARTUP_TIMEOUT) -> bool:
    """Wait for llama-server to be ready."""
    url = f"http://127.0.0.1:{port}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                data = r.json()
                if data.get("status") == "ok":
                    return True
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(1)
    return False


def start_server(llama_server_path: str, model_path: str, tmpdir: str,
                 port: int, extra_args: list[str] = None) -> subprocess.Popen:
    """Start llama-server and return the process handle."""
    cmd = [
        llama_server_path,
        "-m", model_path,
        "--port", str(port),
        "-c", "4096",           # small context for probe eval
        "-ngl", "99",           # offload all layers to GPU
        "--flash-attn", "on",
        "--cache-type-k", "q8_0",
        "--cache-type-v", "q8_0",
        "--reasoning", "off",
        "--no-warmup",
        "-np", "1",             # single slot
    ]
    if extra_args:
        cmd.extend(extra_args)

    print(f"  [CMD] {' '.join(cmd)}", flush=True)

    # Let server output go to a log file so we can debug without pipe deadlocks
    log_path = Path(f"{tmpdir}/rys_server_{port}.log")
    log_file = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    proc._log_file = log_file  # keep reference so it doesn't get GC'd
    proc._log_path = log_path
    print(f"  [PID] Server started as PID {proc.pid}, log: {log_path}", flush=True)
    return proc


def stop_server(proc: subprocess.Popen):
    """Gracefully stop the server."""
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    # Close the log file
    if hasattr(proc, '_log_file'):
        proc._log_file.close()


def dump_server_log(proc: subprocess.Popen, tail_lines: int = 30):
    """Print the last N lines of the server log for debugging."""
    if hasattr(proc, '_log_path') and proc._log_path.exists():
        lines = proc._log_path.read_text().splitlines()
        print(f"  --- Server log (last {tail_lines} lines) ---", file=sys.stderr)
        for line in lines[-tail_lines:]:
            print(f"  | {line}", file=sys.stderr)
        print(f"  --- End server log ---", file=sys.stderr)


def query_model(prompt: str, port: int, max_tokens: int = 256) -> str | None:
    """Send a completion request to llama-server."""
    url = f"http://127.0.0.1:{port}/v1/chat/completions"

    payload = {
        "model": "test",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }

    try:
        r = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
        if r.status_code == 200:
            data = r.json()
            return data["choices"][0]["message"]["content"]
        else:
            print(f"  [WARN] Server returned {r.status_code}", file=sys.stderr)
            return None
    except (requests.ConnectionError, requests.Timeout) as e:
        print(f"  [WARN] Request failed: {e}", file=sys.stderr)
        return None
