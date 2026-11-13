"""Start/stop an OpenAI-compatible server process and run `vllm bench serve` against it.
Enforces one GPU process at a time: refuses to start if any GPU compute process exists,
and checks the GPU is empty again after stopping."""
import json
import os
import signal
import subprocess
import time
import urllib.request

VLLM_BIN = os.environ.get("VLLM_BIN", "/root/.venv-vllm/bin/vllm")


def gpu_processes() -> list[str]:
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True).stdout
    return [line for line in out.splitlines() if line.strip()]


def assert_gpu_idle():
    procs = gpu_processes()
    if procs:
        raise RuntimeError(f"GPU is busy (one GPU process at a time): {procs}")


def wait_healthy(port: int, proc: subprocess.Popen, timeout: float = 900):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with code {proc.returncode} during startup")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:  # noqa: BLE001 - not up yet
            pass
        time.sleep(2)
    raise TimeoutError("server did not become healthy")


def start_server(cmd: list[str], port: int, log_path: str, env: dict | None = None) -> subprocess.Popen:
    assert_gpu_idle()
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    log = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **(env or {})},
                            start_new_session=True)
    t = time.monotonic()
    wait_healthy(port, proc)
    print(f"server healthy after {time.monotonic() - t:.0f}s: {' '.join(cmd)}", flush=True)
    return proc


def stop_server(proc: subprocess.Popen, timeout: float = 60):
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGINT)           # graceful shutdown
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    for _ in range(30):
        if not gpu_processes():
            return
        time.sleep(1)
    raise RuntimeError(f"GPU still busy after stopping server: {gpu_processes()}")


def scrape_metrics(port: int) -> dict[str, float]:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5) as r:
        text = r.read().decode()
    out = {}
    for line in text.splitlines():
        if line and not line.startswith("#"):
            name, _, value = line.rpartition(" ")
            try:
                out[name] = float(value)
            except ValueError:
                pass
    return out


def vllm_bench_serve(port: int, served_model: str, tokenizer: str, result_dir: str, result_filename: str,
                     extra: list[str], log_path: str) -> dict:
    cmd = [VLLM_BIN, "bench", "serve", "--backend", "openai", "--base-url", f"http://127.0.0.1:{port}",
           "--endpoint", "/v1/completions", "--model", served_model, "--tokenizer", tokenizer,
           "--save-result", "--result-dir", result_dir, "--result-filename", result_filename,
           "--disable-tqdm", *extra]
    with open(log_path, "w") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    with open(os.path.join(result_dir, result_filename)) as f:
        result = json.load(f)
    result["_command"] = " ".join(cmd)
    return result
