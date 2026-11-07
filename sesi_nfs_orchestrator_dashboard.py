# file: sesi_nfs_orchestrator_dashboard.py
"""
=============================================================================
SESI NFS ORCHESTRATOR + REAL-TIME WEBSOCKET DASHBOARD
CADO-NFS · msieve · GGNFS · OPMC · Paper-guided Diagnostics
=============================================================================

Purpose:
  Production-style orchestration and monitoring layer for classical NFS tools.

Backends:
  - CADO-NFS
  - msieve
  - GGNFS / factmsieve wrapper

OPMC:
  One shared feature extraction / diagnostic pass supports:
    1. backend selection
    2. parameter recommendation
    3. cost prediction
    4. progress monitoring
    5. anomaly / phase-change detection
    6. dashboard visualization

15-paper grounding:
  - Paper 5: ergodic / AMS / log-scale diagnostics on progress residuals
  - Paper 6: change-point / block-length diagnostics
  - Paper 7: external noise diagnostic hierarchy
  - Paper 8: internal diagnostic battery
  - Paper 9: finite-sample statistical dictionary hooks
  - Paper 11: assumption-light subsampling CI
  - OPMC: shared processing, many computations
  - Revision 21: timing-independent OPMC statistic, but real-time convergence caveats
  - Revision 22: exact finite-time prediction if generating law is known
  - No-Zeno papers: event-rate / restart-budget constraints

Honest scope:
  - This is an orchestration/benchmarking layer, not a new proven GNFS algorithm.
  - CADO-NFS/msieve/GGNFS do the real factorization.
  - OPMC can reduce repeated planning/monitoring work only if benchmarks show it.
  - No Shor equivalence is claimed.
  - No P=NP claim is made.

Install optional dependencies:
  pip install fastapi uvicorn plotly

External tools:
  - CADO-NFS: https://gitlab.inria.fr/cado-nfs/cado-nfs
  - msieve: https://github.com/radii/msieve
  - GGNFS / factmsieve wrappers

Run dashboard:
  uvicorn sesi_nfs_orchestrator_dashboard:app --host 0.0.0.0 --port 8080

Then open:
  http://localhost:8080
=============================================================================
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import random
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("sesi.nfs_orchestrator")
logging.basicConfig(level=logging.INFO)

# Optional FastAPI
try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi import Request
    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False
    FastAPI = None


# =============================================================================
# Basic number theory utilities
# =============================================================================

def is_probable_prime(n: int, rounds: int = 24) -> bool:
    """Miller-Rabin probabilistic primality test."""
    if n < 2:
        return False
    small = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for p in small:
        if n == p:
            return True
        if n % p == 0:
            return False

    d = n - 1
    s = 0
    while d % 2 == 0:
        s += 1
        d //= 2

    for _ in range(rounds):
        a = random.randrange(2, n - 2)
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def validate_factorization(N: int, factors: List[int]) -> Dict[str, Any]:
    product = 1
    prime_flags = []
    for f in factors:
        if not isinstance(f, int) or f <= 1:
            return {"valid": False, "reason": "invalid_factor"}
        product *= f
        prime_flags.append(is_probable_prime(f))
    return {
        "valid": product == N and all(prime_flags),
        "product_equals_N": product == N,
        "prime_flags": prime_flags,
        "product": product,
    }


def generate_semiprime(bits: int) -> Tuple[int, int, int]:
    """Generate random semiprime N=p*q with approximately given bit length."""
    def gen_prime(k):
        while True:
            x = random.getrandbits(k) | (1 << (k - 1)) | 1
            if is_probable_prime(x):
                return x

    p = gen_prime(bits // 2)
    q = gen_prime(bits - bits // 2)
    return p * q, p, q


# =============================================================================
# Backend definitions
# =============================================================================

class BackendKind(str, Enum):
    CADO_NFS = "cado-nfs"
    MSIEVE = "msieve"
    GGNFS = "ggnfs"


class Phase(str, Enum):
    INIT = "init"
    POLYSELECT = "polyselect"
    SIEVING = "sieving"
    FILTERING = "filtering"
    LINEAR_ALGEBRA = "linear_algebra"
    SQRT = "sqrt"
    DONE = "done"
    FAILED = "failed"


@dataclass
class NFSParams:
    threads: int = 4
    degree: Optional[int] = None
    lpb0: Optional[int] = None
    lpb1: Optional[int] = None
    lim0: Optional[int] = None
    lim1: Optional[int] = None
    workdir: str = "/tmp/sesi_nfs"
    timeout_s: int = 3600
    extra_args: List[str] = field(default_factory=list)


@dataclass
class JobState:
    job_id: str
    N: int
    backend: BackendKind
    params: NFSParams
    status: str = "created"
    phase: Phase = Phase.INIT
    pid: Optional[int] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    elapsed_s: float = 0.0
    progress: float = 0.0
    relations_found: int = 0
    relations_needed: int = 0
    factors: List[int] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def public_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["backend"] = self.backend.value
        d["phase"] = self.phase.value
        d["N_digits"] = len(str(self.N))
        d["N_bits"] = self.N.bit_length()
        return d


# =============================================================================
# OPMC + paper-guided planner
# =============================================================================

@dataclass
class OPMCFeatures:
    N_bits: int
    N_digits: int
    small_prime_divisibility: int
    recommended_degree: int
    recommended_lpb: int
    recommended_lim: int
    available_backends: List[str]
    estimated_difficulty: float


class OPMCCostAwarePlanner:
    """
    OPMC shared feature extraction.
    One pass produces features reused by:
      - backend selector
      - parameter selector
      - cost estimator
      - dashboard
      - paper-guided diagnostics

    This is heuristic planning, not a complexity theorem.
    """

    def recommend_degree(self, bits: int) -> int:
        if bits < 80:
            return 4
        if bits < 140:
            return 5
        if bits < 220:
            return 6
        return 7

    def recommend_lpb(self, bits: int) -> int:
        if bits < 80:
            return 20
        if bits < 128:
            return 26
        if bits < 192:
            return 28
        if bits < 256:
            return 30
        return 32

    def recommend_lim(self, bits: int) -> int:
        if bits < 80:
            return 200_000
        if bits < 128:
            return 2_000_000
        if bits < 192:
            return 10_000_000
        if bits < 256:
            return 30_000_000
        return 100_000_000

    def extract_features(self, N: int, available: List[BackendKind]) -> OPMCFeatures:
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
        div_count = sum(1 for p in small_primes if N % p == 0)
        bits = N.bit_length()
        digits = len(str(N))

        degree = self.recommend_degree(bits)
        lpb = self.recommend_lpb(bits)
        lim = self.recommend_lim(bits)

        # Heuristic difficulty proxy, not NFS complexity theorem.
        difficulty = math.log2(max(N, 2)) / 32.0

        return OPMCFeatures(
            N_bits=bits,
            N_digits=digits,
            small_prime_divisibility=div_count,
            recommended_degree=degree,
            recommended_lpb=lpb,
            recommended_lim=lim,
            available_backends=[b.value for b in available],
            estimated_difficulty=difficulty,
        )

    def choose_backend(self, features: OPMCFeatures, available: List[BackendKind]) -> BackendKind:
        # Prefer CADO for larger jobs, msieve for smaller jobs if available.
        if features.N_bits >= 128 and BackendKind.CADO_NFS in available:
            return BackendKind.CADO_NFS
        if BackendKind.MSIEVE in available:
            return BackendKind.MSIEVE
        if BackendKind.CADO_NFS in available:
            return BackendKind.CADO_NFS
        if BackendKind.GGNFS in available:
            return BackendKind.GGNFS
        raise RuntimeError("No NFS backend available.")

    def make_params(self, N: int, backend: BackendKind, threads: int, workdir: str) -> Tuple[NFSParams, OPMCFeatures]:
        available = detect_backends()
        features = self.extract_features(N, available)
        params = NFSParams(
            threads=threads,
            degree=features.recommended_degree,
            lpb0=features.recommended_lpb,
            lpb1=features.recommended_lpb,
            lim0=features.recommended_lim,
            lim1=features.recommended_lim,
            workdir=workdir,
        )
        return params, features


# =============================================================================
# Paper-guided diagnostics on monitoring stream
# =============================================================================

class PaperDiagnostics:
    """
    Lightweight Paper 5/6/7/8/9/11 hooks on monitoring residuals.
    These are diagnostics, not finite-sample proof unless assumptions hold.
    """

    @staticmethod
    def cusum(values: List[float], drift: float = 0.0, threshold: float = 5.0) -> Dict[str, Any]:
        if not values:
            return {"alarms": [], "cusum": []}
        mu = sum(values) / len(values)
        s = 0.0
        traj, alarms = [], []
        for i, x in enumerate(values):
            s = max(0.0, s + (x - mu) - drift)
            traj.append(s)
            if s > threshold:
                alarms.append(i)
        return {"cusum": traj, "alarms": alarms}

    @staticmethod
    def cesaro_stability(values: List[float]) -> Dict[str, Any]:
        if len(values) < 4:
            return {"stable": False, "reason": "insufficient_data"}
        means = []
        acc = 0.0
        for i, x in enumerate(values, 1):
            acc += x
            means.append(acc / i)
        diffs = [abs(means[i] - means[i - 1]) for i in range(1, len(means))]
        tail = diffs[-max(3, len(diffs)//4):]
        return {
            "stable": max(tail) < 1e-3,
            "tail_max_diff": max(tail),
            "cesaro_final": means[-1],
        }

    @staticmethod
    def realized_qv(values: List[float]) -> Dict[str, Any]:
        if len(values) < 3:
            return {"qv": 0.0, "finite_positive": False}
        qv = sum((values[i] - values[i-1])**2 for i in range(1, len(values)))
        return {"qv": qv, "finite_positive": math.isfinite(qv) and qv > 0.0}

    @staticmethod
    def subsampling_ci(values: List[float], block: int = 8, level: float = 0.95) -> Dict[str, Any]:
        if len(values) < block * 2:
            return {"valid": False, "reason": "insufficient_data"}
        means = []
        for i in range(0, len(values) - block + 1, max(1, block // 2)):
            chunk = values[i:i+block]
            means.append(sum(chunk) / len(chunk))
        means.sort()
        alpha = 1 - level
        lo = means[int((alpha/2) * (len(means)-1))]
        hi = means[int((1-alpha/2) * (len(means)-1))]
        return {"valid": True, "ci": [lo, hi], "n_blocks": len(means)}


# =============================================================================
# Backend adapters
# =============================================================================

def detect_backends() -> List[BackendKind]:
    available = []
    if shutil.which("cado-nfs.py") or shutil.which("cado-nfs"):
        available.append(BackendKind.CADO_NFS)
    if shutil.which("msieve"):
        available.append(BackendKind.MSIEVE)
    if shutil.which("factmsieve.py") or shutil.which("factMsieve.pl") or shutil.which("ggnfs"):
        available.append(BackendKind.GGNFS)
    return available


class BackendAdapter:
    def build_command(self, N: int, params: NFSParams) -> List[str]:
        raise NotImplementedError

    def parse_line(self, line: str, state: JobState) -> None:
        lower = line.lower()

        if "poly" in lower or "polyselect" in lower:
            state.phase = Phase.POLYSELECT
        elif "sieve" in lower or "sieving" in lower or "relations" in lower or "rels" in lower:
            state.phase = Phase.SIEVING
        elif "filter" in lower or "merge" in lower:
            state.phase = Phase.FILTERING
        elif "linalg" in lower or "matrix" in lower or "wiedemann" in lower or "lanczos" in lower:
            state.phase = Phase.LINEAR_ALGEBRA
        elif "sqrt" in lower or "square root" in lower:
            state.phase = Phase.SQRT

        # relation parsing
        m = re.search(r"(\d+)\s+relations?", lower)
        if m:
            state.relations_found = max(state.relations_found, int(m.group(1)))
        m = re.search(r"need\s+(\d+)", lower)
        if m:
            state.relations_needed = max(state.relations_needed, int(m.group(1)))

        if state.relations_needed > 0:
            state.progress = min(1.0, state.relations_found / state.relations_needed)

        # factor parsing: plain integer lines or "factor: 123"
        candidates = []
        if re.fullmatch(r"\d+", line.strip()):
            candidates.append(int(line.strip()))
        for m in re.finditer(r"(?:factor|prp\d*|p\d*)\D+(\d+)", lower):
            candidates.append(int(m.group(1)))

        for f in candidates:
            if f > 1 and state.N % f == 0 and f not in state.factors:
                state.factors.append(f)

        if state.factors:
            v = validate_factorization(state.N, state.factors)
            if v["valid"]:
                state.phase = Phase.DONE
                state.progress = 1.0


class CadoNFSAdapter(BackendAdapter):
    def build_command(self, N: int, params: NFSParams) -> List[str]:
        exe = shutil.which("cado-nfs.py") or shutil.which("cado-nfs")
        if not exe:
            raise RuntimeError("CADO-NFS not found")
        Path(params.workdir).mkdir(parents=True, exist_ok=True)

        cmd = [
            exe,
            str(N),
            f"tasks.workdir={params.workdir}",
            f"tasks.threads={params.threads}",
        ]

        # CADO parameter names can vary by version; keep common ones configurable.
        if params.degree is not None:
            cmd.append(f"tasks.polyselect.degree={params.degree}")
        if params.lpb0 is not None:
            cmd.append(f"tasks.sieve.lpb0={params.lpb0}")
        if params.lpb1 is not None:
            cmd.append(f"tasks.sieve.lpb1={params.lpb1}")
        if params.lim0 is not None:
            cmd.append(f"tasks.sieve.lim0={params.lim0}")
        if params.lim1 is not None:
            cmd.append(f"tasks.sieve.lim1={params.lim1}")

        cmd.extend(params.extra_args)
        return cmd


class MsieveAdapter(BackendAdapter):
    def build_command(self, N: int, params: NFSParams) -> List[str]:
        exe = shutil.which("msieve")
        if not exe:
            raise RuntimeError("msieve not found")
        Path(params.workdir).mkdir(parents=True, exist_ok=True)
        log = str(Path(params.workdir) / "msieve.log")
        save = str(Path(params.workdir) / "msieve.dat")

        # Common msieve NFS invocation. Users may override via extra_args.
        cmd = [
            exe,
            "-v",
            "-n",
            "-t", str(params.threads),
            "-l", log,
            "-s", save,
            str(N),
        ]
        cmd.extend(params.extra_args)
        return cmd


class GGNFSAdapter(BackendAdapter):
    def build_command(self, N: int, params: NFSParams) -> List[str]:
        wrapper = shutil.which("factmsieve.py") or shutil.which("factMsieve.pl")
        if wrapper:
            return [wrapper, str(N)] + params.extra_args

        exe = shutil.which("ggnfs")
        if exe:
            return [exe, str(N)] + params.extra_args

        raise RuntimeError("GGNFS/factmsieve wrapper not found")


def get_adapter(backend: BackendKind) -> BackendAdapter:
    if backend == BackendKind.CADO_NFS:
        return CadoNFSAdapter()
    if backend == BackendKind.MSIEVE:
        return MsieveAdapter()
    if backend == BackendKind.GGNFS:
        return GGNFSAdapter()
    raise ValueError(backend)


# =============================================================================
# Job manager + WebSocket broadcaster
# =============================================================================

class ConnectionManager:
    def __init__(self):
        self.active: List[Any] = []

    async def connect(self, ws):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, message: Dict[str, Any]):
        dead = []
        for ws in self.active:
            try:
                await ws.send_text(json.dumps(message, default=str))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


class NFSJobManager:
    def __init__(self):
        self.jobs: Dict[str, JobState] = {}
        self.planner = OPMCCostAwarePlanner()
        self.ws = ConnectionManager()
        self.progress_history: Dict[str, List[float]] = {}

    async def start_job(
        self,
        N: int,
        backend: Optional[BackendKind] = None,
        threads: int = 4,
        workdir: Optional[str] = None,
    ) -> str:
        available = detect_backends()
        if not available:
            raise RuntimeError("No backend found: install CADO-NFS, msieve, or GGNFS.")

        if backend is None:
            features = self.planner.extract_features(N, available)
            backend = self.planner.choose_backend(features, available)

        job_id = f"job_{int(time.time()*1000)}_{random.randint(1000,9999)}"
        workdir = workdir or f"/tmp/sesi_nfs/{job_id}"

        params, features = self.planner.make_params(N, backend, threads, workdir)
        state = JobState(job_id=job_id, N=N, backend=backend, params=params)
        state.status = "running"
        self.jobs[job_id] = state
        self.progress_history[job_id] = []

        asyncio.create_task(self._run_job(job_id, features))
        return job_id

    async def _run_job(self, job_id: str, features: OPMCFeatures):
        state = self.jobs[job_id]
        adapter = get_adapter(state.backend)
        cmd = adapter.build_command(state.N, state.params)

        await self.ws.broadcast({
            "type": "job_started",
            "job": state.public_dict(),
            "cmd": cmd,
            "features": asdict(features),
        })

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=state.params.workdir,
            )
            state.pid = proc.pid

            start = time.time()
            while True:
                if proc.stdout is None:
                    break

                line_b = await proc.stdout.readline()
                if not line_b:
                    break

                line = line_b.decode(errors="replace").rstrip()
                state.logs.append(line)
                state.elapsed_s = time.time() - start
                adapter.parse_line(line, state)

                self.progress_history[job_id].append(state.progress)
                diag = self._diagnostics(job_id)

                await self.ws.broadcast({
                    "type": "log",
                    "job_id": job_id,
                    "line": line,
                    "state": state.public_dict(),
                    "diagnostics": diag,
                })

            rc = await proc.wait()
            state.end_time = time.time()
            state.elapsed_s = state.end_time - state.start_time

            validation = validate_factorization(state.N, state.factors) if state.factors else {"valid": False}

            if rc == 0 and validation.get("valid"):
                state.status = "completed"
                state.phase = Phase.DONE
                state.progress = 1.0
            elif rc == 0 and state.factors:
                state.status = "completed_unverified"
            else:
                state.status = "failed"
                state.phase = Phase.FAILED
                state.error = f"return_code={rc}"

        except Exception as e:
            state.status = "failed"
            state.phase = Phase.FAILED
            state.error = str(e)

        await self.ws.broadcast({
            "type": "job_finished",
            "job": state.public_dict(),
            "validation": validate_factorization(state.N, state.factors) if state.factors else {"valid": False},
        })

    def _diagnostics(self, job_id: str) -> Dict[str, Any]:
        vals = self.progress_history.get(job_id, [])
        # use increments as residual-like stream
        diffs = [vals[i] - vals[i-1] for i in range(1, len(vals))]
        return {
            "paper6_cusum": PaperDiagnostics.cusum(diffs),
            "paper5_ams": PaperDiagnostics.cesaro_stability(vals),
            "paper8_qv": PaperDiagnostics.realized_qv(vals),
            "paper11_subsampling_ci": PaperDiagnostics.subsampling_ci(vals),
        }

    def list_jobs(self) -> List[Dict[str, Any]]:
        return [j.public_dict() for j in self.jobs.values()]

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        j = self.jobs.get(job_id)
        return j.public_dict() if j else None


manager = NFSJobManager()


# =============================================================================
# FastAPI app + dashboard
# =============================================================================

if _HAS_FASTAPI:
    app = FastAPI(title="SESI NFS OPMC Dashboard")

    @app.get("/")
    async def index():
        return HTMLResponse(DASHBOARD_HTML)

    @app.get("/api/backends")
    async def api_backends():
        return {"available": [b.value for b in detect_backends()]}

    @app.get("/api/jobs")
    async def api_jobs():
        return {"jobs": manager.list_jobs()}

    @app.get("/api/job/{job_id}")
    async def api_job(job_id: str):
        j = manager.get_job(job_id)
        if j is None:
            return JSONResponse({"error": "not_found"}, status_code=404)
        return j

    @app.post("/api/start")
    async def api_start(req: Request):
        data = await req.json()
        N = int(data["N"])
        backend = data.get("backend")
        backend = BackendKind(backend) if backend else None
        threads = int(data.get("threads", 4))
        workdir = data.get("workdir")
        job_id = await manager.start_job(N, backend, threads, workdir)
        return {"job_id": job_id}

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket):
        await manager.ws.connect(ws)
        try:
            await ws.send_text(json.dumps({"type": "initial", "jobs": manager.list_jobs()}, default=str))
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            manager.ws.disconnect(ws)


DASHBOARD_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>SESI NFS OPMC Dashboard</title>
  <style>
    body { background:#111; color:#eee; font-family:Arial, sans-serif; margin:20px; }
    input, select, button { padding:8px; margin:4px; background:#222; color:#eee; border:1px solid #555; }
    button { cursor:pointer; }
    .panel { background:#1b1b1b; border:1px solid #333; padding:12px; margin:12px 0; border-radius:8px; }
    .log { height:320px; overflow:auto; background:#050505; padding:10px; font-family:monospace; font-size:12px; }
    .job { border-bottom:1px solid #333; padding:8px 0; }
    .phase { color:#6cf; }
    .ok { color:#6f6; }
    .fail { color:#f66; }
  </style>
</head>
<body>
  <h1>SESI NFS OPMC Real-Time Dashboard</h1>

  <div class="panel">
    <h3>Start Factorization Job</h3>
    <input id="N" size="60" placeholder="Integer N or decimal digits">
    <select id="backend">
      <option value="">auto</option>
      <option value="cado-nfs">CADO-NFS</option>
      <option value="msieve">msieve</option>
      <option value="ggnfs">GGNFS</option>
    </select>
    <input id="threads" type="number" value="4" min="1" max="256">
    <button onclick="startJob()">Start</button>
    <div id="backends"></div>
  </div>

  <div class="panel">
    <h3>Jobs</h3>
    <div id="jobs"></div>
  </div>

  <div class="panel">
    <h3>Live Log</h3>
    <div id="log" class="log"></div>
  </div>

  <div class="panel">
    <h3>Paper-Guided Diagnostics</h3>
    <pre id="diag"></pre>
  </div>

<script>
let ws = new WebSocket(`ws://${location.host}/ws`);

ws.onmessage = (ev) => {
  const msg = JSON.parse(ev.data);
  if (msg.type === "initial") {
    renderJobs(msg.jobs);
  }
  if (msg.type === "job_started") {
    appendLog("START " + msg.job.job_id + " cmd=" + msg.cmd.join(" "));
  }
  if (msg.type === "log") {
    appendLog(msg.line);
    upsertJob(msg.state);
    document.getElementById("diag").textContent = JSON.stringify(msg.diagnostics, null, 2);
  }
  if (msg.type === "job_finished") {
    appendLog("FINISH " + JSON.stringify(msg.job));
    upsertJob(msg.job);
  }
};

function appendLog(s) {
  const el = document.getElementById("log");
  el.textContent += s + "\\n";
  el.scrollTop = el.scrollHeight;
}

function renderJobs(jobs) {
  const el = document.getElementById("jobs");
  el.innerHTML = "";
  jobs.forEach(upsertJob);
}

function upsertJob(j) {
  const el = document.getElementById("jobs");
  let div = document.getElementById(j.job_id);
  if (!div) {
    div = document.createElement("div");
    div.id = j.job_id;
    div.className = "job";
    el.appendChild(div);
  }
  const cls = j.status === "completed" ? "ok" : (j.status === "failed" ? "fail" : "");
  div.innerHTML = `
    <b>${j.job_id}</b>
    | ${j.backend}
    | <span class="phase">${j.phase}</span>
    | <span class="${cls}">${j.status}</span>
    | bits=${j.N_bits}
    | progress=${(100*j.progress).toFixed(2)}%
    | rels=${j.relations_found}/${j.relations_needed}
    | factors=${JSON.stringify(j.factors)}
  `;
}

async function startJob() {
  const N = document.getElementById("N").value;
  const backend = document.getElementById("backend").value || null;
  const threads = document.getElementById("threads").value;
  const res = await fetch("/api/start", {
    method:"POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({N, backend, threads})
  });
  const js = await res.json();
  appendLog("JOB CREATED " + JSON.stringify(js));
}

fetch("/api/backends").then(r=>r.json()).then(js=>{
  document.getElementById("backends").textContent = "Available backends: " + JSON.stringify(js.available);
});
</script>
</body>
</html>
"""


# =============================================================================
# CLI / smoke test
# =============================================================================

def smoke_test():
    print("=" * 72)
    print("SESI NFS Orchestrator — smoke test")
    print("=" * 72)
    print("Available backends:", [b.value for b in detect_backends()])

    N, p, q = generate_semiprime(64)
    print("Generated semiprime:", N, "=", p, "*", q)

    planner = OPMCCostAwarePlanner()
    feats = planner.extract_features(N, detect_backends())
    print("OPMC features:", asdict(feats))

    print("Validation:", validate_factorization(N, [p, q]))

    vals = [0.0, 0.1, 0.2, 0.25, 0.26, 0.27]
    print("CUSUM:", PaperDiagnostics.cusum(vals))
    print("AMS:", PaperDiagnostics.cesaro_stability(vals))
    print("QV:", PaperDiagnostics.realized_qv(vals))
    print("Subsampling CI:", PaperDiagnostics.subsampling_ci(vals))

    if not _HAS_FASTAPI:
        print("FastAPI not installed. Install: pip install fastapi uvicorn")

    print("Done.")


if __name__ == "__main__":
    smoke_test()
