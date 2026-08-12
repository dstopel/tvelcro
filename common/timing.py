"""Wall-clock latency and NVML energy measurement utilities."""

import threading
import time
from typing import Callable, Optional

import numpy as np
import torch


def measure_latency(
    fn: Callable[[], object],
    warmup: int = 10,
    iters: int = 50,
    device: str = "cuda",
) -> dict:
    """Time fn() with per-call CUDA fences. Returns stats in milliseconds."""
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()

    def sync():
        if use_cuda:
            torch.cuda.synchronize()

    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        sync()

        times_ms = []
        for _ in range(iters):
            sync()
            t0 = time.perf_counter()
            fn()
            sync()
            times_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.asarray(times_ms, dtype=np.float64)
    return {
        "median": float(np.median(arr)),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p5": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "iters": int(iters),
    }


def measure_energy(
    fn: Callable[[], object],
    iters: int = 20,
    device: str = "cuda",
    device_index: int = 0,
) -> dict:
    """Energy over iters calls of fn(). Values are None without NVML/GPU."""
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.synchronize()
    with EnergyMeter(device_index=device_index) as em:
        with torch.inference_mode():
            for _ in range(iters):
                fn()
        if use_cuda:
            torch.cuda.synchronize()
    d = em.as_dict()
    d["iters"] = iters
    d["joules_per_iter"] = None if d["joules"] is None else d["joules"] / iters
    return d


class EnergyMeter:
    """Context manager integrating GPU power draw (NVML) over the enclosed region.

    After exit: .joules, .avg_watts, .duration_s. All None when NVML or a GPU
    is unavailable (graceful degradation, never raises for that reason).
    """

    def __init__(self, device_index: int = 0, interval_s: float = 0.05):
        self.device_index = device_index
        self.interval_s = interval_s
        self.joules: Optional[float] = None
        self.avg_watts: Optional[float] = None
        self.duration_s: Optional[float] = None
        self._samples = []  # (t, watts)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._handle = None
        self._nvml = None

    def __enter__(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        except Exception:
            self._nvml = None
            return self

        self._stop.clear()
        self._samples = []
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._t0 = time.perf_counter()
        self._thread.start()
        return self

    def _sample_loop(self):
        while not self._stop.is_set():
            try:
                mw = self._nvml.nvmlDeviceGetPowerUsage(self._handle)
                self._samples.append((time.perf_counter(), mw / 1000.0))
            except Exception:
                break
            self._stop.wait(self.interval_s)

    def __exit__(self, *exc):
        if self._nvml is None:
            return False
        self._stop.set()
        self._thread.join()
        t1 = time.perf_counter()
        self.duration_s = t1 - self._t0
        try:
            self._nvml.nvmlShutdown()
        except Exception:
            pass
        if len(self._samples) >= 2:
            ts = np.asarray([s[0] for s in self._samples])
            ws = np.asarray([s[1] for s in self._samples])
            trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy 1.x/2.x
            self.joules = float(trapezoid(ws, ts))
            self.avg_watts = float(self.joules / (ts[-1] - ts[0]))
        elif len(self._samples) == 1:
            # ponytail: single sample, assume constant power over the region
            self.avg_watts = float(self._samples[0][1])
            self.joules = self.avg_watts * self.duration_s
        return False

    def as_dict(self) -> dict:
        return {
            "joules": self.joules,
            "avg_watts": self.avg_watts,
            "duration_s": self.duration_s,
        }
