"""Logging and monitoring. One decorator = one explicit metric; all record into `meter`, which `Logger.log` drains.

Keys: <split>/<group>/<name>, split in {train, test}, group in {perf, optim, progress, policy} or none (task scores).
Set the split with `with meter.split("test"): ...`; decorators write `<group>/<name>/<metric>` and the meter prefixes it.
A drained key is the mean over the step's calls, except progress/* counts, which are summed.

Layer-specific decorators decorate a method of an nn.Module and take
  modules: glob over `self.named_modules()` whose `*` is the layer index, e.g. "model.layers.*.self_attn"
  layers:  "all" or a list of layer indices
and hook those modules only for the duration of the decorated call. Every decorator takes every=N (record 1 call in N).

Derived per region <name>, from the step's sums (the meter adds them on drain):
  perf/<name>/tflops (flops / latency), perf/<name>/gbps (bytes / latency), perf/<name>/intensity (flops / bytes),
  perf/<name>/memory_bound (intensity below the device's ridge point, from GPU_SPECS),
  perf/<name>/frames_per_s | tokens_per_s (progress/<name>/frames | tokens / latency),
  perf/<name>/<m>_per_frame | <m>_per_token for m in latency_ms, flops, bytes.

Dims: B batch, H heads, Q queries, K keys, T tokens, D hidden, V vocab, F video frames, X height, Y width, C rgb."""
import bisect
import functools
import json
import re
import tempfile
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Literal

import imageio
import numpy as np
import torch
import wandb
from omegaconf import OmegaConf
from torch.profiler import ProfilerActivity

SPLITS = ("train", "test")
GROUPS = ("perf", "optim", "progress", "policy")
Layers = Literal["all"] | list[int]


class Meter:
    def __init__(self, spec: tuple[float, float] | None = None):
        self.current = "train"
        self.sums = {}  # <split>/<key> -> [sum, count]
        self.spec = spec  # (peak TFLOP/s, peak GB/s) for memory_bound; None = gpu_spec() when CUDA is available

    @contextmanager
    def split(self, split: str):
        prev, self.current = self.current, split
        try:
            yield
        finally:
            self.current = prev

    def record(self, key: str, value):
        s = self.sums.setdefault(f"{self.current}/{key}", [0.0, 0])
        s[0] += float(value)
        s[1] += 1

    def drain(self) -> dict:
        sums, self.sums = self.sums, {}
        out = {k: s if k.split("/")[1] == "progress" else s / n for k, (s, n) in sums.items()}
        total = {k: s for k, (s, _) in sums.items()}
        regions = {(k.split("/")[0], "/".join(k.split("/")[2:-1])) for k in total if k.split("/")[1] in ("perf", "progress")}
        for split, name in regions:
            p = f"{split}/perf/{name}/"
            ms, flops, nbytes = (total.get(p + m) for m in ("latency_ms", "flops", "bytes"))
            if flops is not None and ms:
                out[p + "tflops"] = flops / ms / 1e9
            if nbytes is not None and ms:
                out[p + "gbps"] = nbytes / ms / 1e6
            if flops is not None and nbytes:
                out[p + "intensity"] = flops / nbytes
                spec = self.spec or (gpu_spec() if torch.cuda.is_available() else None)
                if spec:
                    out[p + "memory_bound"] = flops / nbytes < 1e3 * spec[0] / spec[1]  # below the ridge point
            for unit in ("frame", "token"):
                n = total.get(f"{split}/progress/{name}/{unit}s")
                if not n:
                    continue
                if ms:
                    out[p + f"{unit}s_per_s"] = n / ms * 1e3
                for m, v in (("latency_ms", ms), ("flops", flops), ("bytes", nbytes)):
                    if v is not None:
                        out[p + f"{m}_per_{unit}"] = v / n
        return out


meter = Meter()


class Logger:
    """WandB + run_dir/metrics.jsonl. name <exp>/<name>, group <exp>, id <exp>-<name>, resume="allow";
    `step` is a logged axis (wandb's internal step rejects steps below its value on resume).
    Values: float | bool | video u8 [F, X, Y, C] (mp4 under run_dir/videos) | line chart {"xs", "ys": {series: ys}}."""

    def __init__(self, cfg):
        self.dir = Path(cfg.run_dir)
        wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, mode=cfg.wandb.mode, name=f"{cfg.exp}/{cfg.name}",
                   group=cfg.exp, id=re.sub(r"[^\w-]", "-", f"{cfg.exp}-{cfg.name}"), resume="allow",
                   dir=cfg.run_dir, config=OmegaConf.to_container(cfg, resolve=True))
        wandb.define_metric("step")
        for split in SPLITS:
            wandb.define_metric(f"{split}/*", step_metric="step")

    def log(self, metrics: dict, step: int):
        """Merge meter.drain(), validate keys, write wandb + metrics.jsonl (scalars)."""
        metrics = {**metrics, **meter.drain()}
        for k in metrics:
            check_key(k)
        out = {k: self.value(k, v, step) for k, v in metrics.items()}
        wandb.log({"step": step, **out})
        with open(self.dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps({"step": step, **{k: v for k, v in out.items() if isinstance(v, float)}}) + "\n")

    def value(self, key, v, step):
        if isinstance(v, dict):
            return wandb.plot.line_series(xs=v["xs"], ys=list(v["ys"].values()), keys=list(v["ys"]), title=key, xname="x")
        if isinstance(v, np.ndarray) and v.ndim == 4:
            path = self.dir / "videos" / f"{key.replace('/', '_')}_{step}.mp4"
            path.parent.mkdir(exist_ok=True)
            imageio.mimwrite(path, v, fps=20)
            return wandb.Video(str(path))
        return float(v)

    def finish(self):
        wandb.finish()


def check_key(key: str):
    """ValueError unless <split>/<...> with split in SPLITS."""
    parts = key.split("/")
    if len(parts) < 2 or parts[0] not in SPLITS:
        raise ValueError(f"metric keys are <split>/<group>/<name> with split in {SPLITS}: {key}")


def _decorator(every: int, measure: Callable):
    """measure(fn, args, kwargs) -> (out, {"<group>/<name>/<metric>": value}). Records 1 call in `every`."""
    def wrap(fn):
        calls = 0

        @functools.wraps(fn)
        def inner(*args, **kwargs):
            nonlocal calls
            calls += 1
            if (calls - 1) % every:
                return fn(*args, **kwargs)
            out, values = measure(fn, args, kwargs)
            for k, v in values.items():
                meter.record(k, v)
            return out
        return inner
    return wrap


_open, _seen = set(), set()


@contextmanager
def _region(name):
    """torch.profiler.record_function(name), opened once even when several *_profiling decorators share the name."""
    _seen.add(name)
    if name in _open:
        yield
        return
    _open.add(name)
    try:
        with torch.profiler.record_function(name):
            yield
    finally:
        _open.discard(name)


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timed(fn, args, kwargs):
    """-> (out, seconds), CUDA-synchronized."""
    _sync()
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    _sync()
    return out, time.perf_counter() - t0


# perf: each opens torch.profiler.record_function(name) once per call, so `profile()` attributes time to the region
def latency_profiling(name: str, every=1):
    """perf/<name>/latency_ms: CUDA-synchronized wall time of the call."""
    def measure(fn, args, kwargs):
        with _region(name):
            out, s = _timed(fn, args, kwargs)
        return out, {f"perf/{name}/latency_ms": 1e3 * s}
    return _decorator(every, measure)


def memory_profiling(name: str, every=1):
    """perf/<name>/peak_mem_gb: peak CUDA memory allocated during the call (not recorded without CUDA)."""
    def measure(fn, args, kwargs):
        if not torch.cuda.is_available():
            return fn(*args, **kwargs), {}
        with _region(name):
            torch.cuda.reset_peak_memory_stats()
            out = fn(*args, **kwargs)
        return out, {f"perf/{name}/peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30}
    return _decorator(every, measure)


def gpu_util_profiling(name: str, every=1):
    """perf/<name>/gpu_util: NVML GPU utilization (% of time a kernel ran) at the end of the call (not recorded without
    CUDA). NVML samples over its own 1/6-1 s window, so use it on calls that last at least that long."""
    def measure(fn, args, kwargs):
        if not torch.cuda.is_available():
            return fn(*args, **kwargs), {}
        with _region(name):
            out = fn(*args, **kwargs)
        return out, {f"perf/{name}/gpu_util": torch.cuda.utilization()}
    return _decorator(every, measure)


def mfu_profiling(name: str, flops_fn: Callable, peak_tflops: float | None = None, every=1):
    """perf/<name>/mfu = flops_fn(*args, **kwargs) / (CUDA-synchronized seconds * peak FLOP/s).
    peak_tflops: the device's dense peak for the dtype used; None = gpu_spec()'s dense BF16 peak."""
    def measure(fn, args, kwargs):
        with _region(name):
            out, s = _timed(fn, args, kwargs)
        peak = peak_tflops or gpu_spec()[0]
        return out, {f"perf/{name}/mfu": flops_fn(*args, **kwargs) / s / (peak * 1e12)}
    return _decorator(every, measure)


# Vendor datasheet peaks: (dense BF16 tensor TFLOP/s, i.e. half the 2:4-sparse figure; memory GB/s).
# T4 has no BF16: FP16. RTX PRO 4500 BF16 is derived from its FP8/FP4 figures (NVIDIA publishes AI TOPS only).
GPU_SPECS = {
    "B200": (2250, 8000),
    "H200": (989, 4800),
    "H100 80GB HBM3": (989, 3350),  # SXM
    "H100 NVL": (835, 3900),
    "H100 PCIe": (756, 2000),
    "A100-SXM4-80GB": (312, 2039),
    "A100-SXM4-40GB": (312, 1555),
    "A100 80GB PCIe": (312, 1935),
    "A100-PCIE-40GB": (312, 1555),
    "RTX PRO 6000 Blackwell Server Edition": (500, 1597),
    "RTX PRO 6000 Blackwell Workstation Edition": (500, 1792),
    "RTX PRO 4500 Blackwell": (202, 896),
    "L40S": (362, 864),
    "A40": (150, 696),
    "A10": (125, 600),
    "L4": (121, 300),
    "T4": (65, 320),
}


def gpu_spec(device_name: str | None = None) -> tuple[float, float]:
    """(peak dense BF16 TFLOP/s, peak GB/s) of `device_name` (default: current CUDA device); longest name match wins,
    so "A100" never matches "A10" and "L40S" never matches "L4". KeyError for a GPU missing from GPU_SPECS."""
    device_name = device_name or torch.cuda.get_device_name()
    return GPU_SPECS[max((k for k in GPU_SPECS if k in device_name), key=len, default=device_name)]


def flops_profiling(name: str, flops_fn: Callable, every=1):
    """perf/<name>/flops = flops_fn(*args, **kwargs): analytic, 2 per multiply-add."""
    def measure(fn, args, kwargs):
        with _region(name):
            out = fn(*args, **kwargs)
        return out, {f"perf/{name}/flops": flops_fn(*args, **kwargs)}
    return _decorator(every, measure)


def bytes_profiling(name: str, bytes_fn: Callable, every=1):
    """perf/<name>/bytes = bytes_fn(*args, **kwargs): analytic memory traffic or resident size."""
    def measure(fn, args, kwargs):
        with _region(name):
            out = fn(*args, **kwargs)
        return out, {f"perf/{name}/bytes": bytes_fn(*args, **kwargs)}
    return _decorator(every, measure)


# progress: denominators for per-frame / per-token metrics and throughput
def count_frames(name: str, frames_fn: Callable, every=1):
    """progress/<name>/frames += frames_fn(output)."""
    def measure(fn, args, kwargs):
        out = fn(*args, **kwargs)
        return out, {f"progress/{name}/frames": frames_fn(out)}
    return _decorator(every, measure)


def count_tokens(name: str, tokens_fn: Callable, every=1):
    """progress/<name>/tokens += tokens_fn(output)."""
    def measure(fn, args, kwargs):
        out = fn(*args, **kwargs)
        return out, {f"progress/{name}/tokens": tokens_fn(out)}
    return _decorator(every, measure)


# policy: model internals
def _layer_modules(model, modules: str, layers: Layers):
    pattern = re.compile(re.escape(modules).replace(r"\*", r"(\d+)"))
    for path, module in model.named_modules():
        if (m := pattern.fullmatch(path)) and (layers == "all" or int(m.group(1)) in layers):
            yield int(m.group(1)), module


def _hooked(modules: str, layers: Layers, every: int, stat: Callable):
    """stat(layer, output) -> {key: value}; forward hooks on the selected layers of `self` for the call.
    A layer that runs several times in one call contributes the mean."""
    def measure(fn, args, kwargs):
        values = defaultdict(list)

        def hook(i):
            def record(module, inputs, output):
                for k, v in stat(i, output).items():
                    values[k].append(v)
            return record

        handles = [m.register_forward_hook(hook(i)) for i, m in _layer_modules(args[0], modules, layers)]
        try:
            out = fn(*args, **kwargs)
        finally:
            for h in handles:
                h.remove()
        return out, {k: sum(v) / len(v) for k, v in values.items()}
    return _decorator(every, measure)


def track_attention_scores(modules: str, q_groups: dict, k_groups: dict, layers: Layers = "all", every=1):
    """policy/attention/l<i>/<q>-><k>: attention mass from query group q to key group k ({name: slice} over Q / K),
    from the hooked module's attention probabilities output[1] [B, H, Q, K] (eager attention)."""
    def stat(i, output):
        p_BHQK = output[1].float()
        return {f"policy/attention/l{i}/{q}->{k}": p_BHQK[:, :, qs][..., ks].sum(-1).mean().item()
                for q, qs in q_groups.items() for k, ks in k_groups.items()}
    return _hooked(modules, layers, every, stat)


def track_attention_entropy(modules: str, layers: Layers = "all", every=1):
    """policy/attention/l<i>/entropy: attention entropy normalized by log(#unmasked keys), from output[1]."""
    def stat(i, output):
        p_BHQK = output[1].float()
        n_BHQ = (p_BHQK > 0).sum(-1).clamp(min=2)
        return {f"policy/attention/l{i}/entropy": (torch.special.entr(p_BHQK).sum(-1) / n_BHQ.log()).mean().item()}
    return _hooked(modules, layers, every, stat)


def track_hidden_states(modules: str, layers: Layers = "all", every=1):
    """policy/hidden/l<i>/norm: mean L2 norm over D of the hooked module's hidden state output (or output[0])."""
    def stat(i, output):
        h_BTD = output[0] if isinstance(output, tuple) else output
        return {f"policy/hidden/l{i}/norm": h_BTD.float().norm(dim=-1).mean().item()}
    return _hooked(modules, layers, every, stat)


def track_entropy(name: str, every=1):
    """policy/<name>/entropy: entropy of softmax(output) over the last dim (logits [..., V])."""
    def measure(fn, args, kwargs):
        out = fn(*args, **kwargs)
        logp = out.float().log_softmax(-1)
        return out, {f"policy/{name}/entropy": -(logp.exp() * logp).sum(-1).mean().item()}
    return _decorator(every, measure)


@contextmanager
def profile(trace_path=None):
    """torch.profiler over the block -> perf/<name>/region_ms for every *_profiling region that ran in it
    (GPU kernel time inside the region's annotation on CUDA, CPU time of the region otherwise)."""
    cuda = torch.cuda.is_available()
    _seen.clear()
    with torch.profiler.profile(activities=[ProfilerActivity.CPU] + [ProfilerActivity.CUDA] * cuda) as prof:
        yield
    if cuda:
        trace_path = trace_path or tempfile.mktemp(suffix=".json")
        prof.export_chrome_trace(str(trace_path))
        ms = gpu_region_ms(json.loads(Path(trace_path).read_text())["traceEvents"], _seen)
    else:
        if trace_path:
            prof.export_chrome_trace(str(trace_path))
        ms = dict.fromkeys(_seen, 0.0)
        for e in prof.events():
            if e.name in ms:
                ms[e.name] += e.cpu_time_total / 1e3
    for name, v in ms.items():
        meter.record(f"perf/{name}/region_ms", v)


def gpu_region_ms(trace_events, names) -> dict:
    """Kernel time inside each region's GPU-side label window (chrome-trace gpu_user_annotation events). CPU-side
    events' child-kernel totals double-count kernels in vLLM, so they are not used."""
    k = sorted((e["ts"], e["ts"] + e["dur"]) for e in trace_events if e.get("cat") == "kernel")
    starts = [a for a, _ in k]
    ms = dict.fromkeys(names, 0.0)
    for e in trace_events:
        if e.get("cat") == "gpu_user_annotation" and e["name"] in ms:
            a, b = e["ts"], e["ts"] + e["dur"]
            i = bisect.bisect_left(starts, a)
            while i < len(k) and k[i][0] < b:
                ms[e["name"]] += (min(k[i][1], b) - k[i][0]) / 1e3
                i += 1
    return ms
