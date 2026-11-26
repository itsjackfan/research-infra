"""Dims: B batch, H heads, T tokens, D hidden, V vocab, F frames, X height, Y width, C rgb."""
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from research_infra import monitoring as mon
from research_infra.monitoring import Logger, Meter


@pytest.fixture(autouse=True)
def meter(monkeypatch):
    m = Meter()
    monkeypatch.setattr(mon, "meter", m)
    return m


@pytest.fixture
def fake_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 2**30)
    monkeypatch.setattr(torch.cuda, "utilization", lambda: 87)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "NVIDIA L4")


class Attn(nn.Module):
    def forward(self, x_BTD):
        B, T, _ = x_BTD.shape
        return x_BTD, torch.full((B, 2, T, T), 1.0 / T)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = Attn()

    def forward(self, x_BTD):
        return 2 * self.attn(x_BTD)[0]


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(Block() for _ in range(3))

    def forward(self, x_BTD):
        for layer in self.layers:
            x_BTD = layer(x_BTD)
        return x_BTD


X_BTD = torch.ones(1, 4, 4)


# meter
def test_meter_split_mean_and_sum(meter):
    meter.record("perf/f/latency_ms", 2.0)
    meter.record("perf/f/latency_ms", 4.0)
    with meter.split("test"):
        meter.record("progress/f/frames", 3)
        meter.record("progress/f/frames", 5)
    assert meter.drain() == {"train/perf/f/latency_ms": 3.0, "test/progress/f/frames": 8}
    assert meter.drain() == {}


def test_meter_derived(meter):
    meter.record("perf/f/latency_ms", 1000.0)
    meter.record("perf/f/flops", 4e12)
    meter.record("perf/f/bytes", 2e9)
    meter.record("progress/f/frames", 10)
    out = meter.drain()
    assert out["train/perf/f/tflops"] == pytest.approx(4.0)
    assert out["train/perf/f/gbps"] == pytest.approx(2.0)
    assert out["train/perf/f/intensity"] == pytest.approx(2000.0)
    assert out["train/perf/f/frames_per_s"] == pytest.approx(10.0)
    assert out["train/perf/f/flops_per_frame"] == pytest.approx(4e11)
    assert out["train/perf/f/latency_ms_per_frame"] == pytest.approx(100.0)
    assert "train/perf/f/tokens_per_s" not in out
    assert "train/perf/f/memory_bound" not in out  # no CUDA, no spec


def test_meter_memory_bound(meter):
    meter.spec = (100.0, 1000.0)  # ridge point 100 FLOP/byte
    meter.record("perf/mm/flops", 4e3)
    meter.record("perf/mm/bytes", 2.0)
    meter.record("perf/copy/flops", 1.0)
    meter.record("perf/copy/bytes", 8.0)
    out = meter.drain()
    assert out["train/perf/mm/memory_bound"] is False and out["train/perf/copy/memory_bound"] is True


@pytest.mark.parametrize("name, spec", [
    ("NVIDIA RTX PRO 6000 Blackwell Server Edition", (500, 1597)), ("NVIDIA RTX PRO 4500 Blackwell", (202, 896)),
    ("NVIDIA A100-SXM4-80GB", (312, 2039)), ("NVIDIA H100 80GB HBM3", (989, 3350)), ("NVIDIA L40S", (362, 864)),
    ("NVIDIA L4", (121, 300)), ("Tesla T4", (65, 320)), ("NVIDIA A10", (125, 600)),
])
def test_gpu_spec(name, spec):
    assert mon.gpu_spec(name) == spec


def test_gpu_spec_unknown():
    with pytest.raises(KeyError):
        mon.gpu_spec("NVIDIA GeForce GTX 1080")


@pytest.mark.parametrize("key", ["train/loss", "test/perf/f/latency_ms", "test/libero/mean_score", "train/optim/lr"])
def test_check_key_accepts(key):
    mon.check_key(key)


@pytest.mark.parametrize("key", ["loss", "perf/f/latency_ms", "eval/loss", "optim/lr"])
def test_check_key_rejects(key):
    with pytest.raises(ValueError):
        mon.check_key(key)


# Logger
@pytest.fixture
def logger(cfg, wandb_mock):
    Path(cfg.run_dir).mkdir(parents=True)
    return Logger(cfg)


def test_logger_init_naming(cfg, wandb_mock, logger):
    kw = wandb_mock.init.call_args.kwargs
    assert kw["project"] == "proj" and kw["name"] == "probe/base" and kw["group"] == "probe"
    assert kw["id"] == "probe-base" and kw["resume"] == "allow" and kw["dir"] == cfg.run_dir
    assert kw["mode"] == "disabled" and kw["config"]["git"]["commit"] == "abc123"
    axes = {c.args[0]: c.kwargs.get("step_metric") for c in wandb_mock.define_metric.call_args_list}
    assert axes == {"step": None, "train/*": "step", "test/*": "step"}


def test_logger_rejects_bad_keys(logger):
    with pytest.raises(ValueError):
        logger.log({"loss": 1.0}, step=0)


def test_logger_values_and_jsonl(cfg, wandb_mock, logger, meter):
    video_FXYC = np.zeros((3, 8, 8, 3), np.uint8)
    chart = {"xs": [0, 1], "ys": {"a": [1.0, 2.0], "b": [3.0, 4.0]}}
    meter.record("perf/f/latency_ms", 5.0)
    logger.log({"train/loss": torch.tensor(1.0), "train/ok": True, "test/suite/video": video_FXYC,
                "test/policy/curve": chart}, step=7)
    logged = wandb_mock.log.call_args.args[0]
    assert "step" not in wandb_mock.log.call_args.kwargs
    assert logged["step"] == 7 and logged["train/loss"] == 1.0 and logged["train/ok"] == 1.0
    assert logged["train/perf/f/latency_ms"] == 5.0
    assert Path(wandb_mock.Video.call_args.args[0]).is_relative_to(cfg.run_dir)
    assert wandb_mock.plot.line_series.call_args.kwargs["keys"] == ["a", "b"]
    line = json.loads((Path(cfg.run_dir) / "metrics.jsonl").read_text().splitlines()[-1])
    assert line == {"step": 7, "train/loss": 1.0, "train/ok": 1.0, "train/perf/f/latency_ms": 5.0}


# perf decorators
def test_latency_profiling(meter):
    @mon.latency_profiling("f")
    def f(x):
        return x + 1

    assert f(1) == 2
    assert meter.drain()["train/perf/f/latency_ms"] > 0


def test_memory_and_gpu_util_need_cuda(meter, monkeypatch, fake_cuda):
    @mon.memory_profiling("f")
    @mon.gpu_util_profiling("f")
    def f():
        return 3

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert f() == 3 and meter.drain() == {}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    f()
    assert meter.drain() == {"train/perf/f/peak_mem_gb": 1.0, "train/perf/f/gpu_util": 87}


def test_mfu_profiling(meter, monkeypatch, fake_cuda):
    clock = iter([1.0, 3.0])
    monkeypatch.setattr(mon.time, "perf_counter", lambda: next(clock))

    @mon.mfu_profiling("mm", lambda n: n * 1e12, peak_tflops=4.0)
    def mm(n):
        return n

    mm(2)  # 2e12 FLOPs in 2 s = 1 TFLOP/s of a 4 TFLOP/s peak
    assert meter.drain() == pytest.approx({"train/perf/mm/mfu": 0.25})


def test_mfu_defaults_to_vendor_peak(meter, monkeypatch, fake_cuda):
    clock = iter([0.0, 1.0])
    monkeypatch.setattr(mon.time, "perf_counter", lambda: next(clock))
    mon.mfu_profiling("mm", lambda: 60.5e12)(lambda: None)()  # half of the L4's 121 TFLOP/s for 1 s
    assert meter.drain() == pytest.approx({"train/perf/mm/mfu": 0.5})


def test_flops_and_bytes_profiling(meter):
    @mon.flops_profiling("mm", lambda a_NK, b_KM: 2 * a_NK.shape[0] * a_NK.shape[1] * b_KM.shape[1])
    @mon.bytes_profiling("mm", lambda a_NK, b_KM: 4 * (a_NK.numel() + b_KM.numel()))
    def mm(a_NK, b_KM):
        return a_NK @ b_KM

    mm(torch.ones(2, 3), torch.ones(3, 4))
    out = meter.drain()
    assert out["train/perf/mm/flops"] == 48 and out["train/perf/mm/bytes"] == 72
    assert out["train/perf/mm/intensity"] == pytest.approx(48 / 72)


def test_count_frames_tokens_and_every(meter):
    @mon.count_frames("rollout", lambda out: len(out), every=2)
    def rollout():
        return [0] * 5

    @mon.count_tokens("gen", lambda out_BT: out_BT.numel())
    def gen():
        return torch.zeros(2, 3)

    for _ in range(3):
        rollout()
    gen()
    assert meter.drain() == {"train/progress/rollout/frames": 10, "train/progress/gen/tokens": 6}


def test_profile_regions(meter):
    @mon.latency_profiling("mm")
    def mm():
        return torch.randn(64, 64) @ torch.randn(64, 64)

    with mon.profile():
        mm()
    assert meter.drain()["train/perf/mm/region_ms"] > 0


# policy decorators
def test_track_attention_entropy_selected_layers(meter):
    class Probed(Toy):
        @mon.track_attention_entropy("layers.*.attn", layers=[0, 2])
        def forward(self, x_BTD):
            return super().forward(x_BTD)

    model = Probed()
    model(X_BTD)
    assert meter.drain() == pytest.approx({"train/policy/attention/l0/entropy": 1.0,
                                           "train/policy/attention/l2/entropy": 1.0})
    assert all(not m._forward_hooks for m in model.modules())


def test_track_attention_scores(meter):
    class Probed(Toy):
        @mon.track_attention_scores("layers.*.attn", {"act": slice(2, 4)}, {"img": slice(0, 2), "act": slice(2, 4)},
                                    layers=[1])
        def forward(self, x_BTD):
            return super().forward(x_BTD)

    Probed()(X_BTD)
    assert meter.drain() == pytest.approx({"train/policy/attention/l1/act->img": 0.5,
                                           "train/policy/attention/l1/act->act": 0.5})


def test_track_hidden_states_all_layers(meter):
    class Probed(Toy):
        @mon.track_hidden_states("layers.*")
        def forward(self, x_BTD):
            return super().forward(x_BTD)

    with meter.split("test"):
        Probed()(X_BTD)
    assert meter.drain() == pytest.approx({"test/policy/hidden/l0/norm": 4.0, "test/policy/hidden/l1/norm": 8.0,
                                           "test/policy/hidden/l2/norm": 16.0})


def test_track_entropy(meter):
    @mon.track_entropy("action")
    def logits():
        return torch.zeros(2, 4)

    logits()
    assert meter.drain() == pytest.approx({"train/policy/action/entropy": math.log(4)})
