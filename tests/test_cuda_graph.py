"""CPU-only tests for the --cuda_graphs wiring (config resolution + plumbing).

These don't touch flashinfer, triton, or a GPU — they cover the pure-Python
argument resolution and signature plumbing that route a run onto the captured
flashinfer paged-KV decode path.
"""
import inspect

from benchmark import resolve_cuda_graph_config
from murmur.inference.local_engine import LocalBatchEngine


# ── --cuda_graphs default resolution ──────────────────────────────────────────

def test_cuda_graph_defaults_speech_eviction():
    # No explicit overrides: tuned cuda-graph defaults apply
    # (speech_window=1536, slide_delay=512), policy=sliding_speech, flashinfer.
    attn, policy, kwargs = resolve_cuda_graph_config(
        speech_window=512,   # the eager default; should be ignored
        slide_delay=32,      # the eager default; should be ignored
        speech_window_explicit=False,
        slide_delay_explicit=False,
    )
    assert attn == "flashinfer"
    assert policy == "sliding_speech"
    assert kwargs["speech_window"] == 1536
    assert kwargs["slide_delay"] == 512
    assert kwargs["page_size"] == 16


def test_cuda_graph_respects_explicit_overrides():
    attn, policy, kwargs = resolve_cuda_graph_config(
        speech_window=2048,
        slide_delay=128,
        speech_window_explicit=True,
        slide_delay_explicit=True,
    )
    assert attn == "flashinfer"
    assert policy == "sliding_speech"
    assert kwargs["speech_window"] == 2048
    assert kwargs["slide_delay"] == 128


def test_cuda_graph_partial_override():
    # Only --speech_window passed: window honored, delay falls back to 512.
    _, _, kwargs = resolve_cuda_graph_config(
        speech_window=1024,
        slide_delay=32,
        speech_window_explicit=True,
        slide_delay_explicit=False,
    )
    assert kwargs["speech_window"] == 1024
    assert kwargs["slide_delay"] == 512


# ── engine signature plumbing ─────────────────────────────────────────────────

def test_engine_accepts_cuda_graph_kwargs():
    for fn in (LocalBatchEngine.__init__, LocalBatchEngine.from_pretrained):
        params = inspect.signature(fn).parameters
        for name in (
            "use_cuda_graph",
            "cuda_graph_max_seq_len",
            "gpu_memory_util",
            "decode_step_impl",
        ):
            assert name in params, f"{fn.__qualname__} missing {name!r}"


def test_engine_cuda_graph_defaults_off():
    # The flags default to the eager (non-cuda-graph) path so existing callers
    # are unaffected.
    params = inspect.signature(LocalBatchEngine.__init__).parameters
    assert params["use_cuda_graph"].default is False
    assert params["decode_step_impl"].default == "triton"
    assert params["gpu_memory_util"].default == 0.9
