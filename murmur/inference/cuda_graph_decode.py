"""Cuda-graph captured decode loop for VibeVoice ASR + FlashInferPagedCache.

Scope: decode phase only. Prefill still runs through the standard model
forward (variable shape, only happens once per batch). After prefill we
replay a pre-captured decode step for each token, eliminating per-step
Python + dispatcher overhead.

Two-phase usage
---------------
1. **At engine init (once)** — build a :class:`CudaGraphDecoder` with the
   max batch size and max sequence length the engine will ever process.
   The constructor allocates a :class:`FlashInferPagedCache` sized for
   those upper bounds, warms up the kernels, and captures one decode step
   into a ``torch.cuda.CUDAGraph``.

2. **Per batch (many)** — call :py:meth:`CudaGraphDecoder.generate` with
   the processor output. It resets cache state, runs prefill eagerly,
   then replays the captured graph per generated token.

Requirements
------------
  * flashinfer attention (``_attn_implementation == "flashinfer"``).
  * Greedy decode; no LogitsProcessor stack.
  * Batches must be ≤ the constructor's ``max_batch_size``; smaller batches
    are zero-padded to that size (extra rows' outputs are discarded).
  * Prefill + max_new_tokens must fit within ``max_seq_len``.
"""
from __future__ import annotations

from typing import Optional

import torch

from .flashinfer_paged import FlashInferPagedCache


class CudaGraphDecoder:
    """Owns the captured decode graph + persistent static buffers + cache."""

    def __init__(
        self,
        model,
        *,
        max_batch_size: int,
        max_seq_len: int,
        dtype: torch.dtype,
        device: torch.device | str,
        page_size: int = 16,
        warmup_steps: int = 2,
        decode_step_impl: str = "triton",
        cache_policy: str | None = None,
        sink_size: int = 0,
        window_size: int = 0,
        speech_window: int = 0,
        slide_delay: int = 0,
    ) -> None:
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.device = torch.device(device)

        dcfg = model.config.decoder_config
        head_dim = getattr(
            dcfg, "head_dim", dcfg.hidden_size // dcfg.num_attention_heads,
        )
        self.cache = FlashInferPagedCache(
            num_layers=dcfg.num_hidden_layers,
            num_qo_heads=dcfg.num_attention_heads,
            num_kv_heads=dcfg.num_key_value_heads,
            head_dim=head_dim,
            batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            dtype=dtype,
            device=self.device,
            page_size=page_size,
            enable_cuda_graph=True,
            decode_step_impl=decode_step_impl,
            cache_policy=cache_policy,
            sink_size=sink_size,
            window_size=window_size,
            speech_window=speech_window,
            slide_delay=slide_delay,
        )

        # Persistent decode-step input buffers — the captured kernels bake
        # in pointers to these, so all per-batch updates are in-place .copy_().
        self._static_input_ids = torch.zeros(
            (max_batch_size, 1), dtype=torch.long, device=self.device,
        )
        self._static_position_ids = torch.zeros(
            (max_batch_size, 1), dtype=torch.long, device=self.device,
        )
        # cache_position is unused on the flashinfer branch but Qwen2Model
        # accepts it; passing a static buffer avoids any internal allocation.
        self._static_cache_position = torch.zeros(
            (1,), dtype=torch.long, device=self.device,
        )

        self._capture(warmup_steps=warmup_steps)

    # ── one-time capture ─────────────────────────────────────────────────────

    def _capture(self, *, warmup_steps: int) -> None:
        """Capture the decode forward into a cuda graph.

        Uses a synthetic post-prefill state (artificial ``seq_lens`` value)
        rather than running a real prefill — capture only needs the kernel
        sequence + buffer pointers, not correct numerical output. Real
        prefill data gets written by per-batch ``generate()`` calls and is
        what the replayed kernels actually read from.
        """
        # Synthetic post-prefill state: all requests at the same fake length.
        # We pick something small (so warmup + capture finish quickly) but
        # within plan limits.
        fake_prefill_len = min(64, self.max_seq_len - self.max_batch_size - 16)
        if fake_prefill_len <= 0:
            raise ValueError(f"max_seq_len {self.max_seq_len} too small for capture")
        synthetic_lens = torch.full(
            (self.max_batch_size,), fake_prefill_len,
            dtype=torch.int32, device=self.device,
        )

        def _prime():
            """Put cache into a 'post-prefill' state without running prefill."""
            self.cache.reset()
            self.cache.seq_lens.copy_(synthetic_lens)
            self.cache._cumulative_length = fake_prefill_len
            self.cache._first_call = False

        def _decode_forward():
            return self.model(
                input_ids=self._static_input_ids,
                position_ids=self._static_position_ids,
                past_key_values=self.cache,
                cache_position=self._static_cache_position,
                use_cache=True,
                return_dict=True,
            )

        # Warmup on a side stream so cublas / flashinfer autotuners settle
        # before capture freezes the kernel selection.
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream), torch.no_grad():
            _prime()
            for _ in range(warmup_steps):
                self.cache.begin_decode_step()
                # static_input_ids stays zero; static_position_ids tracks the
                # new token's position so RoPE picks a well-defined freq.
                self._static_position_ids[:, 0].copy_(self.cache.seq_lens.long() - 1)
                _ = _decode_forward()
        torch.cuda.current_stream().wait_stream(warmup_stream)
        torch.cuda.synchronize()

        # Capture: re-prime, run begin_decode_step + write static inputs,
        # then record kernels.
        _prime()
        self.cache.begin_decode_step()
        self._static_position_ids[:, 0].copy_(self.cache.seq_lens.long() - 1)
        torch.cuda.synchronize()

        dev_index = self.device.index if self.device.index is not None else torch.cuda.current_device()
        free_before, total = torch.cuda.mem_get_info(dev_index)
        reserved_before = torch.cuda.memory_reserved(dev_index)

        self._graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(self._graph):
            captured = _decode_forward()
        self._static_logits = captured.logits  # [max_batch_size, 1, vocab]

        torch.cuda.synchronize()
        free_after, _ = torch.cuda.mem_get_info(dev_index)
        reserved_after = torch.cuda.memory_reserved(dev_index)
        graph_bytes = max(0, free_before - free_after)
        reserved_delta = reserved_after - reserved_before
        import logging
        logging.getLogger("cast").info(
            f"cuda graph captured: graph private-pool ≈ {graph_bytes/1024**2:.1f} MiB "
            f"(driver free-mem delta), reserved-pool delta {reserved_delta/1024**2:+.1f} MiB; "
            f"post-capture device usage {(total-free_after)/1024**3:.2f} / {total/1024**3:.2f} GiB "
            f"({(total-free_after)/total*100:.1f}%), free {free_after/1024**3:.2f} GiB"
        )

        # Leave cache in a clean state — per-batch generate() will reset
        # again, but this keeps the engine sane in between.
        self.cache.reset()

    # ── per-batch path ───────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        inputs: dict,
        *,
        max_new_tokens: int,
        eos_token_id: int,
        pad_token_id: int,
    ) -> torch.Tensor:
        """Run prefill eagerly + replay the captured decode graph per token.

        Returns ``output_ids`` of shape ``(B_actual, T_in + T_gen)``, where
        ``B_actual`` is the input batch (zero-padded internally up to
        ``max_batch_size`` for the captured graph; padding is stripped on
        return).
        """
        input_ids = inputs["input_ids"]
        B_actual, T_in = input_ids.shape
        device = input_ids.device

        if B_actual > self.max_batch_size:
            raise RuntimeError(
                f"actual batch {B_actual} > captured max_batch_size {self.max_batch_size}"
            )

        # Pad batch up to max_batch_size with zero-rows. The padded rows get
        # processed by the captured graph but we discard their outputs.
        if B_actual < self.max_batch_size:
            inputs = self._pad_batch_inputs(inputs, B_actual, T_in, device)
        B = self.max_batch_size

        attention_mask = inputs["attention_mask"]
        prefill_position_ids = attention_mask.long().cumsum(-1) - 1
        prefill_position_ids.masked_fill_(attention_mask == 0, 1)

        # Fresh cache state for this batch. The acoustic mask locates the audio
        # sub-range for the sliding_speech policy (so the surrounding prompt is
        # retained); it's a no-op for other policies.
        with torch.cuda.nvtx.range("cache_reset"):
            self.cache.reset()
            self.cache.set_prefill_lens(
                attention_mask.sum(-1), inputs.get("acoustic_input_mask"),
            )

        # Prefill — eager, variable shape.
        with torch.cuda.nvtx.range("prefill"):
            prefill_outputs = self.model(
                input_ids=inputs["input_ids"],
                position_ids=prefill_position_ids,
                past_key_values=self.cache,
                use_cache=True,
                return_dict=True,
                **{k: v for k, v in inputs.items()
                   if k not in ("input_ids", "attention_mask")},
            )
            next_tokens = prefill_outputs.logits[:, -1, :].argmax(dim=-1)  # [B]
            running_ids = torch.cat([inputs["input_ids"], next_tokens.unsqueeze(1)], dim=1)
            done = (next_tokens == eos_token_id)

        # Real-row done state — pad rows are forced "done" so we don't waste
        # the running_ids slot scanning them and so EOS-detection terminates
        # at the right time.
        if B_actual < B:
            done[B_actual:] = True

        if max_new_tokens <= 1 or bool(done.all()):
            return running_ids[:B_actual]

        # ── Decode replay loop ──────────────────────────────────────────────
        remaining = max_new_tokens - 1
        with torch.cuda.nvtx.range("decode_loop"):
            for _ in range(remaining):
                if bool(done.all()):
                    break
                with torch.cuda.nvtx.range("decode_step"):
                    with torch.cuda.nvtx.range("begin_decode_step"):
                        self.cache.begin_decode_step()
                    with torch.cuda.nvtx.range("static_inputs_copy"):
                        self._static_input_ids[:, 0].copy_(next_tokens)
                        self._static_position_ids[:, 0].copy_(self.cache.seq_lens.long() - 1)
                    with torch.cuda.nvtx.range("graph_replay"):
                        self._graph.replay()
                    with torch.cuda.nvtx.range("sample_next"):
                        next_tokens = self._static_logits[:, -1, :].argmax(dim=-1)
                        next_tokens = torch.where(
                            done, torch.full_like(next_tokens, pad_token_id), next_tokens,
                        )
                        done = done | (next_tokens == eos_token_id)
                        running_ids = torch.cat([running_ids, next_tokens.unsqueeze(1)], dim=1)

        return running_ids[:B_actual]

    # ── helpers ──────────────────────────────────────────────────────────────

    def _pad_batch_inputs(
        self, inputs: dict, B_actual: int, T_in: int, device: torch.device,
    ) -> dict:
        """Zero-pad every per-request input tensor up to ``max_batch_size``."""
        B = self.max_batch_size
        padded: dict = {}
        for k, v in inputs.items():
            if not isinstance(v, torch.Tensor):
                padded[k] = v
                continue
            if v.shape[0] == B_actual:
                # Per-request tensor — pad first dim.
                pad_shape = (B - B_actual,) + tuple(v.shape[1:])
                pad = torch.zeros(pad_shape, dtype=v.dtype, device=device)
                padded[k] = torch.cat([v, pad], dim=0)
            else:
                padded[k] = v
        return padded
