"""Paged-KV cache backed by flashinfer's prefill + decode batched wrappers.

Drop-in replacement for HF's ``DynamicCache`` on the VibeVoice-ASR
LM path when ``_attn_implementation == "flashinfer"``. Each request in
the batch gets a contiguous slice of pages from a per-layer paged pool
(no fragmentation / scheduler — keeps the bookkeeping minimal).

Per forward pass the cache:
  1. On layer 0's ``append_and_attend``, plans the correct wrapper
     (prefill if ``seq_len_new > 1`` else decode) using the post-step
     ``kv_indptr / kv_indices / kv_last_page_len`` buffers.
  2. For every layer: appends new K/V slices into that layer's paged
     buffer via ``flashinfer.append_paged_kv_cache`` and runs the
     planned wrapper.

Variable-length prefill: ``set_prefill_lens`` is called from
``LocalBatchEngine._generate`` with each request's real (unpadded)
prefill length. The cache then ignores left-padding inside flashinfer
— Q/K/V rows for pad columns are stripped before the kernel call, and
per-request ``qo_indptr`` / ``kv_indptr`` reflect real lengths. Without
this, padded K/V would be written into the paged pool and attended to
by real tokens (causal mask doesn't know about padding).

Sizing: callers pass ``batch_size`` and ``max_seq_len``. Total per-layer
memory is ``ceil(max_seq_len/page_size) * batch_size * 2 * page_size *
num_kv_heads * head_dim * dtype_bytes``.
"""
from __future__ import annotations

from typing import Any, Optional

import torch
from transformers import Cache

try:
    import flashinfer
    HAS_FLASHINFER = True
except ImportError:
    flashinfer = None  # type: ignore[assignment]
    HAS_FLASHINFER = False


class FlashInferPagedCache(Cache):
    """Paged-KV cache + attention runner using flashinfer wrappers.

    Unlike HF's ``DynamicCache``, this class owns *both* the K/V storage
    and the attention kernel — Qwen2Attention.forward calls
    :py:meth:`append_and_attend` instead of ``.update(...)`` + a
    separate ``attention_interface(...)``.
    """

    KV_LAYOUT = "NHD"  # paged_kv shape: [num_pages, 2, page_size, num_kv_heads, head_dim]

    def __init__(
        self,
        *,
        num_layers: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        batch_size: int,
        max_seq_len: int,
        dtype: torch.dtype,
        device: torch.device | str,
        page_size: int = 16,
        workspace_mb: int = 128,
        enable_cuda_graph: bool = False,
        decode_step_impl: str = "triton",
        cache_policy: str | None = None,
        sink_size: int = 0,
        window_size: int = 0,
        speech_window: int = 0,
        slide_delay: int = 0,
    ) -> None:
        super().__init__(layers=[])
        if not HAS_FLASHINFER:
            raise ImportError("flashinfer is not installed. pip install flashinfer-python")
        if decode_step_impl not in ("triton", "python"):
            raise ValueError(
                f"decode_step_impl must be 'triton' or 'python', got {decode_step_impl!r}"
            )
        if cache_policy not in (
            None, "streaming", "sliding_speech", "sliding_speech_streaming",
        ):
            raise ValueError(
                f"cache_policy for flashinfer path must be None, 'streaming', "
                f"'sliding_speech', or 'sliding_speech_streaming', got {cache_policy!r}"
            )
        if (
            cache_policy in ("streaming", "sliding_speech", "sliding_speech_streaming")
            and decode_step_impl != "triton"
        ):
            raise ValueError(
                f"cache_policy={cache_policy!r} requires decode_step_impl='triton' "
                f"on the flashinfer path"
            )

        self.num_layers = num_layers
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.batch_size = batch_size
        self.page_size = page_size
        self.dtype = dtype
        self.device = torch.device(device)
        self.enable_cuda_graph = enable_cuda_graph
        self.decode_step_impl = decode_step_impl
        # Streaming-style eviction parameters. Page-aligned (round up so we
        # keep at least the requested token counts).
        self.cache_policy = cache_policy
        self.sink_size = sink_size
        self.window_size = window_size
        self.sink_pages = (sink_size + page_size - 1) // page_size
        self.window_pages = (window_size + page_size - 1) // page_size
        # Sliding-speech-window eviction params. Window is page-aligned; the
        # slide offset is updated each begin_decode_step in Python and passed
        # to the Triton kernel as a regular int (begin_decode_step is outside
        # the cuda graph, so the value can change between captures/replays).
        self.speech_window = speech_window
        self.slide_delay = slide_delay
        self.speech_window_pages = (speech_window + page_size - 1) // page_size
        self._decode_step = 0

        # Per-request page allocation pool: each request reserves
        # ``max_pages_per_req`` contiguous physical pages. Only the prefix
        # holding actual K/V is exposed to flashinfer via ``kv_indices`` each
        # step — flashinfer treats every page in that list as fully populated
        # (except the last one, sized by ``kv_last_page_len``), so we must
        # grow the index list as the sequence advances.
        self.max_pages_per_req = (max_seq_len + page_size - 1) // page_size
        total_pages = batch_size * self.max_pages_per_req

        # Per-layer paged buffer (NHD: K and V interleaved on axis 1).
        self.paged_kv = [
            torch.empty(
                total_pages, 2, page_size, num_kv_heads, head_dim,
                dtype=dtype, device=self.device,
            )
            for _ in range(num_layers)
        ]

        dev = self.device
        # Step-scoped buffers (populated in _begin_step).
        self._step_kv_indices = torch.empty(0, dtype=torch.int32, device=dev)
        self._step_kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=dev)
        self.kv_last_page_len = torch.zeros(batch_size, dtype=torch.int32, device=dev)
        # Per-request real seq length. Heterogeneous across requests once
        # ``set_prefill_lens`` is called with non-uniform lengths.
        self.seq_lens = torch.zeros(batch_size, dtype=torch.int32, device=dev)
        # Per-request unpadded prefill length; populated by ``set_prefill_lens``.
        # If None at first call, falls back to uniform (= padded length),
        # which is correct for batch_size==1 or homogeneous batches.
        self._prefill_lens: Optional[torch.Tensor] = None
        # Logical position used by HF prepare_inputs_for_generation. This
        # tracks the *padded* length because HF slices ``input_ids[:, past:]``
        # against the padded 2D tensor.
        self._cumulative_length = 0
        # First-call flag distinguishes prefill from decode independent of
        # whether seq_len_new happens to be 1.
        self._first_call = True
        # Per-step state — set on layer 0's append_and_attend.
        self._is_prefill_step = False
        # Cached gather/index buffers (built in _begin_step, reused across layers).
        self._step_gather_rows: Optional[torch.Tensor] = None
        self._step_append_batch_indices: Optional[torch.Tensor] = None
        self._step_append_positions: Optional[torch.Tensor] = None
        # When True, ``append_and_attend`` skips its internal _begin_step on
        # layer 0 and uses the persistent decode buffers below. Set by
        # ``begin_decode_step()`` ahead of each captured/replayed step.
        self._step_prepared = False

        # ── Persistent decode buffers (cuda-graph mode only). ────────────────
        # All metadata that ``append_paged_kv_cache`` and ``decode_wrapper.run``
        # read from must live at a fixed address with values updated in-place,
        # otherwise the captured kernels' baked-in pointers would go stale.
        if enable_cuda_graph:
            max_total_pages = batch_size * self.max_pages_per_req
            self._graph_kv_indices = torch.zeros(
                max_total_pages, dtype=torch.int32, device=dev,
            )
            self._graph_kv_indptr = torch.zeros(
                batch_size + 1, dtype=torch.int32, device=dev,
            )
            self._graph_kv_last_page_len = torch.zeros(
                batch_size, dtype=torch.int32, device=dev,
            )
            # Decode appends exactly one token per request — batch indices are
            # constant [0, 1, ..., B-1]; positions update each step.
            self._graph_append_batch_indices = torch.arange(
                batch_size, dtype=torch.int32, device=dev,
            )
            self._graph_append_positions = torch.zeros(
                batch_size, dtype=torch.int32, device=dev,
            )
            # Scratch passed between the two Triton kernels in
            # ``begin_decode_step`` (pages_used per request).
            self._pages_used_scratch = torch.zeros(
                batch_size, dtype=torch.int32, device=dev,
            )
            # Physical pages allocated per request (set by the scalars kernel);
            # the fill kernel reads this to index into the window region.
            self._pages_total_scratch = torch.zeros(
                batch_size, dtype=torch.int32, device=dev,
            )
            # Per-request prefill page count, set by ``set_prefill_lens``.
            # Zeroed by default so the kernel's eviction math is well-defined
            # even before prefill happens (where it never triggers).
            self._prefill_pages_per_req = torch.zeros(
                batch_size, dtype=torch.int32, device=dev,
            )
            # floor(prefill_len / page_size): the physical page that holds the
            # FIRST decoded token. When prefill_len isn't page-aligned the last
            # prefill page is partial and the first decoded tokens continue
            # filling it, so the decoded region starts here (pf-1), not at the
            # ceil-rounded ``_prefill_pages_per_req`` (pf). Used by the
            # sliding-speech eviction path to anchor its decoded region.
            self._prefill_floor_pages_per_req = torch.zeros(
                batch_size, dtype=torch.int32, device=dev,
            )
            # Audio sub-range of the prefill, in pages (sliding-speech only).
            # The prefill is [prompt prefix][audio embeds][prompt suffix +
            # instruction]; the sliding window must move only over the audio
            # pages and ALWAYS retain the prefix (system prompt, <|speech_start|>)
            # and suffix (<|speech_end|>, format instruction, assistant turn),
            # otherwise the model loses its task framing and output schema.
            #   audio_start_page = ceil(audio_start / page_size)  -> first
            #       non-prefix page; pages [0, audio_start_page) are kept as the
            #       prefix.
            #   audio_end_page   = floor(audio_end / page_size)   -> first
            #       suffix page; pages [audio_end_page, prefill_floor) are kept
            #       as the suffix.
            # Both default 0 so a missing acoustic mask degrades to "no
            # eviction" rather than dropping prompt tokens.
            self._audio_start_page_per_req = torch.zeros(
                batch_size, dtype=torch.int32, device=dev,
            )
            self._audio_end_page_per_req = torch.zeros(
                batch_size, dtype=torch.int32, device=dev,
            )

        # Shared workspace + wrappers.
        self._workspace = torch.empty(
            workspace_mb * 1024 * 1024, dtype=torch.uint8, device=dev,
        )
        # Prefill: backend='fa3' is FlashAttention-3 (Hopper+). It has
        # noticeably tighter bf16 precision than the default 'auto' (which
        # picks fa2): in side-by-side tests against SDPA the per-attention max
        # diff drops ~4x, which materially reduces end-to-end WER drift.
        self.prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            self._workspace, self.KV_LAYOUT, backend="fa3",
        )
        # Decode: leave on default backend. fa3 decode crashes with a TMA
        # descriptor init error on small batches in this flashinfer build, and
        # decode is a single output row per request so the precision win of
        # fa3 (which matters most on long-sequence accumulation) is small.
        # use_tensor_cores=True routes through prefill-style tensor-core
        # kernels which support arbitrary group sizes (num_qo_heads /
        # num_kv_heads). The default fast-path decode kernels only support a
        # small enumerated set (1, 4, 8, ...), and Qwen2.5-7B is 28/4 = 7.
        if enable_cuda_graph:
            self.decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                self._workspace, self.KV_LAYOUT,
                use_cuda_graph=True,
                use_tensor_cores=True,
                paged_kv_indptr_buffer=self._graph_kv_indptr,
                paged_kv_indices_buffer=self._graph_kv_indices,
                paged_kv_last_page_len_buffer=self._graph_kv_last_page_len,
            )
        else:
            self.decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                self._workspace, self.KV_LAYOUT, use_tensor_cores=True,
            )

    # ── public hooks ─────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear per-batch state without touching persistent buffers.

        Page contents and the graph metadata buffers (allocations + wrapper
        registration) are reused — they get overwritten by the next batch's
        prefill + decode without any consequence to correctness, so we just
        zero out the bookkeeping tensors and the prefill-len registration.
        """
        self.seq_lens.zero_()
        self._prefill_lens = None
        self._cumulative_length = 0
        self._first_call = True
        self._is_prefill_step = False
        self._step_prepared = False
        self._step_gather_rows = None
        self._step_append_batch_indices = None
        self._step_append_positions = None
        self._decode_step = 0

    def set_prefill_lens(
        self,
        prefill_lens: torch.Tensor,
        acoustic_input_mask: Optional[torch.Tensor] = None,
    ) -> None:
        """Register per-request unpadded prefill lengths.

        Must be called before ``model.generate(...)`` when the input batch
        is left-padded with heterogeneous real lengths. Without it the
        cache assumes every request fills the padded sequence — pad K/V
        end up in the paged pool and corrupt attention.

        ``prefill_lens`` is typically ``inputs["attention_mask"].sum(-1)``.

        ``acoustic_input_mask`` (``inputs["acoustic_input_mask"]``, shape
        ``[B, S_padded]``, 1 on audio-embedding rows) is required by the
        ``sliding_speech`` policy: it locates the audio sub-range so the
        window slides only over audio pages and the surrounding prompt
        (system prefix + format-instruction suffix) is always retained.
        """
        if prefill_lens.shape != (self.batch_size,):
            raise ValueError(
                f"prefill_lens must have shape ({self.batch_size},), got {tuple(prefill_lens.shape)}"
            )
        self._prefill_lens = prefill_lens.to(device=self.device, dtype=torch.int32)
        # Mirror prefill_lens in pages for the eviction kernel. Done here
        # (not at decode time) so it survives across decode steps with no
        # per-step recomputation.
        if self.enable_cuda_graph:
            pf_pages = (self._prefill_lens + self.page_size - 1) // self.page_size
            max_pf_pages = int(pf_pages.max().item())
            if max_pf_pages > self.max_pages_per_req:
                raise RuntimeError(
                    f"prefill needs {max_pf_pages} pages per request but cache "
                    f"max_pages_per_req={self.max_pages_per_req}. Increase "
                    f"gpu_memory_util or cuda_graph_max_seq_len."
                )
            self._prefill_pages_per_req.copy_(pf_pages)
            self._prefill_floor_pages_per_req.copy_(
                self._prefill_lens // self.page_size
            )
            self._set_audio_spans(acoustic_input_mask)

    def _set_audio_spans(self, acoustic_input_mask: Optional[torch.Tensor]) -> None:
        """Derive per-request audio page spans from the acoustic mask.

        The mask is in the left-padded ``[B, S_padded]`` grid; real tokens
        for request i occupy columns ``[S_padded - L_i, S_padded)`` and the
        cache stores them stripped at physical positions ``[0, L_i)``. So a
        masked column ``c`` maps to physical position ``c - (S_padded - L_i)``.

        Stores:
          ``_audio_start_page_per_req = ceil(audio_start / page_size)``
          ``_audio_end_page_per_req   = floor(audio_end   / page_size)``
        Requests with no audio rows (or a missing mask) get spans of 0, which
        the kernel reads as "no audio window" → keeps the full prefill.
        """
        P = self.page_size
        if acoustic_input_mask is None:
            self._audio_start_page_per_req.zero_()
            self._audio_end_page_per_req.zero_()
            return

        mask = acoustic_input_mask.to(device=self.device)
        if mask.shape != (self.batch_size, mask.shape[-1]):
            raise ValueError(
                f"acoustic_input_mask must be [B={self.batch_size}, S], got "
                f"{tuple(mask.shape)}"
            )
        S = mask.shape[-1]
        cols = torch.arange(S, device=self.device, dtype=torch.int32)
        has_audio = mask.any(dim=-1)
        # first/last masked column per request (sentinels make empty rows -> 0 span).
        first = torch.where(mask.bool(), cols, torch.full_like(cols, S)).amin(dim=-1)
        last = torch.where(mask.bool(), cols, torch.full_like(cols, -1)).amax(dim=-1)
        real_start = (S - self._prefill_lens).to(torch.int32)  # left-pad width
        audio_start = (first.to(torch.int32) - real_start).clamp_min(0)
        audio_end = (last.to(torch.int32) + 1 - real_start).clamp_min(0)
        start_pg = (audio_start + P - 1) // P     # ceil -> first non-prefix page
        end_pg = audio_end // P                   # floor -> first suffix page
        start_pg = torch.where(has_audio, start_pg, torch.zeros_like(start_pg))
        end_pg = torch.where(has_audio, end_pg, torch.zeros_like(end_pg))
        self._audio_start_page_per_req.copy_(start_pg.to(torch.int32))
        self._audio_end_page_per_req.copy_(end_pg.to(torch.int32))

    def begin_decode_step(self) -> None:
        """Prepare metadata for one decode step. Call OUTSIDE the cuda graph.

        Splits the work that's normally done inside ``_begin_step`` so the
        captured graph contains only graph-safe GPU work (append_paged_kv_cache
        + decode_wrapper.run). The host-sync + wrapper.plan() lives here and
        runs between replays.

        Updates are in-place into the ``_graph_*`` buffers; the captured
        kernels' pointers stay valid because we never reassign the buffer
        references.
        """
        if not self.enable_cuda_graph:
            raise RuntimeError(
                "begin_decode_step requires enable_cuda_graph=True on the cache."
            )
        if self._first_call:
            raise RuntimeError(
                "begin_decode_step called before prefill; run a prefill forward first."
            )

        with torch.cuda.nvtx.range("page_bookkeeping"):
            self._cumulative_length += 1
            self._decode_step += 1
            self._is_prefill_step = False
            # Sliding-speech window offset (in pages). Computed here in
            # Python because begin_decode_step is OUTSIDE the cuda graph;
            # the value differs per decode step and is passed to the
            # Triton kernel as a regular int.
            if self.cache_policy in ("sliding_speech", "sliding_speech_streaming"):
                raw_offset = max(0, self._decode_step - self.slide_delay) // self.page_size
                # Cap globally so the kernel's per-request min() can't pick
                # something past the prefill region. Per-request cap happens
                # inside the kernel.
                slide_offset_pages = min(
                    raw_offset, self.max_pages_per_req - self.speech_window_pages,
                )
            else:
                slide_offset_pages = 0
            if self.decode_step_impl == "triton":
                # Fused: in-place seq_lens += 1, computes pages_used, last_page_len,
                # append_positions, kv_indptr (exclusive cumsum), and fills the
                # ragged kv_indices buffer — all in two Triton kernel launches with
                # no host sync. Overflow is not checked here because max_pages_per_req
                # was sized at allocation time (see local_engine._derive_cache_max_seq_len).
                from murmur.inference.flashinfer_decode_step_kernel import (
                    triton_begin_decode_step,
                )
                triton_begin_decode_step(
                    self.seq_lens,
                    self._graph_kv_last_page_len,
                    self._graph_append_positions,
                    self._graph_kv_indices,
                    self._graph_kv_indptr,
                    self._pages_used_scratch,
                    self._pages_total_scratch,
                    self._prefill_pages_per_req,
                    self._prefill_floor_pages_per_req,
                    self._audio_start_page_per_req,
                    self._audio_end_page_per_req,
                    batch_size=self.batch_size,
                    max_pages_per_req=self.max_pages_per_req,
                    page_size=self.page_size,
                    sink_pages=self.sink_pages,
                    window_pages=self.window_pages,
                    evict=self.cache_policy in ("streaming", "sliding_speech_streaming"),
                    speech_evict=self.cache_policy in ("sliding_speech", "sliding_speech_streaming"),
                    speech_window_pages=self.speech_window_pages,
                    slide_offset_pages=slide_offset_pages,
                )
            else:
                # Reference Python implementation. Slower (~48 launches +
                # 2 host syncs per step) but kept for A/B testing.
                self.seq_lens += 1
                pages_used_per_req = (self.seq_lens + self.page_size - 1) // self.page_size
                last_page_len_per_req = ((self.seq_lens - 1) % self.page_size) + 1
                if (pages_used_per_req > self.max_pages_per_req).any().item():
                    raise RuntimeError(
                        f"FlashInferPagedCache overflow: a request needs more than "
                        f"{self.max_pages_per_req} pages."
                    )
                self._graph_kv_last_page_len.copy_(last_page_len_per_req)
                self._graph_append_positions.copy_(self.seq_lens - 1)
                with torch.cuda.nvtx.range("pages_used_d2h"):
                    pages_used_cpu = pages_used_per_req.cpu().tolist()
                offset = 0
                for i, n in enumerate(pages_used_cpu):
                    self._graph_kv_indices[offset : offset + n].copy_(
                        torch.arange(n, dtype=torch.int32, device=self.device)
                        + i * self.max_pages_per_req
                    )
                    offset += n
                self._graph_kv_indptr[1:].copy_(
                    pages_used_per_req.cumsum(0).to(torch.int32)
                )

        # Re-plan with the wrapper's pre-registered (persistent) buffers.
        with torch.cuda.nvtx.range("decode_wrapper.plan"):
            self.decode_wrapper.plan(
                self._graph_kv_indptr,
                self._graph_kv_indices,
                self._graph_kv_last_page_len,
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                self.page_size,
                q_data_type=self.dtype,
                kv_data_type=self.dtype,
            )

        # Signal append_and_attend to skip its internal _begin_step and use
        # the graph buffers instead.
        self._step_prepared = True

    # ── core path ────────────────────────────────────────────────────────────

    def _begin_step(self, seq_len_new: int) -> None:
        """Plan the right wrapper for this forward pass. Called from layer 0.

        On prefill, slots in per-request real lengths from ``_prefill_lens``
        (or uniform if unset) and precomputes:
          * ``_step_gather_rows`` — flat row indices to strip pad rows from
            [B, S_padded, ...] tensors.
          * ``_step_append_batch_indices`` / ``_step_append_positions`` —
            per-real-token request id + position-in-request used by
            ``append_paged_kv_cache``.

        On decode, every request advances by exactly ``seq_len_new`` (== 1
        for HF greedy) and the per-request append position is ``seq_lens-1``.
        """
        if seq_len_new <= 0:
            raise ValueError(f"seq_len_new must be > 0, got {seq_len_new}")
        dev = self.device

        if self._first_call:
            if self._prefill_lens is not None:
                per_req_lens = self._prefill_lens
            else:
                per_req_lens = torch.full(
                    (self.batch_size,), seq_len_new, dtype=torch.int32, device=dev,
                )
            if (per_req_lens > seq_len_new).any().item():
                raise ValueError(
                    f"prefill_lens has entries > padded seq_len_new ({seq_len_new})"
                )
            self.seq_lens.copy_(per_req_lens)
            self._is_prefill_step = True
            self._first_call = False
        else:
            self.seq_lens += seq_len_new
            self._is_prefill_step = seq_len_new > 1

        # Per-request page bookkeeping. One host-sync per step is fine
        # because per-request lengths can differ.
        pages_used_per_req = (self.seq_lens + self.page_size - 1) // self.page_size
        last_page_len_per_req = ((self.seq_lens - 1) % self.page_size) + 1
        if (pages_used_per_req > self.max_pages_per_req).any().item():
            raise RuntimeError(
                f"FlashInferPagedCache overflow: a request needs more than "
                f"{self.max_pages_per_req} pages."
            )
        self.kv_last_page_len.copy_(last_page_len_per_req)

        pages_used_cpu = pages_used_per_req.cpu().tolist()
        idx_parts = [
            torch.arange(n, dtype=torch.int32, device=dev) + i * self.max_pages_per_req
            for i, n in enumerate(pages_used_cpu)
        ]
        self._step_kv_indices = (
            torch.cat(idx_parts) if idx_parts
            else torch.empty(0, dtype=torch.int32, device=dev)
        )
        self._step_kv_indptr = torch.zeros(self.batch_size + 1, dtype=torch.int32, device=dev)
        self._step_kv_indptr[1:] = pages_used_per_req.cumsum(0).to(torch.int32)

        if self._is_prefill_step:
            seq_lens_cpu = self.seq_lens.cpu().tolist()

            qo_indptr = torch.zeros(self.batch_size + 1, dtype=torch.int32, device=dev)
            qo_indptr[1:] = self.seq_lens.cumsum(0).to(torch.int32)

            # Strip left-padding: request i's real tokens occupy columns
            # [seq_len_new - L_i .. seq_len_new - 1] in the padded [B, S_padded] grid.
            row_parts = []
            batch_idx_parts = []
            pos_parts = []
            for i, L in enumerate(seq_lens_cpu):
                start = seq_len_new - L
                row_parts.append(
                    torch.arange(start, seq_len_new, dtype=torch.long, device=dev)
                    + i * seq_len_new
                )
                batch_idx_parts.append(torch.full((L,), i, dtype=torch.int32, device=dev))
                pos_parts.append(torch.arange(L, dtype=torch.int32, device=dev))
            self._step_gather_rows = torch.cat(row_parts)
            self._step_append_batch_indices = torch.cat(batch_idx_parts)
            self._step_append_positions = torch.cat(pos_parts)

            self.prefill_wrapper.plan(
                qo_indptr,
                self._step_kv_indptr,
                self._step_kv_indices,
                self.kv_last_page_len,
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                self.page_size,
                causal=True,
                q_data_type=self.dtype,
                kv_data_type=self.dtype,
            )
        else:
            # Decode: 1 token per request. Append slot is the just-vacated
            # ``seq_lens-1`` position (we already incremented seq_lens above).
            self._step_append_batch_indices = torch.arange(
                self.batch_size, dtype=torch.int32, device=dev,
            )
            self._step_append_positions = (self.seq_lens - 1).to(torch.int32)
            self._step_gather_rows = None

            self.decode_wrapper.plan(
                self._step_kv_indptr,
                self._step_kv_indices,
                self.kv_last_page_len,
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                self.page_size,
                q_data_type=self.dtype,
                kv_data_type=self.dtype,
            )

    def append_and_attend(
        self,
        layer_idx: int,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Write K/V into the paged pool then run the planned wrapper.

        Args:
            layer_idx: decoder layer index.
            query: [batch, num_qo_heads, seq_padded, head_dim].
            key:   [batch, num_kv_heads, seq_padded, head_dim].
            value: [batch, num_kv_heads, seq_padded, head_dim].

        Returns:
            [batch, seq_padded, num_qo_heads, head_dim] — channels-last so the
            existing ``attn_output.reshape(*input_shape, -1)`` in
            Qwen2Attention.forward collapses heads*head_dim → hidden_size.
        """
        batch_size, num_qo_heads, seq_len_padded, head_dim = query.shape
        _, num_kv_heads, _, _ = key.shape

        if batch_size != self.batch_size:
            raise ValueError(
                f"FlashInferPagedCache was built for batch={self.batch_size} "
                f"but got batch={batch_size}"
            )

        if layer_idx == 0:
            if self._step_prepared:
                # Metadata already set up by ``begin_decode_step`` outside the
                # cuda graph. Don't touch CPU-side state here; the captured
                # kernels read the graph buffers in-place.
                pass
            else:
                self._begin_step(seq_len_padded)
                self._cumulative_length += seq_len_padded

        # Select metadata buffers: the persistent graph set when a step was
        # pre-prepared, otherwise the per-step transient set.
        if self._step_prepared:
            kv_indices = self._graph_kv_indices
            kv_indptr = self._graph_kv_indptr
            last_page_len = self._graph_kv_last_page_len
            append_batch_indices = self._graph_append_batch_indices
            append_positions = self._graph_append_positions
        else:
            kv_indices = self._step_kv_indices
            kv_indptr = self._step_kv_indptr
            last_page_len = self.kv_last_page_len
            append_batch_indices = self._step_append_batch_indices
            append_positions = self._step_append_positions

        # Flatten K/V/Q to [B*S_padded, heads, head_dim].
        k_flat = key.transpose(1, 2).reshape(-1, num_kv_heads, head_dim)
        v_flat = value.transpose(1, 2).reshape(-1, num_kv_heads, head_dim)
        q_flat = query.transpose(1, 2).reshape(-1, num_qo_heads, head_dim)

        if self._is_prefill_step:
            # Strip pad rows: keep only the real-token rows per request.
            gather = self._step_gather_rows  # [total_real_tokens]
            k_flat = k_flat.index_select(0, gather)
            v_flat = v_flat.index_select(0, gather)
            q_flat = q_flat.index_select(0, gather)

        flashinfer.append_paged_kv_cache(
            k_flat, v_flat,
            append_batch_indices, append_positions,
            self.paged_kv[layer_idx],
            kv_indices, kv_indptr, last_page_len,
            kv_layout=self.KV_LAYOUT,
        )

        if self._is_prefill_step:
            out_flat = self.prefill_wrapper.run(q_flat, self.paged_kv[layer_idx])
            # Scatter real-row outputs back into a [B, S_padded, H, D] tensor.
            # Pad rows stay zero — HF generate reads only the last column
            # (which is real for all left-padded rows) for next-token
            # selection, so pad rows never feed into a decision.
            out = torch.zeros(
                batch_size * seq_len_padded, num_qo_heads, head_dim,
                dtype=out_flat.dtype, device=self.device,
            )
            out.index_copy_(0, self._step_gather_rows, out_flat)
            out = out.view(batch_size, seq_len_padded, num_qo_heads, head_dim)
        else:
            q_flat = q_flat.view(batch_size, num_qo_heads, head_dim)
            out_flat = self.decode_wrapper.run(q_flat, self.paged_kv[layer_idx])
            out = out_flat.unsqueeze(1)  # [B, 1, num_qo_heads, head_dim]

        if layer_idx == self.num_layers - 1 and self._step_prepared:
            # Reset only happens at capture time (not on replay — Python
            # doesn't re-run), but that's fine: the flag's purpose is to
            # select graph-vs-transient buffers at capture time, baking the
            # right pointers into the captured kernels. Replays bypass this.
            self._step_prepared = False

        return out

    # ── HF Cache API ─────────────────────────────────────────────────────────

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "FlashInferPagedCache is driven via append_and_attend(); "
            "Qwen2Attention.forward must route past_key_values directly to that path."
        )

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._cumulative_length

    def get_max_cache_shape(self) -> int:
        return -1
