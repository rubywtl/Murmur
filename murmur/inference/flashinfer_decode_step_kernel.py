"""Fused per-step bookkeeping for FlashInferPagedCache decode.

Replaces the Python loop in ``FlashInferPagedCache.begin_decode_step`` that
does ~48 small launches + 2 host syncs per decode step with two Triton
kernels, no host sync. See [[flashinfer_paged]] for what the buffers mean.

Inputs (all on device, int32):
  seq_lens                : [B]      -- per-request sequence length (in-place +1)
  prefill_pages           : [B]      -- ceil(prefill_len/P) pages (eviction only)
  prefill_floor_pages     : [B]      -- floor(prefill_len/P); page the first
                                       decoded token shares (sliding-speech only)

Outputs (all on device, int32):
  last_page_len           : [B]      -- tokens used in the request's last page
  append_positions        : [B]      -- where to write the new K/V (= seq_lens - 1)
  pages_used_scratch      : [B]      -- *active* pages exposed to flashinfer
  pages_total_scratch     : [B]      -- physical pages allocated per request
                                       (only used by the eviction path to
                                       index into the window region)
  kv_indptr               : [B+1]    -- exclusive cumsum of pages_used; [0] preset to 0 at init
  kv_indices              : [B*M]    -- compressed page table; only [:kv_indptr[B]] is valid

The compressed page table layout matches flashinfer's BatchDecode wrapper:
for request i, ``kv_indices[kv_indptr[i] : kv_indptr[i+1]]`` is the list of
physical page IDs. Each request reserves a contiguous slice of ``M``
physical pages starting at ``i*M``, so page j of request i has physical
ID ``i*M + j``.

Eviction (streaming sink + window) when enabled:
  - All prefill pages stay in kv_indices (positions 0..prefill_pages[i]).
  - First ``sink_pages`` post-prefill pages stay.
  - Last ``window_pages`` physical pages stay (the most-recent decode pages).
  - The middle pages are dropped from kv_indices -- their physical K/V
    storage is untouched (RoPE positions baked in), just not read.
  - Triggers only once the post-prefill region exceeds ``sink + window``.
  - max_pages_per_req still has to fit ``prefill + max_new_tokens`` because
    new tokens still append to physically growing page positions.

Combined eviction (``speech_evict`` + ``evict`` together) when enabled:
  - Prefill side: slides a ``speech_window`` window over the audio sub-range
    (prefix + window + suffix, exactly like the sliding-speech policy), so
    audio-prefill pages get evicted.
  - Decoded side: applies sink + window over the decoded region, so the
    middle of a long decode also gets evicted.
  - The two triggers are independent per request (audio span > speech window;
    decoded span > sink + window), giving four cases per request. The kept
    page layout per request is
        [ prefix ][ audio window ][ suffix ][ decode sink ][ decode window ]
    where the prefill side anchors the decoded side at floor(prefill_len/P).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _scalars_and_indptr_kernel(
    seq_lens_ptr,
    last_page_len_ptr,
    append_positions_ptr,
    pages_used_ptr,         # OUT: active pages (post-eviction)
    pages_total_ptr,        # OUT: physical pages allocated
    kv_indptr_ptr,
    prefill_pages_ptr,
    prefill_floor_pages_ptr,
    audio_start_page_ptr,
    audio_end_page_ptr,
    B: tl.constexpr,
    P: tl.constexpr,
    SINK_PAGES: tl.constexpr,
    WINDOW_PAGES: tl.constexpr,
    EVICT: tl.constexpr,
    SPEECH_EVICT: tl.constexpr,
    COMBINED: tl.constexpr,
    SPEECH_WINDOW_PAGES: tl.constexpr,
    B_BLOCK: tl.constexpr,
):
    """Single block, B_BLOCK lanes (≥ B, power of 2). Updates per-request
    scalars and writes the exclusive cumsum into ``kv_indptr[1:B+1]``.
    ``kv_indptr[0]`` stays 0 (initialised once at allocation time)."""
    offs = tl.arange(0, B_BLOCK)
    mask = offs < B

    sl = tl.load(seq_lens_ptr + offs, mask=mask, other=0) + 1
    tl.store(seq_lens_ptr + offs, sl, mask=mask)

    pages_total = (sl + P - 1) // P
    last_pl = ((sl - 1) % P) + 1

    if COMBINED:
        # Sliding-speech window over the audio prefill AND sink+window over the
        # decoded region. The two triggers fire independently per request; the
        # kept layout is [prefix][audio window][suffix][decode sink][decode
        # window], with the decoded side anchored on floor(prefill_len/P).
        pf_floor = tl.load(prefill_floor_pages_ptr + offs, mask=mask, other=0)
        as_pg = tl.load(audio_start_page_ptr + offs, mask=mask, other=0)
        ae_pg = tl.load(audio_end_page_ptr + offs, mask=mask, other=0)
        audio_pages = ae_pg - as_pg
        decoded_pages = pages_total - pf_floor
        sp = audio_pages > SPEECH_WINDOW_PAGES
        st = decoded_pages > (SINK_PAGES + WINDOW_PAGES)
        # Prefill side: prefix + window + suffix when sliding audio, else the
        # full prefill (identity up to pf_floor).
        pre_active = tl.where(
            sp, as_pg + SPEECH_WINDOW_PAGES + (pf_floor - ae_pg), pf_floor
        )
        # Decoded side: sink + window when the decode is long enough, else all.
        dec_active = tl.where(st, SINK_PAGES + WINDOW_PAGES, decoded_pages)
        active = pre_active + dec_active
    elif EVICT:
        pf = tl.load(prefill_pages_ptr + offs, mask=mask, other=0)
        post = pages_total - pf
        evict_active = pf + SINK_PAGES + WINDOW_PAGES
        active = tl.where(post > SINK_PAGES + WINDOW_PAGES, evict_active, pages_total)
    elif SPEECH_EVICT:
        # Sliding window over the AUDIO sub-range of the prefill, keeping the
        # surrounding prompt at both ends. The kept page layout per request is
        #   [ prefix ][ audio window ][ suffix ][ decoded ]
        # where
        #   prefix  = pages [0, as_pg)              (system prompt, speech_start)
        #   window  = SPEECH_WINDOW_PAGES audio pages, sliding within audio
        #   suffix  = pages [ae_pg, pf_floor)       (speech_end, instruction)
        #   decoded = pages [pf_floor, pages_total) (all decoded tokens)
        #
        # as_pg = ceil(audio_start/P), ae_pg = floor(audio_end/P). The decoded
        # region anchors on pf_floor = floor(prefill_len/P) (the partial last
        # prefill page the first decoded tokens share). Eviction only fires
        # when the audio spans more than the window; otherwise the full prefill
        # is kept (identity), which also covers the no-audio-mask fallback
        # (as_pg == ae_pg == 0).
        pf_floor = tl.load(prefill_floor_pages_ptr + offs, mask=mask, other=0)
        as_pg = tl.load(audio_start_page_ptr + offs, mask=mask, other=0)
        ae_pg = tl.load(audio_end_page_ptr + offs, mask=mask, other=0)
        audio_pages = ae_pg - as_pg
        # prefix(as_pg) + window(SW) + suffix(pf_floor-ae_pg) + decoded(pages_total-pf_floor)
        active_when_evicting = as_pg + SPEECH_WINDOW_PAGES + (pages_total - ae_pg)
        active = tl.where(
            audio_pages > SPEECH_WINDOW_PAGES, active_when_evicting, pages_total
        )
    else:
        active = pages_total

    # flashinfer's append_paged_kv_cache treats append_positions as slot
    # offsets WITHIN the active kv_indices view, not as logical sequence
    # positions. For non-eviction this equals seq_lens-1; with eviction
    # active < pages_total so we have to base off the compressed view.
    appos = (active - 1) * P + (last_pl - 1)

    # Mask off non-real lanes so cumsum doesn't accumulate junk.
    active_masked = tl.where(mask, active, 0)

    tl.store(pages_total_ptr + offs, pages_total, mask=mask)
    tl.store(pages_used_ptr + offs, active_masked, mask=mask)
    tl.store(last_page_len_ptr + offs, last_pl, mask=mask)
    tl.store(append_positions_ptr + offs, appos, mask=mask)

    # tl.cumsum is inclusive; we want kv_indptr[i+1] = inclusive_cumsum[i].
    indptr_vals = tl.cumsum(active_masked, axis=0)
    tl.store(kv_indptr_ptr + 1 + offs, indptr_vals, mask=mask)


@triton.jit
def _kv_indices_fill_kernel(
    pages_used_ptr,
    pages_total_ptr,
    prefill_pages_ptr,
    prefill_floor_pages_ptr,
    audio_start_page_ptr,
    audio_end_page_ptr,
    kv_indptr_ptr,
    kv_indices_ptr,
    slide_offset_pages,     # runtime int (NOT constexpr) — used by SPEECH_EVICT/COMBINED
    B: tl.constexpr,
    M: tl.constexpr,
    SINK_PAGES: tl.constexpr,
    WINDOW_PAGES: tl.constexpr,
    EVICT: tl.constexpr,
    SPEECH_EVICT: tl.constexpr,
    COMBINED: tl.constexpr,
    SPEECH_WINDOW_PAGES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Fills the compressed page table. Each thread maps to one (i, j) cell
    of the logical [B, M] grid; writes only when ``j < pages_used[i]``.
    No race: different requests write into disjoint prefix ranges of
    ``kv_indices`` because ``kv_indptr`` is a monotonic exclusive cumsum
    of pages_used.

    When EVICT (streaming sink+window over decoded):
      - slots [0, prefill_pages + SINK_PAGES) map to identity physical pages
        (i*M + j) -- the prefill + sink region.
      - slots [prefill_pages + SINK_PAGES, active) point at the last
        WINDOW_PAGES physical pages (i*M + pages_total - WINDOW_PAGES + k).

    When SPEECH_EVICT (sliding window over audio, prompt retained both ends):
      view-slot layout per evicting request is [prefix][window][suffix][decoded]:
      - prefix  slots [0, as_pg)               -> identity pages (i*M + j)
                (system prompt + <|speech_start|>).
      - window  slots [as_pg, as_pg+SW)        -> i*M + win_lo + (j - as_pg),
                win_lo = as_pg + slide_eff, slide_eff capped to keep the window
                inside the audio range [as_pg, ae_pg).
      - suffix  slots [as_pg+SW, as_pg+SW+(pf_floor-ae_pg)) -> i*M + ae_pg + k
                (<|speech_end|> + format instruction + assistant turn).
      - decoded slots [.. , active)            -> i*M + pf_floor + k
                (all decoded tokens, anchored on the floor-rounded prefill page
                the first decoded token shares).
      Non-evicting requests (audio_pages <= SW, or no audio mask) fall back to
      the identity mapping.

    When COMBINED (sliding audio window + sink/window over decode):
      view-slot layout per request is [prefix][audio window][suffix][decode
      sink][decode window]. The prefill side (slots [0, pre_active)) uses the
      SPEECH_EVICT mapping; the decoded side (slots [pre_active, active)) uses
      the EVICT mapping anchored on pf_floor:
        - decode sink   slots -> pf_floor + d                (d = j - pre_active)
        - decode window slots -> pt - WINDOW_PAGES + (d - SINK_PAGES)
      Either side independently falls back to identity for requests that don't
      meet that side's eviction trigger."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = B * M
    in_range = offs < total

    i = offs // M
    j = offs - i * M  # offs % M

    i_safe = tl.where(in_range, i, 0)
    active = tl.load(pages_used_ptr + i_safe, mask=in_range, other=0)
    indptr_i = tl.load(kv_indptr_ptr + i_safe, mask=in_range, other=0)

    is_active = in_range & (j < active)
    dest = indptr_i + j

    if COMBINED:
        # Prefill side mirrors SPEECH_EVICT (prefix/window/suffix over audio);
        # decoded side mirrors EVICT (sink + last-window physical pages). The
        # boundary between the two sides is ``pre_active`` view slots in, and
        # the decoded side anchors on the floor-rounded prefill page.
        pf_floor = tl.load(prefill_floor_pages_ptr + i_safe, mask=in_range, other=0)
        as_pg = tl.load(audio_start_page_ptr + i_safe, mask=in_range, other=0)
        ae_pg = tl.load(audio_end_page_ptr + i_safe, mask=in_range, other=0)
        pt = tl.load(pages_total_ptr + i_safe, mask=in_range, other=0)
        audio_pages = ae_pg - as_pg
        decoded_pages = pt - pf_floor
        sp = audio_pages > SPEECH_WINDOW_PAGES
        st = decoded_pages > (SINK_PAGES + WINDOW_PAGES)
        # Where the prefill side ends / the decoded side begins (view slots).
        pre_active = tl.where(
            sp, as_pg + SPEECH_WINDOW_PAGES + (pf_floor - ae_pg), pf_floor
        )
        # --- prefill side (speech layout, identity when not sliding audio) ---
        slide_cap = audio_pages - SPEECH_WINDOW_PAGES
        slide_eff = tl.minimum(slide_offset_pages, slide_cap)
        win_lo = as_pg + slide_eff
        t_win = as_pg
        t_suf = as_pg + SPEECH_WINDOW_PAGES
        in_prefix = j < t_win
        in_awin = (j >= t_win) & (j < t_suf)
        page_pre_speech = tl.where(
            in_prefix, j,
            tl.where(in_awin, win_lo + (j - t_win), ae_pg + (j - t_suf)),
        )
        page_pre = tl.where(sp, page_pre_speech, j)
        # --- decoded side (sink + window, identity when decode is short) ---
        d = j - pre_active
        in_sink = d < SINK_PAGES
        page_dec_evict = tl.where(
            in_sink, pf_floor + d, pt - WINDOW_PAGES + (d - SINK_PAGES)
        )
        page_dec = tl.where(st, page_dec_evict, pf_floor + d)
        # Pick side by view slot.
        is_decoded = j >= pre_active
        page_in_request = tl.where(is_decoded, page_dec, page_pre)
        page_id = i * M + page_in_request
    elif EVICT:
        pf = tl.load(prefill_pages_ptr + i_safe, mask=in_range, other=0)
        pt = tl.load(pages_total_ptr + i_safe, mask=in_range, other=0)
        # Per-request: only requests where post-prefill > sink+window are
        # actually evicting. For non-evicting requests we keep the identity
        # mapping; the window-index path would produce negative page IDs
        # otherwise because pt - WINDOW_PAGES underflows.
        req_evicting = (pt - pf) > (SINK_PAGES + WINDOW_PAGES)
        in_window = req_evicting & (j >= (pf + SINK_PAGES))
        # window slot k in [0, WINDOW_PAGES): physical = pt - WINDOW_PAGES + k.
        window_offset = j - pf - SINK_PAGES
        page_in_request = tl.where(in_window, pt - WINDOW_PAGES + window_offset, j)
        page_id = i * M + page_in_request
    elif SPEECH_EVICT:
        pf_floor = tl.load(prefill_floor_pages_ptr + i_safe, mask=in_range, other=0)
        as_pg = tl.load(audio_start_page_ptr + i_safe, mask=in_range, other=0)
        ae_pg = tl.load(audio_end_page_ptr + i_safe, mask=in_range, other=0)
        audio_pages = ae_pg - as_pg
        req_evicting = audio_pages > SPEECH_WINDOW_PAGES
        # Slide the window within the audio range [as_pg, ae_pg). slide_cap
        # keeps it from running past the suffix; >= 0 whenever evicting since
        # audio_pages > SW there.
        slide_cap = audio_pages - SPEECH_WINDOW_PAGES
        slide_eff = tl.minimum(slide_offset_pages, slide_cap)
        win_lo = as_pg + slide_eff
        # View-slot region boundaries: [prefix | window | suffix | decoded].
        t_win = as_pg
        t_suf = as_pg + SPEECH_WINDOW_PAGES
        t_dec = t_suf + (pf_floor - ae_pg)
        in_prefix = j < t_win
        in_window = (j >= t_win) & (j < t_suf)
        in_suffix = (j >= t_suf) & (j < t_dec)
        # else (j >= t_dec): decoded
        page_window = win_lo + (j - t_win)
        page_suffix = ae_pg + (j - t_suf)
        page_decoded = pf_floor + (j - t_dec)
        page_evict = tl.where(
            in_prefix, j,
            tl.where(in_window, page_window,
            tl.where(in_suffix, page_suffix, page_decoded)),
        )
        # Non-evicting requests keep the identity mapping.
        page_in_request = tl.where(req_evicting, page_evict, j)
        page_id = i * M + page_in_request
    else:
        page_id = offs  # == i * M + j

    tl.store(kv_indices_ptr + dest, page_id, mask=is_active)


def triton_begin_decode_step(
    seq_lens: torch.Tensor,
    last_page_len: torch.Tensor,
    append_positions: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    pages_used_scratch: torch.Tensor,
    pages_total_scratch: torch.Tensor,
    prefill_pages: torch.Tensor,
    prefill_floor_pages: torch.Tensor,
    audio_start_pages: torch.Tensor,
    audio_end_pages: torch.Tensor,
    *,
    batch_size: int,
    max_pages_per_req: int,
    page_size: int,
    sink_pages: int = 0,
    window_pages: int = 0,
    evict: bool = False,
    speech_evict: bool = False,
    speech_window_pages: int = 0,
    slide_offset_pages: int = 0,
    fill_block: int = 256,
) -> None:
    """Run the two fused kernels back-to-back. All tensors must be int32,
    contiguous, and live on the same CUDA device.

    When ``evict=True`` the cache exposes (prefill + sink + window) pages
    once a request exceeds sink + window post-prefill tokens.

    When ``speech_evict=True`` the cache slides a ``speech_window_pages`` window
    over the audio sub-range ``[audio_start_pages, audio_end_pages)`` while
    always retaining the prompt prefix (pages before audio_start) and suffix
    (pages from audio_end to the prefill end), plus all decoded pages. Eviction
    fires only when the audio span exceeds the window. ``slide_offset_pages``
    selects where the window sits within the audio range (clamped per-request);
    the caller bumps it between decode steps.

    When BOTH ``evict`` and ``speech_evict`` are True the combined policy runs:
    the sliding audio window applies to the prefill side and sink + window
    applies to the decoded side, with the two triggers evaluated independently
    per request."""
    combined = evict and speech_evict
    # The pure-policy kernel branches stay mutually exclusive; the combined
    # branch handles the both-True case.
    evict_only = evict and not combined
    speech_only = speech_evict and not combined
    b_block = max(triton.next_power_of_2(batch_size), 2)
    _scalars_and_indptr_kernel[(1,)](
        seq_lens,
        last_page_len,
        append_positions,
        pages_used_scratch,
        pages_total_scratch,
        kv_indptr,
        prefill_pages,
        prefill_floor_pages,
        audio_start_pages,
        audio_end_pages,
        B=batch_size,
        P=page_size,
        SINK_PAGES=sink_pages,
        WINDOW_PAGES=window_pages,
        EVICT=evict_only,
        SPEECH_EVICT=speech_only,
        COMBINED=combined,
        SPEECH_WINDOW_PAGES=speech_window_pages,
        B_BLOCK=b_block,
    )
    total = batch_size * max_pages_per_req
    grid = (triton.cdiv(total, fill_block),)
    _kv_indices_fill_kernel[grid](
        pages_used_scratch,
        pages_total_scratch,
        prefill_pages,
        prefill_floor_pages,
        audio_start_pages,
        audio_end_pages,
        kv_indptr,
        kv_indices,
        slide_offset_pages,
        B=batch_size,
        M=max_pages_per_req,
        SINK_PAGES=sink_pages,
        WINDOW_PAGES=window_pages,
        EVICT=evict_only,
        SPEECH_EVICT=speech_only,
        COMBINED=combined,
        SPEECH_WINDOW_PAGES=speech_window_pages,
        BLOCK=fill_block,
    )
