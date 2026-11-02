"""ASR evaluation metrics: WER, CER, cpWER, tcpWER (MeetEval), DER (pyannote.metrics).

:func:`evaluate` scores a Murmur :class:`~murmur.types.Transcript` against a reference
exactly as in the paper (text normalization, character-level scoring for Mandarin,
tcpWER collar 5 s, DER collar 0.25 s with overlap scored).

Each output segment is a plain dict with these keys (both snake_case and Title Case
spellings are accepted everywhere):
    {
        "Speaker ID"  / "speaker_id"  : str,
        "Start time"  / "start_time"  : float | "HH:MM:SS.mmm",
        "End time"    / "end_time"    : float | "HH:MM:SS.mmm",
        "Content"     / "text"        : str,
    }
(for VibeVoice ASR)
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from jiwer import wer as jiwer_wer

log = logging.getLogger(__name__)


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class MetricsResult:
    wer: float = math.nan
    cer: float = math.nan
    cpwer: float = math.nan
    tcpwer: float = math.nan
    der: float = math.nan

    def as_dict(self) -> dict[str, float]:
        return {
            "wer": self.wer,
            "cer": self.cer,
            "cpwer": self.cpwer,
            "tcpwer": self.tcpwer,
            "der": self.der,
        }

    def __str__(self) -> str:
        pairs = [
            ("WER", self.wer),
            ("CER", self.cer),
            ("cpWER", self.cpwer),
            ("tcpWER", self.tcpwer),
            ("DER", self.der),
        ]
        parts = [f"{name}={val:.3f}%" for name, val in pairs if not math.isnan(val)]
        return " | ".join(parts) if parts else "(no metrics computed)"


# ── Timestamp helper ──────────────────────────────────────────────────────────

def _parse_ts(ts) -> float:
    """Parse ``HH:MM:SS.mmm`` or a numeric value to seconds."""
    if isinstance(ts, (int, float)):
        return float(ts)
    if isinstance(ts, str):
        parts = ts.split(":")
        if len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        try:
            return float(ts)
        except ValueError:
            return 0.0
    return 0.0


def _seg_text(seg: dict) -> str:
    return seg.get("Content", seg.get("text", ""))


def _seg_speaker(seg: dict) -> str:
    return seg.get("Speaker ID", seg.get("speaker_id", "unknown"))


def _seg_start(seg: dict) -> float:
    return _parse_ts(seg.get("Start time", seg.get("start_time", 0.0)))


def _seg_end(seg: dict) -> float:
    return _parse_ts(seg.get("End time", seg.get("end_time", 0.0)))


# ── CJK (character-level) scoring ────────────────────────────────────────────
#
# jiwer/meeteval both tokenize on whitespace by default. Mandarin transcripts
# (AliMeeting, AISHELL-4) have no inter-word spaces, so a naive word-level
# score treats an entire unsegmented sentence as one "word" -- meaningless.
# The standard fix (used by the AliMeeting/AISHELL-4 papers, ESPnet, etc.) is
# to score at the character level: insert a space between every CJK
# character before diffing, which turns "word" error into character error
# with no other code path changes. Latin/digit runs inside a CJK sentence
# (numbers, English loanwords) are left as single tokens, matching how
# humans would read them.
_CJK_RANGES = (
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs
    (0x3400, 0x4DBF),   # CJK Unified Ideographs Extension A
    (0xF900, 0xFAFF),   # CJK Compatibility Ideographs
    (0x3040, 0x30FF),   # Hiragana / Katakana (harmless if unused)
)


def _is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


def cjk_char_split(text: str) -> str:
    """Insert spaces between CJK characters so word-level scorers act as
    character-level scorers. Idempotent (safe to call twice)."""
    out = []
    prev_cjk = False
    for ch in text:
        cur_cjk = _is_cjk(ch)
        if cur_cjk or prev_cjk:
            out.append(" ")
        out.append(ch)
        prev_cjk = cur_cjk
    return re.sub(r"\s+", " ", "".join(out)).strip()


def _cjk_normalizer(seg: dict) -> dict:
    """meeteval ``normalizer=`` callable: space-split the ``words`` key."""
    seg["words"] = cjk_char_split(seg["words"])
    return seg


# ── meeteval STM conversion ───────────────────────────────────────────────────

def _to_stm(segments: list[dict], meeting_id: str):
    from meeteval.io import STM, STMLine

    lines = [
        STMLine(
            filename=meeting_id,
            channel=0,
            speaker_id=_seg_speaker(seg),
            begin_time=_seg_start(seg),
            # meeteval rejects end < start (sub-ms round-off at chunk offsets)
            end_time=max(_seg_end(seg), _seg_start(seg)),
            transcript=_seg_text(seg),
        )
        for seg in segments
    ]
    return STM(lines)


# ── Pure-Python Levenshtein (no jiwer) ───────────────────────────────────────

def _levenshtein(a: list, b: list) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


# ── WER ───────────────────────────────────────────────────────────────────────

def compute_wer(ref: str, hyp: str, cjk: bool = False) -> float:
    if not ref.strip():
        return float("nan")
    if cjk:
        ref, hyp = cjk_char_split(ref), cjk_char_split(hyp or "")
    return round(jiwer_wer([ref], [hyp or ""]) * 100, 3)

# ── CER ───────────────────────────────────────────────────────────────────────

def compute_cer(ref: str, hyp: str) -> float:
    """Character Error Rate (%).

    Uses *jiwer* when available, falls back to pure-Python Levenshtein.
    Language-agnostic already (diffs the raw character stream), so CJK
    datasets need no special-casing here -- this IS the canonical metric
    for Mandarin (WER above is only meaningful once ``cjk=True`` turns it
    into the same character-level score via ``cjk_char_split``).
    """
    if not ref.strip():
        return math.nan
    try:
        from jiwer import cer as _cer

        return round(_cer([ref], [hyp or ""]) * 100, 3)
    except ImportError:
        ref_chars = list(ref)
        hyp_chars = list(hyp or "")
        d = _levenshtein(ref_chars, hyp_chars)
        return round(d / len(ref_chars) * 100, 3)


# ── cpWER ─────────────────────────────────────────────────────────────────────

def compute_cpwer(
    ref_segments: list[dict],
    hyp_segments: list[dict],
    meeting_id: str = "meeting",
    cjk: bool = False,
) -> float:
    """Concatenated minimum-permutation WER (speaker-aware, %).

    ``cjk=True`` (AliMeeting, AISHELL-4) scores at the character level --
    see the module-level comment above ``cjk_char_split`` -- by handing
    meeteval a normalizer that space-splits CJK characters before it diffs.
    Produces what the Mandarin-meeting literature calls "cpCER"; the field
    stays named ``cpwer`` since it's the same quantity (concatenated
    minimum-permutation edit rate), just tokenized per-character.
    """
    # No reference -> undefined (zero denominator). No hypothesis against a
    # real reference -> every reference word is a deletion, i.e. 100%, NOT
    # NaN: a NaN drops the meeting from the mean, so total-failure cases
    # silently improve the average instead of penalising it.
    if not ref_segments:
        return math.nan
    if not hyp_segments:
        return 100.0
    try:
        from meeteval.wer.api import cpwer

        ref_stm = _to_stm(ref_segments, meeting_id)
        hyp_stm = _to_stm(hyp_segments, meeting_id)

        if len(ref_stm) == 0 or len(hyp_stm) == 0:
            return float("nan")

        try:
            result = cpwer(ref_stm, hyp_stm, normalizer=_cjk_normalizer if cjk else None)
            total_errors = sum(v.errors for v in result.values())
            total_length = sum(v.length for v in result.values())
            if total_length == 0:
                return float("nan")
            return round(total_errors / total_length * 100, 3)
        except Exception as e:
            log.warning(f"cpWER computation failed: {e}")
            return float("nan")
        
    except ImportError:
        ref_text = " ".join(_seg_text(s) for s in ref_segments)
        hyp_text = " ".join(_seg_text(s) for s in hyp_segments)
        return compute_wer(ref_text, hyp_text, cjk=cjk)


# ── tcpWER ────────────────────────────────────────────────────────────────────

def compute_tcpwer(
    ref_segments: list[dict],
    hyp_segments: list[dict],
    collar: float = 5.0,
    meeting_id: str = "meeting",
    cjk: bool = False,
) -> float:
    """Time-constrained minimum-permutation WER (%).

    Matches hypothesis segments to reference within ±*collar* seconds.
    ``cjk=True`` scores at the character level -- see ``compute_cpwer``.
    """
    if not ref_segments:
        return math.nan
    if not hyp_segments:
        return 100.0   # all deletions — see compute_cpwer
    try:
        from meeteval.wer.api import tcpwer

        try:
            result = tcpwer(
                _to_stm(ref_segments, meeting_id),
                _to_stm(hyp_segments, meeting_id),
                collar=collar,
                normalizer=_cjk_normalizer if cjk else None,
            )
            total_errors = sum(v.errors for v in result.values())
            total_length = sum(v.length for v in result.values())
            if total_length == 0:
                return math.nan
            return round(total_errors / total_length * 100, 3)
        except Exception as e:
            # meeteval refuses permutation scoring above ~20 speakers. Chunked
            # decoding emits chunk-unique labels (c003:S01), so a long meeting
            # cut into many chunks blows past that (a 44-min AMI meeting at 30s
            # chunks yields 321). Mirror compute_cpwer: degrade to NaN rather
            # than propagate -- an unscoreable tcpWER must not lose the whole
            # meeting's WER/CER and its checkpoint.
            log.warning(f"tcpWER computation failed: {e}")
            return math.nan
    except ImportError:
        return compute_cpwer(ref_segments, hyp_segments, meeting_id, cjk=cjk)


# ── DER ───────────────────────────────────────────────────────────────────────

def compute_der(
    ref_rttm_segments: list[tuple[float, float, str]],
    hyp_segments: list[dict],
    collar: float = 0.25,
) -> float:
    if not ref_rttm_segments:
        return math.nan
    if not hyp_segments:
        return 100.0   # every reference segment is missed speech
    try:
        from pyannote.core import Annotation, Segment
        from pyannote.metrics.diarization import DiarizationErrorRate

        ref_ann = Annotation()
        for start, end_s, spk in ref_rttm_segments:
            ref_ann[Segment(start, end_s)] = spk

        hyp_ann = Annotation()
        for s in hyp_segments:
            hyp_ann[Segment(_seg_start(s), _seg_end(s))] = _seg_speaker(s)

        metric = DiarizationErrorRate(collar=collar, skip_overlap=False)
        der = metric(ref_ann, hyp_ann)
        return round(der * 100, 3)
    except ImportError:
        log.warning(
            "pyannote.metrics not installed — DER will be NaN. "
            "pip install pyannote.metrics"
        )
        return math.nan


def evaluate_segments(
    ref_segments: list[dict],
    hyp_segments: list[dict],
    ref_rttm: list[tuple[float, float, str]] | None = None,
    meeting_id: str = "meeting",
    cjk: bool = False,
) -> MetricsResult:
    """Compute all available metrics in one call.

    Parameters
    ----------
    ref_segments:
        Reference transcript segments.
    hyp_segments:
        Hypothesis (system output) transcript segments.
    ref_rttm:
        Optional list of ``(start_s, end_s, speaker)`` tuples for DER.
        If not provided DER is left as NaN.
    meeting_id:
        Meeting identifier used in STM conversion (for meeteval).
    cjk:
        Score WER/cpWER/tcpWER at the character level (Mandarin meetings:
        AliMeeting, AISHELL-4). CER and DER are unaffected -- CER already
        diffs raw characters, and DER never looks at text.

    Returns
    -------
    MetricsResult
        All metrics.  Any metric whose dependency is missing is NaN.
    """
    ref_text = " ".join(_seg_text(s) for s in ref_segments)
    hyp_text = " ".join(_seg_text(s) for s in hyp_segments)

    return MetricsResult(
        wer=compute_wer(ref_text, hyp_text, cjk=cjk),
        cer=compute_cer(ref_text, hyp_text),
        cpwer=compute_cpwer(ref_segments, hyp_segments, meeting_id, cjk=cjk),
        tcpwer=compute_tcpwer(ref_segments, hyp_segments, meeting_id=meeting_id, cjk=cjk),
        der=compute_der(ref_rttm, hyp_segments) if ref_rttm is not None else math.nan,
    )


# ── Paper scoring ─────────────────────────────────────────────────────────────

import unicodedata  # noqa: E402

_FILLERS = re.compile(r"\b(uh+|um+|mm+|hmm+|hm+|mhm)\b")
_NONSPEECH = re.compile(r"\[[^\]]*\]|<[^>]*>")  # [laughter], <noise>, ...


def normalize(text: str) -> str:
    """Text normalization applied to references and hypotheses before scoring (WhisperX-style)."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = _NONSPEECH.sub("", text)
    text = _FILLERS.sub("", text)
    text = re.sub(r"[^\w\s']", "", text)
    return re.sub(r"\s+", " ", text).strip()


def evaluate(hyp, ref_segments: list[dict], ref_rttm: list[tuple[float, float, str]] | None = None,
             meeting_id: str = "meeting", cjk: bool = False) -> dict:
    """Score one recording: ``hyp`` is a Transcript or a list of Segments.

    ``ref_segments`` are dicts with "Speaker ID", "Start time", "End time" and "Content"
    (already normalized); ``ref_rttm`` is [(start, end, speaker), ...] for DER.
    Returns {"wer", "cer", "cpwer", "tcpwer", "der"} in percent (NaN when not computable).
    """
    import jiwer

    segs = getattr(hyp, "segments", hyp)
    ref_text = " ".join(_seg_text(s) for s in ref_segments)
    hyp_text = normalize(" ".join(s.text for s in segs))
    hyp_dicts = [{"Speaker ID": s.speaker, "Start time": s.start, "End time": s.end, "Content": normalize(s.text)}
                 for s in segs]
    if ref_text.strip():
        r, h = (cjk_char_split(ref_text), cjk_char_split(hyp_text or "")) if cjk else (ref_text, hyp_text)
        wer = round(jiwer.wer(r, h or "") * 100, 3)
        cer = round(jiwer.cer(ref_text, hyp_text or "") * 100, 3)
    else:
        wer = cer = math.nan
    if ref_segments and any(_seg_speaker(s) for s in ref_segments):
        cpwer = compute_cpwer(ref_segments, hyp_dicts, meeting_id=meeting_id, cjk=cjk)
        tcpwer = compute_tcpwer(ref_segments, hyp_dicts, meeting_id=meeting_id, cjk=cjk)
        der = compute_der(ref_rttm, hyp_dicts) if ref_rttm else math.nan
    else:
        cpwer = tcpwer = der = math.nan
    return {"wer": wer, "cer": cer, "cpwer": cpwer, "tcpwer": tcpwer, "der": der}
