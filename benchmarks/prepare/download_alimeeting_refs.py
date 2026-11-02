"""Download + convert AliMeeting (far-field) into CAST's local STM/RTTM/wav
layout, the same shape ``download_ami_refs.py`` produces for AMI.

AliMeeting (ICASSP 2022 M2MeT challenge, OpenSLR 119) ships far-field
(8-channel array) and near-field (per-speaker headset) conditions as
Eval_Ali.tar.gz (~3.4G) / Test_Ali.tar.gz (~8.9G) tarballs -- the Train
archives are 73G/22G and are NOT needed for eval-only benchmarking, so this
script never touches them. Only the far-field condition is converted here
(the SDM analog: channel 0 of the 8-ch array, one wav per meeting); near
field (per-speaker headset, the IHM analog) could be added the same way
later if wanted.

Each meeting's far-field TextGrid has one IntervalTier per speaker
(``lhotse.recipes.ali_meeting`` parses the identical layout); this script
merges all tiers into one time-sorted STM + RTTM, mirroring
``write_stm``/``write_rttm`` in download_ami_refs.py.

Usage:
  # 1. Download the archives (paths below assume OpenSLR's own filenames):
  wget https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/AliMeeting/openlr/Eval_Ali.tar.gz
  wget https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/AliMeeting/openlr/Test_Ali.tar.gz

  # 2. Convert (extracts once into --raw_dir, then writes STM/RTTM/wav):
  python benchmarks/download_alimeeting_refs.py \\
      --archives /path/Eval_Ali.tar.gz /path/Test_Ali.tar.gz \\
      --raw_dir /mnt/storage/<you>/raw_datasets/alimeeting/extracted \\
      --output benchmarks/meeting_datasets/alimeeting/test
"""

from __future__ import annotations

import argparse
import re
import tarfile
from collections import defaultdict
from pathlib import Path

# Non-speech annotation tags. AliMeeting's own transcripts don't use these
# (checked empirically), but AISHELL-4 does (<sil>, <%>, <#>, <$>, &...&
# overlap-speech wrappers) and the two scripts share this normalize() --
# stripped as a tag SPAN first so e.g. "<sil>" doesn't degrade to the bogus
# bare word "sil" once _PUNCT below removes just the angle brackets.
_NONSPEECH = re.compile(r"<[^>]*>|\[[^\]]*\]|&")
_PUNCT = re.compile(r"[^\w\s']", re.UNICODE)   # \w already matches CJK ideographs


def normalize(text: str) -> str:
    """Same punctuation-strip as download_ami_refs.py's normalize(). No
    English filler-word regex here -- Mandarin fillers (呃/嗯/那个) are left
    in verbatim, matching how the AliMeeting/AISHELL-4 papers score CER."""
    text = _NONSPEECH.sub("", text)
    text = _PUNCT.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_far(archive: Path, raw_dir: Path) -> tuple[Path, Path]:
    """Extract only the far-field audio_dir/textgrid_dir members of one
    Eval_Ali.tar.gz / Test_Ali.tar.gz into raw_dir (skips the near-field
    members inside the same tarball -- no need to unpack those for the far
    condition). Idempotent: skips members already on disk."""
    part = archive.stem.replace(".tar", "")   # "Eval_Ali" / "Test_Ali"
    far_root = raw_dir / part / f"{part}_far"
    audio_dir, tg_dir = far_root / "audio_dir", far_root / "textgrid_dir"
    if audio_dir.is_dir() and tg_dir.is_dir() and any(audio_dir.iterdir()):
        print(f"  {part}: already extracted at {far_root}")
        return audio_dir, tg_dir

    print(f"  {part}: extracting far-field members from {archive} ...")
    with tarfile.open(archive) as tf:
        members = [m for m in tf.getmembers() if f"{part}_far/" in m.name]
        tf.extractall(raw_dir, members=members)
    return audio_dir, tg_dir


def _merge_textgrid(tg_path: Path) -> list[dict]:
    import textgrid

    tg = textgrid.TextGrid.fromFile(str(tg_path))
    rows = []
    for tier in tg.tiers:
        spk = tier.name
        for iv in tier.intervals:
            text = normalize(str(iv.mark or ""))
            if not text:
                continue
            rows.append({"speaker": spk, "start": float(iv.minTime), "end": float(iv.maxTime), "text": text})
    return sorted(rows, key=lambda r: r["start"])


def write_stm(rows: list[dict], meeting_id: str, out_path: Path) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f";; STM for {meeting_id} (generated from AliMeeting far-field TextGrid)\n")
        for r in rows:
            f.write(f"{meeting_id} 1 {r['speaker']} {r['start']:.3f} {r['end']:.3f} {r['text']}\n")


def write_rttm(rows: list[dict], meeting_id: str, out_path: Path) -> None:
    with open(out_path, "w") as f:
        for r in rows:
            dur = r["end"] - r["start"]
            if dur <= 0:
                continue
            f.write(f"SPEAKER {meeting_id} 1 {r['start']:.3f} {dur:.3f} <NA> <NA> {r['speaker']} <NA> <NA>\n")


def write_channel0_wav(src_wav: Path, out_path: Path) -> None:
    import soundfile as sf

    data, sr = sf.read(str(src_wav), always_2d=True)   # (frames, channels)
    sf.write(str(out_path), data[:, 0], sr, subtype="PCM_16")


def main() -> None:
    p = argparse.ArgumentParser(description="Download/convert AliMeeting far-field into STM+RTTM+wav")
    p.add_argument("--archives", nargs="+", required=True,
                   help="Path(s) to Eval_Ali.tar.gz / Test_Ali.tar.gz (already downloaded).")
    p.add_argument("--raw_dir", required=True,
                   help="Where to extract the far-field members (large -- put it on a big disk/mount, "
                        "NOT the repo checkout).")
    p.add_argument("--output", default="benchmarks/meeting_datasets/alimeeting/test")
    args = p.parse_args()

    raw_dir = Path(args.raw_dir); raw_dir.mkdir(parents=True, exist_ok=True)
    out = Path(args.output)
    wav_dir, stm_dir, rttm_dir = out / "wav", out / "stm", out / "rttm"
    for d in (wav_dir, stm_dir, rttm_dir):
        d.mkdir(parents=True, exist_ok=True)

    written = 0
    for archive in args.archives:
        audio_dir, tg_dir = _extract_far(Path(archive), raw_dir)
        tg_by_meeting = {tg.stem: tg for tg in sorted(tg_dir.glob("*.TextGrid"))}
        wav_by_meeting: dict[str, Path] = defaultdict(list)
        for wav in sorted(audio_dir.glob("*.wav")):
            # R8001_M8004_MS801.wav -> meeting id R8001_M8004 (strip the mic-id suffix)
            mid = "_".join(wav.stem.split("_")[:2])
            wav_by_meeting[mid] = wav

        for mid, tg_path in tg_by_meeting.items():
            wav_path = wav_by_meeting.get(mid)
            if wav_path is None:
                print(f"  {mid}: no matching far wav -- skipping"); continue
            rows = _merge_textgrid(tg_path)
            if not rows:
                print(f"  {mid}: empty TextGrid -- skipping"); continue
            write_stm(rows, mid, stm_dir / f"{mid}.stm")
            write_rttm(rows, mid, rttm_dir / f"{mid}.rttm")
            write_channel0_wav(wav_path, wav_dir / f"{mid}.wav")
            n_spk = len({r["speaker"] for r in rows})
            print(f"  {mid}: {len(rows)} segments, {n_spk} speakers written")
            written += 1

    print(f"\nDone — wrote {written} meetings to {out}")


if __name__ == "__main__":
    main()
