# Murmur: Efficient Long-Context ASR Inference with Hierarchical Chunking

[![arXiv](https://img.shields.io/badge/arXiv-2606.01483-b31b1b.svg)](https://arxiv.org/abs/2606.01483)

Long-context ASR models such as **VibeVoice-ASR** and **MOSS-Transcribe-Diarize** transcribe an
hour-long meeting in one pass, with speakers and timestamps. That single pass is slow, and one
repetition loop can wipe out the whole transcript. Murmur makes these models faster without
changing what they write:

- **Target chunks (~5 min).** A few minutes of context is enough: 330 s chunks match or beat
  single-pass accuracy, confine a repetition loop to one chunk, and decode in parallel.
- **Draft windows (30 s).** A small short-window ASR (Whisper-tiny/small) transcribes each chunk;
  its words, simple format rules for timestamps and speaker tags, and the model's own output become
  drafts for speculative decoding. The model verifies every token, so the output is its own greedy
  output. Murmur enables draft windows when a decoding step is expensive (VibeVoice-ASR's 7B decoder),
  not for small decoders (MOSS-Transcribe-Diarize's 0.6B).

![Hierarchical chunking](./docs/figures/hierarchy.png)

## Results

End-to-end latency per recording (including diarization) and tcpWER, one NVIDIA H100, vLLM.
Single pass decodes the whole recording at once; Murmur is target chunks + draft windows for
VibeVoice-ASR and target chunks for MOSS-Transcribe-Diarize.

| Model | System | AMI-IHM | AMI-SDM | AliMeeting | AISHELL-4 |
|---|---|---|---|---|---|
| VibeVoice-ASR | single pass | 34.0% / 108.9 s | 43.6% / 99.5 s | 41.7% / 122.5 s | 30.3% / 110.9 s |
| | **Murmur** | **28.8% / 38.0 s** | **36.4% / 38.5 s** | **38.7% / 47.1 s** | 30.5% / **45.6 s** |
| MOSS-Transcribe-Diarize | single pass | 23.2% / 54.2 s | 35.5% / 77.1 s | **13.6%** / 90.8 s | **14.1%** / 50.8 s |
| | **Murmur** | **23.0% / 26.4 s** | **29.0% / 26.4 s** | 17.5% / **31.6 s** | 16.6% / **26.1 s** |

On average Murmur is 2.5x faster (2.0-2.9x) with 2.0 points lower tcpWER. The exception is speaker
attribution for MOSS-Transcribe-Diarize on Mandarin, where its single pass, which diarizes the whole
recording jointly, stays ahead.

## Install

The client and each model's vLLM server live in separate environments, because the two models need
different vLLM builds.

**Client** (the `murmur` package):
```bash
conda create -n murmur python=3.10 -y && conda activate murmur
git clone https://github.com/uw-syfi/Murmur.git && cd Murmur
pip install -e ".[all]"            # diarization (pyannote), draft windows (faster-whisper), scoring
huggingface-cli login              # pyannote/speaker-diarization-3.1 is gated: accept its terms first
```

**VibeVoice-ASR server** (vLLM 0.14 + the VibeVoice vLLM plugin):
```bash
conda create -n vibevoice-vllm python=3.10 -y && conda activate vibevoice-vllm
pip install vllm==0.14.0
git clone https://github.com/microsoft/VibeVoice.git && pip install -e VibeVoice
pip install -e "path/to/Murmur[server]"   # the drafter plugin
```

**MOSS-Transcribe-Diarize server** (a vLLM nightly with the model's code):
```bash
conda create -n moss-vllm python=3.10 -y && conda activate moss-vllm
pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly   # tested: 0.23.1rc1.dev949
pip install -e "path/to/Murmur[server]"
```

Tell the client where each server's `vllm` is, so it can start servers itself:
```bash
export MURMUR_VLLM_VIBEVOICE=$CONDA_PREFIX/../vibevoice-vllm/bin/vllm
export MURMUR_VLLM_MOSS=$CONDA_PREFIX/../moss-vllm/bin/vllm
```

## Quick start

```python
from murmur import Murmur

with Murmur.load("vibevoice") as m:          # starts a vLLM server on a free port
    t = m.transcribe("meeting.wav")
    print(t)                                  # [  12.40-  15.10] SPEAKER_01: ...
    for s in t.segments:
        print(s.speaker, s.start, s.end, s.text)

m = Murmur.load("moss", language="zh")       # Mandarin; MOSS uses target chunks only by default
```

Or start the server yourself (e.g. on another GPU) and point the client at it:
```bash
# in the vibevoice-vllm environment
CUDA_VISIBLE_DEVICES=0 murmur serve vibevoice --port 8000 --draft-dir /tmp/murmur_drafts
# in the murmur environment
murmur transcribe meeting.wav --model vibevoice --url http://localhost:8000 --draft-dir /tmp/murmur_drafts
```
The draft directory is how the client hands each chunk's draft transcript to the server, so client
and server must run on the same machine (or share that directory).

Options of `Murmur.load`: `chunk_s` (target chunk length, default 330), `batch_size` (chunks decoded
concurrently, default 8), `draft_windows` (default: on for VibeVoice-ASR, off for MOSS), `language`
(`"en"` or `"zh"`, picks the draft model), `diarize` (pyannote for chunk boundaries and speaker
linking; without it, chunks are fixed windows and speakers are not linked across chunks), `gpu`.
`transcribe(..., single_pass=True)` runs the single-pass baseline.

## Benchmarks

Prepare the datasets (written to `benchmarks/meeting_datasets/`, or `$MURMUR_DATA`):
```bash
bash benchmarks/prepare/download_ami.sh && python benchmarks/prepare/build_ami_sessions.py   # needs lhotse
python benchmarks/prepare/download_alimeeting_refs.py    # OpenSLR 119
python benchmarks/prepare/download_aishell4_refs.py      # OpenSLR 111
```

Run one configuration:
```bash
python benchmarks/run.py --model vibevoice --dataset ami_ihm --system murmur --mode latency
```
`--system` is `single`, `chunked` (greedy), `murmur` or `ngram` (vLLM's prompt-lookup baseline);
`--mode latency` processes recordings one at a time and reports end-to-end latency per recording,
`--mode throughput` decodes all chunks of a dataset through one pool and reports decoding time.

Scoring follows the paper: WER (CER for Mandarin) with jiwer, cpWER and tcpWER (5 s collar) with
MeetEval, DER with pyannote.metrics (0.25 s collar, overlap scored), averaged over recordings.

## Repository layout

```
murmur/
  api.py          Murmur.load / transcribe: diarize -> chunk -> draft -> decode -> stitch
  models/         VibeVoice-ASR and MOSS-Transcribe-Diarize: requests, output parsing, server flags
  drafting/       vllm_proposer.py (the drafter inside vLLM), whisper_drafts.py (draft windows)
  audio.py        audio loading, pyannote diarization, target-chunk boundaries
  stitch.py       timestamp offsets and cross-chunk speaker linking
  server.py, cli.py
  metrics.py      WER/CER, cpWER/tcpWER, DER
benchmarks/       run.py, meeting_data.py, prepare/
tests/            CPU tests (parsing, chunking, stitching, drafter)
```

## Citation

```bibtex
@article{murmur2026lee,
  title={Murmur: Efficient Long-Context ASR Inference with Hierarchical Chunking},
  author={Lee, Wei-Tzu and Kamahori, Keisuke and Kasikci, Baris},
  journal={arXiv preprint arXiv:2606.01483},
  year={2026},
  url={https://arxiv.org/abs/2606.01483}
}
```
