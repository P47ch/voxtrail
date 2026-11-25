# Local Whisper Turbo + pyannote Community-1 Transcription Pipeline

A fully local, resumable audio/video transcription pipeline built around:

- **OpenAI Whisper Turbo** for speech-to-text
- **pyannote Community-1** for speaker diarization
- **Word-level speaker attribution**
- **Crash-safe external chunk checkpoints**
- **Sequential GPU usage** so Whisper and pyannote are not intentionally kept in VRAM at the same time
- **Offline normal operation** after the models have been downloaded once
- **TXT, JSON, SRT, and VTT output**
- **Single-speaker mode** that skips pyannote and Hugging Face sign-in
- **Quality review report** with timestamps for passages worth checking

The project is designed primarily for long recordings and an NVIDIA GPU with about **8 GB of VRAM**, while still supporting CPU fallback.

## Quick start

After creating the virtual environment and installing `requirements.txt`,
for a multi-speaker recording:

1. Sign in on the [Community-1 model page](https://huggingface.co/pyannote/speaker-diarization-community-1) and accept its access conditions.
2. Run setup from the project directory (Linux, macOS, or WSL):

```bash
bash setup_models.sh
```

On Windows PowerShell or Command Prompt, run `setup_models.cmd` instead.

Setup opens a Hugging Face browser authorization when needed. Hugging Face
stores the credential in your user cache, outside the project. Setup checks
Community-1 access and downloads both models. Normal transcription does not
use the token.

For recordings known to contain only one speaker, run
`bash setup_models.sh --whisper-only` (or `setup_models.cmd --whisper-only` on Windows), then use
`--single-speaker` with `transcribe_local.py`. This path needs no Hugging Face
account or pyannote model.

---

## Table of contents

1. [How the pipeline works](#how-the-pipeline-works)
2. [Requirements](#requirements)
3. [Recommended project layout](#recommended-project-layout)
4. [Python environment setup](#python-environment-setup)
5. [PyTorch and CUDA](#pytorch-and-cuda)
6. [One-time model setup](#one-time-model-setup)
7. [Offline operation](#offline-operation)
8. [Running the pipeline](#running-the-pipeline)
9. [CLI reference](#cli-reference)
10. [Chunking and checkpoints](#chunking-and-checkpoints)
11. [Speaker reconciliation](#speaker-reconciliation)
12. [Output files](#output-files)
13. [Recommended settings](#recommended-settings)
14. [Resume and recomputation behavior](#resume-and-recomputation-behavior)
15. [Troubleshooting](#troubleshooting)
16. [Performance and quality notes](#performance-and-quality-notes)
17. [Development notes](#development-notes)
18. [Public GitHub repository](#public-github-repository)
19. [License](#license)

---

# How the pipeline works

The multi-speaker pipeline is intentionally split into three major stages.

```text
input audio/video
        |
        v
FFmpeg normalization
16 kHz / mono / PCM WAV
        |
        v
+----------------------+
| Stage 1              |
| Whisper Turbo        |
| transcription        |
+----------------------+
        |
        | checkpoint every external chunk
        v
Unload Whisper
clear CUDA memory
        |
        v
+----------------------+
| Stage 2              |
| pyannote Community-1 |
| speaker diarization  |
+----------------------+
        |
        | checkpoint every external chunk
        v
Unload pyannote
clear CUDA memory
        |
        v
+----------------------+
| Stage 3              |
| CPU post-processing  |
+----------------------+
        |
        +--> reconcile speaker identities across chunks
        +--> assign Whisper words to speakers
        +--> build utterances/subtitles
        +--> generate TXT / JSON / SRT / VTT
```

Whisper and pyannote are deliberately run **sequentially**. This is important on GPUs with limited VRAM.

The script explicitly deletes each model and clears CUDA caches before loading the next model.
With `--single-speaker`, stage 2 is skipped.

---

# Requirements

## Recommended Python version

Use:

```text
Python 3.12
```

The supplied `requirements.txt` targets modern versions of Whisper, pyannote, PyTorch, and NumPy.

Create a virtual environment rather than installing the dependencies globally.

## Python dependencies

The project includes:

```text
requirements.txt
```

Core packages include:

- `openai-whisper`
- `pyannote.audio`
- `torch`
- `torchaudio`
- `numpy`

## System dependencies

The following programs are not installed through `pip`.

### FFmpeg

FFmpeg is required during normal operation.

It is used once per input file to normalize the source into:

```text
16 kHz
mono
PCM16 WAV
```

Check that FFmpeg is available:

```bash
ffmpeg -version
```

### Faster word timestamps on Linux and WSL

Whisper Turbo uses Triton for faster word-timestamp alignment when a C
compiler and the development headers matching the virtual environment's Python
version are installed. On Debian/Ubuntu, install them with:

```bash
sudo apt install gcc python3-dev
```

If the virtual environment uses a different Python version from the system
default, install its matching package instead, such as `python3.13-dev` for
Python 3.13. The setup status command reports whether these build tools are
present. Without them, Whisper still transcribes on the GPU, but word
alignment falls back to slower implementations. Whisper's warning may say
"likely due to missing CUDA toolkit" even when the actual missing pieces are
the C compiler or `Python.h`.

### Git LFS

Git LFS is optional. `bash setup_models.sh` downloads Community-1 through
`huggingface_hub` without Git LFS. Manual Git clones still need Git LFS to fetch
the model weights.

---

# Recommended project layout

The script expects model files relative to `transcribe_local.py`.

Recommended layout:

```text
project/
├── transcribe_local.py
├── quality_report.py
├── setup_models.py
├── setup_models.sh
├── setup_models.cmd
├── requirements.txt
├── README.md
│
├── models/
│   ├── whisper/
│   │   └── large-v3-turbo.pt
│   │
│   └── pyannote/
│       └── speaker-diarization-community-1/
│           ├── config.yaml
│           ├── ...
│           └── model files
│
└── output/
```

The paths currently hard-coded by the script are:

```text
models/whisper/large-v3-turbo.pt
models/pyannote/speaker-diarization-community-1/
```

If you move the models somewhere else, update the corresponding constants near the top of `transcribe_local.py`.

---

# Python environment setup

## Linux / macOS

Create the environment:

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Upgrade pip:

```bash
python -m pip install -U pip
```

## Windows PowerShell

Create the environment:

```powershell
py -3.12 -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Upgrade pip:

```powershell
python -m pip install -U pip
```

---

# PyTorch and CUDA

For NVIDIA GPU acceleration, install a **CUDA-enabled PyTorch build** appropriate for your system before installing the remaining project dependencies.

Use the installation command recommended by the official PyTorch installer for your platform, Python version, and supported CUDA runtime.

After installing PyTorch, verify CUDA:

```bash
python -c "import torch; print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA runtime:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Expected result on the intended machine:

```text
CUDA available: True
GPU: <your NVIDIA GPU>
```

Then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

Because `requirements.txt` contains compatible constraints for `torch` and `torchaudio`, an already-installed compatible CUDA build should remain satisfied.

You can check the complete environment with:

```bash
python -c "import torch, whisper, pyannote.audio, numpy; print('torch:', torch.__version__); print('pyannote:', pyannote.audio.__version__); print('numpy:', numpy.__version__); print('CUDA:', torch.cuda.is_available())"
```

## CPU fallback

If CUDA is unavailable, the script automatically falls back to CPU.

That is useful for testing but will normally be much slower for Whisper Turbo and pyannote.

---

# One-time model setup

Model acquisition needs internet access and is separate from normal offline
transcription. For multi-speaker runs, first sign in on the
[Community-1 model page](https://huggingface.co/pyannote/speaker-diarization-community-1)
and accept its access conditions. This is a Hugging Face requirement for the
gated diarization model.

From a terminal in the project directory, run:

```bash
bash setup_models.sh
```

Use `setup_models.cmd` on Windows. Both wrappers find the project's `venv` or
`.venv` environment and run the same setup program. If you are not signed in
for command-line use yet, choose **Log in with your browser** at the prompt,
open the displayed URL, and approve the displayed code. Hugging Face stores
the credential in your user cache (normally `~/.cache/huggingface/token` on
Linux/WSL; `HF_HOME` can change this location). It is never written into the
project. The setup checks access, then downloads Whisper Turbo and Community-1
into `models/`. An interrupted download can be resumed with the same command.

To verify the Whisper checksum and required Community-1 files, and check local
sign-in status without downloading:

```bash
bash setup_models.sh --check
```

For single-speaker use, download/check only Whisper Turbo:

```bash
bash setup_models.sh --whisper-only
bash setup_models.sh --check --whisper-only
```

The Windows equivalents use `setup_models.cmd`. This path does not prompt for
Hugging Face sign-in. The Python dependencies must still be installed first.

To sign in again with another Hugging Face account:

```bash
bash setup_models.sh --login
```

On Windows, use `setup_models.cmd --login`. No manual token creation is needed.

If setup reports that Community-1 access has not been granted, confirm the
model conditions were accepted with the same Hugging Face account used for
sign-in, then rerun setup. There is no Git LFS requirement for this path.

Multi-speaker transcription loads `models/whisper/large-v3-turbo.pt` and
`models/pyannote/speaker-diarization-community-1/` locally. Single-speaker
transcription loads only Whisper. Neither mode reads a Hugging Face token or
uses the network during transcription.

---

# Offline operation

The script sets these variables before importing model libraries:

```text
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
PYANNOTE_METRICS_ENABLED=0
```

FFmpeg is restricted to local file, data, and encryption protocols.

During a normal transcription run:

- Whisper is loaded from `models/whisper/large-v3-turbo.pt`
- Community-1 is loaded from the local pyannote repository for multi-speaker runs
- no Hugging Face token should be required
- no model download should be required
- internet access should not be required

Therefore, make sure the model setup is complete **before** going offline.

---

# Running the pipeline

General form:

```bash
python transcribe_local.py INPUT_FILE [options]
```

The input may be an audio or video file supported by FFmpeg.

---

## Known single speaker

Use this only when the whole recording is known to contain one speaker:

```bash
python3 transcribe_local.py recording.mp3 \
  --language en \
  --single-speaker \
  --output output/recording-single
```

This runs Whisper Turbo and labels every transcribed word `SPEAKER_00`.
pyannote is not loaded, and no diarization checkpoint or Hugging Face token is
needed. The script does not verify the one-speaker assumption. Using both
`--min-speakers 1` and `--max-speakers 1` still runs pyannote; use
`--single-speaker` to skip it.
The quality report is still generated. Changing modes with the same `--output`
reuses compatible Whisper checkpoints and rebuilds the final transcript.
Speaker matching and gap tolerance settings have no effect in this mode.

---

## Recommended first test

Do not start with a multi-hour recording.

First use a short representative sample containing:

- more than one speaker
- normal conversational pauses
- a few speaker changes
- the language you intend to process
- technical terminology if relevant

Example:

```bash
python transcribe_local.py test.mp3 \
  --language it \
  --output output/test \
  --chunk-minutes 10 \
  --context-seconds 15 \
  --max-speakers 6
```

---

## Recommended Italian recording command

```bash
python transcribe_local.py recording.mp3 \
  --language it \
  --output output/recording \
  --chunk-minutes 10 \
  --context-seconds 15 \
  --max-speakers 6
```

---

## Technical vocabulary / names

Whisper can be given an initial prompt containing expected terminology:

```bash
python transcribe_local.py recording.mp3 \
  --language it \
  --output output/recording \
  --chunk-minutes 10 \
  --context-seconds 15 \
  --max-speakers 6 \
  --prompt "Names and terminology: OpenAI, PostgreSQL, Kubernetes, Docker, React, TypeScript"
```

This is especially useful for:

- people's names
- company names
- product names
- acronyms
- programming languages
- technical terminology

Keep the prompt relevant to the recording rather than filling it with unrelated vocabulary.

---

# CLI reference

Run:

```bash
python transcribe_local.py --help
```

for the definitive CLI supported by your local script.

Help and argument validation run before NumPy, PyTorch, and Whisper are imported, so
`--help` does not need the inference packages or local model files.

Current options are:

| Option | Default | Purpose |
|---|---:|---|
| `audio` | required | Input audio or video file |
| `--output` | `output` | Output/job directory |
| `--language` | auto | Whisper language code such as `it`, `en`, `fr`, or `de` |
| `--prompt` | none | Initial Whisper prompt for names, acronyms, products, or technical terms |
| `--chunk-minutes` | `10` | Size of each committed crash-safe core chunk |
| `--context-seconds` | `15` | Context added before/after each core chunk |
| `--min-speakers` | unset | Minimum speakers passed to pyannote for each chunk |
| `--max-speakers` | unset | Maximum speakers passed to pyannote for each chunk |
| `--single-speaker` | off | Assume one speaker and skip pyannote |
| `--speaker-match-threshold` | `0.70` | Embedding cosine-similarity threshold for returning speakers |
| `--overlap-match-seconds` | `0.75` | Minimum shared speech overlap used for adjacent-chunk identity matching |
| `--speaker-gap-tolerance` | `1.0` | Maximum nearby gap used when assigning a Whisper word to a speaker |
| `--utterance-max-seconds` | `15.0` | Maximum generated utterance/subtitle duration |
| `--utterance-gap-seconds` | `1.5` | Silence gap that causes an utterance split |
| `--whisper-verbose` | off | Print Whisper segment output during decoding |
| `--force-whisper` | off | Recompute all Whisper chunk checkpoints |
| `--force-diarization` | off | Recompute all pyannote chunk checkpoints |
| `--rebuild-audio` | off | Recreate the normalized WAV |

---

# Chunking and checkpoints

Long recordings are divided into **external chunks**.

This is separate from the internal windowing already performed by Whisper and pyannote.

The purpose of external chunking is reliability.

Default configuration:

```text
core chunk:       10 minutes
context before:   15 seconds
context after:    15 seconds
```

Conceptually:

```text
                  committed core
              +----------------------+
09:45 ------ 10:00 ---------------- 20:00 ------ 20:15
 context               core                    context
```

Whisper checkpoints retain words from the full extended interval. Final assembly normally
uses words from each chunk's **core**, then matches words near adjacent boundaries.
Matching copies are kept once, including when slightly different timestamps put both
copies outside their respective cores. Consecutive core words with the same text
and separate timing intervals are preserved. Unmatched context words are not committed.

This helps avoid:

- repeated text at chunk boundaries
- clipped words near boundaries
- unstable speaker behavior exactly at a boundary

---

## Checkpoint layout

For an output folder such as:

```text
output/meeting/
```

the script produces checkpoint directories similar to:

```text
output/meeting/
├── work/
│   ├── audio_16k_mono.wav
│   ├── audio_16k_mono.meta.json
│   └── force_run.json (when a force flag is used)
│
└── checkpoints/
    ├── whisper/
    │   ├── chunk_000000.json
    │   ├── chunk_000001.json
    │   └── ...
    │
    └── diarization/
        ├── chunk_000000.json
        ├── chunk_000001.json
        └── ...
```

The `diarization/` checkpoint directory is created only for multi-speaker runs.

Every completed chunk is written immediately using an atomic JSON write. Whisper
checkpoints contain extended words so boundary reconciliation can be rebuilt without
rerunning inference.

That means a crash should normally lose at most the currently-running chunk rather than the whole transcription stage.

---

## Resume after interruption

If the process stops:

```text
00-10 min   complete
10-20 min   complete
20-30 min   complete
30-40 min   complete
40-50 min   interrupted
```

simply run the same command again.

Example:

```bash
python transcribe_local.py recording.mp3 \
  --language it \
  --output output/recording \
  --chunk-minutes 10 \
  --context-seconds 15 \
  --max-speakers 6
```

The script checks existing checkpoint signatures and required fields, then skips
compatible completed chunks. An interrupted `--force-whisper` or
`--force-diarization` pass also resumes its completed chunks.

You do **not** need a special `--resume` flag.

---

# Checkpoint compatibility

A checkpoint is reused only when its stored stage signature, chunk layout, and required
data fields match the current configuration. Incompatible older checkpoints are recomputed.

The Whisper stage signature includes information such as:

- normalized audio identity
- Whisper model identity
- language
- initial prompt
- chunk size
- context size
- important transcription settings

The diarization signature includes information such as:

- normalized audio identity
- local pyannote model directory identity
- minimum speaker constraint
- maximum speaker constraint
- chunk size
- context size
- exclusive diarization mode

This is important because stale results should not silently be mixed with a materially different run.

---

# Speaker reconciliation

A complication of external diarization chunks is that local pyannote labels are not guaranteed to mean the same person in each chunk.

For example:

```text
chunk 1:
SPEAKER_00 = Marco
SPEAKER_01 = Anna

chunk 2:
SPEAKER_00 = Anna
SPEAKER_01 = Marco
```

Simply concatenating local IDs would therefore be incorrect.

The script performs an additional global reconciliation stage.

---

## Strategy 1: shared context audio

Adjacent external chunks intentionally overlap in the context region.

That means the same physical speech can appear in both neighboring diarization calls.

The reconciliation algorithm uses overlapping speaker turns as its strongest identity evidence.

Default:

```text
--overlap-match-seconds 0.75
```

A local speaker with enough temporal overlap with a globally-mapped speaker from the neighboring chunk can inherit that global ID. When several speakers compete, the pipeline chooses the one-to-one assignment with the largest total overlap.

---

## Strategy 2: speaker embeddings

Shared context cannot help when someone disappears for multiple chunks and returns later.

Community-1 speaker embeddings are therefore used as a fallback.

Embeddings are compared with cosine similarity against the accumulated global speaker profiles.

Default:

```text
--speaker-match-threshold 0.70
```

This is a pipeline heuristic, not a universal speaker-recognition threshold.

### If one real person is split into several IDs

Try reducing the threshold:

```bash
--speaker-match-threshold 0.65
```

### If different people are incorrectly merged

Try increasing it:

```bash
--speaker-match-threshold 0.75
```

or:

```bash
--speaker-match-threshold 0.80
```

Do not tune this aggressively from one isolated error. Test against several representative sections of the recording.

---

# Speaker-count constraints

When using external chunks, avoid forcing the total number of meeting participants into every individual chunk.

Example:

```text
meeting participants overall: 6
speakers active in one 10-minute chunk: 2
```

Forcing:

```text
num_speakers = 6
```

for that chunk could degrade diarization.

A safer default is usually an upper bound:

```bash
--max-speakers 6
```

Leave `--min-speakers` unset unless you have a good reason to enforce it.

Use exact constraints only when they accurately describe what is present in each processed chunk.

---

# Word-level speaker attribution

Whisper is run with word timestamps enabled.

After diarization and global speaker reconciliation, each Whisper word is associated with the most suitable speaker turn.

The script primarily uses overlap between:

```text
Whisper word timestamp
```

and:

```text
exclusive pyannote speaker turn
```

If there is no direct overlap, nearby turns can still be considered up to:

```text
--speaker-gap-tolerance 1.0
```

Words that cannot be associated safely are labeled:

```text
UNKNOWN
```

---

# Utterance construction

Word-level results are grouped into readable speaker utterances.

An utterance is split when:

- the speaker changes
- the silence gap exceeds `--utterance-gap-seconds`
- the utterance would exceed `--utterance-max-seconds`

Defaults:

```text
maximum utterance duration: 15 seconds
split after silence:        1.5 seconds
```

These settings primarily affect presentation/subtitle readability rather than model inference.

---

# Output files

A completed job produces:

```text
transcript.txt
transcript.json
transcript.srt
transcript.vtt
words_with_speakers.json
speaker_reconciliation.json
quality_report.txt
quality_report.json
```

---

## `transcript.txt`

Human-readable speaker transcript.

Example:

```text
[00:00:01.320 --> 00:00:08.560] SPEAKER_00
Buongiorno a tutti, oggi parleremo del nuovo progetto.

[00:00:08.720 --> 00:00:14.110] SPEAKER_01
Perfetto. Volevo iniziare dalla parte relativa al database.
```

---

## `transcript.json`

Structured utterance-level transcript.

Each object contains information such as:

```json
{
  "speaker": "SPEAKER_00",
  "start": 1.32,
  "end": 8.56,
  "text": "Buongiorno a tutti, oggi parleremo del nuovo progetto."
}
```

---

## `words_with_speakers.json`

Fine-grained word-level output.

Useful for:

- custom transcript renderers
- downstream alignment
- debugging speaker assignment
- creating other subtitle formats
- analytics

Each word includes its timestamp and assigned global speaker.

---

## `speaker_reconciliation.json`

Debug/report file describing how local pyannote speaker identities were mapped to global speaker IDs.

This is the first file to inspect when speaker IDs appear inconsistent across chunk boundaries.

It also contains the final reconciled speaker turns.

In `--single-speaker` mode, it records the explicit one-speaker assumption and
contains no inferred diarization turns.

---

## `quality_report.txt` and `quality_report.json`

The report is written after every completed transcription. It lists the
recording length, word and utterance counts, per-speaker word counts, and
timestamped review cues for:

- words assigned `UNKNOWN` (multi-speaker mode)
- speaker IDs with fewer than 20 words (multi-speaker mode)
- groups of low Whisper word scores
- speaker matches with embedding similarity near the configured threshold

These are places to listen, not proven errors. Whisper word scores are not an
accuracy percentage. `quality_report.txt` shows up to 20 passages per category;
the JSON file contains all flagged passages.

To create or refresh the report for an existing completed job without running
Whisper or pyannote again:

```bash
python3 quality_report.py output/recording
```

This command reads the saved JSON outputs and Whisper checkpoint metadata.

---

## `transcript.srt`

Standard SRT subtitles containing speaker labels.

---

## `transcript.vtt`

WebVTT subtitles containing speaker labels.

---

# Recommended settings

## General meeting transcription

```bash
python transcribe_local.py meeting.mp3 \
  --language it \
  --output output/meeting \
  --chunk-minutes 10 \
  --context-seconds 15 \
  --max-speakers 6
```

---

## Unstable machine / minimize lost work

Use smaller checkpoints:

```bash
--chunk-minutes 5
```

This increases the number of model invocations and checkpoint files but reduces the maximum amount of work lost if a chunk fails.

---

## Stable machine / long recordings

Start with:

```text
--chunk-minutes 10
--context-seconds 15
```

These are the design defaults for the project.

Do not increase chunk size solely because a recording is long. Checkpoint size is primarily a reliability tradeoff.

---

## Known language

Specify it:

```bash
--language it
```

rather than relying on repeated auto-detection.

This gives the pipeline a known language configuration and makes checkpoint behavior deterministic with respect to language selection.

---

## Technical conversation

Use:

```bash
--prompt "Names and terminology: ..."
```

Include only terms genuinely likely to occur.

---

# Resume and recomputation behavior

Different changes require different levels of recomputation.

## Normal restart after a crash

Each job directory permits one writer at a time. Final exports are staged before
publication. The `job_state.json` file records when the current run has published
the full set of exports and provides the exact duration and chunk layout for the
quality report. Resume an interrupted job before regenerating its quality report
or using its new exports.

Run exactly the same command.

Compatible Whisper chunks are reused; multi-speaker runs also reuse compatible
diarization chunks.

---

## Re-run Whisper only

Use:

```bash
--force-whisper
```

Example:

```bash
python transcribe_local.py recording.mp3 \
  --language it \
  --output output/recording \
  --force-whisper
```

This starts a new Whisper recomputation after a completed run. If a forced run was
interrupted, the same command resumes its completed chunks.

---

## Re-run diarization only

This applies to multi-speaker runs. `--force-diarization` cannot be combined
with `--single-speaker`.

Use:

```bash
--force-diarization
```

Example:

```bash
python transcribe_local.py recording.mp3 \
  --language it \
  --output output/recording \
  --force-diarization
```

Whisper checkpoints can still be reused if compatible. An interrupted forced
diarization pass resumes its completed chunks.

---

## Rebuild normalized audio

Use:

```bash
--rebuild-audio
```

This recreates the 16 kHz mono PCM WAV from the original source. If the rebuilt
WAV is byte-identical, its compatible inference checkpoints remain reusable.

Normally this is unnecessary because the script stores source metadata and reuses a compatible normalized file.

---

## Change speaker reconciliation threshold only

Parameters such as:

```text
--speaker-match-threshold
--overlap-match-seconds
--speaker-gap-tolerance
--utterance-max-seconds
--utterance-gap-seconds
```

belong to post-processing rather than the heavy Whisper/pyannote inference checkpoints.

Therefore you can normally adjust them, rerun the same job, and let the valid model checkpoints be reused while final reconciliation/output is rebuilt.

This is particularly useful when tuning speaker identity matching.

---

# Audio normalization

Every source file is converted once by FFmpeg into:

```text
16 kHz
mono
PCM16 WAV
```

The normalized file is stored inside:

```text
OUTPUT/work/audio_16k_mono.wav
```

The script also writes metadata describing the original source.

If the original source identity still matches, the normalized file is reused.

This gives both Whisper and pyannote the same deterministic audio source and makes chunk boundaries sample-accurate.

---

# Troubleshooting

The commands below use Linux/macOS/WSL paths. On Windows, run
`setup_models.cmd` instead of `bash setup_models.sh` and use
`.venv\Scripts\python.exe` (or `venv\Scripts\python.exe`) instead of the
Unix `bin/python` path.

## `ModuleNotFoundError: No module named 'numpy'` (or another dependency)

The script is running with a Python interpreter that does not have the project
requirements installed. In WSL, check the interpreter explicitly:

```bash
command -v python3
venv/bin/python -m pip show numpy
```

If your environment is named `.venv`, use `.venv/bin/python` instead. Install
dependencies with that same interpreter, then use it to run transcription:

```bash
venv/bin/python -m pip install -r requirements.txt
venv/bin/python transcribe_local.py --help
```

Activating a virtual environment does not repair one that was copied or moved
from another directory. In that case, create a fresh environment in the current
project directory and reinstall `requirements.txt`; see
[Python environment setup](#python-environment-setup). Virtual environments
are local installation artifacts and are excluded from the public repository.

---

## `--help` takes several seconds

The current script parses `--help` before importing NumPy, Torch, Whisper, or
pyannote, so displaying help does not need the models or GPU. Confirm that you
are running the current `transcribe_local.py` and the intended Python
interpreter. A cold WSL startup can still add time before Python begins.

---

## `Required executable not found in PATH: ffmpeg`

FFmpeg is either not installed or not visible in your terminal's `PATH`.

Check:

```bash
ffmpeg -version
```

If that command fails, fix the FFmpeg installation before running the script.

---

## `Whisper Turbo model missing`

Expected path:

```text
models/whisper/large-v3-turbo.pt
```

Run the managed setup command:

```bash
bash setup_models.sh
```

For `--single-speaker`, use `bash setup_models.sh --whisper-only` instead;
that path does not require Hugging Face sign-in.

---

## `pyannote Community-1 model missing/incomplete`

Expected directory:

```text
models/pyannote/speaker-diarization-community-1/
```

Check that:

```text
config.yaml
```

exists and that the download completed. Run `bash setup_models.sh` to sign in,
check access, and finish the download.

---

## `Community-1 access has not been granted`

Hugging Face sign-in succeeded, but the account has not accepted this gated
model's conditions. Open the
[Community-1 model page](https://huggingface.co/pyannote/speaker-diarization-community-1)
with the same account, accept the conditions, then rerun
`bash setup_models.sh`. The saved sign-in will be reused. Use
`bash setup_models.sh --login` only if you need to switch accounts.

---

## `Hugging Face sign-in needs an interactive terminal`

Run `bash setup_models.sh` in a normal Linux/macOS/WSL terminal, or
`setup_models.cmd` in a Windows terminal. On first use, choose browser sign-in
and approve the displayed device code. The sign-in is saved in your user cache
for later setup runs; no token file belongs in this project.

---

## `Some pyannote model files are still Git LFS pointers`

If you cloned manually, fetch its large files with Git LFS. Otherwise, use the
managed setup command:

```bash
bash setup_models.sh
```

Then rerun the transcription command.

---

## Hugging Face / offline errors during normal use

Normal execution is deliberately offline.

Do not solve an incomplete local installation by removing the offline protection first.

Instead run `bash setup_models.sh --check`, then rerun
`bash setup_models.sh` if either model is incomplete.
Also confirm the project virtual environment has the dependencies from
`requirements.txt` installed.

The intent is that credentials and internet access are needed only during model acquisition.

---

## `Failed to launch Triton kernels, likely due to missing CUDA toolkit`

This is a warning about optional fast word-timestamp alignment; transcription
continues using a slower fallback. The message is generic. On this project's
Debian WSL test machine, the actual missing pieces were a C compiler and the
Python development headers. Run `bash setup_models.sh --check` to see whether
they are present. On Debian/Ubuntu, install them with:

```bash
sudo apt install gcc python3-dev
```

Use a matching `python3.X-dev` package if the virtual environment uses a
different Python version than the system default. Start a new transcription
process to verify the warning is gone. Completed chunk checkpoints are reused
when you rerun the same command.

---

## `Whisper 1/13` or a progress bar that starts at 0%

The progress bar is for the **current Whisper chunk**, not the whole recording.
With the default `--chunk-minutes 10`, a recording of about 2 hours 10 minutes
has 13 chunks. A checkpoint is written after each completed chunk; if the
process stops, rerun the same command and completed chunks will be skipped.

Whisper's internal bar can also finish below 100% (for example, at 97%) even
when the chunk has completed. Its decoder sometimes advances past silence or
suspected hallucinations without updating the bar. The reliable completion
signal is the following `Saved ... extended words -> .../chunk_XXXXXX.json`
line and the checkpoint file itself. If the next chunk starts, the previous
one finished. Once all chunks are done, the script continues with diarization
and writes the final transcript files in the job's output directory.

---

## CUDA is reported as unavailable

Test:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
```

If `torch.cuda.is_available()` is `False`, common causes include:

- CPU-only PyTorch installation
- incompatible PyTorch/CUDA package
- NVIDIA driver problems
- running inside an environment without GPU access

The script will still work on CPU, but performance will be much lower.

---

## CUDA out-of-memory

The design already minimizes VRAM pressure by not intentionally keeping Whisper and pyannote loaded simultaneously.

If OOM still occurs:

1. close other GPU-heavy applications
2. confirm no other Python process is occupying VRAM
3. verify the expected GPU is being used
4. restart the Python process to clear abandoned GPU allocations
5. test on a short recording

External `--chunk-minutes` mostly controls checkpoint granularity; it is not guaranteed to reduce every model's peak internal GPU requirement because both Whisper and pyannote also perform their own internal processing.

---

## Transcript repeats at chunk boundaries

The script compares nearby words from both extended chunks, keeps matched words once,
and recovers a match when both timestamps fall just outside their respective cores.

If repetition occurs, inspect:

```text
checkpoints/whisper/
```

and verify that you are running the current script with consistent chunk parameters.

Avoid manually concatenating raw model output from extended/context regions.

---

## Speaker changes near chunk boundaries are wrong

Inspect:

```text
speaker_reconciliation.json
```

Then consider:

```text
--overlap-match-seconds
--speaker-match-threshold
```

Do not immediately force an exact speaker count for every chunk.

---

## Same person gets multiple speaker IDs

Try a slightly lower embedding threshold:

```bash
--speaker-match-threshold 0.65
```

Compare the result with the default `0.70`.

---

## Different people are merged into one speaker ID

Try:

```bash
--speaker-match-threshold 0.75
```

or:

```bash
--speaker-match-threshold 0.80
```

Use representative recordings when deciding which threshold is appropriate.

---

## Words are assigned to `UNKNOWN`

A word can become `UNKNOWN` when no diarization turn overlaps or lies close enough to it.

You may cautiously increase:

```bash
--speaker-gap-tolerance
```

but a large value can assign speech to an incorrect nearby speaker.

---

## Subtitles are too long

Reduce:

```bash
--utterance-max-seconds
```

and/or:

```bash
--utterance-gap-seconds
```

These changes can be tested without rerunning the heavy model stages when their checkpoints remain compatible.

---

# Performance and quality notes

## 8 GB VRAM design

The central multi-speaker GPU design is:

```text
Whisper -> unload -> clear GPU -> pyannote -> unload -> CPU post-processing
```

This avoids intentionally holding both neural pipelines in GPU memory at once.
With `--single-speaker`, the pyannote stage is skipped entirely.

---

## Chunk size is a reliability setting

The default external 10-minute chunk is mainly about:

- resumability
- checkpoint frequency
- limiting rework after a crash

It is not the same as the model's own internal inference window.

---

## Context exists for quality, not output duplication

The 15-second context around each core chunk gives the models surrounding audio near boundaries.

Core words are preferred. A matched word just outside both cores can be recovered at
the boundary, while duplicate context copies are removed.

---

## Speaker IDs are anonymous

Names such as:

```text
SPEAKER_00
SPEAKER_01
```

are stable pipeline IDs, not automatic real-person identification.

The pipeline does not know that `SPEAKER_00` is "Marco" unless a separate application/user mapping is added.

---

## Speaker embeddings are used for stitching

The speaker embeddings are used to maintain identity across external chunks.

They should not be interpreted as an infallible biometric identification mechanism.

Acoustics, noise, microphones, overlapping speech, short speaking duration, and similar voices can all affect matching quality.

---

# Suggested validation workflow

For a multi-speaker recording, before processing hours of audio:

1. Create the virtual environment.
2. Confirm FFmpeg.
3. Confirm CUDA-enabled PyTorch.
4. Accept the Community-1 conditions and run `bash setup_models.sh` (or `setup_models.cmd` on Windows).
5. Run `bash setup_models.sh --check` to verify both models.
6. Run a 2-5 minute sample.
7. Inspect `transcript.txt`.
8. Inspect `words_with_speakers.json`.
9. Inspect `speaker_reconciliation.json`.
10. Test a file long enough to cross at least one external chunk boundary.
11. Interrupt a run deliberately after at least one completed chunk.
12. Rerun the same command and verify completed chunks are skipped.
13. Only then move to multi-hour recordings.

This validates both inference quality and the crash-resume design.
For a known single-speaker recording, use `--whisper-only` during setup and
`--single-speaker` during transcription, then inspect the transcript and
`quality_report.txt`.

---

# Development notes

## Important source files

```text
transcribe_local.py
quality_report.py
job_state.py
local_models.py
setup_models.py
setup_models.sh
setup_models.cmd
test_transcribe_local.py
requirements.txt
README.md
```

## Main implementation areas

`transcribe_local.py` currently contains:

- local model validation
- Git LFS pointer detection
- atomic JSON checkpoint writes
- FFmpeg normalization
- sample-accurate WAV interval reading
- external chunk construction
- Whisper checkpoint generation
- pyannote checkpoint generation
- explicit GPU cleanup
- speaker overlap matching
- speaker embedding matching
- global speaker profile construction
- word-to-speaker assignment
- utterance generation
- TXT writer
- JSON writers
- SRT writer
- WebVTT writer
- CLI validation

Run the focused regression suite with:

```bash
python -m unittest -v test_transcribe_local
```

These tests use lightweight stand-ins for Torch and Whisper and do not download or
load the neural models. A full multi-speaker transcription test still needs
FFmpeg and both local model directories; `--single-speaker` needs only Whisper Turbo.

---

# Current default tuning values

```text
Whisper model:                  Turbo
Audio sample rate:              16000 Hz
External core chunk:            10 minutes
Context:                        15 seconds each side
Embedding speaker threshold:    0.70
Shared-overlap requirement:     0.75 seconds
Word/speaker gap tolerance:     1.0 second
Max utterance duration:         15 seconds
Utterance silence split:        1.5 seconds
```

These values are starting points, especially the speaker matching threshold.

---

# Useful commands

## Show CLI help

```bash
python transcribe_local.py --help
```

## Test CUDA

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Check FFmpeg

```bash
ffmpeg -version
```

## Check package versions

```bash
python -c "import torch, numpy, whisper, pyannote.audio; print('torch', torch.__version__); print('numpy', numpy.__version__); print('pyannote', pyannote.audio.__version__)"
```

## Force only transcription recomputation

```bash
python transcribe_local.py recording.mp3 --output output/recording --force-whisper
```

## Force only diarization recomputation

```bash
python transcribe_local.py recording.mp3 --output output/recording --force-diarization
```

---

# Operational checklist

Before a long unattended job, confirm:

- [ ] virtual environment is active
- [ ] correct `requirements.txt` is installed
- [ ] if using a GPU, `torch.cuda.is_available()` is `True`
- [ ] FFmpeg is available
- [ ] `models/whisper/large-v3-turbo.pt` exists
- [ ] for multi-speaker runs, Community-1 exists and contains actual model files
- [ ] input file is readable
- [ ] output folder is unique/appropriate for the job
- [ ] correct language is supplied when known
- [ ] `--max-speakers` is a realistic upper bound if used
- [ ] short representative test has already succeeded
- [ ] enough disk space exists for normalized WAV plus checkpoints/results

---

# Privacy

Once model acquisition has been completed successfully, this pipeline is intended to run locally.

Normal transcription does not need to upload the recording to OpenAI, Hugging Face, or another transcription service.

The audio, checkpoints, and generated transcripts remain on the machine unless you move or upload them yourself.

---

# Status

The automated suite covers checkpoint handling, chunk boundary merging,
single-speaker mode, model setup paths, and quality report generation. The
pipeline has also been exercised locally on short and multi-hour audio.
Recognition and speaker quality still depend on the recording. For long or
important jobs, review representative passages, speaker changes, and the
timestamped cues in `quality_report.txt`.

---

# Public GitHub repository

The public project consists of `.gitignore`, `LICENSE`, `README.md`, `requirements.txt`,
`transcribe_local.py`, `quality_report.py`, `setup_models.py`, `setup_models.sh`,
`setup_models.cmd`, and `test_transcribe_local.py`.

Keep `venv/`, `.venv/`, `models/`, `output/`, caches, credentials, and input
audio/video files out of the repository. `output/` can contain full transcripts,
the normalized audio, and metadata with an absolute path to the original input.
The included `.gitignore` excludes these local files. The local sample MP3 is
also excluded; publish sample audio only when you have
permission to redistribute it and have documented its license/source.

The model paths in the scripts are relative to the project directory. Model
setup creates that layout on each user's computer, so no personal machine path
or Hugging Face token needs to be committed.

---

# License

The source code and documentation in this repository are licensed under the
[MIT License](LICENSE). Downloaded models and user-provided audio are separate
materials and remain subject to their own terms.
