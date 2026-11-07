# Contributing to Media Sorter

Media Sorter is a local, offline photo/video gallery that clusters faces into people. It runs entirely on your laptop's CPU — nothing is uploaded.

## Setup

1. Install Python 3.11 or 3.12 and add it to PATH.
2. Double-click `setup.bat` — creates the venv on `D:` and installs dependencies.
3. Double-click `models.bat` — downloads AI models (~265 MB, one-time).
4. Double-click `scan.bat` — indexes your photos/videos (prompts for folder on first run).
5. Double-click `serve.bat` — opens the gallery at `http://127.0.0.1:8000/`.

## Project layout

```
app/            FastAPI app: scanning, face engine, clustering, web server
templates/      Jinja2 HTML templates for the gallery UI
tools/          Threshold sweep and other diagnostics
tests/          pytest smoke tests
data/           SQLite DB, thumbnails, face crops (gitignored, local only)
models/         ONNX models (gitignored, downloaded via models.bat)
```

## Running tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Age estimates

Age is a FairFace 9-class estimate, not a fact. It can be wrong on low-res, blurry, profile or sunglasses faces. Manual overrides exist per-face and per-cluster — do not "fix" the model by hardcoding ages.

## Pull requests

- Keep everything on `D:`; nothing should be written to `C:`.
- Add tests under `tests/` for new scanning/clustering behavior.
- Update README if you change CLI flags or batch-file behavior.
