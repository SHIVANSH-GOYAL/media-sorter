# Media Sorter

Local, offline photo/video library that groups pictures by **who is in them**, their **estimated age band**, and **when they were taken**. Everything runs on this machine's CPU. Nothing is uploaded.

## Clone & run

```powershell
git clone https://github.com/<your-username>/media-sorter.git
cd media-sorter
```

Then double-click `setup.bat` → `models.bat` → `scan.bat` → `serve.bat`. Nothing is written to the system drive.

## What it does

- Scans a folder recursively and indexes photos and videos into SQLite.
- Detects faces (SCRFD), computes 512-d ArcFace embeddings (`w600k_r50`), and clusters them into people incrementally.
- Estimates an age band and gender per face with FairFace (9 bands: 0-2, 3-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70+).
- Serves a local gallery with filters for person, age band, date range, media type and single-person media, plus lightbox view.
- Lets you rename people, view "Unsorted" singleton faces, and re-run clustering.
- Re-scans incrementally: unchanged files are skipped by size + mtime, exact copies are skipped by SHA-1.

## Setup (Windows)

### One-time setup (double-click)

Run **`setup.bat`** — it creates the Python virtualenv on the data drive and installs all dependencies. You only need to do this once.

### Download models (about 265 MB)

Double-click **`models.bat`** — it downloads and extracts the face detection, embedding and age models automatically. One-time only.

## Everyday use (double-click)

| Task | File | What it does |
|------|------|--------------|
| Index photos | **`scan.bat [folder]`** | Scans a folder (default `D:\Media`) and indexes photos/videos. First run is slow; re-scans skip unchanged files. |
| Open gallery | **`serve.bat`** | Starts the local gallery at `http://127.0.0.1:8000/`. Close the window to stop. |
| Show counts | **`stats.bat`** | Prints database stats: files, faces, people, unsorted faces. |

### Age estimates — read this

Age is estimated per face with the FairFace model (9 bands: 0-2, 3-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70+). It is **an estimate, not a fact**:

- FairFace can misclassify low-resolution, blurry, profile, sunglasses or extreme-angle faces.
- A single face may land in the wrong band. This is a known FairFace limitation on some faces.
- Once a face joins a cluster with multiple photos, the cluster averages the age predictions across all its faces, so one wrong guess gets washed out.
- The cluster-level age band is what shows in the gallery; individual face ages are still stored but are less reliable.

If an age looks wrong, it's the model's estimate — not a bug in the app.

### Advanced commands (PowerShell)

These need the virtualenv activated (`.\.venv\Scripts\Activate.ps1`):

```powershell
.\.venv\Scripts\python.exe -m app recluster --threshold 0.50
.\.venv\Scripts\python.exe -m app scan "D:\Media" --limit 50          # test run
.\.venv\Scripts\python.exe -m app scan "D:\Media" --no-videos
.\.venv\Scripts\python.exe -m app scan "D:\Media" --no-faces          # files + thumbs only
.\.venv\Scripts\python.exe -m app pytest -q
.\.venv\Scripts\python.exe -m app refresh "D:\Media"
.\.venv\Scripts\python.exe -m app video-scan "D:\Media"
.\.venv\Scripts\python.exe -m app watch "D:\Media" --interval 60
```

`serve` options: `--host` (default `127.0.0.1`), `--port` (default `8000`).

Videos stream with HTTP range support, so they seek while playing. Browsers play `.mp4`; `.mpg` / MPEG-PS files usually cannot be decoded by Chrome or Edge, so the viewer shows a warning and an "Open original" link for every item.

## Gallery

- **People rail**: the 60 largest clusters, one card each with a representative face. Type a new name and press Save.
- **All people page**: `/people` lists every person (60 per page) with search by name, rename forms and a link to their photos. Reachable from "All N people" in the rail header.
- **Unsorted**: faces whose cluster has only one photo. They become a named person automatically once a second matching face arrives.
- **Filters**: person, age band, from/to dates, photo/video type and "only this person" combine with AND. Pagination is newest first and shows 72 items per page.
- **Identity matching**: assignment uses both the normalized cluster centroid and up to 12 strongest face exemplars. `recluster` now runs agglomerative (order-independent) clustering instead of a single greedy pass, so the same person across age gaps, poses and environments merges into one cluster instead of several duplicates. Cosine-distance gate is `0.50` by default (was `0.45`); typical useful range is 0.45-0.55. Lower = stricter (more, smaller people); higher = fewer clusters but more risk of merging two different people. After changing it, run `recluster` so existing faces are regrouped.
- **Age estimates**: after a face joins a cluster, the cluster averages the age predictions across all its faces, so one person's band stays consistent across angles and lighting. It is still an estimate and can be wrong on individual low-res or profile faces — see "Age estimates — read this" above.
- **Fix age from the gallery**: each photo card shows age band chips. Below the chips there's a "Fix age for all faces here" dropdown — pick the right band and hit **Set** to override the model estimate for every face in that photo at once. You can also fix age per person (`/person/{id}`) or per face from a person's page.
- **Readiness check**: `http://127.0.0.1:8000/api/ready` reports whether the venv, models and database are all present — useful if the gallery looks empty or faces aren't detected.
- **Memory profile**: scan results include `peak_python_mb`, the peak Python allocation measured during that scan. It excludes native ONNX/OpenCV allocations, so it is a lower-bound diagnostic, not total process RSS. Real RAM use is higher; ONNX Runtime and OpenCV allocate native memory outside Python's tracker.
- **Recluster**: recomputes all clusters with the current threshold. Custom names are carried over to the new cluster with most overlap. Run it once after changing the threshold; new scans use the updated matching automatically.
- Clicking a card opens the original photo or plays the video.
- **Ask**: search in plain language for people together or separately, places, dates, age bands, photos/videos and favourites.
- **People review**: open a person to rename, merge another cluster into it, or detach an incorrect face.
- **Duplicates**: view-only near-duplicate groups; nothing is deleted automatically.
- **Favourites**: star items and browse them from the navigation bar. Saved searches are available from the Ask page.

## Offline natural-language search

The Ask page always has a deterministic local parser, so searches work with the AI disabled:

```powershell
$env:MEDIA_SORTER_LLM = "off"
\.venv\Scripts\python.exe -m app serve
```

When Ollama is available, the configured local model may turn less predictable wording into a structured search plan. It never writes SQL and its people/place values are validated before use. Use `MEDIA_SORTER_LLM=off` when you want immediate results.

The `refresh` command backfills metadata for existing files. `video-scan` samples existing videos for faces. `watch` periodically rescans configured folders; it does not delete files.

## How clustering works

Each new face embedding is compared by cosine distance to every cluster centroid:

- distance `<= threshold` → joins that cluster and the centroid is updated as a running mean.
- otherwise → starts a new cluster ("Unsorted" until it has two faces).

`recluster` regroups *all* existing faces with agglomerative (order-independent) clustering: it repeatedly merges the pair of clusters with the highest centroid similarity, but only while the merged centroid stays within `threshold` of every member of both clusters. This anti-chaining guard stops unrelated people from being pulled together through borderline faces, while strongest-first merging lets the same person across age gaps, poses and environments come together even if a first-pass scan split them.

Default threshold is `0.50`. Lower is stricter (more, smaller people); higher merges more. Typical useful range is 0.45-0.55. After changing it, run `recluster` so existing faces are regrouped.

To compare thresholds without touching the database, sweep the stored embeddings:

```powershell
.\.venv\Scripts\python.exe tools\threshold_sweep.py --min 0.30 --max 0.65 --step 0.05
```

## What is stored

```
data\library.db    SQLite: media, faces (embedding blobs), clusters, meta
data\thumbs\       480 px JPEG previews, one per file
data\faces\        144 px JPEG face crops
models\            ONNX models
```

`media` keeps the absolute path, size, mtime, SHA-1, EXIF capture time (fallback: file mtime), dimensions and kind. `faces` keeps the box, detection score, 512-d embedding, cluster id, age estimate/band, gender and crop path.

Deleting `data\` rebuilds everything on the next scan. Deleting one thumbnail only breaks that preview.

## Privacy

Fully offline. The scanner and server make no network calls. Models are downloaded once during setup; after that the folder can run with networking off.

## v1 limits / v2 ideas

- Videos get a thumbnail and mtime only - no face tracking inside video.
- Only exact duplicates (SHA-1) are collapsed; near-duplicates and burst shots are kept.
- No automatic file watching: run `scan` again, or add a scheduled task.
- Clusters cannot be manually split/merged in the UI yet; tune the threshold and recluster.
- Age is a coarse 9-class estimate and is wrong on some low-light, sunglasses or profile shots.
- HEIC/HEIF needs `pillow-heif` (installed); unusual RAW formats are skipped.
- No cloud sync, mobile app or multi-user auth. The server binds to localhost by default.
