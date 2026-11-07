import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA = Path(os.environ.get("MEDIA_SORTER_DATA", ROOT / "data"))
MODELS = Path(os.environ.get("MEDIA_SORTER_MODELS", ROOT / "models"))
DB_PATH = Path(os.environ.get("MEDIA_SORTER_DB", DATA / "library.db"))
THUMBS = DATA / "thumbs"
FACES = DATA / "faces"

CLUSTER_THRESHOLD = float(os.environ.get("MEDIA_SORTER_THRESHOLD", "0.50"))
ORT_THREADS = int(os.environ.get("MEDIA_SORTER_THREADS", "4"))
DET_SIZE = int(os.environ.get("MEDIA_SORTER_DET_SIZE", "640"))
MAX_SIDE = int(os.environ.get("MEDIA_SORTER_MAX_SIDE", "1600"))
DET_THRESHOLD = float(os.environ.get("MEDIA_SORTER_DET_THRESHOLD", "0.5"))

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif", ".avif"}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv", ".mpg", ".mpeg", ".webm"}


def ensure_dirs():
    for d in (DATA, THUMBS, FACES, MODELS):
        d.mkdir(parents=True, exist_ok=True)
