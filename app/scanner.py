import hashlib
import json
import os
import time
import tracemalloc
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from . import clustering, config
from . import db as dbmod

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except Exception:
    pass

SKIP_DIRS = {"$recycle.bin", "system volume information", "node_modules", ".git", "__pycache__"}

EXIF_DATETIME_ORIGINAL = 36867
EXIF_DATETIME_DIGITIZED = 36868
EXIF_DATETIME = 306
EXIF_IFD = 0x8769
EXIF_MAKE = 271
EXIF_MODEL = 272


def _age_band(age):
    for limit, label in ((2, "0-2"), (9, "3-9"), (19, "10-19"), (29, "20-29"),
                         (39, "30-39"), (49, "40-49"), (59, "50-59"), (69, "60-69")):
        if age <= limit:
            return label
    return "70+"


def sha1_file(path, chunk=1 << 20):
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _parse_exif_dt(value):
    if not value or not isinstance(value, str):
        return None
    value = value.strip().replace("\x00", "")
    if not value or value.startswith("0000"):
        return None
    try:
        return datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None


def exif_taken_at(img, path):
    taken = None
    try:
        exif = img.getexif()
        for tag in (EXIF_DATETIME_ORIGINAL, EXIF_DATETIME_DIGITIZED, EXIF_DATETIME):
            taken = _parse_exif_dt(exif.get(tag))
            if taken:
                break
        if not taken:
            try:
                sub = exif.get_ifd(EXIF_IFD)
                for tag in (EXIF_DATETIME_ORIGINAL, EXIF_DATETIME_DIGITIZED):
                    taken = _parse_exif_dt(sub.get(tag))
                    if taken:
                        break
            except Exception:
                pass
    except Exception:
        pass
    if taken is None:
        taken = datetime.fromtimestamp(path.stat().st_mtime)
    return taken.strftime("%Y-%m-%d %H:%M:%S")


def exif_camera(img):
    try:
        exif = img.getexif()
        make = str(exif.get(EXIF_MAKE) or "").strip().replace("\x00", "")
        model = str(exif.get(EXIF_MODEL) or "").strip().replace("\x00", "")
    except Exception:
        return None
    if not model and not make:
        return None
    if make and model and not model.lower().startswith(make.lower()):
        return f"{make} {model}"
    return model or make


def dhash(img, size=8):
    gray = img.convert("L").resize((size + 1, size), Image.LANCZOS)
    pixels = np.asarray(gray, dtype=np.int16)
    bits = (pixels[:, 1:] > pixels[:, :-1]).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def place_for(path, root):
    path = Path(path)
    parts = []
    try:
        rel = path.parent.relative_to(Path(root))
        parts = [p for p in rel.parts if p]
    except Exception:
        parts = [path.parent.name] if path.parent.name else []
    if not parts:
        parts = [Path(root).name]
    return " / ".join(parts[-5:])


def iter_files(root):
    root = Path(root)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d
            for d in dirnames
            if not d.startswith(".") and d.lower() not in SKIP_DIRS
        )
        for name in sorted(filenames):
            ext = os.path.splitext(name)[1].lower()
            if ext in config.IMAGE_EXT:
                yield Path(dirpath) / name, "image"
            elif ext in config.VIDEO_EXT:
                yield Path(dirpath) / name, "video"


def _save_thumb(img, dest, size=(480, 480)):
    copy = img.copy()
    copy.thumbnail(size, Image.LANCZOS)
    if copy.mode not in ("RGB", "L"):
        copy = copy.convert("RGB")
    copy.save(dest, "JPEG", quality=82, optimize=True)


def _video_probe(path):
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        duration = frames / fps if fps > 0 else 0.0
        return float(duration)
    finally:
        cap.release()


def _video_thumb(path, dest):
    cap = cv2.VideoCapture(str(path))
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, 1000)
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_MSEC, 0)
            ok, frame = cap.read()
        if not ok or frame is None:
            return None
        h, w = frame.shape[:2]
        scale = min(480 / max(w, h), 1.0)
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(
            dest, "JPEG", quality=82, optimize=True
        )
        return (w, h)
    finally:
        cap.release()


def _remove_files(conn, media_id):
    row = conn.execute("SELECT thumb FROM media WHERE id = ?", (media_id,)).fetchone()
    if row and row["thumb"]:
        p = config.DATA / row["thumb"]
        if p.exists():
            p.unlink(missing_ok=True)
    rows = conn.execute("SELECT thumb FROM faces WHERE media_id = ?", (media_id,)).fetchall()
    for r in rows:
        if r["thumb"]:
            p = config.DATA / r["thumb"]
            if p.exists():
                p.unlink(missing_ok=True)


def _insert_face(conn, media_id, face, threshold, frame_time=None):
    cur = conn.execute(
        "INSERT INTO faces(media_id, bbox, det_score, embedding, age, age_band, gender, age_conf, frame_time) "
        "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            media_id,
            json.dumps([round(v, 2) for v in face["bbox"]]),
            face["det_score"],
            face["embedding"].astype(np.float32).tobytes(),
            round(face["age"], 1),
            face["age_band"],
            face["gender"],
            round(face.get("age_conf") or 0.0, 4),
            frame_time,
        ),
    )
    face_id = int(cur.lastrowid)
    cid = clustering.assign(conn, face["embedding"], threshold)
    conn.execute("UPDATE faces SET cluster_id = ? WHERE id = ?", (cid, face_id))
    consensus = conn.execute(
        "SELECT age, age_band, age_conf FROM faces WHERE cluster_id = ? AND age IS NOT NULL "
        "ORDER BY age_conf DESC, id DESC LIMIT 40", (cid,)
    ).fetchall()
    if consensus:
        weights = np.asarray([max(0.1, float(r["age_conf"] or 0.0)) for r in consensus], dtype=np.float32)
        ages = np.asarray([float(r["age"]) for r in consensus], dtype=np.float32)
        consensus_age = float(np.average(ages, weights=weights))
        consensus_band = _age_band(consensus_age)
        conn.execute("UPDATE faces SET age = ?, age_band = ? WHERE cluster_id = ?", (round(consensus_age, 1), consensus_band, cid))
    face_rel = f"faces/{face_id}.jpg"
    face_img = Image.fromarray(cv2.cvtColor(face["aligned"], cv2.COLOR_BGR2RGB))
    face_img = face_img.resize((144, 144), Image.LANCZOS)
    face_img.save(config.DATA / face_rel, "JPEG", quality=85)
    conn.execute("UPDATE faces SET thumb = ? WHERE id = ?", (face_rel, face_id))
    return face_id


def _index_image(conn, path, media_id, engine, threshold, root):
    with Image.open(path) as raw:
        taken_at = exif_taken_at(raw, path)
        width, height = raw.size
        camera = exif_camera(raw)
        try:
            orientation = raw.getexif().get(274, 1)
        except Exception:
            orientation = 1
        raw.draft("RGB", (config.MAX_SIDE, config.MAX_SIDE))
        img = ImageOps.exif_transpose(raw)
        if orientation in (5, 6, 7, 8):
            width, height = height, width
        img = img.convert("RGB")
        thumb_rel = f"thumbs/{media_id}.jpg"
        _save_thumb(img, config.DATA / thumb_rel)
        hash_value = dhash(img)

        work = img
        if max(width, height) > config.MAX_SIDE:
            ratio = config.MAX_SIDE / max(width, height)
            work = img.resize((max(1, int(width * ratio)), max(1, int(height * ratio))), Image.LANCZOS)
        arr = np.asarray(work)
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    conn.execute(
        "UPDATE media SET taken_at = ?, width = ?, height = ?, thumb = ?, camera = ?, "
        "dhash = ?, place = ?, root = ? WHERE id = ?",
        (taken_at, width, height, thumb_rel, camera, hash_value, place_for(path, root), str(root), media_id),
    )

    face_count = 0
    if engine is not None:
        for face in engine.analyze(bgr):
            _insert_face(conn, media_id, face, threshold)
            face_count += 1
    return face_count


def _video_frames(path, interval, max_frames):
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        duration = frames / fps if fps > 0 else 0.0
        if duration <= 0:
            return
        step = max(1.0, float(interval))
        times = []
        t = step / 2
        while t < duration and len(times) < max_frames:
            times.append(t)
            t += step
        if not times:
            times = [0.0]
        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            if ok and frame is not None:
                yield t, frame
    finally:
        cap.release()


def _video_faces(conn, media_id, path, engine, threshold, interval, max_frames):
    accepted = []
    count = 0
    for t, frame in _video_frames(path, interval, max_frames):
        h, w = frame.shape[:2]
        min_side = max(24.0, 0.035 * max(h, w))
        for face in engine.analyze(frame):
            x1, y1, x2, y2 = face["bbox"]
            if (x2 - x1) < min_side or (y2 - y1) < min_side:
                continue
            emb = face["embedding"]
            if any(float(np.dot(emb, other)) > 0.55 for other in accepted):
                continue
            accepted.append(emb)
            _insert_face(conn, media_id, face, threshold, frame_time=round(t, 2))
            count += 1
    return count


def _update_video(conn, path, media_id, st, root, engine, threshold, do_faces, interval, max_frames):
    thumb_rel = f"thumbs/{media_id}.jpg"
    duration = _video_probe(path)
    dims = _video_thumb(path, config.DATA / thumb_rel)
    taken_at = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        "UPDATE media SET taken_at = ?, duration = ?, place = ?, root = ? WHERE id = ?",
        (taken_at, duration, place_for(path, root), str(root), media_id),
    )
    if dims:
        conn.execute(
            "UPDATE media SET thumb = ?, width = ?, height = ?, error = NULL WHERE id = ?",
            (thumb_rel, dims[0], dims[1], media_id),
        )
    else:
        conn.execute(
            "UPDATE media SET error = ? WHERE id = ?",
            ("video thumbnail failed", media_id),
        )
    if do_faces and engine is not None and dims:
        return _video_faces(conn, media_id, path, engine, threshold, interval, max_frames)
    return 0


def _as_folders(folders):
    if isinstance(folders, (str, Path)):
        return [Path(folders)]
    return [Path(f) for f in folders]


def scan(
    folders,
    engine=None,
    limit=None,
    include_videos=True,
    video_faces=False,
    interval=6.0,
    max_frames=120,
    progress=True,
):
    config.ensure_dirs()
    tracemalloc.start()
    dbmod.init_db()
    roots = _as_folders(folders)
    for root in roots:
        if not root.exists():
            raise FileNotFoundError(root)

    stats = {
        "processed": 0,
        "skipped": 0,
        "duplicates": 0,
        "errors": 0,
        "faces": 0,
        "video_faces": 0,
        "videos": 0,
        "peak_python_mb": 0.0,
    }
    seen = 0
    started = time.monotonic()

    with dbmod.db() as conn:
        threshold = dbmod.threshold(conn)
        for root in roots:
            for path, kind in iter_files(root):
                if limit is not None and seen >= limit:
                    break
                seen += 1
                try:
                    st = path.stat()
                except OSError:
                    stats["errors"] += 1
                    continue
                existing = conn.execute(
                    "SELECT id, size, mtime, error FROM media WHERE path = ?", (str(path),)
                ).fetchone()
                if existing and existing["size"] == st.st_size and abs(existing["mtime"] - st.st_mtime) < 1:
                    stats["skipped"] += 1
                    continue
                if existing:
                    _remove_files(conn, existing["id"])
                    conn.execute("DELETE FROM media WHERE id = ?", (existing["id"],))
                    conn.commit()

                if kind == "video" and not include_videos:
                    continue

                media_id = None
                try:
                    sha1 = sha1_file(path)
                    dup = conn.execute(
                        "SELECT id FROM media WHERE sha1 = ? LIMIT 1", (sha1,)
                    ).fetchone()
                    if dup:
                        stats["duplicates"] += 1
                        continue

                    cur = conn.execute(
                        "INSERT INTO media(path, size, mtime, sha1, kind, indexed_at, root) "
                        "VALUES(?, ?, ?, ?, ?, ?, ?)",
                        (
                            str(path),
                            st.st_size,
                            st.st_mtime,
                            sha1,
                            kind,
                            datetime.now().astimezone().isoformat(timespec="seconds"),
                            str(root),
                        ),
                    )
                    media_id = int(cur.lastrowid)

                    if kind == "image":
                        stats["faces"] += _index_image(conn, path, media_id, engine, threshold, root)
                    else:
                        got = _update_video(
                            conn, path, media_id, st, root, engine, threshold,
                            video_faces, interval, max_frames,
                        )
                        stats["video_faces"] += got
                        stats["videos"] += 1

                    stats["processed"] += 1
                    if stats["processed"] % 10 == 0:
                        conn.commit()
                except Exception as exc:
                    stats["errors"] += 1
                    try:
                        if media_id is not None:
                            _remove_files(conn, media_id)
                            conn.execute("DELETE FROM media WHERE id = ?", (media_id,))
                        conn.execute(
                            "INSERT OR IGNORE INTO media(path, size, mtime, indexed_at, error) "
                            "VALUES(?, ?, ?, ?, ?)",
                            (
                                str(path),
                                st.st_size,
                                st.st_mtime,
                                datetime.now().astimezone().isoformat(timespec="seconds"),
                                f"{type(exc).__name__}: {exc}"[:500],
                            ),
                        )
                    except Exception:
                        pass

                elapsed = time.monotonic() - started
                stats["peak_python_mb"] = round(tracemalloc.get_traced_memory()[1] / (1024 * 1024), 2)
                done = stats["processed"] + stats["skipped"] + stats["duplicates"] + stats["errors"]
                if progress and (done % 25 == 0 or done == 1):
                    rate = stats["processed"] / elapsed * 60 if elapsed > 0 else 0
                    print(
                        f"[{done}] indexed={stats['processed']} skipped={stats['skipped']} "
                        f"dup={stats['duplicates']} err={stats['errors']} "
                        f"faces={stats['faces']} | {rate:.0f} photos/min",
                        flush=True,
                    )
        conn.commit()
        dbmod.set_meta(conn, "last_scan", datetime.now().astimezone().isoformat(timespec="seconds"))
        dbmod.set_roots(conn, dbmod.roots(conn) + [str(r) for r in roots])
        stats["clusters"] = conn.execute(
            "SELECT COUNT(*) AS c FROM clusters WHERE n >= 2"
        ).fetchone()["c"]
        stats["unsorted_faces"] = conn.execute(
            "SELECT COUNT(*) AS c FROM faces f JOIN clusters c ON c.id = f.cluster_id WHERE c.n = 1"
        ).fetchone()["c"]

    stats["elapsed_s"] = round(time.monotonic() - started, 1)
    stats["per_minute"] = round(
        stats["processed"] / stats["elapsed_s"] * 60, 1
    ) if stats["elapsed_s"] > 0 else 0.0
    stats["peak_python_mb"] = round(tracemalloc.get_traced_memory()[1] / (1024 * 1024), 2)
    tracemalloc.stop()
    return stats


def _root_for(path, roots):
    path = Path(path)
    best = None
    for r in roots:
        try:
            path.relative_to(r)
            if best is None or len(str(r)) > len(str(best)):
                best = Path(r)
        except ValueError:
            continue
    return best


def refresh(folders=None, progress=True, force=False):
    config.ensure_dirs()
    dbmod.init_db()
    started = time.monotonic()
    stats = {"images": 0, "videos": 0, "errors": 0}
    with dbmod.db() as conn:
        roots = _as_folders(folders) if folders else [Path(p) for p in dbmod.roots(conn)]
        clause = "" if force else "AND (dhash IS NULL OR place IS NULL)"
        rows = conn.execute(
            f"SELECT id, path, root FROM media WHERE kind = 'image' AND error IS NULL {clause} "
            "ORDER BY id"
        ).fetchall()
        for row in rows:
            try:
                path = Path(row["path"])
                if not path.exists():
                    continue
                root = Path(row["root"]) if row["root"] else _root_for(path, roots)
                with Image.open(path) as raw:
                    camera = exif_camera(raw)
                    raw.draft("RGB", (config.MAX_SIDE, config.MAX_SIDE))
                    img = ImageOps.exif_transpose(raw).convert("RGB")
                    value = dhash(img)
                conn.execute(
                    "UPDATE media SET dhash = ?, camera = COALESCE(?, camera), "
                    "place = COALESCE(?, place), root = COALESCE(?, root) WHERE id = ?",
                    (
                        value,
                        camera,
                        place_for(path, root) if root else None,
                        str(root) if root else None,
                        row["id"],
                    ),
                )
                stats["images"] += 1
                if stats["images"] % 100 == 0:
                    conn.commit()
                    if progress:
                        print(f"[refresh] {stats['images']} images", flush=True)
            except Exception:
                stats["errors"] += 1

        vids = conn.execute(
            "SELECT id, path, root FROM media WHERE kind = 'video' AND error IS NULL "
            + ("" if force else "AND duration IS NULL")
        ).fetchall()
        for row in vids:
            try:
                path = Path(row["path"])
                if not path.exists():
                    continue
                root = Path(row["root"]) if row["root"] else _root_for(path, roots)
                conn.execute(
                    "UPDATE media SET duration = ?, place = COALESCE(?, place), "
                    "root = COALESCE(?, root) WHERE id = ?",
                    (
                        _video_probe(path),
                        place_for(path, root) if root else None,
                        str(root) if root else None,
                        row["id"],
                    ),
                )
                stats["videos"] += 1
            except Exception:
                stats["errors"] += 1
        conn.commit()

    stats["elapsed_s"] = round(time.monotonic() - started, 1)
    return stats


def video_scan(
    folders=None,
    engine=None,
    limit=None,
    interval=6.0,
    max_frames=120,
    progress=True,
):
    config.ensure_dirs()
    dbmod.init_db()
    if engine is None:
        raise ValueError("an engine is required for video face detection")
    stats = {"videos": 0, "faces": 0, "errors": 0}
    started = time.monotonic()
    with dbmod.db() as conn:
        threshold = dbmod.threshold(conn)
        sql = (
            "SELECT id, path, root, size, mtime FROM media WHERE kind = 'video' "
            "AND error IS NULL AND id NOT IN "
            "(SELECT DISTINCT media_id FROM faces WHERE frame_time IS NOT NULL) ORDER BY id"
        )
        rows = conn.execute(sql).fetchall()
        if folders:
            roots = [str(Path(f)) for f in _as_folders(folders)]
            rows = [r for r in rows if any(str(r["path"]).lower().startswith(x.lower()) for x in roots)]
        if limit is not None:
            rows = rows[:limit]
        for i, row in enumerate(rows, 1):
            try:
                path = Path(row["path"])
                if not path.exists():
                    stats["errors"] += 1
                    continue

                class _St:
                    st_size = row["size"]
                    st_mtime = row["mtime"]

                stats["faces"] += _update_video(
                    conn, path, row["id"], _St, row["root"] or path.parent,
                    engine, threshold, True, interval, max_frames,
                )
                stats["videos"] += 1
                conn.commit()
                if progress:
                    print(
                        f"[video {i}/{len(rows)}] {path.name} faces={stats['faces']}",
                        flush=True,
                    )
            except Exception as exc:
                stats["errors"] += 1
                if progress:
                    print(f"[video] failed {row['path']}: {exc}", flush=True)
    stats["elapsed_s"] = round(time.monotonic() - started, 1)
    return stats
