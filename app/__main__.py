import argparse
import os
import sys

from . import config

os.environ.setdefault("HF_HOME", str(config.DATA / "hf"))
os.environ.setdefault("TORCH_HOME", str(config.DATA / "torch"))
os.environ.setdefault("PIP_CACHE_DIR", str(config.ROOT / ".cache" / "pip"))


def cmd_scan(args):
    from . import db as dbmod
    from . import scanner

    dbmod.init_db()
    with dbmod.db() as conn:
        if args.threshold is not None:
            dbmod.set_meta(conn, "cluster_threshold", args.threshold)

    engine = None
    if not args.no_faces:
        from .faceengine import FaceEngine

        print("Loading face models (SCRFD + ArcFace + age/gender)...", flush=True)
        engine = FaceEngine(threads=args.threads)

    print("Scanning " + ", ".join(args.folders), flush=True)
    stats = scanner.scan(
        args.folders,
        engine=engine,
        limit=args.limit,
        include_videos=not args.no_videos,
        video_faces=args.video_faces and not args.no_faces,
        interval=args.interval,
        max_frames=args.max_frames,
        progress=not args.quiet,
    )
    print(
        "Done: {processed} indexed, {skipped} unchanged, {duplicates} duplicates, "
        "{errors} errors, {faces} faces, {videos} videos ({video_faces} video faces), "
        "{clusters} people, {unsorted_faces} unsorted faces".format(**stats)
    )
    print(f"Elapsed: {stats['elapsed_s']}s  Throughput: {stats['per_minute']} photos/min")
    if args.no_faces:
        print("Faces were skipped (--no-faces): no face rows or clusters were created.")


def cmd_refresh(args):
    from . import db as dbmod
    from . import scanner

    dbmod.init_db()
    stats = scanner.refresh(args.folders or None, progress=not args.quiet, force=args.force)
    print(f"Refreshed {stats['images']} images and {stats['videos']} videos ({stats['errors']} errors).")


def cmd_video_scan(args):
    from . import db as dbmod
    from . import scanner
    from .faceengine import FaceEngine

    dbmod.init_db()
    print("Loading face models (SCRFD + ArcFace + age/gender)...", flush=True)
    engine = FaceEngine(threads=args.threads)
    stats = scanner.video_scan(
        args.folders or None,
        engine=engine,
        limit=args.limit,
        interval=args.interval,
        max_frames=args.max_frames,
        progress=not args.quiet,
    )
    print(f"Done: {stats['videos']} videos scanned, {stats['faces']} faces found, {stats['errors']} errors.")


def cmd_watch(args):
    import time

    from . import db as dbmod
    from . import scanner

    dbmod.init_db()
    engine = None
    if not args.no_faces:
        from .faceengine import FaceEngine

        print("Loading face models (SCRFD + ArcFace + age/gender)...", flush=True)
        engine = FaceEngine(threads=args.threads)

    print(f"Watching every {args.interval}s. Press Ctrl+C to stop.", flush=True)
    while True:
        with dbmod.db() as conn:
            folders = args.folders or dbmod.roots(conn)
        if not folders:
            print("No folders to watch: scan a folder first.", flush=True)
            return
        try:
            stats = scanner.scan(
                folders,
                engine=engine,
                include_videos=not args.no_videos,
                video_faces=args.video_faces and not args.no_faces,
                interval=args.sample,
                progress=False,
            )
            if stats["processed"] or stats["faces"] or stats["video_faces"]:
                print(
                    f"Added {stats['processed']} files, {stats['skipped']} unchanged, "
                    f"{stats['faces']} faces, {stats['video_faces']} video faces.",
                    flush=True,
                )
            scanner.refresh(folders, progress=False)
        except Exception as exc:
            print(f"Watch pass failed: {exc}", flush=True)
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("Stopped.", flush=True)
            return



def cmd_serve(args):
    import uvicorn

    from . import db as dbmod

    dbmod.init_db()
    print(f"Gallery: http://{args.host}:{args.port}/", flush=True)
    uvicorn.run("app.web:app", host=args.host, port=args.port, log_level="info")


def cmd_recluster(args):
    from . import clustering
    from . import db as dbmod

    dbmod.init_db()
    with dbmod.db() as conn:
        if args.threshold is not None:
            dbmod.set_meta(conn, "cluster_threshold", args.threshold)
        threshold = dbmod.threshold(conn)
        result = clustering.recluster(conn, threshold, args.exemplars, args.max_iter)
    print(
        f"Reclustered {result['faces']} faces into {result['clusters']} clusters "
        f"(threshold {threshold}, {args.exemplars} exemplars, {result.get('iterations', 0)} iterations)."
    )


def cmd_stats(args):
    from . import db as dbmod

    dbmod.init_db()
    with dbmod.db() as conn:
        media = conn.execute("SELECT COUNT(*) AS c FROM media").fetchone()["c"]
        videos = conn.execute("SELECT COUNT(*) AS c FROM media WHERE kind = 'video'").fetchone()["c"]
        favorites = conn.execute("SELECT COUNT(*) AS c FROM media WHERE favorite = 1").fetchone()["c"]
        duplicates = conn.execute("SELECT COUNT(*) AS c FROM media WHERE dhash IS NOT NULL").fetchone()["c"]
        faces = conn.execute("SELECT COUNT(*) AS c FROM faces").fetchone()["c"]
        people = conn.execute("SELECT COUNT(*) AS c FROM clusters WHERE n >= 2").fetchone()["c"]
        unsorted_faces = conn.execute(
            "SELECT COUNT(*) AS c FROM faces f JOIN clusters c ON c.id = f.cluster_id WHERE c.n = 1"
        ).fetchone()["c"]
        errors = conn.execute("SELECT COUNT(*) AS c FROM media WHERE error IS NOT NULL").fetchone()["c"]
        first = conn.execute("SELECT MIN(taken_at) AS d FROM media").fetchone()["d"]
        last = conn.execute("SELECT MAX(taken_at) AS d FROM media").fetchone()["d"]
        root = dbmod.get_meta(conn, "last_scan_root", "-")
        scanned = dbmod.get_meta(conn, "last_scan", "-")
        roots = dbmod.roots(conn)
        threshold = dbmod.threshold(conn)
    print(f"Database: {config.DB_PATH}")
    print(f"Indexed files: {media}  (videos: {videos}, errors: {errors})")
    print(f"Hashed for duplicate checks: {duplicates}  Favourites: {favorites}")
    print(f"Faces: {faces}  People (2+ faces): {people}  Unsorted faces: {unsorted_faces}")
    print(f"Date range: {first} .. {last}")
    print(f"Last scan: {scanned}  root: {root}")
    print("Folders: " + (", ".join(roots) if roots else "-"))
    print(f"Cluster threshold (cosine distance): {threshold}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m app", description="Local face-aware media sorter")
    sub = parser.add_subparsers(dest="command", required=True)

    p_scan = sub.add_parser("scan", help="Index one or more folders into the local database")
    p_scan.add_argument("folders", nargs="+")
    p_scan.add_argument("--limit", type=int, default=None, help="Stop after N files (for testing)")
    p_scan.add_argument("--no-faces", action="store_true", help="Index files and thumbnails only")
    p_scan.add_argument("--no-videos", action="store_true", help="Skip video files")
    p_scan.add_argument("--video-faces", action="store_true", help="Also find faces inside videos (slower)")
    p_scan.add_argument("--interval", type=float, default=6.0, help="Seconds between sampled video frames")
    p_scan.add_argument("--max-frames", type=int, default=120)
    p_scan.add_argument("--threshold", type=float, default=None, help="Cosine distance threshold for a new person")
    p_scan.add_argument("--threads", type=int, default=config.ORT_THREADS)
    p_scan.add_argument("--quiet", action="store_true")
    p_scan.set_defaults(func=cmd_scan)

    p_refresh = sub.add_parser("refresh", help="Backfill hashes, places, cameras and video details")
    p_refresh.add_argument("folders", nargs="*")
    p_refresh.add_argument("--force", action="store_true", help="Re-read every file, not only the missing fields")
    p_refresh.add_argument("--quiet", action="store_true")
    p_refresh.set_defaults(func=cmd_refresh)

    p_video = sub.add_parser("video-scan", help="Find faces inside videos already in the library")
    p_video.add_argument("folders", nargs="*")
    p_video.add_argument("--limit", type=int, default=None)
    p_video.add_argument("--interval", type=float, default=6.0)
    p_video.add_argument("--max-frames", type=int, default=120)
    p_video.add_argument("--threads", type=int, default=config.ORT_THREADS)
    p_video.add_argument("--quiet", action="store_true")
    p_video.set_defaults(func=cmd_video_scan)

    p_watch = sub.add_parser("watch", help="Keep indexing new files in the background")
    p_watch.add_argument("folders", nargs="*")
    p_watch.add_argument("--interval", type=int, default=300, help="Seconds between passes")
    p_watch.add_argument("--sample", type=float, default=6.0, help="Seconds between sampled video frames")
    p_watch.add_argument("--video-faces", action="store_true")
    p_watch.add_argument("--no-faces", action="store_true")
    p_watch.add_argument("--no-videos", action="store_true")
    p_watch.add_argument("--threads", type=int, default=config.ORT_THREADS)
    p_watch.set_defaults(func=cmd_watch)

    p_serve = sub.add_parser("serve", help="Run the local gallery")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.set_defaults(func=cmd_serve)

    p_rec = sub.add_parser("recluster", help="Recompute face clusters")
    p_rec.add_argument("--threshold", type=float, default=None)
    p_rec.add_argument("--exemplars", type=int, default=4, help="Quality-ranked, diverse face examples per cluster")
    p_rec.add_argument("--max-iter", type=int, default=10, help="Maximum merge rounds")
    p_rec.set_defaults(func=cmd_recluster)

    p_stats = sub.add_parser("stats", help="Show database counts")
    p_stats.set_defaults(func=cmd_stats)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
