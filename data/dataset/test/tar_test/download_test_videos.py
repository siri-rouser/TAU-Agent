#!/usr/bin/env python3
"""
Download and trim YouTube source videos for the AI City 2026 Track 3
TAR test set.

The test split references 80 short clips taken from 17 public YouTube
videos. This script reads `clip_manifest.csv` (sibling of this script),
downloads each YouTube source once, and trims it into the per-clip files
that the test annotations reference.

After a successful run, the layout under `--out` is::

    <out>/tar_test/v=<ytid>_<start>-<end>.mp4   # 80 files

where each `<out>/tar_test/<id>.mp4` matches a `video_id` in
`test/test.json`. Load the test annotations with `media_root` set to
`<out>` and every `video_id` resolves directly.

Usage::

    python download_test_videos.py --out /mnt/faster0/yl4300/aicity_output
    python download_test_videos.py --install-deps    # installs yt-dlp
    # Fetch a single source video (note: YouTube ids that start with `-`
    # must be passed with the `=` form so argparse doesn't read them as a flag):
    python download_test_videos.py --out /mnt/faster0/yl4300/aicity_output --only-ytid=-3nwOfm1Pdk
    # Age-restricted videos may require browser cookies:
    python download_test_videos.py --out /mnt/faster0/yl4300/aicity_output --cookies-from-browser chrome

Requirements:
    - `yt-dlp` on PATH or installed via pip (`pip install yt-dlp`).
    - `ffmpeg` on PATH (used for accurate trimming). The companion
      `download_videos.py` script for the training set also requires ffmpeg.

Source videos are hosted by their original YouTube uploaders. By using
them, you agree to YouTube's terms of service and to the AI City Challenge
2026 Track 3 rules (no redistribution).
"""

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

REQUIRED_PKGS = ["yt_dlp"]


# ---- helpers ----

def find_ffmpeg() -> str:
    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        return sys_ffmpeg
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        return get_ffmpeg_exe()
    except ImportError:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install via your package manager "
            "(`apt install ffmpeg`, `brew install ffmpeg`) or "
            "`pip install imageio-ffmpeg`."
        )


def install_deps():
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", "yt-dlp"]
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def check_missing_deps():
    import importlib.util
    return [p for p in REQUIRED_PKGS if importlib.util.find_spec(p) is None]


def parse_ts(ts: str) -> float:
    """Parse 'M:SS' or 'H:MM:SS' into seconds (float)."""
    parts = ts.strip().split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    return float(ts)


def read_csv(csv_path: Path):
    """Yield dicts of {video_id, youtube_url, youtube_id, start, end}.

    `youtube_id` is parsed from the URL.
    """
    import re
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            url = row["youtube_url"].strip()
            m = re.search(r"[?&]v=([A-Za-z0-9_-]+)", url)
            if not m:
                raise ValueError(f"Cannot extract YouTube id from URL: {url!r}")
            yield {
                "video_id": row["video_id"].strip(),
                "youtube_url": url,
                "youtube_id": m.group(1),
                "start": row["start"].strip(),
                "end": row["end"].strip(),
            }


# ---- yt-dlp + ffmpeg ----

def download_source(
    ytid: str,
    tmp_dir: Path,
    cookies_from_browser: str | None = None,
    cookies_file: Path | None = None,
) -> Path:
    """Download a single YouTube video to ``tmp_dir/<ytid>.mp4``.

    Prefers a single muxed mp4 stream so no separate audio merge is needed.
    """
    from yt_dlp import YoutubeDL

    out_path = tmp_dir / f"{ytid}.mp4"
    if out_path.exists():
        return out_path

    ydl_opts = {
        # Pick the best mp4 with both video+audio in one file. Fallback to
        # bv+ba merge if no single-file mp4 is available.
        "format": "best[ext=mp4]/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best",
        "outtmpl": str(out_path.with_suffix("")) + ".%(ext)s",
        "merge_output_format": "mp4",
        "quiet": True,
        "noprogress": True,
        "no_warnings": True,
    }
    if cookies_from_browser:
        browser, _, profile = cookies_from_browser.partition(":")
        ydl_opts["cookiesfrombrowser"] = (browser, None, profile or None, None)
    if cookies_file:
        ydl_opts["cookiefile"] = str(cookies_file)

    url = f"https://www.youtube.com/watch?v={ytid}"
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # yt-dlp may have written .mp4 or .mkv depending on streams available.
    if not out_path.exists():
        candidates = list(tmp_dir.glob(f"{ytid}.*"))
        if not candidates:
            raise RuntimeError(f"yt-dlp produced no output for {ytid}")
        candidates[0].rename(out_path)
    return out_path


def trim_clip(src: Path, dst: Path, start: str, end: str, ffmpeg: str) -> None:
    """Trim ``src`` from ``start`` to ``end`` (M:SS or H:MM:SS) into ``dst``.

    Re-encodes to H.264/yuv420p for broad VLM/codec compatibility. Audio is
    dropped (``-an``) to match the original test-set processing convention
    and avoid the few KB/s of extra storage for streams VLMs ignore anyway.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-ss", str(parse_ts(start)),
        "-to", str(parse_ts(end)),
        "-i", str(src),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-an",  # drop audio
        str(dst),
    ]
    subprocess.run(cmd, check=True)


# ---- main ----

def main(argv=None):
    script_dir = Path(__file__).resolve().parent
    default_csv = script_dir / "clip_manifest.csv"

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--out", type=Path,
                    help="Root output directory (videos land at <out>/tar_test/).")
    ap.add_argument("--csv", type=Path, default=default_csv,
                    help=f"Path to clip_manifest.csv (default: {default_csv}).")
    ap.add_argument("--only-ytid", nargs="+", default=None,
                    help="Only fetch clips from these YouTube ids.")
    ap.add_argument("--cookies-from-browser", default=None,
                    help="Browser cookie source for yt-dlp, e.g. 'chrome', 'firefox', or 'chrome:Default'.")
    ap.add_argument("--cookies", type=Path, default=None,
                    help="Path to a Netscape-format cookies.txt file for yt-dlp.")
    ap.add_argument("--keep-source", action="store_true",
                    help="Keep the full-length YouTube downloads under "
                         "<out>/tar_test/_source/ instead of deleting them "
                         "after trimming (useful for debugging).")
    ap.add_argument("--install-deps", action="store_true",
                    help="pip install yt-dlp and exit.")
    args = ap.parse_args(argv)

    if args.install_deps:
        install_deps()
        return 0

    missing = check_missing_deps()
    if missing:
        print(f"Missing Python packages: {missing}")
        print(f"Run: python {sys.argv[0]} --install-deps")
        return 1
    if not args.csv.exists():
        ap.error(f"CSV not found: {args.csv}")
    if not args.out:
        ap.error("--out is required")
    if args.cookies and not args.cookies.exists():
        ap.error(f"cookies file not found: {args.cookies}")

    ffmpeg = find_ffmpeg()

    out_root = args.out / "tar_test"
    out_root.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_root / "_source"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Group clips by YouTube id so each source is downloaded once.
    clips_by_yt: dict[str, list[dict]] = {}
    for clip in read_csv(args.csv):
        if args.only_ytid and clip["youtube_id"] not in args.only_ytid:
            continue
        clips_by_yt.setdefault(clip["youtube_id"], []).append(clip)

    if not clips_by_yt:
        print("No clips selected; nothing to do.")
        return 0

    n_clips = sum(len(v) for v in clips_by_yt.values())
    print(f"Fetching {n_clips} clips from {len(clips_by_yt)} YouTube source videos.")

    ok, failed, skipped = 0, [], 0
    for ytid, clips in sorted(clips_by_yt.items()):
        # Skip download if every target clip for this ytid is already present.
        targets = [args.out / c["video_id"] for c in clips]
        if all(t.exists() for t in targets):
            skipped += len(clips)
            print(f"\n=== {ytid}: all {len(clips)} clip(s) already present; skipping. ===")
            continue

        print(f"\n=== {ytid}: downloading source + trimming {len(clips)} clip(s) ===")
        try:
            src = download_source(
                ytid,
                tmp_dir,
                cookies_from_browser=args.cookies_from_browser,
                cookies_file=args.cookies,
            )
        except Exception as e:
            print(f"  [fail] yt-dlp could not fetch {ytid}: {e}")
            failed.extend(c["video_id"] for c in clips)
            continue

        for clip in clips:
            dst = args.out / clip["video_id"]
            if dst.exists():
                skipped += 1
                continue
            try:
                trim_clip(src, dst, clip["start"], clip["end"], ffmpeg)
                print(f"  [ok] {clip['video_id']} ({clip['start']}–{clip['end']})")
                ok += 1
            except subprocess.CalledProcessError as e:
                print(f"  [fail] trim {clip['video_id']}: ffmpeg exit {e.returncode}")
                failed.append(clip["video_id"])

        if not args.keep_source:
            try:
                src.unlink()
            except OSError:
                pass

    # Final cleanup of the source dir if empty.
    if not args.keep_source:
        try:
            tmp_dir.rmdir()
        except OSError:
            pass  # not empty (some sources kept due to failures)

    print("\n=== Summary ===")
    print(f"  ok      : {ok}")
    print(f"  skipped : {skipped} (already present)")
    if failed:
        print(f"  failed  : {len(failed)}")
        for v in failed[:10]:
            print(f"    - {v}")
        if len(failed) > 10:
            print(f"    ... and {len(failed) - 10} more")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
