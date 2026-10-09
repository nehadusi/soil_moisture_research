"""Stage 0: pull one frame per second out of each drone video with ffmpeg.

Frames go to frames/<video name with spaces -> _>/<name>_t<milliseconds>.jpg.
Videos whose frame folder already has frames are skipped, so this is safe to re-run.

Usage: python src/extract_frames.py <videos_dir> [fps]
"""
import subprocess
import sys
from pathlib import Path

videos = Path(sys.argv[1])
fps = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
for f in sorted(videos.glob("9.1*.MP4")) + sorted(videos.glob("9.1*.mp4")):
    name = f.stem.replace(" ", "_")
    out = Path("frames") / name
    if out.exists() and any(out.glob("*_t*.jpg")):
        print(f"skip {name} (frames already there)")
        continue
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / f"{name}_%05d.jpg"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(f), "-vf", f"fps={fps}", "-q:v", "2", str(tmp)], check=True)
    # rename 00001, 00002 ... to the timestamp in ms (frame k is at (k-1)/fps seconds)
    for p in sorted(out.glob(f"{name}_[0-9][0-9][0-9][0-9][0-9].jpg")):
        k = int(p.stem.rsplit("_", 1)[1])
        p.rename(out / f"{name}_t{int(round((k - 1) / fps * 1000)):06d}.jpg")
    print(f"{name}: {len(list(out.glob('*_t*.jpg')))} frames")
