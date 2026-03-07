"""
Download compilation videos per class using yt-dlp.

Folder structure created:
  data/compilations/{label}/{index}_{video_id}.mp4

Add more labels/URLs to VIDEOS as needed.
Install: pip install yt-dlp
"""
import subprocess
from pathlib import Path

VIDEOS = {
    "seoi": [
        "https://www.youtube.com/watch?v=Gv-Cr9tsM9w",
        "https://youtu.be/mzJScxzp68A",
        "https://www.youtube.com/watch?v=o-qBMFVW2SE",
        "https://www.youtube.com/watch?v=l8fTnDSwN2E",
        "https://www.youtube.com/watch?v=5zZQ8TJ0tg0",
        "https://www.youtube.com/watch?v=qDn9vMT7yK8",
    ],
}


def download(url: str, out_dir: Path, index: int):
    out_template = str(out_dir / f"{index:02d}_%(id)s.%(ext)s")
    result = subprocess.run(
        [
            "yt-dlp",
            "-f", "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4][height<=720]/best",
            "--merge-output-format", "mp4",
            "--no-playlist",
            "-o", out_template,
            url,
        ],
        check=False,
    )
    if result.returncode != 0:
        print(f"  WARNING: yt-dlp failed for {url}")


def main():
    for label, urls in VIDEOS.items():
        out_dir = Path(f"data/compilations/{label}")
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Downloading {len(urls)} video(s) for label '{label}' ===")
        for i, url in enumerate(urls):
            print(f"  [{i+1}/{len(urls)}] {url}")
            download(url, out_dir, i)

    print("\nAll downloads complete.")
    print("Next: python src/track_people.py")


if __name__ == "__main__":
    main()
