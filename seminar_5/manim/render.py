"""Render short MP4 videos, GIFs and the exact final-frame posters."""

import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SCENES = ["PipelineLCM", "ArchitectureLCM", "ConsistencyMath", "CompareSampling", "LocalGeneration"]


def main():
    """Render all scenes and transcode reusable assets.

    Raises:
        RuntimeError: If a rendered scene cannot be located.
    """
    for name in ["videos", "gifs", "images"]:
        (ROOT / "assets" / name).mkdir(parents=True, exist_ok=True)
    media = ROOT / "manim/media"
    metadata = []
    for scene in SCENES:
        subprocess.run(
            [sys.executable, "-m", "manim", "-qm", "--disable_caching",
             "--media_dir", str(media), str(ROOT / "manim/lcm_scenes.py"), scene],
            check=True, cwd=ROOT,
        )
        matches = list(media.glob(f"videos/lcm_scenes/720p30/{scene}.mp4"))
        if len(matches) != 1:
            raise RuntimeError(f"Vídeo ausente: {scene}")
        video = ROOT / f"assets/videos/{scene}.mp4"
        shutil.copyfile(matches[0], video)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-filter_complex",
                        "fps=10,scale=640:-1:flags=lanczos,split[s0][s1];"
                        "[s0]palettegen[p];[s1][p]paletteuse",
                        "-loop", "0", str(ROOT / f"assets/gifs/{scene}.gif")], check=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-sseof", "-0.15", "-i", str(video),
                        "-frames:v", "1", str(ROOT / f"assets/images/{scene}.png")], check=True)
        probe = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "json", str(video)], text=True
        )
        metadata.append({"scene": scene, "duration_seconds": float(json.loads(probe)["format"]["duration"]),
                         "video": f"assets/videos/{scene}.mp4", "resolution": "1280x720",
                         "fps": 30, "gif_fps": 10})
    (ROOT / "assets/videos/manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("Cinco MP4, GIFs e posters renderizados.", flush=True)


if __name__ == "__main__":
    main()
