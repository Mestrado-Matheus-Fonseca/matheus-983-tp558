"""Renderize três cenas e exporte MP4/GIF para slides (requer ffmpeg)."""

import shutil
import subprocess
import sys
from pathlib import Path


if __name__ == "__main__":
    seminar = Path(__file__).resolve().parents[1]
    output = seminar / "data/outputs/manim"
    cache = output / "cache"
    output.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("Instale ffmpeg para exportar os GIFs.")
    scenes = ["TransformacoesRotulos", "Misturas", "Resultados"]
    subprocess.run([
        sys.executable, "-m", "manim", "-qm", "--disable_caching",
        "--media_dir", str(cache), str(seminar / "manim/augmentation_scenes.py"), *scenes,
    ], check=True)
    for scene in scenes:
        source = cache / "videos/augmentation_scenes/720p30" / f"{scene}.mp4"
        video = output / f"{scene}.mp4"
        shutil.copy2(source, video)
        subprocess.run([
            ffmpeg, "-y", "-v", "error", "-i", str(video),
            "-filter_complex",
            "[0:v]fps=10,scale=800:-1:flags=lanczos,split[a][b];"
            "[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=sierra2_4a",
            "-loop", "0", str(video.with_suffix(".gif")),
        ], check=True)
        print(f"Gerados: {video.name} e {video.stem}.gif", flush=True)
