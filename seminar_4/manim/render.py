"""Renderize MP4 720p30 e GIF 800px/10fps, no padrão do seminário 3."""

from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SCENES = ["PipelineDiffuse2Seg", "PropagacaoPrompt", "HierarquiaNMS", "ResultadosLocais"]


def main():
    out = ROOT / "data/outputs/manim"
    out.mkdir(parents=True, exist_ok=True)
    cache = ROOT / "manim/cache"
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("FFmpeg necessário para exportar GIFs.")
    subprocess.run([sys.executable, "-m", "manim", "-qm", "--disable_caching", "--progress_bar", "none",
                    "-v", "WARNING", "--media_dir", str(cache), str(ROOT / "manim/diffuse2seg_scenes.py"),
                    *SCENES], check=True)
    for scene in SCENES:
        video = out / f"{scene}.mp4"
        shutil.copy2(cache / "videos/diffuse2seg_scenes/720p30" / video.name, video)
        subprocess.run([ffmpeg, "-y", "-v", "error", "-i", str(video), "-filter_complex",
                        "[0:v]fps=10,scale=800:-1:flags=lanczos,split[a][b];"
                        "[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=sierra2_4a",
                        "-loop", "0", str(video.with_suffix(".gif"))], check=True)
        print(f"Gerados: {video.name}, {scene}.gif", flush=True)


if __name__ == "__main__":
    main()
