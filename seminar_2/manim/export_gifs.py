import argparse
import shutil
import subprocess
from pathlib import Path


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "outputs" / "manim"


def main():
    parser = argparse.ArgumentParser(description="Converte os vídeos Manim em GIFs para slides.")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--width", type=int, default=960)
    args = parser.parse_args()

    ffmpeg = shutil.which("ffmpeg")
    videos = sorted(OUTPUT_DIR.glob("*.mp4"))
    if not ffmpeg:
        raise SystemExit("ffmpeg não encontrado no PATH.")
    if not videos:
        raise SystemExit(f"Nenhum MP4 encontrado em {OUTPUT_DIR}.")

    video_filter = (
        f"[0:v]fps={args.fps},scale={args.width}:-1:flags=lanczos,split[a][b];"
        "[a]palettegen=stats_mode=diff[p];"
        "[b][p]paletteuse=dither=sierra2_4a"
    )
    for video in videos:
        gif = video.with_suffix(".gif")
        subprocess.run(
            [ffmpeg, "-y", "-i", str(video), "-filter_complex", video_filter, "-loop", "0", str(gif)],
            check=True,
        )
        print(f"Gerado: {gif}")


if __name__ == "__main__":
    main()
