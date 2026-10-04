"""Cenas 720p30 no padrão escuro dos ciclos 2/3; métricas/frames medidos."""

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
from manim import (
    Arrow, Create, Dot, DOWN, FadeIn, FadeOut, ImageMobject,
    LEFT, Rectangle, RIGHT, Scene, Text, Transform, UP, VGroup,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments/results"
BG, BLUE, GREEN, YELLOW, GRAY = "#101827", "#60a5fa", "#34d399", "#fbbf24", "#b9c7db"
Text.set_default(font="DejaVu Sans")


def heading(scene, title, footer):
    scene.camera.background_color = BG
    title_text = Text(title, font_size=32)
    footer_text = Text(footer, font_size=16, color=GRAY)
    title_text.scale(min(1, 13.2/title_text.width))
    footer_text.scale(min(1, 13.2/footer_text.width))
    scene.add(title_text.to_edge(UP, buff=.28))
    scene.add(footer_text.to_edge(DOWN, buff=.18))


def block(label, xy, color, width=2.3):
    rect = Rectangle(width=width, height=.95, color=color, fill_color=color, fill_opacity=.12)
    text = Text(label, font_size=21)
    text.scale(min(1, (width-.3)/text.width, .72/text.height))
    return VGroup(rect, text).move_to([*xy, 0])


def map_image(values, height=3.6, cmap="viridis"):
    norm = (values-values.min()) / max(float(np.ptp(values)), 1e-12)
    rgb = (matplotlib.colormaps[cmap](norm)[..., :3] * 255).astype(np.uint8)
    return ImageMobject(rgb).set_height(height)


class PipelineDiffuse2Seg(Scene):
    def construct(self):
        heading(self, "Diffuse2Seg: de imagem a supervisão", "Diagrama próprio • método descrito na Figura 2 / Seções 3.3–3.5")
        labels = ["Imagem RGB", "VAE + U-Net\nSD2 congelado", "Afinidade A", "Grade +\npropagação", "KL + níveis\ncomponentes + NMS", "Pseudo-máscaras", "Mask2Former\nRodada 1", "Global + crops\nenriquecimento", "Rodada 2\nZero-shot"]
        locations = [(-4.5, 1.9), (0, 1.9), (4.5, 1.9), (4.5, .1), (0, .1), (-4.5, .1), (-4.5, -1.7), (0, -1.7), (4.5, -1.7)]
        boxes = [block(name, loc, [BLUE, GREEN, YELLOW][i // 3], width=3.4) for i, (name, loc) in enumerate(zip(labels, locations, strict=True))]
        self.play(FadeIn(boxes[0]))
        for i in range(1, len(boxes)):
            direction = boxes[i].get_center() - boxes[i-1].get_center()
            horizontal = abs(direction[0]) > .1
            source = boxes[i-1].get_right() if direction[0] > 0 else boxes[i-1].get_left()
            target = boxes[i].get_left() if direction[0] > 0 else boxes[i].get_right()
            if not horizontal:
                source, target = boxes[i-1].get_bottom(), boxes[i].get_top()
            arrow = Arrow(source, target, buff=.08, color=GRAY)
            self.play(Create(arrow), FadeIn(boxes[i]), run_time=.65)
            dot = Dot(arrow.get_start(), color=GREEN, radius=.05)
            self.add(dot)
            self.play(dot.animate.move_to(arrow.get_end()), run_time=.35)
            self.remove(dot)
        self.wait(3)


class PropagacaoPrompt(Scene):
    def construct(self):
        heading(self, "Um ponto se transforma em mapa suave", "Reprodução parcial • afinidade RGB • p=1,6 • cor reescalada por frame")
        data = np.load(RESULTS / "propagation_snapshots.npz")
        input_data = np.load(RESULTS / "synthetic_seed7_p1.6.npz")
        image = ImageMobject((input_data["image"] * 255).astype(np.uint8)).set_height(3.8).shift(LEFT*3.4)
        panel = map_image(data["maps"][0, 9].reshape(24, 24), 3.8).shift(RIGHT*3.4)
        caption = Text("Iteração 0: prompt one-hot", font_size=23, color=YELLOW).move_to([0, -2.4, 0])
        self.play(FadeIn(image), FadeIn(panel), FadeIn(caption))
        self.wait(1)
        for index, step in enumerate(data["steps"][1:], 1):
            target = map_image(data["maps"][index, 9].reshape(24, 24), 3.8).move_to(panel)
            new_caption = Text(f"Iteração {step}: mapa calculado pelo solver", font_size=23, color=YELLOW).move_to(caption)
            # Fade entre snapshots; interpolação visual não são iterações extras do solver.
            self.play(FadeOut(panel), FadeIn(target), Transform(caption, new_caption), run_time=.5)
            panel = target
            self.wait(1.2)
        self.wait(2)


class HierarquiaNMS(Scene):
    def construct(self):
        heading(self, "Cortes KL mudam a granularidade", "Dados do proxy • p=2 • 6 cortes • NMS por área • sem difusão")
        data = np.load(RESULTS / "synthetic_seed7_p2.0.npz")
        config = json.loads((RESULTS / "protocol.json").read_text())["config"]
        panel = map_image(data["levels"][0], 3.8, "tab20")
        title = Text(f"h={config['thresholds'][0]:.3f}: partição fina", font_size=27, color=GREEN).move_to([0, 2.35, 0])
        self.play(FadeIn(panel), FadeIn(title))
        for i in range(1, len(data["levels"])):
            target = map_image(data["levels"][i], 3.8, "tab20")
            caption = Text(f"h={config['thresholds'][i]:.3f}: {len(np.unique(data['levels'][i]))} grupos", font_size=27, color=GREEN).move_to(title)
            self.play(FadeOut(panel), FadeIn(target), Transform(title, caption), run_time=.5)
            panel = target
            self.wait(1)
        self.play(FadeOut(panel), FadeOut(title))
        rows = list(csv.DictReader((RESULTS / "metrics.csv").open(encoding="utf-8")))
        record = next(r for r in rows if r["seed"] == "7" and r["p"] == "2.0")
        explanation = VGroup(Text(f"{record['candidates_before_nms']} candidatos nos 6 níveis", font_size=32),
                             Text("Ordenar por área → rejeitar IoU > 0,9", font_size=28, color=YELLOW),
                             Text(f"{record['proposals']} máscaras finais • semente 7", font_size=31, color=GREEN)).arrange(DOWN, buff=.45)
        self.play(FadeIn(explanation))
        self.wait(3)


class ResultadosLocais(Scene):
    def construct(self):
        heading(self, "O proxy não reproduziu o ganho do paper", "Recall toy, matching 1:1 • mesma cena com 3 sementes de ruído • sem avaliação COCO")
        data = json.loads((RESULTS / "protocol.json").read_text())
        for i, row in enumerate(data["summary"]):
            x = -4 + i*4
            height = row["recall_mean"]*4
            bar = Rectangle(width=1.4, height=height, color=[GRAY, BLUE, GREEN][i], fill_opacity=.9)
            bar.move_to([x, -1.8+height/2, 0])
            value = Text(f"{100*row['recall_mean']:.1f}% ± {100*row['recall_std']:.1f}", font_size=23).next_to(bar, UP)
            label = Text(f"p={row['p']}", font_size=27).move_to([x, -2.25, 0])
            self.play(Create(bar), FadeIn(value), FadeIn(label), run_time=.7)
        self.wait(2)
        caption = Text("Afinidade e calibração importam; p<2 não garante melhora.", font_size=26, color=YELLOW).move_to([0, 2.55, 0])
        self.play(FadeIn(caption))
        self.wait(4)
