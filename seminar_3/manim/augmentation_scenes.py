"""Cenas baseadas nos artefatos e medições dos notebooks 01–03."""

import csv
from pathlib import Path

import numpy as np
from manim import (
    DOWN, LEFT, RIGHT, UP, BLUE, GREEN, WHITE, YELLOW,
    FadeIn, FadeOut, Group, ImageMobject, Rectangle, Scene, Text,
    Transform, UpdateFromAlphaFunc, VGroup, ValueTracker, always_redraw,
)
from PIL import Image

OUT = Path(__file__).resolve().parents[1] / "data/outputs"
BACKGROUND = "#101827"


def picture(name, height=3.3):
    return ImageMobject(str(OUT / name)).set_height(height)


def heading(scene, title, subtitle):
    scene.camera.background_color = BACKGROUND
    scene.add(Text(title, font_size=34).to_edge(UP, buff=0.3))
    scene.add(Text(subtitle, font_size=19, color="#b9c7db").to_edge(DOWN, buff=0.25))


class TransformacoesRotulos(Scene):
    def construct(self):
        heading(self, "Augmentation: imagem e rótulo caminham juntos",
                "Demonstração sintética • Buslaev et al. (2018), seção II")
        original = picture("original.png", 2.7).move_to([-3.1, 0.7, 0])
        mask = picture("mask_original.png", 1.45).move_to([-3.1, -1.65, 0])
        label = Text("Original", font_size=26).move_to([-3.1, 2.5, 0])
        self.play(FadeIn(original), FadeIn(mask), FadeIn(label))
        self.add(Text("máscara", font_size=18).move_to([-5.4, -1.65, 0]))
        current = None
        notes = {
            "HorizontalFlip": "Flip: área preservada; posição alterada",
            "Rotacao": "Rotação: máscara usa classes discretas",
            "Brilho": "Brilho: muda pixels; preserva máscara",
            "Recorte": "Recorte: imagem e máscara perdem a mesma região",
        }
        for name, note in notes.items():
            transformed = picture(f"{name}.png", 2.7).move_to([3.1, 0.7, 0])
            transformed_mask = picture(f"mask_{name}.png", 1.45).move_to([3.1, -1.65, 0])
            title = Text(name, font_size=26, color=GREEN).move_to([3.1, 2.5, 0])
            caption = Text(note, font_size=22, color=YELLOW).move_to([0, -2.9, 0])
            group = Group(transformed, transformed_mask, title, caption)
            if current is not None:
                self.play(FadeOut(current), run_time=0.3)
            self.play(FadeIn(group), run_time=0.7)
            self.wait(2)
            current = group
        self.play(FadeOut(current), run_time=0.3)
        good = Text("Flip + máscara correta: IoU = 1,00", font_size=25, color=GREEN).move_to([2.4, 1, 0])
        bad = Text("Máscara esquecida: IoU = 0,00", font_size=25, color="#ff8080").move_to([2.4, -0.2, 0])
        self.play(FadeIn(good), FadeIn(bad))
        self.wait(3)


class Misturas(Scene):
    def construct(self):
        heading(self, "Cutout, Mixup e CutMix",
                "Mecanismos ilustrativos • Yang et al. (2023), seções 2.2–2.3")
        a = np.asarray(Image.open(OUT / "mix_A.png").convert("RGB"))
        b = np.asarray(Image.open(OUT / "mix_B.png").convert("RGB"))
        left = ImageMobject(a).set_height(2.2).move_to([-4.5, 0.5, 0])
        right = ImageMobject(b).set_height(2.2).move_to([4.5, 0.5, 0])
        mixed = ImageMobject(a).set_height(2.6).move_to([0, 0.5, 0])
        self.play(FadeIn(left), FadeIn(right), FadeIn(mixed))
        self.add(Text("Classe A", font_size=24).move_to([-4.5, 2, 0]),
                 Text("Classe B", font_size=24).move_to([4.5, 2, 0]))
        tracker = ValueTracker(1)
        label = always_redraw(lambda: Text(
            f"Mixup: λ = {tracker.get_value():.2f}   |   y = ({tracker.get_value():.2f}, {1-tracker.get_value():.2f})",
            font_size=27, color=YELLOW,
        ).move_to([0, -1.65, 0]))
        self.add(label)

        def update_mix(mobject, alpha):
            weight = 1 - alpha
            array = np.rint(weight * a + (1-weight) * b).astype(np.uint8)
            mobject.become(ImageMobject(array).set_height(2.6).move_to([0, 0.5, 0]))

        self.play(UpdateFromAlphaFunc(mixed, update_mix), tracker.animate.set_value(0), run_time=5)
        self.wait(1)
        label.clear_updaters()
        self.play(FadeOut(label), FadeOut(mixed))
        cutout = picture("mix_Cutout.png", 2.6).move_to([0, 0.5, 0])
        caption = Text("Cutout: apaga uma região; mantém y = (1, 0)", font_size=26).move_to([0, -1.65, 0])
        self.play(FadeIn(cutout), FadeIn(caption))
        self.wait(2.5)
        cutmix = picture("mix_CutMix.png", 2.6).move_to([0, 0.5, 0])
        new_caption = Text("CutMix: troca 1/3 da área; y = (0,67, 0,33)", font_size=26, color=GREEN).move_to([0, -1.65, 0])
        self.play(FadeOut(cutout), FadeIn(cutmix), Transform(caption, new_caption))
        self.wait(3)


class Resultados(Scene):
    def construct(self):
        heading(self, "Augmentation funciona? Vamos medir",
                "Medição local: MNIST / MLP • Não reproduz as acurácias do survey")
        with (OUT / "mnist_summary.csv").open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        colors = [BLUE, GREEN, YELLOW, "#b394ff", "#ff8080"]
        baseline = -1.65
        self.add(Text("Acurácia (%) • média ± desvio entre 3 sementes", font_size=21).move_to([0, 2.8, 0]))
        for tick in [0, 50, 100]:
            line = Rectangle(width=11.7, height=0.01, stroke_width=0, fill_color=WHITE, fill_opacity=0.25)
            line.move_to([0.15, baseline + tick * 0.032, 0])
            self.add(line, Text(str(tick), font_size=17).move_to([-6.1, baseline + tick * 0.032, 0]))
        chart = None
        for metric, title in [("test_accuracy", "Teste limpo"), ("shifted_accuracy", "Teste deslocado +3 pixels")]:
            group = VGroup(Text(title, font_size=27, color=YELLOW).move_to([0, 2.25, 0]))
            for i, (row, color) in enumerate(zip(rows, colors, strict=True)):
                mean = float(row[f"{metric}_mean"])
                std = float(row[f"{metric}_std"])
                x = -4.7 + i * 2.35
                bar = Rectangle(width=1.4, height=mean * 0.032, fill_color=color,
                                fill_opacity=0.9, stroke_width=0).move_to([x, baseline + mean*0.016, 0])
                value = Text(f"{mean:.1f} ± {std:.1f}", font_size=19).next_to(bar, UP, buff=0.12)
                name = Text(row["policy"].replace(" ", "\n", 1), font_size=19).move_to([x, -2.25, 0])
                group.add(bar, value, name)
            if chart is not None:
                self.play(FadeOut(chart), run_time=0.5)
            self.play(FadeIn(group, shift=UP * 0.2), run_time=1.2)
            self.wait(4)
            chart = group
        self.play(FadeOut(chart))
        lesson = Text("A política precisa respeitar a tarefa.\nMais transformações não garantem mais acertos.",
                      font_size=32, color=GREEN).move_to([0, 0.4, 0])
        self.play(FadeIn(lesson))
        self.wait(3)
