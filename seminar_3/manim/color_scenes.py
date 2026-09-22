"""Cenas coloridas baseadas nas saídas dos notebooks 05 e 06."""

import csv
from pathlib import Path

from manim import (
    BLUE,
    DOWN,
    FadeIn,
    FadeOut,
    GREEN,
    Group,
    ImageMobject,
    Rectangle,
    Scene,
    Text,
    Transform,
    UP,
    VGroup,
    WHITE,
    YELLOW,
)


OUT = Path(__file__).resolve().parents[1] / "data/outputs/color"
BACKGROUND = "#101827"


def heading(scene, title, footer):
    scene.camera.background_color = BACKGROUND
    scene.add(Text(title, font_size=34).to_edge(UP, buff=0.25))
    scene.add(Text(footer, font_size=17, color="#b9c7db").to_edge(DOWN, buff=0.18))


def image_panel(filename, height=4.4):
    image = ImageMobject(str(OUT / filename)).set_height(height)
    border = Rectangle(width=image.width + 0.08, height=image.height + 0.08, color="#94a3b8")
    border.move_to(image)
    return Group(image, border)


class CocoTransformacoes(Scene):
    def construct(self):
        heading(
            self,
            "Como cada transformação interfere na imagem",
            "COCO 2017 image 39769 • máscara verde • caixas amarelas",
        )
        effects = [
            ("Original", "coco_original.png", "pixels e rótulos originais"),
            ("Flip horizontal", "coco_flip_horizontal.png", "imagem, máscara e caixas espelhadas"),
            ("Rotação + escala", "coco_rotacao_+_escala.png", "geometria e coordenadas mudam juntas"),
            ("Cor e contraste", "coco_cor_e_contraste.png", "geometria preservada; distribuição RGB muda"),
            ("Desfoque", "coco_desfoque.png", "detalhes e bordas perdem alta frequência"),
            ("Escala de cinza", "coco_escala_de_cinza.png", "informação cromática removida"),
            ("Cutout", "coco_cutout.png", "regiões ocultas; rótulos permanecem"),
        ]
        panel = image_panel(effects[0][1]).shift(DOWN * 0.25)
        label = VGroup(
            Text(effects[0][0], font_size=29, color=YELLOW),
            Text(effects[0][2], font_size=20),
        ).arrange(DOWN, buff=0.12).to_edge(UP, buff=0.85)
        self.play(FadeIn(panel), FadeIn(label))
        self.wait(1.5)
        for name, filename, explanation in effects[1:]:
            new_panel = image_panel(filename).shift(DOWN * 0.25)
            new_label = VGroup(
                Text(name, font_size=29, color=YELLOW),
                Text(explanation, font_size=20),
            ).arrange(DOWN, buff=0.12).to_edge(UP, buff=0.85)
            self.play(Transform(panel, new_panel), Transform(label, new_label), run_time=0.8)
            self.wait(1.8)
        self.wait(1)


class CifarMisturas(Scene):
    def construct(self):
        heading(
            self,
            "Mixup e CutMix também transformam o rótulo",
            "CIFAR-10 • exemplo didático com gato e cachorro",
        )
        left = image_panel("cifar_gato.png", 2.45).move_to([-4.5, 0.6, 0])
        right = image_panel("cifar_cachorro.png", 2.45).move_to([4.5, 0.6, 0])
        middle = image_panel("cifar_mixup.png", 3.05).move_to([0, 0.6, 0])
        names = VGroup(
            Text("Gato\ny=(1, 0)", font_size=23).next_to(left, UP),
            Text("Mixup\ny=(0,60, 0,40)", font_size=23, color=YELLOW).next_to(middle, UP),
            Text("Cachorro\ny=(0, 1)", font_size=23).next_to(right, UP),
        )
        caption = Text("Combinação convexa de todos os pixels", font_size=24).move_to([0, -1.65, 0])
        self.play(FadeIn(left), FadeIn(right), FadeIn(middle), FadeIn(names), FadeIn(caption))
        self.wait(3)
        cutmix = image_panel("cifar_cutmix.png", 3.05).move_to(middle)
        new_name = Text("CutMix\ny=(0,61, 0,39)", font_size=23, color=GREEN).next_to(cutmix, UP)
        new_caption = Text("Troca de região; pesos definidos pela área", font_size=24).move_to(caption)
        self.play(Transform(middle, cutmix), Transform(names[1], new_name), Transform(caption, new_caption))
        self.wait(4)


class CifarResultados(Scene):
    def construct(self):
        heading(
            self,
            "Ganhos medidos no CIFAR-10 completo",
            "CNN pequena • 45 mil treino • 10 mil teste • média de 3 sementes",
        )
        with (OUT / "cifar10_summary.csv").open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        short = ["Sem aug.", "Geometria", "+ Cor", "+ Cutout", "+ Mixup", "+ CutMix"]
        colors = ["#64748b", BLUE, "#f59e0b", "#14b8a6", "#8b5cf6", "#ec4899"]
        baseline = -1.7
        chart = VGroup(Text("Acurácia no teste limpo (%)", font_size=25, color=YELLOW).move_to([0, 2.7, 0]))
        for tick in [0, 25, 50, 75, 100]:
            line = Rectangle(width=11.5, height=0.008, stroke_width=0, fill_color=WHITE, fill_opacity=0.2)
            line.move_to([0.2, baseline + tick * 0.034, 0])
            chart.add(line, Text(str(tick), font_size=16).move_to([-6.05, baseline + tick * 0.034, 0]))
        for index, (row, name, color) in enumerate(zip(rows, short, colors, strict=True)):
            mean = float(row["clean_accuracy_mean"])
            std = float(row["clean_accuracy_std"])
            x = -4.75 + index * 1.9
            bar = Rectangle(width=1.1, height=mean * 0.034, fill_color=color, fill_opacity=0.9, stroke_width=0)
            bar.move_to([x, baseline + mean * 0.017, 0])
            chart.add(bar)
            chart.add(Text(f"{mean:.1f} ± {std:.1f}", font_size=17).next_to(bar, UP, buff=0.1))
            chart.add(Text(name, font_size=17).move_to([x, -2.35, 0]))
        self.play(FadeIn(chart, shift=UP * 0.2), run_time=1.2)
        self.wait(5)
        self.play(FadeOut(chart))
        best = max(rows, key=lambda row: float(row["clean_accuracy_mean"]))
        baseline_row = rows[0]
        gain = float(best["clean_accuracy_mean"]) - float(baseline_row["clean_accuracy_mean"])
        conclusion = VGroup(
            Text("Resultado desta execução", font_size=35, color=GREEN),
            Text(best["policy"], font_size=31),
            Text(f"{float(best['clean_accuracy_mean']):.1f}% no teste limpo", font_size=29),
            Text(f"{gain:+.1f} pontos percentuais sobre sem augmentation", font_size=24, color=YELLOW),
        ).arrange(DOWN, buff=0.3)
        self.play(FadeIn(conclusion))
        self.wait(4)
