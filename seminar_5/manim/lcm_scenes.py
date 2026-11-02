"""Short Manim scenes following the dark visual language of seminars 3 and 4."""

from pathlib import Path

import numpy as np
from manim import (
    Arrow, Create, DOWN, Dot, FadeIn, ImageMobject, Line,
    Rectangle, Scene, Text, Transform, UP, VGroup,
)

ROOT = Path(__file__).resolve().parents[1]
BG, BLUE, GREEN, YELLOW, GRAY = "#101827", "#60a5fa", "#34d399", "#fbbf24", "#b9c7db"
Text.set_default(font="DejaVu Sans")


def heading(scene, title, footer):
    """Set a consistent scene title and attribution.

    Args:
        scene: Manim scene.
        title: Heading.
        footer: Provenance or scope label.
    """
    scene.camera.background_color = BG
    for text, edge, size, color in [(title, UP, 32, "#FFFFFF"), (footer, DOWN, 16, GRAY)]:
        label = Text(text, font_size=size, color=color)
        label.scale(min(1, 13 / label.width))
        scene.add(label.to_edge(edge, buff=.25))


def block(label, x, y, width=3.2, color=BLUE):
    """Create a labeled process block.

    Args:
        label: Block text.
        x: Horizontal coordinate.
        y: Vertical coordinate.
        width: Rectangle width.
        color: Outline and tint color.

    Returns:
        Group containing the rectangle and label.
    """
    box = Rectangle(width=width, height=1, color=color, fill_color=color, fill_opacity=.12)
    text = Text(label, font_size=22)
    text.scale(min(1, (width - .25) / text.width, .78 / text.height))
    return VGroup(box, text).move_to([x, y, 0])


class PipelineLCM(Scene):
    """Show training and inference as distinct flows."""

    def construct(self):
        """Animate the central LCD training dependencies."""
        heading(self, "LCM: professor no treino, aluno na geração",
                "Esquema próprio • Luo et al., v1, Algoritmos 1 e 3")
        top = [block("Imagem + condição\nEncoder → z₀", -4.5, 1.7),
               block("Adicionar ruído → zₜ", 0, 1.7),
               block("Professor CFG + DDIM\n→ estado em s < t", 4.5, 1.7)]
        bottom = [block("Aluno fθ(zₜ)", -4.5, -.15, color=GREEN),
                  block("Perda de consistência", 0, -.15, color=GREEN),
                  block("Alvo EMA fθ⁻(zₛ)", 4.5, -.15, color=YELLOW)]
        for index, item in enumerate(top):
            self.play(FadeIn(item), run_time=.65)
            if index:
                self.play(Create(Arrow(top[index-1].get_right(), item.get_left(), buff=.15)),
                          run_time=.4)
        self.play(*[FadeIn(item) for item in bottom])
        self.play(Create(Arrow(top[1].get_bottom(), bottom[0].get_top(), buff=.12)),
                  Create(Arrow(top[2].get_bottom(), bottom[2].get_top(), buff=.12)))
        self.play(Create(Arrow(bottom[0].get_right(), bottom[1].get_left(), buff=.12)),
                  Create(Arrow(bottom[2].get_left(), bottom[1].get_right(), buff=.12)))
        loss = Text("Mesma previsão limpa • gradiente só no aluno", font_size=25, color=GREEN)
        loss.move_to([0, -1.4, 0])
        self.play(FadeIn(loss))
        inference = Text("Inferência: ruído → aluno (1–4 passos) → decoder → imagem",
                         font_size=25, color=BLUE).move_to([0, -2.4, 0])
        self.play(FadeIn(inference))
        self.wait(3)


class ArchitectureLCM(Scene):
    """Explain what is reused and what is learned."""

    def construct(self):
        """Animate the student conditioning and boundary output."""
        heading(self, "Arquitetura: preservar a rede, aprender consistência",
                "Paper: VAE/U-Net/texto • demo: PCA/MLP/classe • não são a mesma arquitetura")
        inputs = [block("Latente ruidoso zₜ", -4.5, 1.7),
                  block("Condição c", -4.5, .2), block("Tempo t + guidance ω", -4.5, -1.3)]
        center = block("Rede do aluno\ninicializada do professor", 0, .2, width=3.6, color=GREEN)
        output = block("Boundary + previsão z₀\n→ decoder → imagem", 4.5, .2, color=YELLOW)
        for item in inputs:
            self.play(FadeIn(item), run_time=.6)
        self.play(FadeIn(center))
        for item in inputs:
            self.play(Create(Arrow(item.get_right(), center.get_left(), buff=.18)), run_time=.5)
        self.play(FadeIn(output), Create(Arrow(center.get_right(), output.get_left(), buff=.18)))
        note = Text("LCD atualiza o aluno. EMA acompanha os pesos. Professor fica congelado.",
                    font_size=23, color=GRAY).move_to([0, -2.5, 0])
        self.play(FadeIn(note))
        self.wait(3)


class ConsistencyMath(Scene):
    """Explain consistency without pretending the drawing is a numerical ODE solution."""

    def construct(self):
        """Animate two temporal points, one endpoint and an EMA arithmetic example."""
        heading(self, "Consistência: dois pontos, o mesmo destino",
                "Ilustração conceitual • pontos devem pertencer à mesma trajetória guiada")
        line = Line([-5, .6, 0], [5, .6, 0], color=GRAY)
        destination = Dot([-5, .6, 0], color=GREEN, radius=.13)
        early = Dot([1.2, .6, 0], color=YELLOW, radius=.12)
        late = Dot([5, .6, 0], color=BLUE, radius=.12)
        labels = [Text(label, font_size=23, color=color).move_to([x, 1.3, 0])
                  for label, x, color in [("z₀", -5, GREEN), ("zₛ", 1.2, YELLOW), ("zₜ", 5, BLUE)]]
        self.play(Create(line), FadeIn(destination), FadeIn(early), FadeIn(late),
                  *[FadeIn(label) for label in labels])
        self.play(Create(Arrow([4.8, .1, 0], [-4.8, .1, 0], color=BLUE, buff=0)),
                  run_time=1.5)
        self.play(Create(Arrow([1, -.4, 0], [-4.8, -.4, 0], color=YELLOW, buff=0)),
                  run_time=1.2)
        formula = Text("fθ(zₜ, t, c, ω) ≈ fθ⁻(zₛ, s, c, ω)",
                       font_size=32, color=GREEN).move_to([0, -1.25, 0])
        self.play(FadeIn(formula))
        boundary = Text("Âncora: fθ(z, 0) = z", font_size=26).move_to([0, -2, 0])
        ema = Text("EMA: 0,9 × 2 + 0,1 × 4 = 2,2", font_size=25, color=YELLOW)
        ema.move_to([0, -2.7, 0])
        self.play(FadeIn(boundary), FadeIn(ema))
        self.wait(3)


class CompareSampling(Scene):
    """Compare equivalent network evaluations without claiming hardware speedups."""

    def construct(self):
        """Animate the call count for a 50-step teacher and 4-step LCM."""
        heading(self, "Menos avaliações de rede na inferência",
                "Contagem teórica • velocidade real depende de rede, batch, hardware e decoder")
        title1 = Text("DDIM 50 passos + CFG: 100 avaliações de ramo",
                      font_size=27, color=BLUE).move_to([0, 2, 0])
        title2 = Text("LCM 4 passos: 4 avaliações do aluno",
                      font_size=27, color=GREEN).move_to([0, -.2, 0])
        self.play(FadeIn(title1))
        dots = VGroup(*[Dot([-5.6 + (i % 25) * .46, 1.3 - (i // 25) * .25, 0],
                           radius=.055, color=BLUE) for i in range(100)])
        self.play(FadeIn(dots), run_time=2)
        self.play(FadeIn(title2))
        for index in range(4):
            box = block(f"Prever z₀\npasso {index + 1}", -4.5 + 3 * index, -1.3,
                        width=2.5, color=GREEN)
            self.play(FadeIn(box), run_time=.65)
        note = Text("Entre passos LCM: reinjetar ruído em nível menor", font_size=25, color=YELLOW)
        self.play(FadeIn(note.move_to([0, -2.6, 0])))
        self.wait(3)


class LocalGeneration(Scene):
    """Show saved, actually executed model outputs."""

    def construct(self):
        """Animate clean predictions for the same initial noise and requested classes."""
        heading(self, "Geração local: DDIM e LCM k=20",
                "MNIST/PCA/MLP • seed 101 • classes 0–9 • previsões limpas, não ground truth")
        saved = np.load(ROOT / "experiments/results/samples.npz")
        selection = np.arange(10) * 30
        panels = []
        for row, key in enumerate(["ddim_history", "lcm_k20_history"]):
            label = Text("DDIM" if row == 0 else "LCM k=20", font_size=25,
                         color=BLUE if row == 0 else GREEN).move_to([-5.4, 1.3 - 2.3 * row, 0])
            self.add(label)
            for col in range(10):
                pixels = (saved[key][0, selection[col]].reshape(14, 14) * 255).astype(np.uint8)
                image = ImageMobject(pixels).set_height(.95).move_to(
                    [-4.1 + col * 1.02, 1.3 - 2.3 * row, 0]
                )
                self.add(image)
                panels.append((image, key, col))
        caption = Text("Passo 1/4", font_size=29, color=YELLOW).move_to([0, -2.6, 0])
        self.add(caption)
        self.wait(1.7)
        for step in range(1, 4):
            animations = []
            for current, key, col in panels:
                pixels = (saved[key][step, selection[col]].reshape(14, 14) * 255).astype(np.uint8)
                target = ImageMobject(pixels).set_height(.95).move_to(current)
                animations.append(Transform(current, target))
            new_caption = Text(f"Passo {step + 1}/4", font_size=29, color=YELLOW).move_to(caption)
            self.play(*animations, Transform(caption, new_caption), run_time=.8)
            self.wait(1.3)
        self.wait(2)
