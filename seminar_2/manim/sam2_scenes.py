import csv
from pathlib import Path

import numpy as np
from manim import (
    BLACK,
    DOWN,
    LEFT,
    ORIGIN,
    RIGHT,
    UP,
    WHITE,
    Arrow,
    Create,
    CurvedArrow,
    Dot,
    FadeIn,
    FadeOut,
    FadeTransform,
    GrowArrow,
    ImageMobject,
    Indicate,
    Rectangle,
    Scene,
    Text,
    Transform,
    VGroup,
    Write,
)
from PIL import Image


SEMINAR_DIR = Path(__file__).resolve().parents[1]
FRAME_DIR = SEMINAR_DIR / "data" / "raw" / "video_demo"
MASK_DIR = SEMINAR_DIR / "data" / "outputs" / "sam2_masks"
METRICS_PATH = SEMINAR_DIR / "data" / "outputs" / "demo_iou_by_frame.csv"

BACKGROUND = "#0f172a"
PANEL = "#1e293b"
BLUE = "#38bdf8"
GREEN = "#22c55e"
YELLOW = "#facc15"
ORANGE = "#f59e0b"
RED = "#ef4444"
PURPLE = "#a78bfa"
GRAY = "#94a3b8"


def load_demo():
    frame_paths = sorted(FRAME_DIR.glob("*.jpg"))
    mask_paths = sorted(MASK_DIR.glob("*.png"))
    if not frame_paths or len(frame_paths) != len(mask_paths):
        raise FileNotFoundError("Execute primeiro o notebook 03_sam2_inference.ipynb com SAM 2.")

    frames = [np.asarray(Image.open(path).convert("RGB")).copy() for path in frame_paths]
    masks = [np.asarray(Image.open(path)) > 0 for path in mask_paths]
    with METRICS_PATH.open(encoding="utf-8") as stream:
        ious = [float(row["iou"]) for row in csv.DictReader(stream)]
    if len(ious) != len(frames):
        raise ValueError("Quantidade de métricas diferente da quantidade de frames.")
    return frames, masks, ious


def mask_rgb(mask):
    return np.repeat((mask * 255).astype(np.uint8)[..., None], 3, axis=2)


def overlay(frame, mask):
    result = frame.copy()
    color = np.array([14, 210, 190], dtype=np.float32)
    result[mask] = (0.45 * result[mask] + 0.55 * color).astype(np.uint8)
    return result


def panel_image(array, x_position):
    image = ImageMobject(array).set_height(2.55).move_to([x_position, 0.55, 0])
    border = Rectangle(width=image.width + 0.12, height=image.height + 0.12, color=GRAY, stroke_width=1.5)
    border.move_to(image)
    return image, border


def prompt_position(image, pixel_x, pixel_y, frame_width, frame_height):
    return np.array(
        [
            image.get_left()[0] + pixel_x / frame_width * image.width,
            image.get_top()[1] - pixel_y / frame_height * image.height,
            0,
        ]
    )


class SAM2UsageDemo(Scene):
    def construct(self):
        self.camera.background_color = BACKGROUND
        frames, masks, ious = load_demo()
        selected = [0, 5, 6, 7, 8, 11]

        title = Text("SAM 2: um prompt, um masklet", font_size=38, weight="BOLD").to_edge(UP)
        subtitle = Text("segmentação interativa com memória temporal", font_size=23, color=GRAY)
        subtitle.next_to(title, DOWN, buff=0.12)
        labels = VGroup(
            Text("Frame original", font_size=23),
            Text("Máscara prevista", font_size=23),
            Text("Sobreposição", font_size=23),
        )
        for label, x_position in zip(labels, [-4.55, 0, 4.55], strict=True):
            label.move_to([x_position, 2.22, 0])

        frame_image, frame_border = panel_image(frames[0], -4.55)
        mask_image, mask_border = panel_image(mask_rgb(masks[0]), 0)
        overlay_image, overlay_border = panel_image(overlay(frames[0], masks[0]), 4.55)
        prompt = Dot(
            prompt_position(frame_image, 55, 128, frames[0].shape[1], frames[0].shape[0]),
            radius=0.09,
            color=GREEN,
        ).set_stroke(WHITE, width=2)
        prompt_label = Text("clique positivo", font_size=20, color=GREEN).next_to(frame_image, DOWN, buff=0.18)

        timeline_cells = VGroup(
            *[
                Rectangle(width=0.62, height=0.42, color=GRAY, stroke_width=1.4).set_fill(PANEL, opacity=1)
                for _ in range(len(frames))
            ]
        ).arrange(RIGHT, buff=0.09).move_to([0, -2.28, 0])
        timeline_numbers = VGroup(
            *[
                Text(str(index), font_size=15, color=WHITE).move_to(cell)
                for index, cell in enumerate(timeline_cells)
            ]
        )
        memory_label = Text("memória streaming", font_size=20, color=PURPLE).next_to(timeline_cells, UP, buff=0.12)

        status = Text("Frame 0  |  IoU 0.996  |  segmentação consistente", font_size=23, color=GREEN)
        status.move_to([0, -3.2, 0])

        self.play(Write(title), FadeIn(subtitle))
        self.play(FadeIn(labels), FadeIn(frame_image), Create(frame_border))
        self.play(FadeIn(prompt), FadeIn(prompt_label))
        self.play(FadeIn(mask_image), Create(mask_border), FadeIn(overlay_image), Create(overlay_border))
        self.play(FadeIn(memory_label), FadeIn(timeline_cells), FadeIn(timeline_numbers), FadeIn(status))
        self.play(timeline_cells[0].animate.set_fill(PURPLE, opacity=0.9), run_time=0.4)
        self.play(FadeOut(prompt), FadeOut(prompt_label), run_time=0.45)

        current_frame = frame_image
        current_mask = mask_image
        current_overlay = overlay_image
        current_status = status
        current_timeline = 0

        for frame_index in selected[1:]:
            next_frame, _ = panel_image(frames[frame_index], -4.55)
            next_mask, _ = panel_image(mask_rgb(masks[frame_index]), 0)
            next_overlay, _ = panel_image(overlay(frames[frame_index], masks[frame_index]), 4.55)
            if frame_index <= 5:
                message, color = "segmentação consistente", GREEN
            elif frame_index == 6:
                message, color = "subsegmentação", ORANGE
            else:
                message, color = "masklet perdido", RED
            next_status = Text(
                f"Frame {frame_index}  |  IoU {ious[frame_index]:.3f}  |  {message}",
                font_size=23,
                color=color,
            ).move_to(current_status)

            timeline_animations = [
                timeline_cells[index].animate.set_fill(PURPLE if index == frame_index else PANEL, opacity=1)
                for index in {current_timeline, frame_index}
            ]
            self.play(
                FadeTransform(current_frame, next_frame),
                FadeTransform(current_mask, next_mask),
                FadeTransform(current_overlay, next_overlay),
                Transform(current_status, next_status),
                *timeline_animations,
                run_time=1.05,
            )
            current_frame, current_mask, current_overlay = next_frame, next_mask, next_overlay
            current_timeline = frame_index

        correction = Dot(
            prompt_position(current_frame, 319, 118, frames[11].shape[1], frames[11].shape[0]),
            radius=0.09,
            color=YELLOW,
        ).set_stroke(WHITE, width=2)
        correction_text = Text(
            "Adicionar um clique corretivo no frame da falha",
            font_size=25,
            color=YELLOW,
        ).move_to([0, -3.2, 0])
        self.play(FadeOut(current_status), FadeIn(correction), FadeIn(correction_text))
        self.play(Indicate(correction, color=YELLOW, scale_factor=1.8))
        self.wait(1.5)


def flow_box(label, position, color, width=2.15, font_size=21):
    box = Rectangle(width=width, height=0.95, color=color, stroke_width=2)
    box.set_fill(PANEL, opacity=1).move_to(position)
    text = Text(label, font_size=font_size, color=WHITE, line_spacing=0.8).move_to(box)
    return VGroup(box, text)


class SAM2ArchitectureFlow(Scene):
    def construct(self):
        self.camera.background_color = BACKGROUND
        title = Text("Fluxo do SAM 2", font_size=40, weight="BOLD").to_edge(UP)
        subtitle = Text("o frame atual consulta a memória antes de gerar a máscara", font_size=23, color=GRAY)
        subtitle.next_to(title, DOWN, buff=0.12)

        frame = flow_box("Frame", [-5.65, 0.25, 0], BLUE, width=1.5)
        encoder = flow_box("Encoder\nHiera", [-3.5, 0.25, 0], GREEN, width=1.95)
        attention = flow_box("Memory\nattention", [-0.95, 0.25, 0], YELLOW, width=2.2)
        decoder = flow_box("Mask\ndecoder", [1.7, 0.25, 0], RED, width=1.95)
        mask = flow_box("Máscara", [4.25, 0.25, 0], PURPLE, width=1.65)

        prompt = flow_box("Prompt encoder\nponto • caixa\n• máscara", [1.7, -2.0, 0], BLUE, width=2.2, font_size=15)
        memory_encoder = flow_box("Memory\nencoder", [4.25, -2.0, 0], ORANGE)
        memory_bank = flow_box("Memory bank\nframes + prompts", [-0.95, -2.0, 0], PURPLE, width=2.2, font_size=18)

        main_arrows = [
            Arrow(frame.get_right(), encoder.get_left(), buff=0.12, color=GRAY),
            Arrow(encoder.get_right(), attention.get_left(), buff=0.12, color=GRAY),
            Arrow(attention.get_right(), decoder.get_left(), buff=0.12, color=GRAY),
            Arrow(decoder.get_right(), mask.get_left(), buff=0.12, color=GRAY),
        ]
        prompt_arrow = Arrow(prompt.get_top(), decoder.get_bottom(), buff=0.12, color=BLUE)
        mask_to_memory = Arrow(mask.get_bottom(), memory_encoder.get_top(), buff=0.12, color=ORANGE)
        memory_to_bank = CurvedArrow(
            memory_encoder.get_bottom(), memory_bank.get_bottom(), angle=-1.2, color=PURPLE
        )
        bank_to_attention = Arrow(memory_bank.get_top(), attention.get_bottom(), buff=0.12, color=PURPLE)

        conclusion = Text("Um prompt pode orientar vários frames", font_size=25, color=YELLOW)
        conclusion.move_to([0, 1.65, 0])

        self.play(Write(title), FadeIn(subtitle))
        self.play(FadeIn(frame))
        for box, arrow in zip([encoder, attention, decoder, mask], main_arrows, strict=True):
            self.play(GrowArrow(arrow), FadeIn(box), run_time=0.65)

        self.play(FadeIn(prompt), GrowArrow(prompt_arrow))
        self.play(Indicate(decoder, color=RED), run_time=0.8)
        self.play(GrowArrow(mask_to_memory), FadeIn(memory_encoder))
        self.play(Create(memory_to_bank), FadeIn(memory_bank))
        self.play(GrowArrow(bank_to_attention))
        self.play(Indicate(memory_bank, color=PURPLE), Indicate(attention, color=YELLOW))
        self.play(Write(conclusion))
        self.wait(1.5)


if __name__ == "__main__":
    frames, masks, ious = load_demo()
    assert len(frames) == len(masks) == len(ious) == 12
    assert np.isclose(ious[:5], 1, atol=0.01).all()
