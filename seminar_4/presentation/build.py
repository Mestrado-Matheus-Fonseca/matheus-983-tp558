"""Gere diagramas, gráficos auditáveis, PPTX editável e PDF a partir do roteiro."""

import csv
from decimal import Decimal, ROUND_HALF_UP
import json
from pathlib import Path
import subprocess
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.textpath import TextPath
import numpy as np
from PIL import Image

from pptx import write_pptx

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "assets/figures"
OUT = ROOT / "data/outputs"
PRES = ROOT / "presentation"
BLUE, GREEN, ORANGE, GRAY = "#2563eb", "#14b8a6", "#f59e0b", "#64748b"


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def diagram(name, labels, rows, caption=""):
    """Diagrama funcional em linhas alternadas, com rótulos legíveis."""
    fig, ax = plt.subplots(figsize=(14, 5.1))
    ax.set(xlim=(0, 14), ylim=(0, 5))
    ax.axis("off")
    coords, colors = [], [BLUE, GREEN, ORANGE]
    for row, count in enumerate(rows):
        xs = np.linspace(1.9, 12.1, count) if count > 1 else [7]
        if row % 2:
            xs = xs[::-1]
        coords.extend([(float(x), 4.15-row*1.55, colors[row % 3]) for x in xs])
    for (x, y, color), label in zip(coords, labels, strict=True):
        ax.add_patch(FancyBboxPatch((x-1.4, y-.52), 2.8, 1.04, boxstyle="round,pad=0.05",
                                   facecolor=color+"18", edgecolor=color, linewidth=2))
        ax.text(x, y, label, ha="center", va="center", fontsize=16, color="#172033")
    for (x1, y1, _), (x2, y2, _) in zip(coords[:-1], coords[1:], strict=True):
        if y1 == y2:
            direction = 1 if x2 > x1 else -1
            start, end = (x1+direction*1.5, y1), (x2-direction*1.5, y2)
        else:
            start, end = (x1, y1-.6), (x2, y2+.6)
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=18,
                                    color=GRAY, linewidth=2))
    if caption:
        ax.text(7, .1, caption, ha="center", fontsize=14, color=GRAY)
    save(fig, name)


def branches(name, nodes, connections, caption=""):
    """Desenhe saídas paralelas sem sugerir uma cadeia entre elas."""
    fig, ax = plt.subplots(figsize=(14, 5.1))
    ax.set(xlim=(0, 14), ylim=(0, 5))
    ax.axis("off")
    boxes = []
    for text, x, y, color in nodes:
        box = FancyBboxPatch((x-1.4, y-.58), 2.8, 1.16, boxstyle="round,pad=0.05",
                             facecolor=color+"18", edgecolor=color, linewidth=2)
        ax.add_patch(box)
        boxes.append(box)
        ax.text(x, y, text, ha="center", va="center", fontsize=16)
    for source, target in connections:
        ax.add_patch(FancyArrowPatch(nodes[source][1:3], nodes[target][1:3], patchA=boxes[source],
                                    patchB=boxes[target], arrowstyle="-|>", mutation_scale=18,
                                    color=GRAY, linewidth=2))
    if caption:
        ax.text(7, .05, caption, ha="center", fontsize=14, color=GRAY)
    save(fig, name)


def diagrams():
    diagram("pipeline", ["RGB → resize\n1120 × 1120", "VAE / U-Net\nSD2 congelado", "Autoatenção\nafinidade A", "Grade one-hot\npropagação p", "Normalize / KL\nclustering / níveis",
                         "Upsample / argmax\ncomponentes / NMS", "Pseudo-máscaras\nrefino opcional", "Mask2Former\nRodada 1", "Global + crops\nnovos rótulos", "Rodada 2\ninferência zero-shot"], [3, 4, 3])
    diagram("extraction", ["Imagem RGB", "Resize 1120²\nVAE → latent", "U-Net SD2\nt = 150", "Cabeças / camadas\nautoatenção A"], [4],
            "Uma passagem • embeddings de texto nulos • sem adicionar ruído forward ao latent")
    branches("student", [("Imagem", 1.9, 4.1, BLUE), ("ResNet-50\nfeatures\nmultiescala", 5.3, 4.1, BLUE),
                          ("Pixel decoder", 8.7, 4.1, BLUE), ("Decoder\n2.000 queries", 12.1, 4.1, BLUE),
                          ("Máscaras", 8.7, 2.25, GREEN), ("Objectness", 12.1, .9, GREEN),
                          ("Pseudo-rótulos", 1.9, 1.25, ORANGE), ("Loss Dice / BCE\nclassificação", 5.3, 1.25, ORANGE)],
             [(0, 1), (1, 2), (2, 3), (3, 4), (3, 5), (4, 7), (5, 7), (6, 7)],
            "Arquitetura funcional do estudante • a extração SD2 fornece rótulos para treinamento")
    diagram("crop_training", ["Rodada 1\npseudo-rótulos", "Inferência global\npais previstos", "Recortar / ampliar\n1024 × 1024", "Predizer filhos", "Checar contenção\n≥ 0,70", "Mesclar rótulos\nIoU > 0,50", "Rodada 2\n100k imagens"], [4, 3])
    branches("hierarchy", [("Mapas suaves", 1.9, 4.1, BLUE), ("Normalize soma 1", 5.3, 4.1, BLUE),
                            ("KL simétrica\nentre prompts", 8.7, 4.1, BLUE), ("Ligação média", 12.1, 4.1, BLUE),
                            ("h pequeno\nmais grupos / partes", 10.8, 2.3, GREEN),
                            ("h grande\nmenos grupos / objetos", 6.7, 2.3, GREEN),
                            ("Cada nível:\nupsample / argmax\ncomponentes", 2.5, 1, ORANGE)],
             [(0, 1), (1, 2), (2, 3), (3, 4), (3, 5), (4, 6), (5, 6)])
    diagram("sar_application", ["Pesquisa atual\nSAM ajustado SAR", "Ablar crops\nmesmo checkpoint", "Auditar\npseudo-máscaras", "Treinar / testar\nIoU, Dice, FP"], [4],
            "Proposta futura • SD2→SAR é hipótese separada • nenhuma transferência foi validada")
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.set(xlim=(0, 13), ylim=(0, 4.5))
    ax.axis("off")
    ax.add_patch(Rectangle((.3, 2.3), 5.5, 1.8, color="#bfdbfe"))
    ax.add_patch(Rectangle((.3, .4), 5.5, 1.9, color="#a7f3d0"))
    ax.add_patch(Rectangle((2.1, 1), 2.3, 1.8, color=ORANGE))
    ax.add_patch(Rectangle((3.4, 1.7), .6, .7, color="#fef3c7"))
    ax.text(.7, 3.1, "Céu: stuff", fontsize=20)
    ax.text(.7, .7, "Solo: stuff", fontsize=20)
    ax.text(7, 3.4, "Objeto inteiro → things", fontsize=23, color=BLUE)
    ax.text(7, 2.3, "Região interna → parte", fontsize=23, color=ORANGE)
    ax.text(7, 1, "Sem nome de classe\nMáscaras entre níveis podem se sobrepor", fontsize=18)
    save(fig, "entities")
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.axis("off")
    ax.text(.03, .78, "DINO / UnSAM", fontsize=24, color=GRAY, transform=ax.transAxes)
    ax.text(.03, .52, "DiffSeg / M2N2", fontsize=24, color=BLUE, transform=ax.transAxes)
    ax.text(.03, .26, "Diffuse2Seg", fontsize=24, color=GREEN, transform=ax.transAxes)
    for y, text in zip([.78, .52, .26], ["Descoberta SSL de objetos; hierarquias e detector no UnSAM",
                                        "Afinidades generativas; fusão ou prompts interativos",
                                        "Grade automática + propagação + várias granularidades"], strict=True):
        ax.text(.34, y, textwrap.fill(text, 52), fontsize=19, transform=ax.transAxes, va="center")
    save(fig, "prior_work")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    n = 12
    a = np.full((n, n), .02)
    a[:6, :6] = .15
    a[6:, 6:] = .15
    a /= a.sum(1, keepdims=True)
    axes[0].imshow(a, cmap="Blues")
    axes[0].set_title("A ilustrativa: duas regiões afins", fontsize=18)
    axes[0].set_xlabel("Token j", fontsize=16)
    axes[0].set_ylabel("Token i", fontsize=16)
    axes[1].axis("off")
    axes[1].text(.02, .81, r"$A=\sum_l w_l\frac{1}{n_H}\sum_h A^{(l,h)}$", fontsize=28, va="center")
    axes[1].text(.02, .43, "w₁ = 0,85    w₂ = 0,15\nTemperatura de atenção: 0,55", fontsize=23, linespacing=1.5, va="top")
    axes[1].text(.02, .1, "Esquema conceitual; não é atenção SD2 medida", fontsize=15, color=GRAY)
    fig.tight_layout()
    save(fig, "affinity")
    fig, ax = plt.subplots(figsize=(14, 4.7))
    ax.axis("off")
    ax.text(.5, .83, r"$E(f)=\frac{1}{p}\sum_i\left[\sum_j A_{ij}(f_j-f_i)^2\right]^{p/2}+\frac{\lambda}{2}\|f-f^0\|_2^2$",
            fontsize=27, ha="center", va="center", transform=ax.transAxes)
    ax.text(.08, .44, "Coerência sobre o grafo\np controla a regularização", fontsize=24, color=BLUE,
            transform=ax.transAxes, linespacing=1.5, va="top")
    ax.text(.64, .44, "Ancoragem ao ponto\nλ preserva o prompt", fontsize=24, color=GREEN,
            transform=ax.transAxes, linespacing=1.5, va="top")
    ax.text(.5, .09, "Gauss-Jacobi • parar quando a atualização quadrática é pequena", ha="center",
            fontsize=21, transform=ax.transAxes, color=GRAY)
    save(fig, "energy")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax in axes:
        ax.set(xlim=(0, 5), ylim=(0, 4))
        ax.axis("off")
        ax.add_patch(Rectangle((.5, .4), 3.5, 2.8, facecolor=BLUE+"30", edgecolor=BLUE, linewidth=2))
    axes[0].add_patch(Rectangle((.6, .5), 3.4, 2.6, facecolor=GREEN+"40", edgecolor=GREEN, linewidth=2))
    axes[0].set_title("Candidatos quase idênticos → rejeitar", fontsize=18)
    axes[1].add_patch(Rectangle((1.2, 1), .9, 1, facecolor=GREEN+"60", edgecolor=GREEN, linewidth=2))
    axes[1].set_title("Parte dentro do objeto → pode preservar", fontsize=18)
    fig.tight_layout()
    save(fig, "nms")
    branches("datasets", [("Pseudo-rótulos / treino\nSA-1B\n100k / 200k / 400k", 1.9, 3, BLUE),
                           ("Validar parâmetros\nHoldout SA-1B\n1.000 imagens", 5.3, 3, BLUE),
                           ("Avaliar gerador\nSA-1B / ADE / Entity\nUVO / PACO", 8.7, 3, GREEN),
                           ("Avaliar estudante\nCOCO / LVIS / ADE\nEntity / SA-1B / PACO", 12.1, 3, GREEN)], [],
            "Parâmetros fixos entre domínios • zero-shot avalia transferência sem ajuste ao alvo")
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.axis("off")
    ax.text(.04, .74, "IoU = interseção / união", fontsize=27, color=BLUE, transform=ax.transAxes)
    ax.text(.04, .40, "Uma proposta cobre o objeto\nse a IoU supera o limiar", fontsize=23, transform=ax.transAxes)
    ax.text(.56, .74, "0,50  0,55  …  0,90  0,95", fontsize=25, color=GREEN, transform=ax.transAxes)
    ax.text(.56, .40, "Calcular recall em cada limiar\n→ média com até 1.000 propostas", fontsize=23,
            transform=ax.transAxes)
    ax.text(.5, .08, "AR alto não garante precisão, poucos falsos positivos ou a classe desejada", ha="center",
            fontsize=19, color=GRAY, transform=ax.transAxes)
    save(fig, "metrics")


def rounded_average(values):
    return float((sum(Decimal(str(v)) for v in values) / Decimal(len(values))).quantize(
        Decimal(".1"), rounding=ROUND_HALF_UP))


def grouped_bars(ax, columns, series, colors, maximum):
    x = np.arange(len(columns))
    width = .78 / len(series)
    for i, ((name, values), color) in enumerate(zip(series.items(), colors, strict=True)):
        bars = ax.bar(x + (i-(len(series)-1)/2)*width, values, width=width, color=color, label=name)
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=13)
    ax.set_xticks(x, columns, fontsize=15)
    ax.set_ylim(0, maximum)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_ylabel("AR1000 (%) • resultado publicado", fontsize=15)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=14, loc="upper center", bbox_to_anchor=(.5, 1.18), ncol=len(series), frameon=False)


def charts():
    paper = json.loads((ROOT / "data/paper_results.json").read_text())
    processed = ROOT / "data/processed/paper_results"
    processed.mkdir(parents=True, exist_ok=True)
    for table in ["table1", "table2", "table5a", "table5b"]:
        columns = paper.get(table+"_columns", ["Things", "Stuff+Things", "Parts"])
        with (processed / f"{table}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["method", *columns])
            writer.writerows([[k, *v] for k, v in paper[table].items()])
    with (processed / "table3.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", "images", "detector", *paper["table3_columns"], "Things_mean", "StuffThings_mean"])
        for row in paper["table3"]:
            writer.writerow([row["method"], row["images"], row["detector"], *row["values"],
                             rounded_average(row["values"][:2]), rounded_average(row["values"][2:5])])
    for name, header in [("levels", ["levels", "AR1000", "ARS", "ARM", "ARL", "mAP"]),
                         ("nms", ["nms_iou", "AR1000", "ARS", "ARM", "ARL", "mAP"])]:
        with (processed / f"table4_{name}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(paper[f"table4_{name}"])
    gains = {domain: round(ours-base, 1) for domain, ours, base in zip(paper["table2_columns"],
                paper["table2"]["Diffuse2Seg"], paper["table2"]["UnSAM"], strict=True)}
    assert list(gains.values()) == [4.3, 6.5, 4.5, 7.1, 4.3]
    (processed / "recalculated_gains.json").write_text(json.dumps(gains, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(13, 4.5))
    grouped_bars(ax, ["Pequenos", "Médios", "Grandes", "Todos"],
                 {k: paper["table1"][k] for k in ["UnSAM", "DiffSeg", "Diffuse2Seg"]}, [GRAY, BLUE, GREEN], 49)
    ax.set_ylabel("Recall por escala / AR1000 (%)", fontsize=15)
    fig.tight_layout()
    save(fig, "paper_sizes")
    fig, ax = plt.subplots(figsize=(13, 4.5))
    grouped_bars(ax, paper["table2_columns"], {k: paper["table2"][k] for k in ["UnSAM", "DiffSeg", "Diffuse2Seg"]},
                 [GRAY, BLUE, GREEN], 39)
    fig.tight_layout()
    save(fig, "paper_domains")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for ax, rows, title in zip(axes, [[paper["table3"][2], paper["table3"][3]], paper["table3"][-2:]],
                               ["Sem detector: 200k SOHES / 100k Diffuse2Seg", "Com CutLER: 400k imagens para ambos"], strict=True):
        series = {r["method"]: [rounded_average(r["values"][:2]), rounded_average(r["values"][2:5]), r["values"][5]] for r in rows}
        grouped_bars(ax, ["Things", "Stuff+Things", "Parts"], series, [GRAY, GREEN], 56)
        ax.set_title(title, fontsize=14, pad=40)
    fig.tight_layout()
    save(fig, "paper_students")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    for ax, data, xlabel in zip(axes[:2], [paper["table4_levels"], paper["table4_nms"]],
                               ["Número de níveis", "Limiar NMS"], strict=True):
        values = [r[1] for r in data]
        bars = ax.bar([str(r[0]) for r in data], values, color=BLUE)
        ax.bar_label(bars, fmt="%.1f", fontsize=14, padding=4)
        ax.set(xlabel=xlabel, ylabel="AR1000 (%)", ylim=(0, 25))
        ax.tick_params(labelsize=14)
        ax.spines[["top", "right"]].set_visible(False)
    values = [v[4] for v in paper["table5a"].values()]
    bars = axes[2].bar(["Globais", "+ crops", "+ CutLER"], values, color=[GRAY, BLUE, GREEN])
    axes[2].bar_label(bars, fmt="%.1f", fontsize=14, padding=4)
    axes[2].set(xlabel="Fonte dos rótulos • SA-1B", ylabel="AR1000 (%)", ylim=(0, 55))
    axes[2].tick_params(labelsize=13)
    axes[2].spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "paper_ablations")
    local = json.loads((ROOT / "experiments/results/ablations.json").read_text())
    fig, ax = plt.subplots(figsize=(12, 4.2))
    bars = ax.bar([f"{r['levels']} níveis\nNMS {r['nms_iou']}" for r in local],
                  [100*r["toy_recall_50_95"] for r in local], color=[GRAY, BLUE, GRAY, BLUE])
    ax.bar_label(bars, fmt="%.1f", fontsize=16, padding=4)
    ax.set_ylim(0, 40)
    ax.set_ylabel("Recall toy (%)", fontsize=17)
    ax.tick_params(labelsize=16)
    ax.set_title("Ablação local • p=1,6 • semente 7 • matching 1:1", fontsize=17)
    fig.tight_layout()
    save(fig, "toy_ablations")
    data = np.load(ROOT / "experiments/results/synthetic_seed7_p1.6.npz")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    for ax, values, title in zip(axes, [data["image"], data["maps"][9].reshape(24, 24), data["levels"][0]],
                                 ["Imagem sintética", "Mapa suave • prompt 9", "Partição fina • p=1,6"], strict=True):
        ax.imshow(values, cmap="tab20" if "Partição" in title else "viridis")
        ax.set_title(title, fontsize=18)
        ax.axis("off")
    fig.tight_layout()
    save(fig, "toy_overview")


def extract_originals():
    # Coordenadas na página rasterizada com lado maior de 1600 px; recortes científicos.
    for page, name, box in [(2, "paper_figure1", (145, 165, 945, 425)),
                            (13, "paper_figure5", (145, 155, 955, 295))]:
        x, y, w, h = box
        subprocess.run(["pdftoppm", "-f", str(page), "-l", str(page), "-singlefile", "-scale-to", "1600",
                        "-x", str(x), "-y", str(y), "-W", str(w), "-H", str(h), "-png",
                        str(ROOT / "articles/2609.06491v1.pdf"), str(FIG / name)], check=True)


def text_element(text, x, y, w, h, size=20, color="#172033", bold=False):
    return {"type": "text", "text": text, "x": x, "y": y, "w": w, "h": h,
            "size": size, "color": color, "bold": bold}


def wrap_measured(text, inches, points):
    """Quebre linhas pela largura real da fonte usada no PDF/PPTX."""
    prop = FontProperties(family="DejaVu Sans", size=points)
    lines, current = [], ""
    for word in text.split():
        candidate = (current + " " + word).strip()
        if current and TextPath((0, 0), candidate, prop=prop).get_extents().width > inches*72:
            lines.append(current)
            current = word
        else:
            current = candidate
    return "\n".join([*lines, current])


def slide_elements(slide, index, total):
    width = 13.333333
    source = slide["source"]
    elements = [{"type": "rect", "x": .5, "y": .35, "w": .72, "h": .06, "fill": BLUE},
                text_element(textwrap.fill(slide["title"], 62), .5, .57, 12.3, .85, 28, bold=True),
                text_element("Inatel • TP558", .5, 7.02, 2, .23, 11, BLUE, True),
                text_element(textwrap.shorten(source, width=103, placeholder="…"), 2.4, 7.02, 10.1, .23, 10, GRAY),
                text_element(f"{index:02d}/{total}", 12.53, 7.02, .6, .23, 10, GRAY)]
    figure_name = slide.get("figure")
    if slide.get("subtitle"):
        elements.append(text_element(slide["subtitle"], .5, 1.08, 12, .28, 17, GRAY))
    if figure_name:
        figure = FIG / figure_name
        if not figure.exists():
            figure = OUT / figure_name
        if not figure.exists():
            raise FileNotFoundError(figure)
        with Image.open(figure) as image:
            ratio = image.width/image.height
        w, h = 12.25, 4.25
        if w/h > ratio:
            w = h * ratio
        else:
            h = w / ratio
        elements.append({"type": "image", "path": str(figure), "x": (width-w)/2, "y": 1.35+(4.25-h)/2,
                         "w": w, "h": h})
        for i, bullet in enumerate(slide["bullets"]):
            x = .5 + i*4.17
            elements.append({"type": "rect", "x": x, "y": 5.78, "w": 3.96, "h": .90, "fill": "#eff6ff"})
            elements.append(text_element(wrap_measured(bullet, 3.55, 16), x+.14, 5.9, 3.68, .73, 16))
    else:
        for i, bullet in enumerate(slide["bullets"]):
            y = 2 + i*1.18
            elements.append(text_element(f"{i+1:02d}", .55, y, .75, .65, 28, BLUE, True))
            elements.append(text_element(textwrap.fill(bullet, 60), 1.55, y, 11, 1, 25))
    return elements


def draw_slide(elements):
    fig = plt.figure(figsize=(13.333333, 7.5), facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, 13.333333), ylim=(0, 7.5))
    ax.axis("off")
    for e in elements:
        if e["type"] == "text":
            ax.text(e["x"], 7.5-e["y"], e["text"], fontsize=e["size"], color=e["color"],
                    weight="bold" if e.get("bold") else "normal", ha="left", va="top", linespacing=1.1,
                    family="DejaVu Sans")
        elif e["type"] == "rect":
            ax.add_patch(Rectangle((e["x"], 7.5-e["y"]-e["h"]), e["w"], e["h"], facecolor=e["fill"], edgecolor="none"))
        elif e["type"] == "image":
            ax.imshow(Image.open(e["path"]), extent=(e["x"], e["x"]+e["w"], 7.5-e["y"]-e["h"], 7.5-e["y"]),
                      aspect="auto", interpolation="lanczos")
    return fig


def report():
    protocol = json.loads((ROOT / "experiments/results/protocol.json").read_text())
    lines = ["# Relatório dos experimentos executados", "", "Tipo: **reprodução parcial dos algoritmos com afinidade proxy**. "
             "Nenhuma extração SD2, inferência oficial Diffuse2Seg ou treinamento Mask2Former foi executado.", "",
             "| p | Recall toy médio (%) | Std amostral (p.p.) | Mean best IoU |", "|---|---:|---:|---:|"]
    for row in protocol["summary"]:
        lines.append(f"| {row['p']} | {100*row['recall_mean']:.2f} | {100*row['recall_std']:.2f} | {row['iou_mean']:.4f} |")
    lines.extend(["", "As sementes 7/23/42 modificam somente ruído RGB numa mesma cena/GT; não são três cenas independentes. "
                  "O melhor p neste proxy é 2. Não houve reprodução do ganho não linear publicado. "
                  "Não ajustar parâmetros pelo melhor resultado desta demonstração e depois tratá-la como teste independente.", "",
                  "## Configuração e métrica", "", "Entrada 24×24, 576 nós, 16 prompts automáticos. "
                  "Afinidade simétrica é kernel Gaussiano de diferença RGB e distância espacial, "
                  "sem GT. Normalização global pelo maior grau preserva simetria. "
                  "λ=0,03; epsilon de gradiente=1e−8; tolerância de atualização quadrática=1e−8; "
                  "máximo400 iterações; mínimo3 pixels; NMS0,9. "
                  "Seis thresholds fixos aproximadamente log-espaçados entre0,186 e2,99. "
                  "Dados e configurações estão em protocol.json e nos NPZ.", "",
                  "A cena tem cinco regiões disjuntas; a referência também inclui a máscara inteira do retângulo "
                  "com sua parte, totalizando seis referências multigranulares. O GT entra apenas na avaliação. "
                  "Recall toy é média de recall nos dez limiares IoU0,50…0,95, com matching greedy 1:1 por IoU, "
                  "sem scores de objectness, API COCO ou categoria. Cada referência e cada proposta só casa uma vez. "
                  "Mean best IoU permite a melhor proposta por referência e é uma medida complementar. "
                  "Nenhuma dessas métricas é o AR1000 oficial do artigo.", "",
                  "## Verificação e convergência", "",
                  f"O caso p=2 foi comparado à solução fechada `[2L+λI]f=λf⁰`: erro máximo "
                  f"{protocol['checks']['linear_closed_form_max_error']:.3e}. "
                  "Também são verificados simetria KL, mapas iguais, IoU perfeito, ausência de propostas e não duplicação de matching. "
                  "As nove execuções atingiram o critério de atualização; isso é diagnóstico numérico, não prova de segmentação correta. "
                  "O epsilon evita potência negativa de gradiente zero. O solver usa grafo simétrico, "
                  "enquanto o tratamento de simetria da atenção original precisaria ser confirmado.", "",
                  "## Ablação local", "", "| Níveis | NMS | Candidatos | Propostas | Recall toy (%) |",
                  "|---:|---:|---:|---:|---:|"])
    for row in json.loads((ROOT / "experiments/results/ablations.json").read_text()):
        lines.append(f"| {row['levels']} | {row['nms_iou']} | {row['candidates_before_nms']} | {row['proposals']} | {100*row['toy_recall_50_95']:.2f} |")
    lines.extend(["", "p=1,6, semente7. Mais níveis aumentam candidatos sem aumentar recall neste caso. "
                  "NMS0,9 preserva propostas úteis que NMS0,5 rejeita.", "",
                  "## Diferenças para o artigo", "", "| Elemento | Artigo | Execução local |",
                  "|---|---|---|", "| Representação | SD2 VAE/U-Net congelados | RGB/posição, sem features aprendidas |",
                  "| Resolução | RGB1120²; latent140² | 24², sem VAE |",
                  "| Ancoragem/parada | λ1e−5; update²1e−4 | λ0,03; update²1e−8 |",
                  "| Saída | Upsampling, componentes, NMS, CascadePSP opcional | Componentes4-conexos e NMS na grade original |",
                  "| Avaliação | Benchmarks naturais, AR1000 | Uma cena toy, métrica própria |",
                  "| Treinamento | Duas rodadas Mask2Former | Não realizado |", "",
                  "As diferenças de λ, escala de A, epsilon e parada alteram a geometria dos mapas e as distâncias KL. "
                  "Manter os thresholds do artigo não calibra automaticamente um grafo RGB. "
                  "No p=1,6, grupos colapsam e entidades desaparecem; o mapa suave não é suficiente para preservar a hierarquia.", "",
                  "## Demonstração em fotografia", "",
                  f"Imagem COCO39769 existente no ciclo3, com grafo reduzido a24²: p=1,6 gera "
                  f"{protocol['coco_concept']['proposals']} proposta final; p=2 gera "
                  f"{protocol['coco_concept']['linear_comparison']['proposals']} propostas. "
                  "A figura apresenta a fotografia original e as duas partições finas calculadas na grade. "
                  "Não se usam anotações e não se mediu recall COCO. As partições mostram agrupamento por aparência, "
                  "sem demonstrar descoberta semântica de gatos ou controles remotos.", "",
                  "## Por que o pipeline integral não foi executado", "",
                  "O ambiente disponível registra PyTorch CPU, sem CUDA e sem diffusers/checkpoints SD2. "
                  "Não foi localizado código/checkpoint oficial verificável do Diffuse2Seg na consulta. "
                  "O experimento original usa 100k–400k imagens e treinamento em quatro A100; "
                  "a afinidade densa de19600tokens sozinha tem1,43GiB emFP32. "
                  "CPU não torna toda inferência SD2 impossível, mas uma configuração arbitrária não validaria o resultado oficial. "
                  "Priorizou-se a alternativa parcial solicitada, sem download de pesos e sem prometer resultados de benchmarks.", "",
                  "## Conclusão sustentada", "", "A implementação demonstra operações e reproduz o caso linear verificável. "
                  "Os resultados medidos deixam claro que a afinidade é central para a hipótese Diffuse2Seg. "
                  "Não validam nem refutam seus números originais, e não constituem evidência SAR."])
    (ROOT / "notes/experimental_report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    (PRES / "slides").mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 14})
    diagrams()
    charts()
    extract_originals()
    slides = json.loads((PRES / "slides.json").read_text())
    for slide in slides:
        if slide.get("figure") == "toy_pipeline.png":
            slide["figure"] = "toy_overview.png"
    # Roteiro é a fonte editável, independente do formato de saída.
    (PRES / "slides.json").write_text(json.dumps(slides, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    native_slides, guide = [], ["# Roteiro de fala e estrutura dos slides", "",
                               f"{len(slides)} slides; os quatro últimos são apoio. Tempo principal estimado: "
                               f"{sum(s['seconds'] for s in slides)/60:.1f} minutos, incluindo demonstrações curtas.", "",
                               "PPTX contém texto editável e notas; o PDF é alternativa estática. "
                               "Vídeos/GIFs são arquivos separados, indicados no roteiro; não estão embutidos no PPTX.", ""]
    with PdfPages(PRES / "TP558_Diffuse2Seg.pdf") as pdf:
        for i, slide in enumerate(slides, 1):
            elements = slide_elements(slide, i, len(slides))
            fig = draw_slide(elements)
            fig.savefig(PRES / "slides" / f"slide_{i:02d}.png", dpi=135)
            pdf.savefig(fig)
            plt.close(fig)
            native_slides.append({"elements": elements, "notes": slide["notes"] + "\n\nFonte: " + slide["source"]})
            guide.extend([f"## {i:02d}. {slide['title']}", "", "\n".join("- " + b for b in slide["bullets"]), "",
                          f"**Fonte/tipo:** {slide['source']}", "", f"**Fala ({slide['seconds']}s):** {slide['notes']}", ""])
            if slide.get("figure"):
                where = "../assets/figures" if (FIG / slide["figure"]).exists() else "../data/outputs"
                guide.extend([f"**Visual:** [{slide['figure']}]({where}/{slide['figure']})", ""])
            if slide.get("video"):
                video = slide["video"]
                guide.extend([f"**Animação:** [{video}.mp4](../data/outputs/manim/{video}.mp4) "
                              f"ou [{video}.gif](../data/outputs/manim/{video}.gif)", ""])
    write_pptx(PRES / "TP558_Diffuse2Seg.pptx", native_slides, ROOT.parent / "seminar_1/TP558_X_Template.pptx")
    (PRES / "speaker_guide.md").write_text("\n".join(guide), encoding="utf-8")
    report()
    print(f"Gerados {len(slides)} slides PPTX/PDF, notas, gráficos e diagramas.")


if __name__ == "__main__":
    main()
