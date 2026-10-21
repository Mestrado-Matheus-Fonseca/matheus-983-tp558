"""Create publication-style figures exclusively from recorded source values."""

import csv
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from publication_plot_style import apply_publication_style

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets/figures"
COLORS = ["#365f85", "#8b6550", "#778562", "#8a7394", "#b58034"]


def save(figure, name):
    """Save the same figure in PNG and vector PDF.

    Args:
        figure: Matplotlib figure.
        name: Output basename.
    """
    for extension in ["png", "pdf"]:
        figure.savefig(OUT / f"{name}.{extension}", dpi=220, bbox_inches="tight")
    plt.close(figure)


def clean(axis, xlabel, ylabel):
    """Apply readable labels and restrained axes.

    Args:
        axis: Matplotlib axes.
        xlabel: Horizontal label.
        ylabel: Vertical label.
    """
    axis.set(xlabel=xlabel, ylabel=ylabel)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", linestyle="--", alpha=0.2)


def main():
    """Generate plots, image grids and a measured experimental report."""
    OUT.mkdir(parents=True, exist_ok=True)
    images = ROOT / "assets/images"
    images.mkdir(parents=True, exist_ok=True)
    paper_pdf = ROOT / "articles/2310.04378v1.pdf"
    for page in [2, 8]:
        subprocess.run(["pdftoppm", "-f", str(page), "-singlefile", "-scale-to", "1400",
                        "-png", str(paper_pdf), str(images / f"paper_page{page}")], check=True)
    for name, top, height in [("paper_figure2", 280, 510), ("paper_figure3", 810, 460)]:
        subprocess.run(["pdftoppm", "-f", "8", "-singlefile", "-scale-to", "2800",
                        "-x", "370", "-y", str(top), "-W", "1420", "-H", str(height),
                        "-png", str(paper_pdf), str(images / name)], check=True)
    apply_publication_style()
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 12,
                         "axes.titlesize": 14, "axes.labelsize": 12})
    paper = json.loads((ROOT / "data/paper_results.json").read_text())
    for metric in ["FID", "CLIP"]:
        figure, axes = plt.subplots(1, 2, figsize=(11.5, 4), layout="constrained")
        for axis, resolution in zip(axes, ["512", "768"]):
            for index, (method, values) in enumerate(paper[resolution].items()):
                axis.plot(paper["steps"], values[metric], "o-", label=method,
                          color=COLORS[index], lw=1.7, ms=4)
            clean(axis, "Passos de inferência", f"{metric} {'↓' if metric == 'FID' else '↑'}")
            axis.set_title(f"Artigo: {resolution} × {resolution}")
            axis.set_xticks(paper["steps"])
        axes[1].legend(frameon=False, fontsize=10)
        save(figure, f"paper_{metric.lower()}")
    records = list(csv.DictReader((ROOT / "experiments/results/metrics.csv").open()))
    curves = list(csv.DictReader((ROOT / "experiments/results/training.csv").open()))
    figure, axes = plt.subplots(1, 3, figsize=(11.5, 3.7), layout="constrained")
    for axis, phase, title in zip(axes, ["teacher", "lcm_k1", "lcm_k20"],
                                 ["Professor: ruído", "LCD: k = 1", "LCD: k = 20"]):
        rows = [r for r in curves if r["phase"] == phase]
        axis.plot([int(r["step"]) for r in rows], [float(r["loss"]) for r in rows],
                  color=COLORS[0], lw=1.2)
        axis.set_title(title)
        axis.set_yscale("log")
        clean(axis, "Atualização", "MSE de treino (log)")
    save(figure, "training")
    aggregate = []
    for method in ["ddim", "lcm_k1", "lcm_k20"]:
        for count in sorted({int(r["steps"]) for r in records if r["method"] == method}):
            rows = [r for r in records if r["method"] == method and int(r["steps"]) == count]
            item = {"method": method, "steps": count, "n_generation_seeds": len(rows)}
            for metric in ["class_agreement", "latent_mse_vs_ddim50", "latent_variance",
                           "ms_per_sample", "batch_seconds"]:
                values = [float(r[metric]) for r in rows]
                item[f"{metric}_mean"] = float(np.mean(values))
                item[f"{metric}_std"] = float(np.std(values, ddof=1))
            aggregate.append(item)
    (ROOT / "experiments/results/aggregate.json").write_text(
        json.dumps(aggregate, indent=2) + "\n"
    )
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.1), layout="constrained")
    for method, label, color in zip(["ddim", "lcm_k1", "lcm_k20"],
                                   ["DDIM", "LCM k=1", "LCM k=20"], COLORS):
        rows = [r for r in aggregate if r["method"] == method and r["steps"] <= 8]
        x = [r["steps"] for r in rows]
        axes[0].errorbar(x, [r["class_agreement_mean"] * 100 for r in rows],
                        yerr=[r["class_agreement_std"] * 100 for r in rows],
                        fmt="o-", label=label, color=color, capsize=3)
        axes[1].errorbar(x, [r["ms_per_sample_mean"] for r in rows],
                        yerr=[r["ms_per_sample_std"] for r in rows],
                        fmt="o-", label=label, color=color, capsize=3)
    clean(axes[0], "Passos", "Aderência à classe (%)")
    clean(axes[1], "Passos", "Tempo por amostra (ms)")
    for axis in axes:
        axis.set_xticks([1, 2, 4, 8])
        axis.legend(frameon=False, fontsize=10)
    axes[0].set_ylim(55, 101)
    axes[0].set_title("Demonstração MNIST")
    axes[1].set_title("CPU · batch 300 · só amostragem latente")
    save(figure, "local_results")
    samples = np.load(ROOT / "experiments/results/samples.npz")
    data = np.load(ROOT / "data/mnist_subset.npz")
    selected = np.arange(10) * 30  # First sample of each class, no manual selection.
    rows = [("Real (não pareado)", data["test_images"][np.arange(10) * 100]),
            ("DDIM · 1 passo", samples["ddim_1"][selected]),
            ("DDIM · 4 passos", samples["ddim_4"][selected]),
            ("DDIM · 50 passos", samples["ddim_50"][selected]),
            ("LCM k20 · 1 passo", samples["lcm_k20_1"][selected]),
            ("LCM k20 · 2 passos", samples["lcm_k20_2"][selected]),
            ("LCM k20 · 4 passos", samples["lcm_k20_4"][selected])]
    figure, axes = plt.subplots(len(rows), 10, figsize=(11.5, 7), layout="constrained")
    for row, (label, values) in enumerate(rows):
        for col in range(10):
            axes[row, col].imshow(values[col].reshape(14, 14), cmap="gray", vmin=0, vmax=1,
                                  interpolation="nearest")
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            for spine in axes[row, col].spines.values():
                spine.set_visible(False)
            if row == 0:
                axes[row, col].set_title(str(col), fontsize=12)
        axes[row, 0].set_ylabel(label, rotation=0, ha="right", va="center", fontsize=10)
    save(figure, "generation_grid")
    mean, basis, scales = data["mean"], data["basis"], data["scales"]
    real = data["test_images"][np.arange(10) * 100]
    latent = (real - mean) @ basis.T / scales
    rebuilt = np.clip((latent * scales) @ basis + mean, 0, 1)
    figure, axes = plt.subplots(2, 10, figsize=(11, 2.8), layout="constrained")
    for row, values in enumerate([real, rebuilt]):
        for col in range(10):
            axes[row, col].imshow(values[col].reshape(14, 14), cmap="gray", vmin=0, vmax=1)
            axes[row, col].axis("off")
    axes[0, 0].set_title("Real", loc="left")
    axes[1, 0].set_title("PCA", loc="left")
    save(figure, "pca_reconstruction")
    # Clean mathematical formula asset, with no unsupported trajectory illustration.
    figure = plt.figure(figsize=(11.5, 2.9))
    figure.text(.05, .68, r"$f_\theta(z_t,c,\omega,t)=c_{\rm skip}(t)z_t+"
                r"c_{\rm out}(t)\frac{z_t-\sigma_t\hat{\epsilon}_\theta}{\alpha_t}$",
                fontsize=27)
    figure.text(.05, .25, r"$\mathcal{L}=\|f_\theta(z_t,c,\omega,t)-"
                r"\mathrm{stopgrad}\,f_{\theta^-}(\hat z_s,c,\omega,s)\|^2$",
                fontsize=25)
    save(figure, "equations")
    protocol = json.loads((ROOT / "experiments/results/protocol.json").read_text())
    lines = [
        "# Relatório experimental executado", "",
        "**Classificação: demonstração conceitual de LCD, não reprodução quantitativa LAION.**",
        "", "Dados: 6000 treino, 1000 validação e 1000 teste. PCA e classificador ajustados",
        "somente no treino. Professor com 5000 updates e dois alunos com 5000 cada.",
        "Seed de treino 42. Seeds de geração 101, 202 e 303, 300 amostras por seed.",
        "", "| Método | Passos | Aderência à classe (%) ± DP | Amostragem (ms/imagem) ± DP |",
        "|---|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(f"| {row['method']} | {row['steps']} | "
                     f"{100 * row['class_agreement_mean']:.2f} ± "
                     f"{100 * row['class_agreement_std']:.2f} | "
                     f"{row['ms_per_sample_mean']:.5f} ± {row['ms_per_sample_std']:.5f} |")
    lines += [
        "", "DP é desvio padrão amostral entre três sementes de geração, não intervalo",
        "de confiança nem variabilidade de três treinamentos independentes.", "",
        f"Classificador local em reais: validação {protocol['classifier_accuracy_real']['val']:.3%}, "
        f"teste {protocol['classifier_accuracy_real']['test']:.3%}.",
        "Aderência de sintéticos pode ser maior por serem mais fáceis ou menos diversos.",
        "Não é acurácia do gerador em imagens reais, FID, CLIP ou medida de utilidade SAR.",
        "", "A latência exclui PCA decode, classificador e I/O. É uma medida curta em CPU,",
        "após warmup. Não representa latência ponta a ponta ou desempenho de Stable Diffusion.",
        "", "## Artigo versus demonstração", "",
        "| Experimento | Artigo | Local | Comparação direta |",
        "|---|---|---|---|",
        "| FID LCM 512, 4 passos | 11,10 | Não calculado | N/A — configuração experimental diferente |",
        "| CLIP LCM 512, 4 passos | 28,69 | Não calculado | N/A — configuração experimental diferente |",
        "| Aderência de classe MNIST | Não reportada | Tabela acima | N/A — configuração experimental diferente |",
        "| IoU/Dice SAR após aumento | Não reportado | Não executado | Hipótese para pesquisa futura |",
        "", "## Interpretação", "",
        "O experimento demonstra treino DDPM, orientação CFG, DDIM, destilação com EMA",
        "e inferência LCM. A ablação k=1/20 tem inicialização e orçamento iguais.",
        "Perda menor em k=1 não assegura melhor geração. Não há inferência de",
        "significância estatística nem seleção de hiperparâmetros pelo teste.",
        "", "MSE latente contra DDIM-50 está no CSV, com limite de interpretação na auditoria.",
        "Os exemplos visuais usam a primeira amostra de cada classe da seed 101, sem",
        "seleção manual. As imagens reais são referências não pareadas, não ground truth",
        "da geração. O baseline local usa DDIM, apesar de o professor ter perda DDPM.",
        "", "## Tempo de treinamento observado", "",
    ]
    lines.extend(f"- {key}: {value:.3f} s." for key, value in protocol["durations"].items())
    lines += ["", "Configuração integral, versões e hashes: `experiments/results/protocol.json`.",
              "Perdas: `training.csv`. Métricas por semente: `metrics.csv`.",
              "Checkpoints: `experiments/checkpoints/`. Nenhum checkpoint SAR foi criado.", ""]
    (ROOT / "notes/experimental_report.md").write_text("\n".join(lines))
    print(f"Figuras e relatório gerados em {OUT}")


if __name__ == "__main__":
    main()
