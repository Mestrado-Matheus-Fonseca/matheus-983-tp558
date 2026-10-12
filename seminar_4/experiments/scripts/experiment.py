"""Reprodução parcial dos Algoritmos 1/2: grafo RGB, sem Stable Diffusion."""

import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.ndimage import label
from scipy.spatial.distance import squareform

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/outputs"
RESULTS = ROOT / "experiments/results"
CONFIG = {"grid": 24, "prompt_spacing": 6, "p_values": [2.0, 1.6, 1.2],
          "lambda": 0.03, "gradient_epsilon": 1e-8, "stop_squared_update": 1e-8,
          "max_iterations": 400, "thresholds": [0.186, 0.323, 0.561, 0.974, 1.691, 2.99],
          "min_area": 3, "nms_iou": 0.9, "color_sigma": 0.18,
          "spatial_sigma": 3.0, "seeds": [7, 23, 42], "proposal_cap": 1000}


def affinity(image):
    """Grafo simétrico de similaridade de RGB e posição; nunca usa GT."""
    n = image.shape[0]
    rgb = image.reshape(-1, 3)
    coords = np.stack(np.indices((n, n)), axis=-1).reshape(-1, 2)
    color_d2 = ((rgb[:, None] - rgb[None, :]) ** 2).sum(-1)
    space_d2 = ((coords[:, None] - coords[None, :]) ** 2).sum(-1)
    a = np.exp(-color_d2 / (2 * CONFIG["color_sigma"] ** 2)
               -space_d2 / (2 * CONFIG["spatial_sigma"] ** 2))
    np.fill_diagonal(a, 0)
    return a / a.sum(1).max()  # Escala global preserva simetria.


def prompts(n):
    positions = [(y, x) for y in range(3, n, CONFIG["prompt_spacing"])
                 for x in range(3, n, CONFIG["prompt_spacing"])]
    return np.eye(n * n)[[y * n + x for y, x in positions]], positions


def propagate(a, f0, p, anchoring=None, tolerance=None):
    """Gauss-Jacobi vetorizado sobre os prompts, Eq. 2/Alg. 1 com epsilon."""
    anchoring = CONFIG["lambda"] if anchoring is None else anchoring
    tolerance = CONFIG["stop_squared_update"] if tolerance is None else tolerance
    f = f0.copy()
    degree = a.sum(1)
    history, snapshots = [], {0: f.copy()}
    converged = False
    for iteration in range(1, CONFIG["max_iterations"] + 1):
        # g_i² = sum_j A_ij (f_j-f_i)²; evita tensor K x N x N.
        variance = (f * f) @ a.T - 2 * f * (f @ a.T) + f * f * degree
        g_power = np.maximum(variance, CONFIG["gradient_epsilon"] ** 2) ** ((p - 2) / 2)
        denominator = anchoring + g_power * degree + g_power @ a.T
        numerator = anchoring * f0 + g_power * (f @ a.T) + (g_power * f) @ a.T
        updated = numerator / denominator
        residual = ((updated - f) ** 2).sum(1)
        history.append(float(residual.max()))
        f = updated
        if iteration in [1, 5, 20, 80, 200]:
            snapshots[iteration] = f.copy()
        if np.all(residual <= tolerance):
            converged = True
            break
    snapshots[iteration] = f.copy()
    return f, {"iterations": iteration, "converged": converged,
               "max_squared_update": history[-1], "history": history}, snapshots


def symmetric_kl(maps):
    probs = np.maximum(maps, 1e-12)
    probs /= probs.sum(1, keepdims=True)
    log_probs = np.log(probs)
    cross = probs @ log_probs.T
    kl = np.diag(cross)[:, None] - cross
    distance = np.maximum((kl + kl.T) / 2, 0)
    np.fill_diagonal(distance, 0)
    return probs, distance


def iou_matrix(predictions, targets):
    pred = np.asarray(predictions, dtype=float).reshape(len(predictions), -1)
    gt = np.asarray(targets, dtype=float).reshape(len(targets), -1)
    intersection = pred @ gt.T
    return intersection / np.maximum(pred.sum(1)[:, None] + gt.sum(1)[None, :] - intersection, 1)


def merge(maps, n, thresholds=None, nms=None):
    probs, distance = symmetric_kl(maps)
    hierarchy = linkage(squareform(distance, checks=True), method="average")
    candidates, levels = [], []
    for h in CONFIG["thresholds"] if thresholds is None else thresholds:
        clusters = fcluster(hierarchy, h, criterion="distance")
        averages = np.stack([probs[clusters == c].mean(0) for c in np.unique(clusters)])
        segmentation = averages.argmax(0).reshape(n, n)
        levels.append(segmentation)
        for c in np.unique(segmentation):
            components, count = label(segmentation == c)  # conectividade 4, explicitada.
            for component in range(1, count + 1):
                mask = components == component
                if mask.sum() >= CONFIG["min_area"]:
                    candidates.append(mask)
    accepted = []
    for mask in sorted(candidates, key=lambda m: int(m.sum()), reverse=True):
        if len(accepted) >= CONFIG["proposal_cap"]:
            break
        if not accepted or iou_matrix([mask], accepted).max() <= (CONFIG["nms_iou"] if nms is None else nms):
            accepted.append(mask)
    return accepted, levels, distance, len(candidates)


def evaluate(predictions, targets):
    """Recall toy com matching 1:1 por IoU; NÃO é COCO AR1000 oficial."""
    overlaps = iou_matrix(predictions, targets) if len(predictions) else np.zeros((0, len(targets)))
    recalls = []
    for threshold in np.linspace(0.5, 0.95, 10):
        used_pred, used_gt = set(), set()
        pairs = [(overlaps[i, j], i, j) for i, j in zip(*np.where(overlaps >= threshold), strict=True)]
        for _, i, j in sorted(pairs, reverse=True):
            if i not in used_pred and j not in used_gt:
                used_pred.add(i)
                used_gt.add(j)
        recalls.append(len(used_gt) / len(targets))
    best = overlaps.max(0) if len(predictions) else np.zeros(len(targets))
    return {"toy_recall_50_95": float(np.mean(recalls)), "mean_best_iou": float(best.mean()),
            "proposals": len(predictions), "recall_50": recalls[0]}


def synthetic(seed):
    n = CONFIG["grid"]
    y, x = np.indices((n, n))
    regions = np.zeros((n, n), dtype=int)
    regions[y < 8] = 1
    regions[(x - 7) ** 2 + (y - 15) ** 2 <= 16] = 2
    regions[(x >= 15) & (x <= 20) & (y >= 11) & (y <= 19)] = 3
    regions[(x >= 17) & (x <= 18) & (y >= 12) & (y <= 14)] = 4
    palette = np.array([[.30, .48, .24], [.24, .55, .86], [.90, .38, .22],
                        [.68, .60, .20], [.93, .81, .27]])
    image = np.clip(palette[regions] + np.random.default_rng(seed).normal(0, .035, (n, n, 3)), 0, 1)
    targets = [regions == c for c in range(5)]
    targets.append((regions == 3) | (regions == 4))  # GT multi-granular: parte e objeto.
    return image, regions, targets


def save_fig(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def checks():
    a = np.array([[0, .8, .1], [.8, 0, .2], [.1, .2, 0]])
    f0 = np.eye(3)[[0]]
    computed, status, _ = propagate(a, f0, 2, anchoring=.5, tolerance=1e-18)
    expected = np.linalg.solve(2 * (np.diag(a.sum(1)) - a) + .5 * np.eye(3), .5 * f0[0])
    error = float(np.max(np.abs(computed[0] - expected)))
    assert error < 1e-7 and status["converged"], (computed, expected)
    _, distance = symmetric_kl(np.array([[.2, .8], [.2, .8], [.8, .2]]))
    assert np.allclose(distance, distance.T) and np.isclose(distance[0, 1], 0)
    gt = [np.eye(3, dtype=bool)]
    assert evaluate(gt, gt)["toy_recall_50_95"] == 1
    assert evaluate([], gt)["toy_recall_50_95"] == 0
    assert evaluate(gt, gt * 2)["toy_recall_50_95"] == .5  # uma máscara não casa duas vezes.
    return {"linear_closed_form_max_error": error, "checks": "passed"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    check_result = checks()
    rows, example = [], None
    for seed in CONFIG["seeds"]:
        image, regions, targets = synthetic(seed)
        a = affinity(image)
        f0, positions = prompts(CONFIG["grid"])
        for p in CONFIG["p_values"]:
            maps, status, snapshots = propagate(a, f0, p)
            proposals, levels, distance, count = merge(maps, CONFIG["grid"])
            row = {"seed": seed, "p": p, **evaluate(proposals, targets),
                   **{k: status[k] for k in ["iterations", "converged", "max_squared_update"]},
                   "candidates_before_nms": count}
            rows.append(row)
            if seed == 7 and p == 1.6:
                example = (image, regions, maps, proposals, levels, distance, snapshots, positions, a)
            np.savez_compressed(RESULTS / f"synthetic_seed{seed}_p{p}.npz",
                                image=image, regions=regions, gt=np.stack(targets), maps=maps,
                                proposals=np.stack(proposals), levels=np.stack(levels),
                                squared_update=np.array(status["history"]))
    with (RESULTS / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    image, regions, maps, proposals, levels, distance, snapshots, positions, a = example
    np.savez_compressed(RESULTS / "propagation_snapshots.npz", steps=np.array(list(snapshots)),
                        maps=np.stack(list(snapshots.values())), positions=np.array(positions))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image)
    axes[0].set_title("Entrada • semente 7")
    for ax, p in zip(axes[1:], [2.0, 1.6], strict=True):
        result = np.load(RESULTS / f"synthetic_seed7_p{p}.npz")
        ax.imshow(result["levels"][0], cmap="tab20")
        ax.set_title(f"p = {p} • mesmo grafo/λ/cortes KL")
    for ax in axes:
        ax.axis("off")
    fig.suptitle("Partição fina observada • p=1,6 perde entidades neste proxy")
    fig.tight_layout()
    save_fig(fig, "toy_comparison")
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, data, title in zip(axes.flat, [image, regions, maps[9].reshape(24, 24), levels[0], levels[3], levels[-1]],
                               ["Entrada sintética RGB", "Referência conhecida (GT)", "Mapa suave: prompt 9",
                                "Granularidade fina", "Granularidade intermediária", "Granularidade grossa"], strict=True):
        ax.imshow(data, cmap="tab20" if "Granularidade" in title or "GT" in title else "viridis")
        ax.set_title(title, fontsize=12)
        ax.axis("off")
    fig.suptitle("Reprodução parcial • afinidade RGB/posição • sem difusão ou treinamento", fontsize=15)
    fig.tight_layout()
    save_fig(fig, "toy_pipeline")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
    axes[0].imshow(image)
    axes[0].scatter([x for y, x in positions], [y for y, x in positions], c="white", s=20)
    axes[0].set_title("16 prompts automáticos")
    axes[1].imshow(a, cmap="magma")
    axes[1].set_title("Afinidade proxy: 576 × 576")
    axes[2].imshow(distance, cmap="magma")
    axes[2].set_title("KL simétrica: 16 × 16")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    save_fig(fig, "toy_representations")
    snapshot_keys = list(snapshots)
    fig, axes = plt.subplots(1, len(snapshot_keys), figsize=(14, 2.8))
    for ax, step in zip(axes, snapshot_keys, strict=True):
        ax.imshow(snapshots[step][9].reshape(24, 24), cmap="viridis")
        ax.set_title(f"Iteração {step}")
        ax.axis("off")
    fig.suptitle("Mapa do prompt 9 • cores normalizadas por painel • dados reais do solver")
    fig.tight_layout()
    save_fig(fig, "toy_propagation")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    summaries = []
    for p in CONFIG["p_values"]:
        group = [r for r in rows if r["p"] == p]
        values = np.array([r["toy_recall_50_95"] for r in group])
        summaries.append({"p": p, "recall_mean": float(values.mean()), "recall_std": float(values.std(ddof=1)),
                          "iou_mean": float(np.mean([r["mean_best_iou"] for r in group]))})
        run = np.load(RESULTS / f"synthetic_seed7_p{p}.npz")
        axes[1].semilogy(np.arange(1, len(run["squared_update"]) + 1), run["squared_update"], label=f"p={p}")
    axes[0].bar([f"p={s['p']}" for s in summaries], [100*s["recall_mean"] for s in summaries],
                yerr=[100*s["recall_std"] for s in summaries], color=["#64748b", "#2563eb", "#14b8a6"], capsize=6)
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("Recall toy médio (%) • matching 1:1")
    axes[0].set_title("3 sementes • dispersão, sem teste de significância")
    axes[1].axhline(CONFIG["stop_squared_update"], color="gray", linestyle="--")
    axes[1].set_xlabel("Iteração • semente 7")
    axes[1].set_ylabel("Maior atualização quadrática por prompt")
    axes[1].legend()
    fig.tight_layout()
    save_fig(fig, "toy_results")
    ablations = []
    for h in [3, 6]:
        for nms in [.5, .9]:
            proposed, _, _, count = merge(maps, 24, np.geomspace(.186, 2.99, h), nms)
            ablations.append({"levels": h, "nms_iou": nms, **evaluate(proposed, synthetic(7)[2]),
                              "candidates_before_nms": count})
    with (RESULTS / "ablations.json").open("w", encoding="utf-8") as stream:
        json.dump(ablations, stream, indent=2)
    # Fotografia existente: nenhuma anotação do seminário 3 é lida pelo algoritmo.
    coco = ROOT.parent / "seminar_3/data/raw/coco_demo/000000039769.jpg"
    real_status = {"executed": False}
    if coco.exists():
        original_photo = Image.open(coco).convert("RGB")
        real_image = np.asarray(original_photo.resize((24, 24), Image.Resampling.BILINEAR)) / 255
        real_maps, solver_status, _ = propagate(affinity(real_image), prompts(24)[0], 1.6)
        real_proposals, real_levels, _, _ = merge(real_maps, 24)
        np.savez_compressed(RESULTS / "coco_concept.npz", image=real_image,
                            maps=real_maps, levels=np.stack(real_levels), proposals=np.stack(real_proposals))
        linear_maps, linear_status, _ = propagate(affinity(real_image), prompts(24)[0], 2.0)
        linear_proposals, linear_levels, _, _ = merge(linear_maps, 24)
        np.savez_compressed(RESULTS / "coco_concept_linear.npz", image=real_image,
                            maps=linear_maps, levels=np.stack(linear_levels), proposals=np.stack(linear_proposals))
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, data, title in zip(axes, [np.asarray(original_photo), linear_levels[0], real_levels[0]],
                                   ["COCO 39769 • fotografia original", "Proxy p=2 • partição fina", "Proxy p=1,6 • partição fina"], strict=True):
            ax.imshow(data, cmap="tab20")
            ax.set_title(title)
            ax.axis("off")
        fig.suptitle("Demonstração conceitual • grafo RGB 24² • sem avaliação COCO")
        fig.tight_layout()
        save_fig(fig, "coco_concept")
        real_status = {"executed": True, "image": "COCO val2017 39769",
                       "source_sha256": hashlib.sha256(coco.read_bytes()).hexdigest(),
                       "proposals": len(real_proposals), "iterations": solver_status["iterations"],
                       "converged": solver_status["converged"], "metrics": "not_evaluated",
                       "linear_comparison": {"proposals": len(linear_proposals),
                                             "iterations": linear_status["iterations"],
                                             "converged": linear_status["converged"]}}
    record = {"type": "partial_algorithm_reproduction_with_proxy_affinity", "config": CONFIG,
              "summary": summaries, "checks": check_result, "coco_concept": real_status,
              "elapsed_seconds": time.perf_counter()-started, "python": platform.python_version(),
              "platform": platform.platform(), "dependencies": {x: importlib.metadata.version(x)
              for x in ["numpy", "scipy", "matplotlib", "Pillow", "manim"]},
              "paper_sha256": hashlib.sha256((ROOT / "articles/2609.06491v1.pdf").read_bytes()).hexdigest()}
    (RESULTS / "protocol.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summaries, "checks": check_result, "elapsed_seconds": record["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
