"""CPU demonstration of latent consistency distillation, not Stable Diffusion."""

import copy
import csv
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
CONFIG = json.loads((ROOT / "experiments/configs/demo.json").read_text())
RESULTS = ROOT / "experiments/results"
CHECKPOINTS = ROOT / "experiments/checkpoints"


def digest(path):
    """Compute a file digest.

    Args:
        path: File to hash.

    Returns:
        SHA-256 hexadecimal string.
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def schedule(config=CONFIG):
    """Construct the explicitly local scaled-linear VP noise schedule.

    Args:
        config: Experiment configuration.

    Returns:
        Alpha and sigma tensors including an exact clean endpoint at index zero.
    """
    betas = torch.linspace(
        config["beta_start"] ** 0.5, config["beta_end"] ** 0.5, config["timesteps"]
    ).square()
    cumulative = torch.cat([torch.ones(1), torch.cumprod(1 - betas, dim=0)])
    return cumulative.sqrt(), (1 - cumulative).sqrt()


def ddim_step(z, epsilon, t, s, alpha, sigma):
    """Take a deterministic DDIM step without clipping.

    Args:
        z: Noisy latent batch.
        epsilon: Predicted noise batch.
        t: Source time indices, one per example.
        s: Target time indices.
        alpha: Signal schedule.
        sigma: Noise schedule.

    Returns:
        Estimated latent at the target times.
    """
    x0 = (z - sigma[t, None] * epsilon) / alpha[t, None]
    return alpha[s, None] * x0 + sigma[s, None] * epsilon


class NoiseMLP(nn.Module):
    """Small class-conditioned noise predictor with an added guidance projection."""

    def __init__(self, config=CONFIG):
        """Initialize the local MLP.

        Args:
            config: Dimensions and time schedule for the experiment.
        """
        super().__init__()
        self.config = config
        self.classes = nn.Embedding(11, 16)
        self.guidance = nn.Linear(16, 16, bias=False)
        nn.init.zeros_(self.guidance.weight)
        self.net = nn.Sequential(
            nn.Linear(config["latent_dim"] + 32, config["hidden_dim"]),
            nn.SiLU(),
            nn.Linear(config["hidden_dim"], config["hidden_dim"]),
            nn.SiLU(),
            nn.Linear(config["hidden_dim"], config["latent_dim"]),
        )

    def forward(self, z, t, labels, omega=None):
        """Predict noise for a latent batch.

        Args:
            z: Latents of shape batch by latent dimension.
            t: Integer time indices.
            labels: Class indices, with 10 representing the null condition.
            omega: Optional distilled guidance values.

        Returns:
            Predicted noise with the same shape as z.
        """
        frequencies = 2 ** torch.arange(8, dtype=z.dtype, device=z.device)
        phase = t[:, None] / self.config["timesteps"] * frequencies[None] * torch.pi
        time_features = torch.cat([phase.sin(), phase.cos()], dim=1)
        if omega is not None:
            w_phase = omega[:, None] * frequencies[None] * 0.25
            time_features = time_features + self.guidance(
                torch.cat([w_phase.sin(), w_phase.cos()], dim=1)
            )
        return self.net(torch.cat([z, self.classes(labels), time_features], dim=1))


def consistency(model, z, t, labels, omega, alpha, sigma):
    """Evaluate the boundary-conditioned epsilon parameterization in paper Eq. 9.

    Args:
        model: Student noise predictor.
        z: Noisy latent batch.
        t: Time indices.
        labels: Class condition.
        omega: Guidance condition.
        alpha: Signal schedule.
        sigma: Noise schedule.

    Returns:
        Predicted clean latent, exactly z for t equal to zero.
    """
    noise = model(z, t, labels, omega)
    clean = (z - sigma[t, None] * noise) / alpha[t, None]
    scaled = t[:, None] / model.config["timesteps"] * model.config["boundary_time_scale"]
    variance = model.config["sigma_data"] ** 2
    skip = variance / (scaled.square() + variance)
    out = scaled / (scaled.square() + variance).sqrt()
    return skip * z + out * clean


@torch.no_grad()
def ema_update(target, online, rate):
    """Update an EMA target without building an autograd graph.

    Args:
        target: Target module.
        online: Online module.
        rate: Weight assigned to previous target parameters.
    """
    for target_parameter, online_parameter in zip(target.parameters(), online.parameters()):
        target_parameter.mul_(rate).add_(online_parameter, alpha=1 - rate)


def numerical_checks():
    """Check independent analytical identities before any training.

    Returns:
        Errors and boolean invariants for DDIM, guidance, boundary and EMA.
    """
    torch.manual_seed(19)
    a, s = schedule()
    x = torch.randn(5, CONFIG["latent_dim"])
    eps = torch.randn_like(x)
    t = torch.tensor([100, 300, 500, 700, 1000])
    lower = torch.tensor([0, 100, 300, 500, 700])
    z = a[t, None] * x + s[t, None] * eps
    actual = ddim_step(z, eps, t, lower, a, s)
    expected = a[lower, None] * x + s[lower, None] * eps
    model = NoiseMLP()
    boundary = consistency(model, x, torch.zeros(5, dtype=torch.long),
                           torch.arange(5), torch.ones(5), a, s)
    other = torch.randn_like(eps)
    omega = 1.3
    guided = ddim_step(z, (1 + omega) * eps - omega * other, t, lower, a, s)
    cond = ddim_step(z, eps, t, lower, a, s)
    uncond = ddim_step(z, other, t, lower, a, s)
    target = copy.deepcopy(model).requires_grad_(False)
    before = [p.detach().clone() for p in target.parameters()]
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.2)
    ema_update(target, model, 0.9)
    error = max(float((p - (0.9 * b + 0.1 * q)).abs().max().detach())
                for p, b, q in zip(target.parameters(), before, model.parameters()))
    return {
        "ddim_max_error": float((actual - expected).abs().max()),
        "boundary_max_error": float((boundary - x).abs().max().detach()),
        "guidance_max_error": float((guided - ((1 + omega) * cond - omega * uncond)).abs().max()),
        "ema_max_error": error,
        "target_detached": all(not p.requires_grad and p.grad is None for p in target.parameters()),
    }


def prepare_data(config=CONFIG):
    """Load the portable subset or construct it from the existing MNIST archive.

    Args:
        config: Counts, dimensions and split seed.

    Returns:
        A dictionary of arrays with images, labels, source indices and PCA factors.

    Raises:
        FileNotFoundError: If neither the bundled subset nor source archive exists.
        ValueError: If a saved subset was prepared with a different configuration.
    """
    path = ROOT / "data/mnist_subset.npz"
    selection = {k: config[k] for k in
                 ["seed", "train_per_class", "val_per_class", "test_per_class",
                  "image_size", "latent_dim"]}
    signature = json.dumps(selection, sort_keys=True)
    if path.exists():
        data = dict(np.load(path, allow_pickle=False))
        if str(data["selection_config"]) != signature:
            raise ValueError("Subset/config incompatíveis. Preserve o subset e prepare outro explicitamente.")
        return data
    raw = ROOT.parent / "seminar_1/data/mnist/source/MNIST/raw"
    names = ["train-images-idx3-ubyte", "train-labels-idx1-ubyte",
             "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"]
    if any(not (raw / name).exists() for name in names):
        raise FileNotFoundError("Forneça data/mnist_subset.npz ou o MNIST original em " + str(raw))
    arrays = [np.frombuffer((raw / name).read_bytes(), dtype=np.uint8,
                           offset=16 if "images" in name else 8).copy() for name in names]
    images, labels, test_images, test_labels = arrays
    images = images.reshape(-1, 28, 28)
    test_images = test_images.reshape(-1, 28, 28)
    generator = np.random.default_rng(config["seed"])
    train_indices, val_indices, test_indices = [], [], []
    for label in range(10):
        indices = generator.permutation(np.where(labels == label)[0])
        cut = config["train_per_class"]
        train_indices.extend(indices[:cut])
        val_indices.extend(indices[cut:cut + config["val_per_class"]])
        test_indices.extend(generator.permutation(np.where(test_labels == label)[0])[
            :config["test_per_class"]])
    data = {"selection_config": np.array(signature)}
    for split, ids, source_x, source_y in [
        ("train", train_indices, images, labels),
        ("val", val_indices, images, labels),
        ("test", test_indices, test_images, test_labels),
    ]:
        ids = np.array(ids)
        resized = F.interpolate(torch.tensor(source_x[ids], dtype=torch.float32)[:, None] / 255,
                                size=(config["image_size"], config["image_size"]),
                                mode="area").flatten(1).numpy()
        data[f"{split}_images"] = resized
        data[f"{split}_labels"] = source_y[ids].astype(np.int64)
        data[f"{split}_source_indices"] = ids
    mean = data["train_images"].mean(0)
    _, _, vectors = np.linalg.svd(data["train_images"] - mean, full_matrices=False)
    basis = vectors[:config["latent_dim"]]
    scales = ((data["train_images"] - mean) @ basis.T).std(0).clip(1e-4)
    data.update(mean=mean, basis=basis, scales=scales)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    provenance = {
        "source": "MNIST, arquivos IDX reutilizados somente para leitura do Seminário 1",
        "url": "https://storage.googleapis.com/cvdf-datasets/mnist/",
        "sha256_original": {name: digest(raw / name) for name in names},
        "selection": selection,
        "split": "treino e validação disjuntos no train oficial, teste no t10k oficial",
        "preprocessing": "uint8/255, area 28x28→14x14, PCA/whitening ajustados só no treino",
    }
    (path.parent / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n"
    )
    return data


def decode(z, data):
    """Decode PCA latents to bounded grayscale pixels.

    Args:
        z: Latent tensor.
        data: Training-only PCA mean, basis and scale arrays.

    Returns:
        Pixel tensor in [0, 1].
    """
    return ((z * torch.tensor(data["scales"])) @ torch.tensor(data["basis"])
            + torch.tensor(data["mean"])).clamp(0, 1)


@torch.no_grad()
def sample(model, labels, steps, seed, mode="lcm", omega=1.0, return_history=False):
    """Generate latent samples using LCM Algorithm 3 or guided deterministic DDIM.

    Args:
        model: Teacher for DDIM or EMA student for LCM.
        labels: Requested class indices.
        steps: Number of model sampling steps.
        seed: Generator seed, shared across methods for the initial noise.
        mode: Either lcm or ddim.
        omega: Paper convention, (1 + omega) conditional minus omega unconditional.
        return_history: Whether to return each clean prediction.

    Returns:
        Final latents, and optionally a stack of clean predictions.

    Raises:
        ValueError: If the mode or number of steps is invalid.
    """
    if mode not in {"lcm", "ddim"} or not 1 <= steps <= model.config["timesteps"]:
        raise ValueError("Modo ou número de passos inválido")
    model.eval()
    alpha, sigma = schedule(model.config)
    rng = torch.Generator().manual_seed(seed)
    z = torch.randn((len(labels), model.config["latent_dim"]), generator=rng)
    times = torch.linspace(model.config["timesteps"], 0, steps + 1).round().long()
    history = []
    w = torch.full((len(labels),), float(omega))
    for index, value in enumerate(times[:-1]):
        t = value.repeat(len(labels))
        lower = times[index + 1].repeat(len(labels))
        if mode == "ddim":
            noise = (1 + w[:, None]) * model(z, t, labels) - w[:, None] * model(
                z, t, torch.full_like(labels, 10)
            )
            clean = (z - sigma[t, None] * noise) / alpha[t, None]
            z = ddim_step(z, noise, t, lower, alpha, sigma)
        else:
            clean = consistency(model, z, t, labels, w, alpha, sigma)
            z = clean if index == steps - 1 else (
                alpha[lower, None] * clean + sigma[lower, None] *
                torch.randn(z.shape, generator=rng)
            )
        history.append(clean)
    return (z, torch.stack(history)) if return_history else z


def train(config=CONFIG):
    """Execute the complete demonstration and save measured evidence.

    Args:
        config: Fixed local configuration, distinct from the paper's settings.
    """
    RESULTS.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    checks = numerical_checks()
    assert checks["ddim_max_error"] < 1e-5 and checks["boundary_max_error"] == 0
    assert checks["guidance_max_error"] < 1e-5 and checks["target_detached"]
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    data = prepare_data(config)
    assert not np.intersect1d(data["train_source_indices"], data["val_source_indices"]).size
    images = torch.tensor(data["train_images"])
    labels = torch.tensor(data["train_labels"])
    latents = torch.tensor(
        (data["train_images"] - data["mean"]) @ data["basis"].T / data["scales"]
    )
    alpha, sigma = schedule(config)
    curves, durations = [], {}
    start = time.perf_counter()
    classifier = nn.Sequential(nn.Linear(config["image_size"] ** 2, 128), nn.ReLU(),
                               nn.Linear(128, 10))
    optimizer = torch.optim.Adam(classifier.parameters(), lr=config["classifier_lr"])
    for step in range(config["classifier_steps"]):
        ids = torch.randint(len(images), (config["batch_size"],))
        loss = F.cross_entropy(classifier(images[ids]), labels[ids])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    classifier.eval()
    classifier_scores = {}
    with torch.no_grad():
        for split in ["val", "test"]:
            prediction = classifier(torch.tensor(data[f"{split}_images"])).argmax(1)
            classifier_scores[split] = float(
                (prediction == torch.tensor(data[f"{split}_labels"])).float().mean()
            )
    durations["classifier_seconds"] = time.perf_counter() - start
    torch.manual_seed(config["seed"])
    teacher = NoiseMLP(config)
    optimizer = torch.optim.Adam(teacher.parameters(), lr=config["teacher_lr"])
    start = time.perf_counter()
    for step in range(config["teacher_steps"]):
        ids = torch.randint(len(latents), (config["batch_size"],))
        t = torch.randint(1, config["timesteps"] + 1, (len(ids),))
        noise = torch.randn_like(latents[ids])
        z = alpha[t, None] * latents[ids] + sigma[t, None] * noise
        condition = labels[ids].clone()
        condition[torch.rand(len(ids)) < 0.1] = 10
        loss = F.mse_loss(teacher(z, t, condition), noise)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1.0)
        optimizer.step()
        if step % 100 == 0 or step == config["teacher_steps"] - 1:
            curves.append({"phase": "teacher", "step": step + 1, "loss": float(loss.detach())})
        if (step + 1) % 1000 == 0:
            print(f"Professor {step + 1}/{config['teacher_steps']} loss={loss.item():.4f}", flush=True)
    durations["teacher_seconds"] = time.perf_counter() - start
    teacher.eval().requires_grad_(False)
    torch.save({"state_dict": teacher.state_dict(), "config": config}, CHECKPOINTS / "teacher.pt")
    torch.save(classifier.state_dict(), CHECKPOINTS / "classifier.pt")
    students = {}
    for skip in config["skip_intervals"]:
        torch.manual_seed(config["seed"] + 1)
        online = copy.deepcopy(teacher).train().requires_grad_(True)
        target = copy.deepcopy(online).eval().requires_grad_(False)
        optimizer = torch.optim.Adam(online.parameters(), lr=config["student_lr"])
        start = time.perf_counter()
        for step in range(config["student_steps"]):
            ids = torch.randint(len(latents), (config["batch_size"],))
            t = torch.randint(skip, config["timesteps"] + 1, (len(ids),))
            lower = t - skip
            omega = torch.rand(len(ids)) * (config["guidance_max"] - config["guidance_min"])
            omega += config["guidance_min"]
            noisy = alpha[t, None] * latents[ids] + sigma[t, None] * torch.randn_like(latents[ids])
            condition = labels[ids]
            with torch.no_grad():
                epsilon = ((1 + omega[:, None]) * teacher(noisy, t, condition)
                           - omega[:, None] * teacher(noisy, t, torch.full_like(condition, 10)))
                lower_z = ddim_step(noisy, epsilon, t, lower, alpha, sigma)
                goal = consistency(target, lower_z, lower, condition, omega, alpha, sigma)
            prediction = consistency(online, noisy, t, condition, omega, alpha, sigma)
            loss = F.mse_loss(prediction, goal)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(online.parameters(), 1.0)
            optimizer.step()
            ema_update(target, online, config["ema"])
            if step % 100 == 0 or step == config["student_steps"] - 1:
                curves.append({"phase": f"lcm_k{skip}", "step": step + 1,
                               "loss": float(loss.detach())})
            if (step + 1) % 1000 == 0:
                print(f"LCM k={skip} {step + 1}/{config['student_steps']} loss={loss.item():.4f}",
                      flush=True)
        durations[f"lcm_k{skip}_seconds"] = time.perf_counter() - start
        students[skip] = target
        torch.save({"state_dict": target.state_dict(), "config": config},
                   CHECKPOINTS / f"lcm_k{skip}.pt")
    evaluation_labels = torch.arange(10).repeat_interleave(config["samples_per_class"])
    records, samples = [], {"labels": evaluation_labels.numpy()}
    with torch.no_grad():
        sample(teacher, evaluation_labels, 4, 0, "ddim")  # Warmup, excluded from timings.
        for seed in config["sample_seeds"]:
            reference = sample(teacher, evaluation_labels, 50, seed, "ddim",
                               config["eval_guidance"])
            for name, model, mode, counts in [
                ("ddim", teacher, "ddim", [1, 2, 4, 8, 50]),
                *[(f"lcm_k{k}", student, "lcm", config["inference_steps"])
                  for k, student in students.items()],
            ]:
                for count in counts:
                    begin = time.perf_counter()
                    z, history = sample(model, evaluation_labels, count, seed, mode,
                                        config["eval_guidance"], True)
                    elapsed = time.perf_counter() - begin
                    pixels = decode(z, data)
                    agreement = float((classifier(pixels).argmax(1) == evaluation_labels).float().mean())
                    record = {
                        "method": name, "steps": count, "seed": seed,
                        "samples": len(evaluation_labels),
                        "class_agreement": agreement,
                        "latent_mse_vs_ddim50": float(F.mse_loss(z, reference)),
                        "latent_variance": float(z.var(0).mean()),
                        "batch_seconds": elapsed,
                        "ms_per_sample": elapsed * 1000 / len(evaluation_labels),
                        "network_evaluations": count * (2 if mode == "ddim" else 1),
                    }
                    assert all(np.isfinite(value) for value in record.values()
                               if isinstance(value, (int, float)))
                    records.append(record)
                    if seed == config["sample_seeds"][0]:
                        samples[f"{name}_{count}"] = pixels.numpy()
                        if count == 4:
                            samples[f"{name}_history"] = torch.stack(
                                [decode(frame, data) for frame in history]).numpy()
    for filename, rows in [("metrics.csv", records), ("training.csv", curves)]:
        with (RESULTS / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    np.savez_compressed(RESULTS / "samples.npz", **samples)
    versions = {package: importlib.metadata.version(package) for package in
                ["torch", "numpy", "matplotlib", "manim", "nbformat", "nbclient", "pillow"]}
    protocol = {
        "classification": config["classification"],
        "config": config, "python": platform.python_version(), "platform": platform.platform(),
        "processor": platform.processor(),
        "hardware": "CPU, CUDA disponível: " + str(torch.cuda.is_available()),
        "threads": torch.get_num_threads(), "versions": versions, "durations": durations,
        "classifier_accuracy_real": classifier_scores,
        "checks": checks, "data_sha256": digest(ROOT / "data/mnist_subset.npz"),
        "paper_sha256": digest(ROOT / "articles/2310.04378v1.pdf"),
        "script_sha256": digest(Path(__file__)),
        "config_sha256": digest(ROOT / "experiments/configs/demo.json"),
        "checkpoint_sha256": {p.name: digest(p) for p in CHECKPOINTS.glob("*.pt")},
        "timing_scope": "batch de 300, amostragem latente, exclui PCA decode/classificador/I/O",
        "statistical_scope": "1 treino por modelo, 3 sementes de geração, sem inferência de significância",
        "selection": "config fixada antes do teste, sem seleção por métricas de teste",
    }
    (RESULTS / "protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"completed": True, "durations": durations,
                      "classifier_accuracy_real": classifier_scores}, indent=2), flush=True)


if __name__ == "__main__":
    train()
