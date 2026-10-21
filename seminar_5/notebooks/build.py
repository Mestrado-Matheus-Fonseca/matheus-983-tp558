"""Create and execute the seven teaching notebooks with a temporary local kernel."""

import json
import os
from pathlib import Path
import sys

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
M = nbf.v4.new_markdown_cell
C = nbf.v4.new_code_cell
BOOT = """from pathlib import Path
import sys, json, csv
import numpy as np
import torch
from IPython.display import display, Markdown, Image, Video
ROOT = next(p for p in [Path.cwd(), Path.cwd().parent, Path.cwd() / 'seminar_5']
            if (p / 'experiments/configs/demo.json').exists())
sys.path.insert(0, str(ROOT / 'experiments/scripts'))
import experiment as exp
torch.set_num_threads(2)
config = json.loads((ROOT / 'experiments/configs/demo.json').read_text())
protocol = json.loads((ROOT / 'experiments/results/protocol.json').read_text())
print('Demonstração conceitual MNIST/PCA/MLP. Não é reprodução de FID/CLIP LAION.')
"""


def image_cell(path):
    """Create an embedding cell for a generated figure.

    Args:
        path: Image path relative to the seminar root.

    Returns:
        Notebook code cell.
    """
    return C(f"display(Image(filename=str(ROOT / '{path}')))")


def note_cell(name):
    """Create a source-backed explanatory Markdown cell.

    Args:
        name: Notes filename.

    Returns:
        Markdown cell with the actual note contents.
    """
    return M((ROOT / "notes" / name).read_text())


def main():
    """Write all notebooks and execute every code cell with nbclient.

    Raises:
        CellExecutionError: If any notebook code cell fails.
    """
    notebook_dir = ROOT / "notebooks"
    specs = [
        ("01_article_overview", [
            M("# 01 · Artigo, narrativa e limites\n\nLeia antes dos experimentos. "
              "A versão estudada é arXiv:2310.04378v1."),
            C(BOOT), note_cell("didactic_summary.md"), note_cell("technical_summary.md"),
            C("paper = json.loads((ROOT / 'data/paper_results.json').read_text())\n"
              "print(paper['source'])\n"
              "print('Diferença FID DDIM−LCM, 512/4 passos:', "
              "round(paper['512']['DDIM']['FID'][2] - paper['512']['LCM']['FID'][2], 2))"),
            image_cell("assets/figures/paper_fid.png"),
            M("Este gráfico é transcrição das Tabelas 1/2, não uma execução local em LAION."),
        ]),
        ("02_data_pipeline", [
            M("# 02 · Dados, splits e compressão latente"),
            C(BOOT),
            M("6000 imagens de treino, 1000 de validação e 1000 de teste. "
              "Treino/validação vêm do train oficial, teste do t10k. "
              "Imagens 28×28 são reduzidas por média de área a 14×14. "
              "PCA e whitening são ajustados somente no treino. A condição é a classe."),
            C("data = exp.prepare_data()\n"
              "for split in ['train', 'val', 'test']:\n"
              "    print(split, data[f'{split}_images'].shape, "
              "np.bincount(data[f'{split}_labels']).tolist())\n"
              "assert not np.intersect1d(data['train_source_indices'], "
              "data['val_source_indices']).size\n"
              "print(json.dumps(json.loads((ROOT / 'data/provenance.json').read_text()), "
              "indent=2, ensure_ascii=False))"),
            M("A compressão é um autoencoder linear. Não possui KL, posterior variacional "
              "nem decoder convolucional de Stable Diffusion. "
              "O rótulo não participa do ajuste PCA."),
            C("x = data['test_images']\n"
              "z = (x - data['mean']) @ data['basis'].T / data['scales']\n"
              "reconstructed = exp.decode(torch.tensor(z), data).numpy()\n"
              "print('Entrada / latente / reconstrução:', x.shape, z.shape, reconstructed.shape)\n"
              "print('MSE reconstrução no teste:', float(np.mean((x - reconstructed)**2)))"),
            image_cell("assets/figures/pca_reconstruction.png"),
            M("As reconstruções permitem avaliar o gargalo de compressão. "
              "Para SAR, a largura das pistas precisa de uma auditoria específica."),
        ]),
        ("03_method", [
            M("# 03 · LCD: professor, solver, aluno e EMA"),
            C(BOOT),
            M("A célula abaixo executa uma atualização real de LCD em uma **cópia em memória** "
              "do aluno salvo. Ela não substitui os checkpoints nem as métricas entregues. "
              "O alvo EMA é congelado, mas atualizado por média dos pesos."),
            image_cell("assets/images/PipelineLCM.png"),
            C("import copy\n"
              "data = exp.prepare_data()\n"
              "teacher = exp.NoiseMLP(config)\n"
              "teacher.load_state_dict(torch.load(ROOT / 'experiments/checkpoints/teacher.pt', "
              "weights_only=True)['state_dict'])\n"
              "teacher.eval().requires_grad_(False)\n"
              "online = copy.deepcopy(teacher).train().requires_grad_(True)\n"
              "target = copy.deepcopy(teacher).eval().requires_grad_(False)\n"
              "torch.manual_seed(11)\n"
              "ids = torch.randperm(len(data['train_images']))[:16]\n"
              "z0 = torch.tensor((data['train_images'] - data['mean']) @ data['basis'].T "
              "/ data['scales'])[ids]\n"
              "labels = torch.tensor(data['train_labels'])[ids]\n"
              "a, s = exp.schedule()\n"
              "t = torch.randint(20, 1001, (16,))\n"
              "low = t - 20\n"
              "w = torch.rand(16) * 2\n"
              "noisy = a[t, None] * z0 + s[t, None] * torch.randn_like(z0)\n"
              "with torch.no_grad():\n"
              "    epsilon = (1+w[:, None])*teacher(noisy,t,labels) - "
              "w[:, None]*teacher(noisy,t,torch.full_like(labels,10))\n"
              "    earlier = exp.ddim_step(noisy, epsilon, t, low, a, s)\n"
              "    goal = exp.consistency(target, earlier, low, labels, w, a, s)\n"
              "prediction = exp.consistency(online, noisy, t, labels, w, a, s)\n"
              "loss = torch.nn.functional.mse_loss(prediction, goal)\n"
              "optimizer = torch.optim.Adam(online.parameters(), lr=config['student_lr'])\n"
              "optimizer.zero_grad()\nloss.backward()\n"
              "print('Loss:', loss.item(), 'alvo requer gradiente:', goal.requires_grad)\n"
              "assert all(p.grad is None for p in teacher.parameters())\n"
              "assert all(p.grad is None for p in target.parameters())\n"
              "assert any(p.grad is not None for p in online.parameters())\n"
              "optimizer.step()\nexp.ema_update(target, online, config['ema'])\n"
              "print('Uma atualização LCD e EMA executada, sem alterar checkpoints.')"),
            M("Inspecione `experiments/scripts/experiment.py`: `ddim_step` devolve o "
              "estado final. A função Ψ do artigo representa o incremento. "
              "Somar novamente o latente ao estado final seria incorreto."),
        ]),
        ("04_mathematics", [
            M("# 04 · Equações, exemplos e verificações"),
            C(BOOT), note_cell("mathematics.md"),
            C("checks = exp.numerical_checks()\nprint(json.dumps(checks, indent=2))\n"
              "assert checks['ddim_max_error'] < 1e-5\n"
              "assert checks['boundary_max_error'] == 0\n"
              "assert checks['guidance_max_error'] < 1e-5\n"
              "assert checks['target_detached']"),
            C("z0, noise = 2.0, -1.0\n"
              "zt = .8*z0 + .6*noise\n"
              "recovered = (zt - .6*noise)/.8\n"
              "guided_noise = (1+2)*.2 - 2*.5\n"
              "ema_weight = .9*2 + .1*4\n"
              "lcf_pair = (.6*z0+.8*noise, .8*z0+.6*noise)\n"
              "print({'z_t':zt, 'z0_reconstruído':recovered, 'CFG':guided_noise, "
              "'EMA':ema_weight, 'par_LCF':lcf_pair})\n"
              "assert recovered == z0"),
            C("display(Video(str(ROOT / 'assets/videos/ConsistencyMath.mp4'), embed=True, width=850))"),
        ]),
        ("05_training", [
            M("# 05 · Treinamento efetivamente executado\n\n"
              "A receita local está explícita. Nenhum hiperparâmetro local é atribuído ao paper."),
            C(BOOT),
            C("print(json.dumps(config, indent=2, ensure_ascii=False))\n"
              "print('Durações (s):', protocol['durations'])\n"
              "print('Hardware:', protocol['hardware'], protocol['platform'])\n"
              "print('Python/bibliotecas:', protocol['python'], protocol['versions'])\n"
              "print('SHA-256 checkpoints:', protocol['checkpoint_sha256'])"),
            image_cell("assets/figures/training.png"),
            M("As perdas têm escalas diferentes. k=1 pode ter perda menor por comparar "
              "estados próximos. A geração é avaliada separadamente. "
              "Os modelos foram treinados uma vez por configuração, sem seleção no teste."),
            C("RUN_FULL_TRAINING = False\n"
              "if RUN_FULL_TRAINING:\n"
              "    exp.train(config)\n"
              "else:\n"
              "    print('Treino completo já executado. Para refazer e atualizar checkpoints, "
              "mude RUN_FULL_TRAINING para True ou execute python3 seminar_5/run.py experiment.')"),
            M("Após refazer treino, regenere figuras, animações, slides e notebooks. "
              "Não mantenha gráficos ou narrativa numérica de uma execução antiga."),
        ]),
        ("06_inference", [
            M("# 06 · Inferência e inspeção das representações"),
            C(BOOT),
            M("Input: ruído normal e classe. Estado intermediário: latente PCA. "
              "Output: imagem 14×14. O LCM faz nova previsão limpa após reinjetar ruído. "
              "O baseline local é DDIM, não amostragem ancestral DDPM."),
            C("data = exp.prepare_data()\n"
              "student = exp.NoiseMLP(config)\n"
              "student.load_state_dict(torch.load(ROOT / 'experiments/checkpoints/lcm_k20.pt', "
              "weights_only=True)['state_dict'])\n"
              "labels = torch.arange(10)\n"
              "latent, history = exp.sample(student, labels, 4, 101, return_history=True)\n"
              "repeated = exp.sample(student, labels, 4, 101)\n"
              "assert torch.equal(latent, repeated)\n"
              "pixels = exp.decode(latent, data)\n"
              "print('Latente:', latent.shape, 'histórico:', history.shape, 'pixels:', pixels.shape)\n"
              "print('Reprodutibilidade bit a bit da inferência nesta máquina: OK')"),
            C("import matplotlib.pyplot as plt\n"
              "fig, axes = plt.subplots(1, 10, figsize=(12, 2))\n"
              "for i, ax in enumerate(axes):\n"
              "    ax.imshow(pixels[i].reshape(14,14), cmap='gray', vmin=0, vmax=1)\n"
              "    ax.set_title(str(i))\n    ax.axis('off')\n"
              "plt.show()"),
            C("display(Video(str(ROOT / 'assets/videos/LocalGeneration.mp4'), embed=True, width=850))"),
            M("O vídeo usa o batch de 300 salvo na avaliação. A grade desta célula usa batch "
              "de 10 e faz nova inferência. Eles não são apresentados como a mesma execução."),
        ]),
        ("07_results", [
            M("# 07 · Resultados, comparação e relação com a pesquisa"),
            C(BOOT), note_cell("experimental_report.md"),
            image_cell("assets/figures/local_results.png"),
            image_cell("assets/figures/generation_grid.png"),
            C("rows = list(csv.DictReader((ROOT / 'experiments/results/metrics.csv').open()))\n"
              "for method in ['ddim', 'lcm_k1', 'lcm_k20']:\n"
              "    values = [float(r['class_agreement']) for r in rows "
              "if r['method']==method and int(r['steps'])==4]\n"
              "    print(method, '4 passos:', np.mean(values), 'DP amostral:', np.std(values, ddof=1))\n"
              "print('FID e CLIP locais: N/A — configuração experimental diferente')"),
            note_cell("methodological_audit.md"),
            note_cell("research_application.md"),
        ]),
    ]
    kernel_root = ROOT / ".cache/jupyter"
    kernel_dir = kernel_root / "kernels/seminar5"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Seminário 5", "language": "python",
    }))
    os.environ["JUPYTER_PATH"] = str(kernel_root)
    os.environ["JUPYTER_RUNTIME_DIR"] = str(ROOT / ".cache/jupyter/runtime")
    os.environ["IPYTHONDIR"] = str(ROOT / ".cache/ipython")
    for name, cells in specs:
        notebook = nbf.v4.new_notebook(cells=cells)
        notebook.metadata.kernelspec = {
            "display_name": "Python 3", "language": "python", "name": "python3",
        }
        notebook.metadata.language_info = {"name": "python", "version": sys.version.split()[0]}
        path = notebook_dir / f"{name}.ipynb"
        nbf.write(notebook, path)
        NotebookClient(notebook, timeout=180, kernel_name="seminar5",
                       resources={"metadata": {"path": str(ROOT)}}).execute()
        nbf.validate(notebook)
        nbf.write(notebook, path)
        print(f"Executado: {path.name}", flush=True)


if __name__ == "__main__":
    main()
