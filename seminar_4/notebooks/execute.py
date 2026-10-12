"""Crie/execute dois notebooks com kernel temporário; sem instalação global."""

import json
import os
from pathlib import Path
import sys
import tempfile

from nbclient import NotebookClient
import nbformat

ROOT = Path(__file__).resolve().parents[1]


def notebook(cells):
    result = nbformat.v4.new_notebook()
    result.cells = [nbformat.v4.new_markdown_cell(source) if kind == "md"
                    else nbformat.v4.new_code_cell(source) for kind, source in cells]
    result.metadata.kernelspec = {"display_name": "Python 3 — TP558", "language": "python", "name": "python3"}
    return result


SETUP = '''from pathlib import Path
import json
from IPython.display import display, Image
import os
root = Path(os.environ.get("SEMINAR4_ROOT", "seminar_4")).resolve()
if not root.exists():
    root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
assert (root / "articles/2609.06491v1.pdf").exists()
print("Seminário:", root)'''


def main():
    first = notebook([
        ("md", "# 01 — Entender o artigo e auditar os resultados\n\n"
         "**Diffuse2Seg, arXiv:2609.06491v1.** Os números abaixo são transcritos do artigo; "
         "não são resultados produzidos por execução de seus modelos. "
         "A análise completa está em [analysis.md](../notes/analysis.md)."),
        ("code", SETUP),
        ("md", "## 1. Hipótese e pipeline\n\nUm gerador pré-treinado captura relações de entidades. "
         "O artigo usa sua autoatenção congelada para criar pseudo-máscaras e treina um estudante. "
         "O estudante Mask2Former e o gerador SD2 são modelos distintos. A extração usa "
         "t=150, imagem sem ruído forward e embeddings de texto nulos."),
        ("code", 'display(Image(filename=str(root / "assets/figures/pipeline.png"), width=1100))'),
        ("md", "## 2. Propagação\n\n"
         r"$E(f)=\frac1p\sum_i[\sum_j A_{ij}(f_j-f_i)^2]^{p/2}+\frac\lambda2\|f-f^0\|_2^2$."
         "\n\np=2 é linear; p<2 altera o comportamento de regularização. "
         "Os autores usam p=1,6 para favorecer precisão dos rótulos, não o máximo recall. "
         "Grade → mapas suaves → normalização → KL simétrica → ligação média → cortes → "
         "upsampling/argmax → componentes → NMS. O refinador de borda é opcional."),
        ("code", 'display(Image(filename=str(root / "assets/figures/energy.png"), width=1000))'),
        ("md", "## 3. Auditoria das tabelas originais\n\nAs tabelas 1–5 têm transcrição em JSON "
         "e CSV. Não inferimos valores ausentes em curvas. Todos os valores desta seção são em %."),
        ("code", '''from IPython.display import Markdown
paper = json.loads((root / "data/paper_results.json").read_text())
columns = paper["table2_columns"]
lines = ["| Método | " + " | ".join(columns) + " |", "|---|" + "---:|"*len(columns)]
lines.extend("| " + name + " | " + " | ".join(map(str, values)) + " |" for name, values in paper["table2"].items())
display(Markdown("\\n".join(lines)))
gains = [round(o-b, 1) for o, b in zip(paper["table2"]["Diffuse2Seg"], paper["table2"]["UnSAM"], strict=True)]
assert gains == [4.3, 6.5, 4.5, 7.1, 4.3]
print("Ganhos recalculados vs. UnSAM (p.p.):", dict(zip(columns, gains, strict=True)))'''),
        ("code", 'display(Image(filename=str(root / "assets/figures/paper_sizes.png"), width=1000))\n'
         'display(Image(filename=str(root / "assets/figures/paper_students.png"), width=1100))'),
        ("md", "## 4. Interpretação crítica\n\n- AR1000 mede recall de até 1.000 propostas; "
         "não garante precisão nem classe correta.\n- Gerador: AR20,7 na Tabela1; estudante: "
         "outros valores na Tabela3.\n- UnSAM é melhor em objetos pequenos no gerador.\n"
         "- Ganhos de fine-tuning pressupõem pré-treinamento e pseudo-rótulos em grande escala.\n"
         "- Figura5b é normalizada por métrica; não é curva de treino ou AR absoluto."),
        ("code", '''n = 140*140
bytes_fp32 = n*n*4
print(f"Uma afinidade densa: {n:,} tokens, {bytes_fp32/2**30:.3f} GiB FP32")
print("Cálculo teórico; sem U-Net, cabeças e buffers. Não é medida de VRAM.")'''),
        ("md", "## 5. Relação com a pesquisa SAR\n\n"
         "[Análise fundamentada](../notes/research_application.md): pistas são uma classe estreita e alongada; "
         "as entidades genéricas precisam de seleção semântica. RGB→SAR exige avaliar mudança de domínio. "
         "Crops e enriquecimento de pseudo-rótulos podem ser testados antes de trocar arquitetura. "
         "Nenhum resultado SAR foi reproduzido nesta atividade."),
    ])
    second = notebook([
        ("md", "# 02 — Reprodução parcial e materiais visuais\n\n"
         "Executamos os mecanismos de propagação/fusão em um grafo **proxy RGB/posição**. "
         "Não carregamos SD2 ou checkpoints Diffuse2Seg. A geometria é uma cena toy; "
         "três sementes variam somente ruído. [Protocolo e diferenças](../notes/experimental_report.md)."),
        ("code", SETUP),
        ("md", "## 1. Executar e verificar\n\nA referência conhecida não entra na afinidade ou nos prompts. "
         "O caso linear é comparado à solução fechada; os controles incluem matching sem duplicação."),
        ("code", '''import importlib.util
spec = importlib.util.spec_from_file_location("seminar4_experiment", root / "experiments/scripts/experiment.py")
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)
experiment.main()'''),
        ("md", "## 2. Entradas, intermediários e saídas\n\nAs figuras mostram mapas realmente "
         "calculados pelo solver. As cores de snapshots são normalizadas por painel: "
         "não comparar amplitude entre frames."),
        ("code", 'display(Image(filename=str(root / "data/outputs/toy_pipeline.png"), width=1100))\n'
         'display(Image(filename=str(root / "data/outputs/toy_comparison.png"), width=1100))'),
        ("md", "## 3. Resultados medidos\n\nRecall toy nos limiares IoU0,50…0,95, matching greedy "
         "1:1 por IoU. **Não é COCO AR1000**. Reportamos mean ± std amostral, sem teste de significância."),
        ("code", '''protocol = json.loads((root / "experiments/results/protocol.json").read_text())
for row in protocol["summary"]:
    print(f"p={row['p']}: {100*row['recall_mean']:.2f}% ± {100*row['recall_std']:.2f} p.p.")
assert protocol["checks"]["checks"] == "passed"
assert protocol["checks"]["linear_closed_form_max_error"] < 1e-7
display(Image(filename=str(root / "data/outputs/toy_results.png"), width=1100))'''),
        ("md", "O ganho publicado de propagação não linear **não foi reproduzido neste proxy**. "
         "O p atua em conjunto com afinidade, λ, escala, parada e fusão KL. p=2 foi melhor aqui. "
         "Esta discrepância não valida nem refuta o benchmark SD2; mostra o limite da simplificação."),
        ("code", 'display(Image(filename=str(root / "assets/figures/toy_ablations.png"), width=1000))\n'
         'display(Image(filename=str(root / "data/outputs/coco_concept.png"), width=1100))'),
        ("md", "## 4. Animações Manim prontas\n\nAs quatro cenas seguem o fundo escuro do ciclo3. "
         "Pipeline é um diagrama do paper; propagação/hierarquia/resultados usam dados do proxy. "
         "MP4s não ficam embutidos no PPTX; inserir manualmente nos slides indicados no roteiro. "
         "Regenerar: `python3 seminar_4/run.py animations`."),
        ("code", '''from IPython.display import Video
scenes = ["PipelineDiffuse2Seg", "PropagacaoPrompt", "HierarquiaNMS", "ResultadosLocais"]
for scene in scenes:
    video = root / "data/outputs/manim" / f"{scene}.mp4"
    assert video.exists(), video
    print(scene, round(video.stat().st_size/1024), "KiB")
    display(Video(filename=str(video), embed=True, width=960))'''),
        ("md", "## 5. Apresentação e documentação\n\n"
         "[PPTX](../presentation/TP558_Diffuse2Seg.pptx) • "
         "[PDF](../presentation/TP558_Diffuse2Seg.pdf) • "
         "[Roteiro de fala](../presentation/speaker_guide.md) • [README](../README.md). "
         "Slides: 32 principais + 4 de apoio; cerca de35min. "
         "As referências de resultados publicados e resultados locais estão explícitas."),
    ])
    with tempfile.TemporaryDirectory(prefix="seminar4-jupyter-") as temporary:
        kernel = Path(temporary) / "kernels/seminar4"
        kernel.mkdir(parents=True)
        (kernel / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Seminário4", "language": "python",
            "env": {"SEMINAR4_ROOT": str(ROOT)}
        }), encoding="utf-8")
        os.environ["JUPYTER_PATH"] = temporary + os.pathsep + os.environ.get("JUPYTER_PATH", "")
        os.environ["JUPYTER_RUNTIME_DIR"] = str(Path(temporary) / "runtime")
        for name, nb in [("01_article_pipeline.ipynb", first), ("02_reproduction_and_media.ipynb", second)]:
            client = NotebookClient(nb, timeout=180, kernel_name="seminar4",
                                    resources={"metadata": {"path": str(ROOT.parent)}})
            client.execute()
            nbformat.write(nb, ROOT / "notebooks" / name)
            print(f"Executado e salvo: {name}", flush=True)


if __name__ == "__main__":
    main()
