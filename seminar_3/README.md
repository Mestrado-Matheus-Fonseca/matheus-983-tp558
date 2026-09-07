# Seminário 3 — Image Data Augmentation

Quatro notebooks simples, com explicações em português, verificações executáveis,
figuras, tabelas e animações Manim para apoiar a apresentação dos dois artigos.
Os notebooks são entregues com as saídas da execução. Todos os experimentos usam CPU.

## Ordem de execução

| Notebook | Evidência produzida |
|---|---|
| [01 — Transformações e rótulos](notebooks/01_transformacoes_e_rotulos.ipynb) | Geometria sincronizada em imagem, máscara e caixa; teste de IoU; Cutout, Mixup e CutMix |
| [02 — Experimento MNIST](notebooks/02_experimento_mnist.ipynb) | Treinamento real de uma MLP, cinco políticas, três sementes, testes limpo e deslocado |
| [03 — Tabelas e benchmark](notebooks/03_tabelas_e_benchmark.ipynb) | Auditoria das Tabelas 2–4 do survey e Tabela I de Albumentations; benchmark local |
| [04 — Animações Manim](notebooks/04_animacoes_manim.ipynb) | Renderização, reprodução de MP4/GIF e roteiro de fala |

Na raiz do repositório:

```bash
uv sync --group seminar3
uv run --group seminar3 seminar_3/run_notebooks.py
```

Para abrir no JupyterLab:

```bash
uv run --group seminar3 --with jupyterlab jupyter lab seminar_3/notebooks
```

No VS Code, selecione o interpretador `.venv/bin/python` como kernel. Os caminhos
funcionam a partir da raiz do repositório ou da pasta dos notebooks. O MNIST é
reutilizado de `seminar_1/data/mnist/source`; caso ausente, torchvision faz o
download. As demonstrações de imagem usam dados sintéticos criados no notebook 01.
Não são necessários checkpoints ou GPU. O treinamento tem 15 execuções pequenas;
o tempo total depende da CPU e inclui a renderização das três cenas.

Manim já é uma dependência do projeto. O grupo `seminar3` acrescenta
Albumentations **2.0.8**, nbformat e nbclient. O `uv.lock` fixa a resolução completa.
O sistema precisa de **FFmpeg** para os GIFs e de **Cairo/Pango** para Manim
(em Debian/Ubuntu: `ffmpeg libcairo2-dev libpango1.0-dev pkg-config`). As cenas
usam `Text`, dispensando LaTeX.

## Vídeos e GIFs prontos

| Cena | MP4 (720p, 30 FPS) | GIF (800 px, 10 FPS) |
|---|---|---|
| Imagem e rótulo | [TransformacoesRotulos.mp4](data/outputs/manim/TransformacoesRotulos.mp4) | [TransformacoesRotulos.gif](data/outputs/manim/TransformacoesRotulos.gif) |
| Cutout, Mixup e CutMix | [Misturas.mp4](data/outputs/manim/Misturas.mp4) | [Misturas.gif](data/outputs/manim/Misturas.gif) |
| Resultados locais | [Resultados.mp4](data/outputs/manim/Resultados.mp4) | [Resultados.gif](data/outputs/manim/Resultados.gif) |

Para regenerar apenas as animações, após os notebooks 01–03:

```bash
uv run --group seminar3 seminar_3/manim/render.py
```

As cenas estão em [manim/augmentation_scenes.py](manim/augmentation_scenes.py).
Os arquivos finais são preservados em `data/outputs/manim/`; somente caches e
fragmentos intermediários são ignorados pelo Git. As cenas usam os pixels e as
métricas produzidos pelos notebooks, sem resultados fictícios.

## O que foi reproduzido

- **Mecanismos:** transformações, preservação de rótulos e mistura de imagens,
  com verificações numéricas sobre dados conhecidos.
- **Experimento independente:** MNIST, MLP 784→128→10, 2.000 exemplos de treino,
  1.000 de validação, 2.000 de teste; Adam, 15 épocas, lote 128; sementes 7/23/42.
  Split estratificado fixo, inicialização e ordem dos lotes pareadas por semente.
  Usa a última época, sem seleção pelo teste. Médias e desvios não são intervalos
  de confiança e não demonstram significância estatística com três sementes.
- **Análise dos números publicados:** transcrição auditável em CSV e gráficos
  dos ganhos recalculados. A auditoria encontrou cinco divergências entre a
  coluna de ganho e a subtração das colunas com/sem augmentation; ambos os
  valores são preservados. As linhas do CenterNet mostram piora.
- **Velocidade local:** três operações equivalentes, entrada/saída NumPy RGB,
  conversões Pillow e cópia de saída incluídas; aquecimento, cinco repetições,
  mediana e quartis. Hardware e versões ficam registrados em JSON.

Não foi refeito o treinamento de Wide-ResNet/DenseNet/Shake-ResNet em
CIFAR/SVHN, nem de segmentadores em VOC ou detectores em COCO. O benchmark atual
não replica o ambiente de 2018. A transcrição não é uma nova medição; os ganhos
recalculados não determinam qual campo do artigo contém erro. O MNIST permite
testar as ideias com baixo custo, sem reivindicar as acurácias do survey.

## Resultado da execução entregue

Acurácia em %, média ± desvio padrão amostral entre as três sementes:

| Política | Teste limpo | Teste deslocado +3 px |
|---|---:|---:|
| Sem augmentation | 90,07 ± 0,30 | 39,53 ± 0,38 |
| Affine leve | 89,42 ± 0,93 | 65,38 ± 0,36 |
| Cutout | 89,60 ± 0,23 | 40,20 ± 1,08 |
| Mixup | 88,92 ± 0,32 | 39,57 ± 0,53 |
| Rotação 180° | 82,95 ± 0,48 | 43,92 ± 0,96 |

Neste protocolo, nenhuma política superou a referência no teste limpo. Affine
leve aumentou a robustez à translação em **25,85 pontos percentuais**, com queda
de 0,65 p.p. no teste limpo. O controle de rotação perdeu 7,12 p.p. no teste
limpo. Isso permite apresentar tanto o benefício de uma invariância adequada
quanto os limites do aumento de dados, sem selecionar somente resultados favoráveis.

Os valores completos estão em [mnist_summary.csv](data/outputs/mnist_summary.csv).
Esta tabela registra a execução entregue; ao modificar parâmetros ou reexecutar
com outras versões, consulte os CSVs regenerados. Os quatro notebooks foram
executados integralmente, sem células com erro. Foram verificados também os
três MP4s (1280×720, 30 FPS) e os três GIFs (800×450, 10 FPS), com duração entre
15,5 e 17 segundos por cena.

## Arquivos para os slides e auditoria

- `data/outputs/transformacoes.png`, `misturas.png`: exemplos visuais.
- `data/outputs/mnist_accuracy.png`, `mnist_learning.png`: resultados locais.
- `data/outputs/survey_gains.png`, `paper_speedup.png`: dados dos artigos.
- `data/outputs/benchmark_local.png`: tempos medidos nesta máquina.
- `data/outputs/*results.csv`, `*curves.csv`, `*summary.csv`, `benchmark_raw.csv`:
  medições por execução, época e repetição.
- `data/outputs/*protocol.json`: parâmetros, ambiente e versões.
- `data/processed/mnist_split.npz`: índices exatos do split.
- `data/processed/survey_tables_2_3_4.csv`, `albumentations_table_1.csv`:
  transcrições das tabelas; valores ausentes de Albumentations são `nan`.

As verificações estão nas próprias células: geometria, rótulos, conservação dos
pesos de mistura, separação dos splits, execução do treinamento, auditoria
aritmética, equivalência das saídas do benchmark e existência das mídias.

## Fontes

1. Yang et al., *Image Data Augmentation for Deep Learning: A Survey*,
   [arXiv:2204.08610v2, 5 nov. 2023](https://arxiv.org/abs/2204.08610v2).
   §§2.1–2.3, Figura 1, Tabelas 2–4 (páginas 4–5 do PDF local).
2. Buslaev et al., *Albumentations: fast and flexible image augmentations*,
   [arXiv:1809.06839v1, 18 set. 2018](https://arxiv.org/abs/1809.06839v1).
   §II, Figuras 1–4 e Tabela I (página 3). O PDF fornecido é a versão de 2018,
   não a publicação expandida de 2020.
3. [Código oficial de Albumentations 2.0.8](https://github.com/albumentations-team/albumentations/tree/2.0.8)
   e [documentação de bounding boxes](https://albumentations.ai/docs/3-basic-usage/bounding-boxes-augmentations/).

Os dois PDFs originais permanecem em `articles/`.
