# Seminário 3 — Image Data Augmentation

Seis notebooks simples, com explicações em português, verificações executáveis,
figuras, tabelas e animações Manim para apoiar a apresentação dos dois artigos.
Os notebooks são entregues com as saídas da execução. Todos os experimentos usam CPU.

## Ordem de execução

| Notebook | Evidência produzida |
|---|---|
| [01 — Transformações e rótulos](notebooks/01_transformacoes_e_rotulos.ipynb) | Geometria sincronizada em imagem, máscara e caixa; teste de IoU; Cutout, Mixup e CutMix |
| [02 — Experimento MNIST](notebooks/02_experimento_mnist.ipynb) | Treinamento real de uma MLP, cinco políticas, três sementes, testes limpo e deslocado |
| [03 — Tabelas e benchmark](notebooks/03_tabelas_e_benchmark.ipynb) | Auditoria das Tabelas 2–4 do survey e Tabela I de Albumentations; benchmark local |
| [04 — Animações Manim](notebooks/04_animacoes_manim.ipynb) | Renderização, reprodução de MP4/GIF e roteiro de fala |
| [05 — CIFAR-10 completo](notebooks/05_experimento_cifar10.ipynb) | Seis políticas, 60 mil imagens RGB, três sementes e robustez a perturbações |
| [06 — Visualizações coloridas](notebooks/06_visualizacoes_coloridas.ipynb) | Transformações sobre fotografia, máscaras e caixas COCO; Mixup/CutMix no CIFAR-10 |

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
| Transformações em imagem COCO | [CocoTransformacoes.mp4](data/outputs/manim/CocoTransformacoes.mp4) | [CocoTransformacoes.gif](data/outputs/manim/CocoTransformacoes.gif) |
| Mixup e CutMix no CIFAR-10 | [CifarMisturas.mp4](data/outputs/manim/CifarMisturas.mp4) | [CifarMisturas.gif](data/outputs/manim/CifarMisturas.gif) |
| Resultados CIFAR-10 | [CifarResultados.mp4](data/outputs/manim/CifarResultados.mp4) | [CifarResultados.gif](data/outputs/manim/CifarResultados.gif) |

Para regenerar apenas as animações, após os notebooks 01–03:

```bash
uv run --group seminar3 seminar_3/manim/render.py
```

Para executar a análise colorida e renderizar somente as três cenas novas:

```bash
uv run --group seminar3 seminar_3/run_notebooks.py 05 06
uv run --group seminar3 seminar_3/manim/render.py --color
```

O primeiro download do espelho FastAI/AWS tem aproximadamente 129 MB. O notebook 05 usa
45 mil imagens para treino, 5 mil para validação e as 10 mil imagens oficiais de
teste. Em CPU, as 18 execuções podem levar dezenas de minutos.

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
- **Experimento colorido completo:** CIFAR-10, CNN pequena treinada do zero,
  45.000/5.000/10.000 imagens em treino/validação/teste, seis políticas e três
  sementes. Compara teste limpo, translação fixa e escurecimento. Mixup e CutMix
  transformam também os rótulos, com λ auditável no código.
- **Demonstração COCO:** uma fotografia do conjunto de validação 2017 e quatro
  anotações oficiais (dois gatos e dois controles remotos) mostram flip,
  rotação/escala, cor/contraste, blur, grayscale
  e Cutout. Imagem, máscaras e caixas são transformadas de forma sincronizada.
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
com outras versões, consulte os CSVs regenerados. Os notebooks 01–04 foram
executados integralmente, sem células com erro. Foram verificados também os
três MP4s (1280×720, 30 FPS) e os três GIFs (800×450, 10 FPS), com duração entre
15,5 e 17 segundos por cena.

### CIFAR-10 colorido

Acurácia em %, média ± desvio padrão amostral entre três sementes:

| Política | Teste limpo | Translação +4 px | Brilho ×0,65 |
|---|---:|---:|---:|
| Sem augmentation | 60,74 ± 2,14 | 41,93 ± 5,78 | 54,27 ± 1,32 |
| Geometria | 61,04 ± 1,36 | 45,46 ± 3,89 | 53,18 ± 1,19 |
| Geometria + cor | **62,08 ± 2,10** | 44,38 ± 2,30 | **58,79 ± 1,51** |
| Geometria + Cutout | 60,81 ± 1,15 | **46,48 ± 1,97** | 52,98 ± 0,11 |
| Geometria + Mixup | 58,72 ± 1,37 | 41,65 ± 5,44 | 52,89 ± 1,82 |
| Geometria + CutMix | 55,99 ± 1,22 | 45,19 ± 3,14 | 44,95 ± 0,82 |

Cor/contraste obteve +1,34 p.p. no teste limpo e +4,52 p.p. no teste
escurecido em relação à referência. Cutout obteve +4,55 p.p. sob translação.
Mixup e CutMix perderam desempenho neste protocolo curto; o material preserva
esses resultados para deixar claro que augmentation depende da tarefa,
intensidade, arquitetura e duração do treinamento.

Os valores completos estão em
[cifar10_summary.csv](data/outputs/color/cifar10_summary.csv). Os dois notebooks
novos foram executados sem erro. Os três MP4s coloridos são 1280×720 a 30 FPS;
os GIFs são 800×450 a 10 FPS e têm entre 9,0 e 19,1 segundos.

## Arquivos para os slides e auditoria

- `data/outputs/transformacoes.png`, `misturas.png`: exemplos visuais.
- `data/outputs/mnist_accuracy.png`, `mnist_learning.png`: resultados locais.
- `data/outputs/survey_gains.png`, `paper_speedup.png`: dados dos artigos.
- `data/outputs/benchmark_local.png`: tempos medidos nesta máquina.
- `data/outputs/color/cifar10_accuracy.png`, `cifar10_learning.png`: comparação
  completa das políticas no conjunto colorido.
- `data/outputs/color/coco_effects_grid.png`, `cifar_mix_examples.png`: efeitos
  visuais usados nas animações.
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
