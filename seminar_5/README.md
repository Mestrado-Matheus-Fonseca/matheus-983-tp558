# Seminário 5 · Latent Consistency Models

Material em português sobre **Latent Consistency Models: Synthesizing
High-Resolution Images with Few-Step Inference**, Luo et al.,
[arXiv:2310.04378v1](https://arxiv.org/abs/2310.04378v1), publicado como preprint
em 6 de outubro de 2023. A fonte fixa é o [PDF local](articles/2310.04378v1.pdf).

> [!IMPORTANT]
> A implementação é uma **demonstração conceitual executada de LCD** com
> MNIST/PCA/MLP. As tabelas FID/CLIP são resultados do artigo. Não foram
> reproduzidas quantitativamente. O uso de DDPM para aumento SAR e a comparação
> posterior com consistência/LCM são propostas para a pesquisa de mestrado.

## Começar pelo material pronto

| Material | Conteúdo |
|---|---|
| [Apresentação editável](presentation/TP558_LCM.pptx) | 33 slides, notas de fala e cinco MP4 incorporados |
| [Apresentação PDF](presentation/TP558_LCM.pdf) | Versão estática, com os posters dos vídeos |
| [Roteiro](presentation/speaker_notes.md) | Explicação por slide, duração sugerida de 30–35 minutos |
| [Resumo técnico](notes/technical_summary.md) | Método, configurações, resultados publicados, ablações e limites |
| [Resumo didático](notes/didactic_summary.md) | Narrativa acadêmica e versão para a fala |
| [Matemática](notes/mathematics.md) | Equações, variáveis, exemplos e ligação ao código |
| [Relatório executado](notes/experimental_report.md) | Métricas locais, tempos e comparação de escopo |
| [Aplicação ao mestrado](notes/research_application.md) | DDPM para aumento SAR e comparação futura com consistência/LCM |
| [Auditoria](notes/methodological_audit.md) | Diferenças entre paper, código público, checkpoint e demo |
| [Fontes](notes/sources.md) | Proveniência, citações, versão do artigo e referências visuais |

No PowerPoint, clique no vídeo para reproduzi-lo. O suporte a vídeo depende do
visualizador. Os MP4 externos abaixo permitem reproduzir cada animação separadamente.
No PDF os vídeos são estáticos. O deck mantém proporção 16:9, fundo branco,
tipografia e cores dos seminários anteriores.

## Notebooks executados

1. [Artigo e visão geral](notebooks/01_article_overview.ipynb)
2. [Dados, preprocessing e PCA](notebooks/02_data_pipeline.ipynb)
3. [Método e uma atualização LCD](notebooks/03_method.ipynb)
4. [Matemática e verificações](notebooks/04_mathematics.ipynb)
5. [Treinamento e checkpoints](notebooks/05_training.ipynb)
6. [Inferência determinística](notebooks/06_inference.ipynb)
7. [Resultados, auditoria e pesquisa SAR](notebooks/07_results.ipynb)

Todos contêm saídas salvas. O notebook de treinamento carrega os registros de
treinos efetivamente realizados. Para refazer o treino completo, use o comando
`experiment` ou habilite `RUN_FULL_TRAINING` nesse notebook.
A célula do notebook 03 faz uma atualização real em cópias em memória e preserva
os checkpoints entregues.

## O que foi executado

```text
MNIST 28×28 → redução 14×14 → PCA/whitening 32D
                                   ↓
                         Professor DDPM condicional
                                   ↓
                      DDIM + CFG para construir pares
                                   ↓
                         Aluno LCD + alvo EMA
                                   ↓
                      Inferência LCM em 1, 2, 4, 8 passos
                                   ↓
                         Decoder PCA → imagem
```

- 6000 imagens de treino, 1000 de validação e 1000 de teste, balanceadas por classe.
- PCA e classificador de avaliação ajustados somente no treino.
- Professor: MLP treinado com perda DDPM, 5000 atualizações.
- Dois alunos: 5000 atualizações cada, ablação de skipping k=1 e k=20.
- Baseline de inferência: **DDIM**, incluindo referência de 50 passos.
- Uma seed de treino por configuração. Três seeds de geração, 300 imagens cada.
- CPU, duas threads. Sem CUDA, modelos Stable Diffusion ou coleta LAION.

Aderência à classe é calculada com um classificador local. Não equivale a FID,
CLIP, qualidade perceptual nem desempenho do SAM. A latência mede apenas a
amostragem latente em batch. Médias, desvios e medidas por seed estão no
[relatório](notes/experimental_report.md) e no [CSV](experiments/results/metrics.csv).
Não há inferência de significância estatística entre treinamentos.

## Instalação

Ambiente observado: Python **3.13.11**, Intel Core Ultra 9 185H, PyTorch CPU.
As versões estão em [requirements.txt](requirements.txt) e o ambiente observado
em [environment.json](experiments/results/environment.json).

Dependências do sistema: FFmpeg/ffprobe, Poppler (`pdfinfo`, `pdftoppm`),
LibreOffice para exportar PDF e Cairo/Pango para Manim. Em Debian/Ubuntu, os
pacotes correspondentes são `ffmpeg`, `poppler-utils`, `libreoffice-impress`,
`libcairo2-dev`, `libpango1.0-dev`, `pkg-config` e `fonts-dejavu-core`.
As cenas usam `Text`, portanto não exigem instalação LaTeX.

A partir da raiz `matheus-983-tp558`, crie um ambiente exclusivo:

```bash
uv venv --python 3.13 seminar_5/.venv
uv pip install --python seminar_5/.venv/bin/python -r seminar_5/requirements.txt
```

O launcher prefere `seminar_5/.venv`. Neste computador, o link do Python do
ambiente antigo do repositório estava quebrado. O launcher informa quando usa
o Python 3.13 do uv com os pacotes existentes, sem modificar esse ambiente.
`python-pptx` foi instalado em `seminar_5/.deps` para esta execução.
Essa particularidade local não é necessária no ambiente independente acima.

Os notebooks podem ser abertos no VS Code/Jupyter escolhendo o Python do
ambiente criado. O runner executa um kernel temporário usando `nbclient`.
Ambientes com sandbox precisam permitir sockets locais para o kernel.

## Reproduzir

O subset MNIST já está incluído em `data/mnist_subset.npz`. Não há download de
dados nem modelos ao executar o experimento. A origem, hashes dos arquivos IDX
e a seleção estão em [provenance.json](data/provenance.json).
Os índices de cada split também estão no NPZ.

Para refazer todas as etapas, da raiz do repositório:

```bash
python3 seminar_5/run.py all
```

Esse comando atualiza checkpoints, métricas, figuras, vídeos, notebooks e slides
dentro de `seminar_5/`. O tempo depende da máquina. As durações de treino medidas
na execução entregue estão no protocolo. A renderização e a exportação acrescentam
tempo ao processo completo.

Etapas individuais, na ordem:

```bash
python3 seminar_5/run.py experiment
python3 seminar_5/run.py figures
python3 seminar_5/run.py animations
python3 seminar_5/run.py materials
python3 seminar_5/run.py notebooks
python3 seminar_5/run.py verify
```

`materials` exporta PPTX e PDF usando LibreOffice headless. Se a política do
ambiente bloquear o processo, a etapa termina com erro em vez de afirmar que
o PDF foi gerado. A execução dos notebooks também termina com erro se o kernel
for bloqueado.

Para alterar a narrativa, edite [slides.json](presentation/slides.json) e execute
`materials`. Para mudar o experimento, edite [demo.json](experiments/configs/demo.json)
e regenere todas as etapas. O subset contém a assinatura de sua configuração
de seleção. Se contagens, seed de seleção, resolução ou dimensão latente
mudarem, o script rejeita um subset incompatível em vez de reutilizá-lo
silenciosamente. Preserve o subset anterior e prepare explicitamente outro
a partir dos IDX originais.

Para reproduzir os **mesmos dados**, use o NPZ entregue. Para reconstruí-lo dos
originais, o script espera os quatro arquivos IDX em
`seminar_1/data/mnist/source/MNIST/raw/`, acessados somente para leitura.
O próprio pipeline de experimentos não depende dos slides antigos.

## Vídeos e GIFs

| Conteúdo | MP4 1280×720, 30 fps | GIF |
|---|---|---|
| Pipeline de LCD e inferência | [PipelineLCM](assets/videos/PipelineLCM.mp4) | [GIF](assets/gifs/PipelineLCM.gif) |
| Arquitetura e condições | [ArchitectureLCM](assets/videos/ArchitectureLCM.mp4) | [GIF](assets/gifs/ArchitectureLCM.gif) |
| Consistência, boundary e EMA | [ConsistencyMath](assets/videos/ConsistencyMath.mp4) | [GIF](assets/gifs/ConsistencyMath.gif) |
| Comparação de avaliações de rede | [CompareSampling](assets/videos/CompareSampling.mp4) | [GIF](assets/gifs/CompareSampling.gif) |
| Saídas intermediárias locais | [LocalGeneration](assets/videos/LocalGeneration.mp4) | [GIF](assets/gifs/LocalGeneration.gif) |

Código em [lcm_scenes.py](manim/lcm_scenes.py), renderização em
[render.py](manim/render.py). Os posters são os frames finais dos próprios vídeos.
As durações estão no [manifesto](assets/videos/manifest.json).
Gráficos estão disponíveis em PNG e PDF vetorial em `assets/figures/`.

## Resultados e rastreabilidade

| Arquivo | Uso |
|---|---|
| [paper_results.json](data/paper_results.json) | Tabelas 1/2 publicadas, separadas dos resultados locais |
| [metrics.csv](experiments/results/metrics.csv) | 39 configurações/seeds avaliadas |
| [aggregate.json](experiments/results/aggregate.json) | Médias e desvios por configuração |
| [training.csv](experiments/results/training.csv) | Perdas registradas durante os treinos |
| [protocol.json](experiments/results/protocol.json) | Configuração, versões, hardware, tempos e hashes |
| [samples.npz](experiments/results/samples.npz) | Imagens geradas e históricos para as animações |
| [Checkpoints](experiments/checkpoints/) | Professor, classificador e alunos EMA k=1/20 |
| [verification.json](experiments/results/verification.json) | Resultado da verificação automatizada |

`verify` verifica identidades DDIM/guidance/boundary/EMA, ausência de gradiente
no alvo, splits e PCA, hashes, determinismo de inferência, contagem de avaliações,
regeração das imagens salvas, agregação das métricas, execução dos notebooks,
integridade de vídeos/GIFs e páginas/slides.

## Relação com a pesquisa

A entrega mais recente do projeto descreve segmentação de pistas SAR com SAM
e YOLO. A orientação do usuário acrescenta **DDPM para aumento do treino** e
comparação posterior com **consistência/LCM**.

A análise propõe primeiro validar pares SAR–máscara sintéticos e seu efeito no
segmentador. Depois, destilar o mesmo professor e medir custo por par aceito e
IoU/Dice em teste real independente. DDPM em pixels pede inicialmente
consistência em pixels. O nome LCM aplica-se a uma comparação em latentes com
autoencoder validado. Consulte a [análise e os dois protocolos propostos](notes/research_application.md).

## Organização

```text
seminar_5/
├── articles/                 # Paper original e snapshots de fontes oficiais
├── data/                     # Subset MNIST, proveniência e tabelas do paper
├── experiments/
│   ├── configs/              # Configuração explícita da demonstração
│   ├── scripts/              # Treinamento, inferência e verificação
│   ├── checkpoints/          # Pesos realmente treinados
│   └── results/              # CSV, JSON e NPZ medidos
├── notebooks/                # Sete notebooks executados e gerador/runner
├── manim/                    # Cenas e renderização
├── assets/                   # Figuras, imagens, vídeos e GIFs
├── presentation/             # PPTX, PDF, roteiro, narrativa e geradores
├── notes/                    # Análise científica, matemática e auditoria
├── requirements.txt
└── run.py
```

Os seminários anteriores foram utilizados somente como referência.
O código e os materiais deste trabalho ficam restritos ao Seminário 5.
