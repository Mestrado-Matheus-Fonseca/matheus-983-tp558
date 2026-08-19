# Seminário 2 - Segment Anything Model 2 (SAM 2)

Este projeto transforma os três artigos de `articles/` em um roteiro prático e
reprodutível. Ele combina:

- explicação da arquitetura e da segmentação orientada por prompts;
- um experimento controlado de imagem e vídeo com máscaras de referência;
- inferência opcional com a implementação oficial do SAM 2;
- reprodução gráfica dos principais números do paper original;
- figuras em PNG e PDF prontas para a apresentação.

## Estrutura

```text
seminar_2/
├── articles/                 # PDFs fornecidos na disciplina
├── data/
│   ├── raw/                  # frames sintéticos ou dados externos
│   ├── processed/            # máscaras e tabelas processadas
│   ├── models/               # código/checkpoints locais do SAM 2
│   └── outputs/              # métricas e figuras geradas
└── notebooks/
    ├── 01_introduction_sam2.ipynb
    ├── 02_data_preparation.ipynb
    ├── 03_sam2_inference.ipynb
    ├── 03_1_sam2_efficiency_comparison.ipynb
    ├── 04_results_reproduction.ipynb
    └── 05_visualizations_for_seminar.ipynb
```

Os diretórios de dados mantêm somente `.gitkeep` no Git. Arquivos gerados,
datasets e pesos não são versionados.

## Execução básica

Na raiz do repositório:

```bash
uv sync
uv run --with jupyter jupyter lab seminar_2/notebooks
```

Execute os notebooks em ordem. O caminho básico usa apenas as dependências já
declaradas e funciona em CPU. O notebook 03 usa um baseline de cor quando o
SAM 2 não está instalado; esse baseline valida o fluxo e as métricas, mas não é
apresentado como resultado do SAM 2. O notebook 03.1 requer o SAM 2 e compara
seleção de instância e tempo com um baseline clássico orientado por cor.

## Inferência com o SAM 2 oficial

A reprodução neural requer o código oficial e um checkpoint. A variante Tiny é
a escolha mais leve para a demonstração:

```bash
git clone https://github.com/facebookresearch/sam2.git seminar_2/data/models/sam2-source
uv pip install --python .venv/bin/python -e seminar_2/data/models/sam2-source
curl -L https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt \
  -o seminar_2/data/models/sam2.1_hiera_tiny.pt
```

Depois, reinicie o kernel e execute novamente o notebook 03. Ele detecta o
pacote e o checkpoint, seleciona CUDA quando disponível e salva as máscaras e
métricas em `data/outputs/`.

## Escopo da reprodução

As Tabelas 1, 2, 4, 5 e 6 de *SAM 2: Segment Anything in Images and Videos*
são transcritas, verificadas e visualizadas no notebook 04. Isso reproduz os
resultados publicados, não o treinamento nem o benchmark completo. Reexecutar
os experimentos integrais requer SA-V, MOSE, DAVIS, LVOS e YouTube-VOS, além de
infraestrutura comparável à descrita no artigo (as medições de velocidade foram
feitas em uma NVIDIA A100).

## Artigos usados

1. Ravi et al. (2024), *SAM 2: Segment Anything in Images and Videos*.
2. Zhang et al., *A Survey on Segment Anything Model (SAM): Vision Foundation
   Model Meets Prompt Engineering*.
3. *SAM2 for Image and Video Segmentation: A Comprehensive Survey*.

Cada notebook indica as seções, figuras ou tabelas do PDF usadas como fonte.
Código e checkpoints oficiais: <https://github.com/facebookresearch/sam2>.

## Vídeos Manim

As cenas `SAM2UsageDemo` e `SAM2ArchitectureFlow` estão em
`manim/sam2_scenes.py`. Para renderizá-las em 720p/30 FPS:

```bash
uv run manim -qm --media_dir seminar_2/data/outputs/manim \
  seminar_2/manim/sam2_scenes.py SAM2UsageDemo SAM2ArchitectureFlow
```

Os MP4s usam os frames, máscaras e métricas produzidos pelo notebook 03.
