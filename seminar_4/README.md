# Seminário 4 — Diffuse2Seg

Material em português para **Diffuse2Seg: Diffusion Models Can Segment Anything Without Supervision**, Hümmer et al., [arXiv:2609.06491v1](https://arxiv.org/abs/2609.06491v1). A versão fixa é o [PDF local](articles/2609.06491v1.pdf). Foram usados os ciclos 2/3 como referência, sem alterá-los.

## Começar pela apresentação

- [PPTX — 36 slides, texto editável e notas de fala](presentation/TP558_Diffuse2Seg.pptx)
- [PDF — os mesmos 36 slides](presentation/TP558_Diffuse2Seg.pdf)
- [Roteiro completo de fala e referências por slide](presentation/speaker_guide.md)
- [Fonte editável dos slides](presentation/slides.json)

São **32 slides principais + 4 de apoio**, cerca de **35 minutos** de fala. Títulos/bullets do PPTX são editáveis; gráficos e diagramas entram como imagens, com PNG/SVG e scripts disponíveis. O arquivo foi aberto e exportado pelo LibreOffice para verificar compatibilidade. Vídeos/GIFs ficam separados: inserir os arquivos abaixo nos slides indicados no roteiro, ou reproduzi-los pelo notebook. O PDF tem os visuais estáticos. Para uma fala de 20–25 minutos, usar principalmente 1–6, 8–14, 16–19, 23, 25–26 e 28–32, resumindo protocolo/dados oralmente.

O fundo claro, azul e identificação TP558 seguem a apresentação anterior; as animações mantêm fundo escuro `#101827`, MP4 720p30 e GIF de 800 px a 10 FPS. O tema do template do primeiro ciclo é apenas lido na exportação PPTX.

## Análise e evidências

| Material | Conteúdo |
|---|---|
| [Análise do artigo](notes/analysis.md) | Problema, hipótese, metodologia, modelos, treino/inferência, datasets, métricas, tabelas, ablações, contribuições e crítica |
| [Relatório experimental](notes/experimental_report.md) | Resultados efetivamente medidos, configuração, verificações e diferenças frente ao artigo |
| [Aplicabilidade à pesquisa SAR](notes/research_application.md) | Relação com `project/initial_project.pdf`; componentes reutilizáveis, adaptações e protocolo futuro |
| [Fontes/proveniência](notes/sources.md) | PDF, páginas, consulta pública, fotografia e referências dos ciclos anteriores |
| [Tabelas originais em JSON](data/paper_results.json) | Transcrição manual das Tabelas 1–5; valores publicados, não resultados locais |
| [CSVs originais e ganhos recalculados](data/processed/paper_results/) | Versões para auditar gráficos e diferenças em pontos percentuais |

## Reprodução realizada: escopo explícito

Reproduzimos parcialmente os **Algoritmos 1/2**: prompts one-hot, propagação Gauss-Jacobi p-Laplaciana, normalização, KL simétrica, clustering de ligação média em vários cortes, componentes conectados e NMS por área. A afinidade SD2 foi substituída por um **kernel de RGB/posição** em grade 24×24. São 16 prompts, seis máscaras GT multigranulares conhecidas, três valores p e três sementes que variam ruído na mesma cena. O GT entra apenas na avaliação.

| p | Recall toy médio (%) | Std amostral (p.p.) | Mean best IoU |
|---|---:|---:|---:|
| 2,0 | 56,11 | 9,62 | 0,7348 |
| 1,6 | 25,00 | 0,00 | 0,3568 |
| 1,2 | 45,00 | 0,00 | 0,5269 |

**O ganho não linear do artigo não foi reproduzido neste proxy.** Afinidade, escala, λ, parada e fusão influenciam o resultado. Recall toy usa matching greedy 1:1 por IoU entre 0,50 e 0,95 e **não é AR1000 COCO oficial**. Não comparar esses valores aos números SA-1B do artigo como um benchmark comum. Três sementes de uma cena não demonstram generalização nem significância estatística.

O caso linear foi verificado contra a solução fechada, erro máximo ≈1,9×10⁻⁹. Há checks para KL e matching sem duplicação, histórico de atualização, métricas por execução e mapas NPZ. A ablação local de níveis/NMS também está salva. A fotografia COCO 39769, reutilizada somente para leitura do ciclo3, é uma **demonstração conceitual sem avaliação COCO**.

Não executamos extração SD2, refinamento CascadePSP, treinamento Mask2Former ou inferência de checkpoints oficiais Diffuse2Seg. O ambiente tem PyTorch CPU; não foram localizados código/pesos oficiais verificáveis na consulta registrada. O treino publicado usa 100k–400k imagens e quatro A100. A alternativa parcial permite demonstrar as operações sem inventar resultados. As lacunas para reprodução integral estão documentadas no relatório.

Os materiais distinguem **resultado original**, **reprodução parcial**, **demonstração conceitual**, **cálculo teórico** e **proposta futura**. Não há resultado SAR produzido neste seminário.

## Notebooks com saídas

1. [01 — Artigo, pipeline e auditoria](notebooks/01_article_pipeline.ipynb)
2. [02 — Reprodução, gráficos e vídeos](notebooks/02_reproduction_and_media.ipynb)

Ambos foram executados e salvos com outputs. Imagens e vídeos no notebook são embutidos para visualização independente. O notebook02 executa novamente os experimentos; diagramas/gráficos de apresentação são gerados pelo comando `materials`.

## Vídeos/GIFs prontos

| Conceito / slide | MP4 | GIF |
|---|---|---|
| Pipeline completo / 5 | [PipelineDiffuse2Seg](data/outputs/manim/PipelineDiffuse2Seg.mp4) | [GIF](data/outputs/manim/PipelineDiffuse2Seg.gif) |
| Propagação medida / 8 | [PropagacaoPrompt](data/outputs/manim/PropagacaoPrompt.mp4) | [GIF](data/outputs/manim/PropagacaoPrompt.gif) |
| Hierarquia e NMS / 11–12 | [HierarquiaNMS](data/outputs/manim/HierarquiaNMS.mp4) | [GIF](data/outputs/manim/HierarquiaNMS.gif) |
| Resultados locais / 25 | [ResultadosLocais](data/outputs/manim/ResultadosLocais.mp4) | [GIF](data/outputs/manim/ResultadosLocais.gif) |

Propagação e resultados usam os dados realmente calculados. A cena de hierarquia usa p=2 no proxy, não a configuração SD2 oficial. Os mapas de propagação têm cores normalizadas por frame e transições visuais entre snapshots; essas transições não são iterações extras do solver. O pipeline e os diagramas de arquitetura são explicações esquemáticas, não medições.

## Regenerar tudo

Na raiz `matheus-983-tp558`, com o ambiente Python disponível:

```bash
python3 seminar_4/run.py all
```

Ou por etapa, nesta ordem:

```bash
python3 seminar_4/run.py experiment
python3 seminar_4/run.py materials
python3 seminar_4/run.py animations
python3 seminar_4/run.py notebook
python3 seminar_4/run.py verify
```

`run.py` usa primeiro `seminar_4/.venv`, depois o `.venv` do repositório. Neste computador o link do Python do ambiente antigo estava quebrado por apontar para uma instalação VS Code removida: o launcher usa o Python3.13 instalado pelo uv e os pacotes já existentes, sem alterar o ambiente anterior. Caches ficam somente em `seminar_4/.cache` e `seminar_4/manim/cache`, ignorados pelo Git. O script de notebook cria um kernel temporário; sua execução precisa permitir os sockets locais do Jupyter.

### Ambiente independente em outro computador

```bash
uv venv --python 3.13 seminar_4/.venv
uv pip install --python seminar_4/.venv/bin/python -r seminar_4/requirements.txt
python3 seminar_4/run.py all
```

[requirements.txt](requirements.txt) fixa as versões observadas; [protocol.json](experiments/results/protocol.json) registra Python, bibliotecas, configuração e SHA-256 do paper/fotografia. Dependências do sistema: **FFmpeg**, **Poppler** (`pdftoppm`/`pdfinfo`) e **Cairo/Pango** para Manim. Em Debian/Ubuntu: `ffmpeg poppler-utils libcairo2-dev libpango1.0-dev pkg-config`. Não há requisito LaTeX: Manim usa `Text` e fórmulas estáticas usam mathtext. Não há necessidade de torch, checkpoints ou GPU para este experimento parcial.

O experimento leva segundos na máquina utilizada; renderizar animações e notebooks leva mais tempo e depende do computador. Execuções atualizam protocolos/outputs, mantendo fontes e dados do artigo separados. Se a fotografia do ciclo3 não estiver disponível, a etapa conceitual é opcional no script experimental, mas deve ser fornecida para regenerar o conjunto completo de materiais já entregue.

## Estrutura

```text
seminar_4/
├── articles/                   # PDF original existente
├── notes/                      # análise, relatório, pesquisa, fontes
├── presentation/               # PPTX/PDF, fonte JSON, roteiro, previews
├── notebooks/                  # duas análises executadas e runner
├── experiments/scripts/        # algoritmo e verificação dos artefatos
├── experiments/results/        # métricas CSV, protocolo JSON e mapas NPZ
├── assets/figures/              # diagramas/gráficos PNG + SVG; figuras originais atribuídas
├── data/processed/paper_results/# CSVs transcritos e ganhos recalculados
├── data/outputs/                # visuais locais e MP4/GIF Manim
├── manim/                      # cenas e renderização
├── requirements.txt
└── run.py
```

Para modificar a narrativa, editar `presentation/slides.json` e executar `materials`. Para alterar a análise experimental, editar `experiments/scripts/experiment.py`, regenerar experimentos e visuais, e revisar as frases numéricas do roteiro/README: resultados alterados não devem ficar ligados a texto antigo. Os dados originais do artigo permanecem em `data/paper_results.json`.
