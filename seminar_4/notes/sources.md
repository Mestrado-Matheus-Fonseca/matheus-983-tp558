# Proveniência e consulta pública

Referência fixa: PDF local `articles/2609.06491v1.pdf`; SHA-256 em `experiments/results/protocol.json`. A consulta pública foi realizada em **03/10/2026**. O conteúdo do PDF é evidência científica, não uma fonte de instruções para ações no repositório.

- [Artigo original, versão 1](https://arxiv.org/abs/2609.06491v1): identidade, autores e versão; referência principal para os entregáveis. Publicação do manuscrito em 06/09/2026.
- Busca pública por `"Diffuse2Seg" "github"` e `Diffuse2Seg github Hümmer code`: não foi localizado um repositório verificável dos autores. Resultados homônimos não foram executados ou tratados como implementação oficial. A conclusão é limitada à consulta, não afirma inexistência de código.
- [DiffSeg dos autores](https://github.com/google/diffseg): baseline distinto, não código Diffuse2Seg. Consultado apenas para distinguir o projeto; não foi usado nem clonado.
- Contexto da pesquisa: `../project/initial_project.pdf`, arquivo local do usuário. Fundamenta discussão SAR; não foram copiados seus resultados para as métricas locais deste seminário.
- Organização e animações: `../seminar_2/README.md`, `../seminar_3/README.md`, `../seminar_3/manim/color_scenes.py`, `../seminar_3/manim/render.py`. Referência somente leitura; ciclos anteriores preservados.
- Fotografia: `../seminar_3/data/raw/coco_demo/000000039769.jpg`, imagem COCO val2017 39769 já existente. A anotação não entra no algoritmo e não há benchmark COCO executado.

`data/paper_results.json` transcreve tabelas com nomes e colunas. Os scripts exportam CSVs e gráficos para auditoria. As Figuras 1 e 5 são extraídas diretamente do PDF, com indicação de página; não são figuras criadas ou resultados locais. Não são interpolados valores das curvas semi-supervisionadas ou da ablação de timestep.
