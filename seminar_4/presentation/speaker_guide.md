# Roteiro de fala e estrutura dos slides

36 slides; os quatro últimos são apoio. Tempo principal estimado: 35.3 minutos, incluindo demonstrações curtas.

PPTX contém texto editável e notas; o PDF é alternativa estática. Vídeos/GIFs são arquivos separados, indicados no roteiro; não estão embutidos no PPTX.

## 01. Diffuse2Seg

- TP558 • quarto ciclo de seminários
- Matheus Henrique Fonseca Afonso
- Artigo de Hümmer et al. • arXiv:2609.06491v1

**Fonte/tipo:** Artigo original, pp. 1–5

**Fala (40s):** Abrir com a pergunta: um modelo que aprendeu a gerar cenas já sabe onde estão suas entidades? A apresentação separa os resultados publicados da nossa reprodução parcial. O paper é um preprint na versão fornecida, não atribuir venue ou revisão por pares não comprovados.

**Visual:** [pipeline.png](../assets/figures/pipeline.png)

## 02. O que significa segmentar qualquer coisa?

- Things: objetos contáveis
- Stuff: regiões como céu e solo
- Partes e objetos podem coexistir

**Fonte/tipo:** Sec. 1 e 3.1

**Fala (55s):** Usar o desenho próprio: céu e solo, objeto e parte. As máscaras são agnósticas a classes e podem sobrepor-se entre níveis. Distinguir segmentação de entidades de segmentação semântica de uma classe, como pista de pouso. Não afirmar que o método atribui nomes às entidades.

**Visual:** [entities.png](../assets/figures/entities.png)

## 03. O gargalo são os rótulos

- SAM: cerca de 11M imagens / 1B máscaras
- Anotação densa custa tempo e trabalho
- Hipótese: reutilizar estrutura aprendida na geração

**Fonte/tipo:** Introdução, pp. 1–3

**Fala (55s):** As cenas não são só objetos de primeiro plano. O gerador precisa representar composição, relações e fundo; por isso sua autoatenção pode cobrir stuff melhor. A hipótese é testada via qualidade dos rótulos e utilidade no treinamento, não apenas por visualizar atenção.

## 04. Onde as abordagens anteriores param

- DINO / UnSAM: forte viés para objetos
- DiffSeg: fusão de atenção em grupos
- M2N2: máscara interativa por prompt

**Fonte/tipo:** Sec. 2 e 4.1

**Fala (65s):** UnSAM combina descoberta de instâncias com CutLER e subdivisão bottom-up; SOHES também usa hierarquias e é o baseline sem detector. DiffSeg e M2N2 são adaptados pelos autores para a comparação de entidades. Diffuse2Seg acrescenta automação de prompts, propagação e múltiplos níveis; não inventa a difusão nem o Mask2Former.

**Visual:** [prior_work.png](../assets/figures/prior_work.png)

## 05. A proposta em três etapas

- 1. SD2 congelado → afinidade
- 2. Grade + propagação → pseudo-máscaras
- 3. Treinar e enriquecer o estudante

**Fonte/tipo:** Figura 2; Sec. 3.2–3.5

**Fala (75s):** Reproduzir o vídeo PipelineDiffuse2Seg se houver tempo. Azul é extração; verde é criação de rótulos; amarelo é estudante. O gerador pode segmentar diretamente, mas a inferência zero-shot da Tabela 3 é do estudante treinado. O refinador CascadePSP é opcional e deve ser distinguido do NMS.

**Visual:** [pipeline.png](../assets/figures/pipeline.png)

**Animação:** [PipelineDiffuse2Seg.mp4](../data/outputs/manim/PipelineDiffuse2Seg.mp4) ou [PipelineDiffuse2Seg.gif](../data/outputs/manim/PipelineDiffuse2Seg.gif)

## 06. Uma passagem, não uma geração completa

- RGB 1120² → latent de aproximadamente 140²
- U-Net em t=150; texto nulo
- Extrair autoatenção de alta resolução

**Fonte/tipo:** Sec. 3.3 e 4.1, pp. 5–10

**Fala (65s):** O texto diz que o latent da imagem não recebe ruído forward. Mostrar contraste conceitual com o ciclo DDPM estudado no seminário 1: aqui não se gera uma nova imagem. A difusão fica congelada. O identificador/revisão exato SD2 não está especificado; não chamar nosso proxy RGB de latent SD2.

**Visual:** [extraction.png](../assets/figures/extraction.png)

## 07. Da autoatenção ao grafo

- Aij representa afinidade entre tokens
- Média das cabeças; pesos 0,85 / 0,15
- Uma matriz A guia todos os prompts

**Fonte/tipo:** Equação 1; Apêndice A.1

**Fala (60s):** N tokens produzem N por N afinidades. Self-attention relaciona posições da imagem; cross-attention com palavras não entra no método agnóstico a classes. A figura pequena é conceitual, não uma atenção SD2 medida. A temperatura é 0,55; a fórmula exata de aplicação não é detalhada pelo artigo.

**Visual:** [affinity.png](../assets/figures/affinity.png)

## 08. Por que propagar em vez de aplicar threshold?

- Um ponto é um vetor one-hot
- Afinidades fortes difundem informação
- Regularização busca mapas coerentes

**Fonte/tipo:** Sec. 3.4; figura local com afinidade proxy

**Fala (70s):** O paper considera as atenções esparsas e threshold isolado insuficiente. Exibir snapshots computados, não supor que sejam resultados SD2. A animação interpola visualmente entre snapshots, não simula novas iterações. Cores são reescaladas por frame, servem para estrutura e não amplitude absoluta.

**Visual:** [toy_propagation.png](../data/outputs/toy_propagation.png)

**Animação:** [PropagacaoPrompt.mp4](../data/outputs/manim/PropagacaoPrompt.mp4) ou [PropagacaoPrompt.gif](../data/outputs/manim/PropagacaoPrompt.gif)

## 09. A energia equilibra coerência e ancoragem

- p=2: propagação linear
- p<2: comportamento que preserva diferenças
- λ mantém ligação com o prompt inicial

**Fonte/tipo:** Equação 2; Algoritmo 1

**Fala (80s):** Ler a energia em duas partes: regularização no grafo e fidelidade ao f0. O p atua no gradiente local; a proteção de bordas depende da representação. No paper p=1,6, λ=1e−5 e parada por norma quadrática 1e−4. Na reprodução usamos parâmetros diferentes e epsilon explícito. No grafo simétrico, p=2 permite verificar solução fechada com Laplaciano.

**Visual:** [energy.png](../assets/figures/energy.png)

## 10. Vários pontos podem descobrir a mesma região

- Normalizar cada mapa para soma 1
- Comparar com KL simétrica
- Clustering aglomerativo por ligação média

**Fonte/tipo:** Equação 3 e Algoritmo 2; figura local

**Fala (60s):** A distância compara distribuições de propagação, não coordenadas dos prompts. A grade é automática. Mostrar as dimensões proxy: 576 por 576 na afinidade e 16 por 16 na distância KL. Estas não são as dimensões do experimento original.

**Visual:** [toy_representations.png](../data/outputs/toy_representations.png)

## 11. Cortes diferentes recuperam escalas diferentes

- h pequeno → mais grupos
- h grande → fusão em grupos grossos
- Média → upsampling → argmax → componentes

**Fonte/tipo:** Sec. 3.4 e Algoritmo 2

**Fala (70s):** O desenho explica a hierarquia; a animação usa as partições realmente calculadas no proxy p=2. O artigo agrega seis cortes KL de 0,186 a 2,99. Uma partição em um nível não basta: as propostas de todos os níveis podem representar parte e objeto. No proxy não há upsampling porque entrada e grafo já são 24 por 24.

**Visual:** [hierarchy.png](../assets/figures/hierarchy.png)

**Animação:** [HierarquiaNMS.mp4](../data/outputs/manim/HierarquiaNMS.mp4) ou [HierarquiaNMS.gif](../data/outputs/manim/HierarquiaNMS.gif)

## 12. Deduplicar sem apagar a hierarquia

- Filtrar componentes pequenos
- Ordenar candidatos por área
- Rejeitar IoU > 0,9; até 1.000 máscaras

**Fonte/tipo:** Sec. 3.4; Apêndice A.1–A.2

**Fala (65s):** A figura usa retângulos esquemáticos para explicar por que uma parte pode ter IoU baixa com o objeto inteiro. Um limiar mais baixo elimina mais candidatos, prejudicando recall. CascadePSP é refinamento de borda opcional após gerar as propostas; não é um módulo do U-Net. O artigo usa área mínima 100 px; nosso proxy usa 3 px.

**Visual:** [nms.png](../assets/figures/nms.png)

## 13. O estudante é outro modelo

- Mask2Former + ResNet-50
- Queries → máscaras e objectness
- Loss = 5 Dice + 5 máscara + 2 classe

**Fonte/tipo:** Sec. 3.5; Apêndice A.3

**Fala (65s):** Backbone produz features multiescala; pixel decoder as organiza e decoder de queries produz máscaras. Desenho próprio em nível funcional, sem inventar blocos específicos. ResNet inicia de DINO; decoder da primeira rodada é aleatório. Treinar estudante reduz dependência da geração cara para novas imagens.

**Visual:** [student.png](../assets/figures/student.png)

## 14. Rodada 2: olhar novamente para as partes

- Inferência global encontra pais
- Crops ampliados podem revelar filhos
- Validar contenção e enriquecer rótulos

**Fonte/tipo:** Sec. 3.5; Tabela 5; Apêndice A.3

**Fala (75s):** Predict-and-conquer: score mínimo, fração de área do pai entre 0,003 e 0,30, filhos com contenção mínima 0,70. Substitui rótulos com IoU maior que 0,5 e acrescenta demais. Round 2 inicializa do Round 1 e treina em 100k imagens, mesmo quando Round 1 usa mais. Esse enriquecimento não depende de CutLER, que é adicionado somente na variante correspondente.

**Visual:** [crop_training.png](../assets/figures/crop_training.png)

## 15. O que foi treinado e onde foi avaliado

- SA-1B: 100k / 200k / 400k imagens
- Holdout de 1.000 fixa os parâmetros
- Zero-shot: objetos, entidades e partes

**Fonte/tipo:** Sec. 4.1; Tabela 3

**Fala (65s):** Separar UVO, usado no gerador, de COCO/LVIS, acrescentados no estudante. O artigo exclui PartImageNet para reduzir vantagem de overlap ImageNet para DINO/CutLER. Zero-shot significa sem adaptação aos datasets alvo da avaliação, não sem pré-treinamento de modelos. Ver discussão de possível overlap dos corpora em notes/analysis.md.

**Visual:** [datasets.png](../assets/figures/datasets.png)

## 16. AR1000 mede cobertura de propostas

- Até 1.000 máscaras por imagem
- Recall médio em IoU 0,50…0,95
- ARS / ARM / ARL mostram efeito da escala

**Fonte/tipo:** Sec. 4.1

**Fala (70s):** Duas máscaras podem ter coberturas e qualidade diferentes: IoU controla se a correspondência é aceitável. AR não substitui precisão nem qualidade semântica; mil propostas podem encontrar um objeto sem poucas máscaras úteis. A métrica toy local tem matching próprio e não é COCO AR1000 oficial.

**Visual:** [metrics.png](../assets/figures/metrics.png)

## 17. Rótulos melhores, sobretudo em objetos grandes

- AR: 20,7 vs. 16,4 do UnSAM
- +4,3 pontos percentuais no agregado
- Pequenos: 2,6 vs. 5,7; limitação real

**Fonte/tipo:** Resultado ORIGINAL • Tabela 1, p. 10

**Fala (65s):** Mostrar a comparação por tamanho. A avaliação é do gerador, sem refinamento, não do estudante. A diferença de 4,3 é em pontos percentuais; não porcentagem relativa. Não esconder a desvantagem em ARS, que é relevante para pistas estreitas e partes.

**Visual:** [paper_sizes.png](../assets/figures/paper_sizes.png)

## 18. Os ganhos transferem para cinco domínios

- +4,3 a +7,1 p.p. frente ao UnSAM
- Inclui stuff e partes
- CutLER segue maior em ADE20K e UVO

**Fonte/tipo:** Resultado ORIGINAL • Tabela 2, p. 11

**Fala (70s):** UnSAM é referência anterior, mas CutLER é detector treinado: Diffuse2Seg não supera toda linha da tabela em todos os datasets. No gráfico comparar UnSAM, DiffSeg e Diffuse2Seg; CutLER foi preservado nos CSVs e discutido no texto. UVO trata imagens extraídas de vídeos; o método aqui não modela consistência temporal.

**Visual:** [paper_domains.png](../assets/figures/paper_domains.png)

## 19. No estudante, a fonte dos rótulos importa

- Sem detector: +7,7 p.p. em stuff+things
- Com CutLER / 400k: 43,8 vs. 41,7
- Things / 400k: 41,1 vs. 41,3

**Fonte/tipo:** Resultado ORIGINAL • Tabela 3, p. 12

**Fala (80s):** A coluna sem detector compara Diffuse2Seg 100k a SOHES 200k. Outra comparação é UnSAM e Diffuse2Seg ambos com CutLER e 400k. Não misturar essas condições. Means foram recalculados das seis colunas; pequenas diferenças de arredondamento são esperadas. Não afirmar superioridade universal sobre SAM ou UnSAM.

**Visual:** [paper_students.png](../assets/figures/paper_students.png)

## 20. O que as imagens do artigo mostram

- Comparação qualitativa selecionada pelos autores
- Cobertura de fundo e limites de entidades
- Exemplos não substituem avaliação sistemática

**Fonte/tipo:** Resultado ORIGINAL • Figura 1, p. 2

**Fala (60s):** Na esquerda estão pseudo-rótulos sem refinamento; na direita predições do estudante. Apontar diferenças em fundo e divisão das entidades, mas não tratar cores como identidades semânticas. O material é extraído do PDF com atribuição. Não afirmar que esses exemplos foram reproduzidos.

**Visual:** [paper_figure1.png](../assets/figures/paper_figure1.png)

## 21. Ablações: preservar propostas melhora recall

- 3 → 6 níveis: +1,8 p.p.
- NMS 0,5 → 0,9: +3,1 p.p.
- Crops em SA-1B: +3,7 p.p.

**Fonte/tipo:** Resultado ORIGINAL • Tabelas 4–5

**Fala (75s):** São três controles diferentes; não somar ganhos. Os valores da Tabela 4 não coincidem com 20,7 do resultado final e não se conhece pelo PDF uma equivalência completa das configurações. Na Figura 5 p=1,2 favorece recall, p=1,6 favorece mAP; explicar compromisso. O slide de backup mostra a figura original sem inventar pontos.

**Visual:** [paper_ablations.png](../assets/figures/paper_ablations.png)

## 22. Poucos rótulos humanos: qual é a comparação?

- Referência supervisionada 100k: 51,1 mAR
- 100 imagens de ajuste: 93,5% da referência
- 10k: ganho reportado de 1,4 p.p.

**Fonte/tipo:** Resultado ORIGINAL • Sec. 4.4 / Figura 4

**Fala (65s):** O encoder fica congelado e o decoder é ajustado sobre inicialização não supervisionada. É uma comparação de fine-tuning, não aprendizado do zero com 100 exemplos. O texto diz que 5k alcançam a referência. Não desenhamos curva de valores ausentes. Comparação UnSAM usa checkpoint público 0,2M; Diffuse2Seg usa 0,4M, conforme Apêndice A.3. Isso limita equivalência de budgets.

## 23. Nossa reprodução: o que entrou no teste

- Algoritmos de propagação e fusão
- Grafo RGB/posição, 24² pixels, 16 prompts
- Sem SD2, CascadePSP ou treinamento

**Fonte/tipo:** REPRODUÇÃO PARCIAL • scripts e NPZ locais

**Fala (80s):** Mostrar a cena conhecida com céu, solo, círculo, objeto retangular e uma parte. Há seis máscaras de referência incluindo o objeto inteiro e a parte. GT serve apenas à avaliação; não constrói o grafo nem escolhe prompts. A demonstração do pipeline em p=1,6 perde objetos; isso é um resultado e não defeito escondido da figura.

**Visual:** [toy_overview.png](../assets/figures/toy_overview.png)

## 24. Protocolo pequeno, mas verificável

- p=2 / 1,6 / 1,2; sementes 7 / 23 / 42
- Mesmos grafo, λ, prompts e cortes por semente
- Caso linear validado por solução fechada

**Fonte/tipo:** REPRODUÇÃO PARCIAL • protocol.json / métricas

**Fala (70s):** As sementes alteram ruído RGB na mesma geometria, não são três datasets independentes. λ=0,03, epsilon=1e−8, critério de parada=1e−8 e até 400 iterações. Esses parâmetros diferem dos do artigo porque o grafo/resolução são diferentes. Cada configuração convergiu segundo atualização, mas isso não prova minimização global. Verificação p=2 contra solve do sistema linear, erro máximo ~1,9e−9.

**Visual:** [toy_representations.png](../data/outputs/toy_representations.png)

## 25. O resultado local foi diferente do paper

- p=2: 56,1% ± 9,6 de recall toy
- p=1,6: 25,0%; p=1,2: 45,0%
- Não houve reprodução do ganho de p<2

**Fonte/tipo:** RESULTADO LOCAL • 9 execuções • mean ± std amostral

**Fala (85s):** Mostrar a animação com números lidos do protocolo. Recall toy usa IoU 0,50…0,95 e matching greedy 1:1 por IoU, não o ranking/evaluador COCO. O p não pode ser transferido isolado da escala da afinidade, λ, eps e critério de parada. Esses resultados não refutam o benchmark original, mas limitam a demonstração própria. Não declarar significância com três sementes.

**Visual:** [toy_results.png](../data/outputs/toy_results.png)

**Animação:** [ResultadosLocais.mp4](../data/outputs/manim/ResultadosLocais.mp4) ou [ResultadosLocais.gif](../data/outputs/manim/ResultadosLocais.gif)

## 26. Por que os resultados não são comparáveis?

- Autoatenção SD2 foi substituída por similaridade RGB
- Mesma cena toy; nenhum benchmark do paper
- Pior fusão em p=1,6: entidades desaparecem

**Fonte/tipo:** RESULTADO LOCAL • semente 7; análise própria

**Fala (75s):** Figura compara partições finas calculadas com p=2 e p=1,6. Nosso grafo não tem estrutura semântica generativa; proximidade espacial e cor não equivalem a atenção. Como demonstração, ele permite aprender as operações; como evidência de Diffuse2Seg, é insuficiente. Não comparar 56,1% toy a 20,7% SA-1B em gráfico como se fossem a mesma tarefa.

**Visual:** [toy_comparison.png](../data/outputs/toy_comparison.png)

## 27. Fotografia real: apenas demonstração conceitual

- Imagem COCO 39769 já usada no ciclo 3
- Partições vêm do mesmo proxy RGB
- Não há benchmark COCO ou identificação de classe

**Fonte/tipo:** DEMONSTRAÇÃO CONCEITUAL • imagem COCO / algoritmo local

**Fala (55s):** A imagem foi reduzida a 24 por 24 para o grafo. Não se usam anotações para gerar máscaras, e não se calculou ARCOCO. Apontar a falta de semântica: partição por cor não garante separar gatos e controles. É uma forma concreta de mostrar por que a representação SD2 é central na hipótese dos autores.

**Visual:** [coco_concept.png](../data/outputs/coco_concept.png)

## 28. Limitações e custo: onde o método pode falhar

- Latent: objetos pequenos podem desaparecer
- Texturas podem produzir sobresegmentação
- Afinidade densa 140²: 1,43 GiB só em FP32

**Fonte/tipo:** Sec. 5; cálculo teórico próprio da afinidade

**Fala (70s):** 1,43 GiB é 19.600² elementos vezes 4 bytes, sem U-Net, buffers ou cabeças. Não é medição de VRAM nem runtime. Treinamento publicado: batch16 em quatro A100. Falta relatório completo de custo e variância no paper. A inferência do estudante evita repetir o gerador, mas não remove o custo de criar sua supervisão.

## 29. O que isso oferece para pistas SAR?

- Reutilizar crops e enriquecimento de pseudo-rótulos
- Pistas estreitas desafiam a resolução latente
- SD2 RGB → SAR exige validar mudança de domínio

**Fonte/tipo:** Análise própria; project/initial_project.pdf

**Fala (85s):** Nossa pesquisa mede IoU/Dice de uma classe específica. Diffuse2Seg não decide que uma região é pista; a transferência requer seleção/validação semântica. Copiar intensidade em três canais só resolve formato. Não há benchmark SAR no artigo. Conceito mais prático é crops sobre SAM existente; usar affinities SD2 é uma hipótese mais incerta.

**Visual:** [sar_application.png](../assets/figures/sar_application.png)

## 30. Experimento futuro com controle

- 1. Congelar split, checkpoint e baseline SAM
- 2. Avaliar crops; depois qualidade dos pseudo-rótulos
- 3. Teste final: IoU/Dice, falsos positivos e custo

**Fonte/tipo:** Proposta futura • notes/research_application.md

**Fala (65s):** Uma alteração por ablação, seleção apenas na validação, separação geográfica quando possível, mesmo orçamento/arquitetura/inicialização. Medir aceitação das pseudo-máscaras e erros de largura. Adicionar imagens sem pista quando disponíveis para medir falso positivo. Nenhum desses experimentos SAR foi executado neste seminário.

## 31. Conclusões

- Representações generativas podem supervisionar segmentação
- Propagação + hierarquia conectam pontos a entidades
- Nossa reprodução explica operações, não valida o benchmark

**Fonte/tipo:** Síntese do artigo e dos experimentos locais

**Fala (55s):** Retomar a pergunta inicial: o paper oferece evidência de estrutura reutilizável e rótulos úteis; a reprodução local confirma a operacionalização e mostra dependência da afinidade. Para SAR, tratar transferência como hipótese e priorizar avaliação de crops. Perguntar à audiência como avaliar utilidade de pseudo-rótulos além de recall de propostas.

## 32. Referências e materiais

- Hümmer et al. • arXiv:2609.06491v1
- Código, protocolos e resultados em seminar_4
- README: execução, vídeos e notas técnicas

**Fonte/tipo:** Fontes em notes/sources.md

**Fala (30s):** Referência principal https://arxiv.org/abs/2609.06491v1. Baselines e fundamentos estão nas referências do próprio artigo. As animações são próprias; figuras originais têm página atribuída. Códigos não oficiais com nomes semelhantes não foram tratados como implementação do artigo.

## 33. Backup: timestep e compromisso p / precisão

- p=1,2 favorece recall; p=1,6 favorece mAP
- Curvas de timestep são normalizadas por métrica
- Não extraímos valores intermediários dessas curvas

**Fonte/tipo:** Resultado ORIGINAL • Figura 5, p. 13

**Fala (0s):** A figura original tem eixo da direita normalizado por máximo individual. Não comparar esses pontos como valores absolutos de AR ou mAP. Evitar inferir uma curva de treinamento: é sweep de hiperparâmetros, não loss por época.

**Visual:** [paper_figure5.png](../assets/figures/paper_figure5.png)

## 34. Backup: configuração para treino integral

- AdamW; LR 5e−5; weight decay 0,05
- 1024²; batch16; 2.000 queries; 4 × A100
- Round 1 ~8 épocas; Round 2 em 100k imagens

**Fonte/tipo:** Apêndice A.3, p. 21

**Fala (0s):** Backbone multiplier0,1, clipping0,01, warmup5000 e schedule90%/96%. Reprodução numérica fiel exigiria checkpoints identificados, acesso SA-1B, variantes de refinamento e validação da implementação. Esses recursos não foram carregados nem treinados.

## 35. Backup: ablação local de fusão / NMS

- 3 / 6 níveis têm mesmo recall neste toy
- NMS0,5: 16,7%; NMS0,9: 25,0%
- Mais candidatos não garantem melhores máscaras

**Fonte/tipo:** RESULTADO LOCAL • p=1,6 • semente 7

**Fala (0s):** Valores vêm de ablations.json. A contagem pré-NMS muda com seis níveis; o recall final não muda porque as propostas úteis não aumentam. Não somar esse ganho ao efeito de p. É uma única cena/seed, sem conclusão estatística.

**Visual:** [toy_ablations.png](../assets/figures/toy_ablations.png)

## 36. Backup: linearidade e detalhes numéricos

- p=2: [2L + λI]f = λf⁰ no grafo simétrico
- Erro máximo do check: cerca de 1,9e−9
- Gradiente zero: epsilon explícito no proxy

**Fonte/tipo:** Verificação própria • checks() / protocol.json

**Fala (0s):** O fator2 vem de contar pares simétricos na energia descrita. Atenção real pode ser assimétrica; uma reprodução exata precisa esclarecer o tratamento. A regularização de gradientes no código usa piso de epsilon para evitar potência negativa de zero. Parada por atualização pequena não prova ótimo global não linear.
