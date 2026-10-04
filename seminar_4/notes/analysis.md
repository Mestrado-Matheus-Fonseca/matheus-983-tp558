# Análise técnica — Diffuse2Seg

**Fonte principal:** Hümmer, Sicking, Hüger e Gottschalk, *Diffuse2Seg: Diffusion Models Can Segment Anything Without Supervision*, arXiv:2609.06491v1. O PDF local de 22 páginas é a versão de referência. Página pública: <https://arxiv.org/abs/2609.06491v1>. Este documento distingue afirmações dos autores, cálculos sobre tabelas e interpretação crítica própria.

## Problema, motivação e hipótese

O problema é segmentação de entidades em mundo aberto: produzir máscaras sem um vocabulário fechado de classes, cobrindo objetos contáveis (*things*), regiões como céu/solo (*stuff*) e partes em várias escalas. As máscaras podem se sobrepor entre níveis da hierarquia. A tarefa não exige nomear o objeto; uma máscara de carro não precisa vir acompanhada de “carro”. Isso difere tanto de segmentação semântica supervisionada quanto de segmentação em vocabulário aberto condicionada por texto.

SAM demonstra a utilidade desse objetivo, mas seu treinamento usa SA-1B, aproximadamente 11 milhões de imagens e mais de um bilhão de máscaras. Os autores procuram uma fonte alternativa de supervisão que dispense máscaras manuais na geração inicial. Abordagens baseadas em DINO tendem a privilegiar objetos de primeiro plano. Geradores de imagens precisam representar também fundo, composição e estrutura da cena.

A hipótese central é que a autoatenção de um modelo de difusão pré-treinado contém afinidades suficientes para agrupar entidades e preservar seus limites. Pontos automáticos, propagação não linear e fusão em múltiplas escalas transformariam essa representação em pseudo-máscaras úteis para treinar um segmentador generalista.

“Sem supervisão” se refere à ausência de máscaras humanas na geração e no treinamento não supervisionado estudado. SD2 já foi pré-treinado em imagem/texto; ResNet-50 tem inicialização DINO; CascadePSP é um refinador pré-treinado; a variante com CutLER usa propostas de um detector aprendido. A seleção de hiperparâmetros usa um holdout SA-1B com métricas contra referência. Portanto, não interpretar o título como ausência absoluta de dados, modelos aprendidos ou informação de validação.

## Metodologia: três etapas com papéis diferentes

### 1. Extrair afinidade de difusão congelada — Sec. 3.3, pp. 5–6

Uma imagem RGB é redimensionada para 1120 × 1120 e codificada pelo VAE de SD2. O latent tem aproximadamente 140 × 140 posições, N = 19.600. Os autores executam uma única passagem do U-Net em t = 150, com embeddings de texto nulos. O texto explicita que não adiciona ruído do processo forward ao latent. Não se trata de sintetizar uma imagem com uma trajetória completa de amostragem.

Extraem-se matrizes de autoatenção A^(l,h), N × N, nas duas camadas selecionadas da resolução mais alta do decoder. Primeiro faz-se média entre cabeças; depois a combinação ponderada entre camadas:

`A = Σ_l w_l (1/n_H) Σ_h A^(l,h)`, com `w_1 = 0,85`, `w_2 = 0,15`.

Usa-se temperatura de atenção 0,55. A Sec. 4.1 fala em suavizar, enquanto o Apêndice A.1 fala em tornar a distribuição mais concentrada. Uma temperatura menor que 1 aplicada aos logits antes do softmax usualmente concentra a distribuição. A implementação exata depende do ponto onde a temperatura entra; o PDF não especifica integralmente esse detalhe nem o identificador/revisão do checkpoint SD2. Não simular uma extração oficial com escolhas arbitrárias.

### 2. Converter prompts em máscaras — Sec. 3.4, pp. 6–8; Algoritmos 1/2, p. 19

Uma grade equidistante, espaçamento de seis posições no latent, define os prompts. Cada prompt é um vetor one-hot f⁰; todos são propagados independentemente e podem ser processados em paralelo. O objetivo é:

`E(f) = (1/p) Σ_i [Σ_j A_ij (f_j − f_i)²]^(p/2) + (λ/2)||f − f⁰||²`.

O primeiro termo promove coerência no grafo; o segundo preserva ligação com o ponto inicial. p = 2 corresponde à propagação linear; p < 2 favorece comportamento que limita a suavização entre regiões distintas. Isso depende da qualidade da afinidade e não garante contornos corretos em qualquer imagem.

O Algoritmo 1 usa Gauss-Jacobi: calcula a norma local do gradiente g_i, os coeficientes `γ_ij = A_ij (g_i^(p−2) + g_j^(p−2))` e atualiza `f_i = (λ f⁰_i + Σ_j γ_ij f_j)/(λ + Σ_j γ_ij)`. Configuração: p = 1,6; λ = 10⁻⁵; parar quando `||f_next − f||² ≤ 10⁻⁴`. Para p < 2, gradiente nulo requer tratamento numérico; esse epsilon não é detalhado no PDF. Autoatenção também pode ser assimétrica, enquanto a interpretação usual da energia de um grafo precisa considerar a simetria. São detalhes a verificar no código oficial antes de uma reprodução fiel.

Cada mapa é normalizado para uma distribuição de soma 1. Calcula-se a divergência KL simétrica entre pares: `(KL(p_k||p_l) + KL(p_l||p_k))/2`. Mapas semelhantes são agrupados com clustering aglomerativo de ligação média. Seis cortes logaritmicamente espaçados entre 0,186 e 2,99 produzem escalas distintas. Em cada grupo faz-se média dos mapas; após upsampling para a resolução original, argmax fornece uma partição. Componentes conectados separam instâncias espacialmente desconectadas. Filtra-se área inferior a 100 pixels.

Reúnem-se candidatos de todos os níveis. NMS em ordem decrescente de área elimina um candidato somente quando sua IoU com alguma máscara aceita é maior que 0,9, preservando até 1.000 propostas. É NMS por área; não supor um score de confiança de SAM. Um refinamento opcional com CascadePSP melhora limites. Os autores descartam máscaras refinadas quase globais (fração de área > 0,9) ou com IoU entre máscara original e refinada < 0,5; usam crops até 480 px, em vez dos 900 px do UnSAM.

### 3. Destilar em segmentador e enriquecer — Sec. 3.5; Apêndice A.3

Mask2Former com backbone ResNet-50 é treinado de modo agnóstico a classes. A difusão atua como gerador de supervisão; a inferência do estudante não precisa repetir a extração SD2. A arquitetura Mask2Former usa features multiescala, pixel decoder e decoder de queries para produzir máscaras e objectness. O artigo não propõe um novo bloco neural; a contribuição está na fonte de rótulos e no treinamento.

A perda combina Dice, BCE/máscara e classificação de objectness, pesos 5, 5 e 2. Round 1 inicia ResNet-50 com DINO e decoder aleatório. AdamW, LR 5 × 10⁻⁵, weight decay 0,05, multiplicador LR do backbone 0,1, batch 16 em quatro A100, clipping 0,01, entrada 1024 × 1024 e 2.000 queries. Aproximadamente oito épocas, warmup de 5.000 iterações e decaimentos em 90%/96% do schedule. Esses custos explicam por que o treinamento integral não é uma demonstração leve em CPU.

Há copy-paste hierárquico (p = 0,5), preferência por objetos pequenos (probabilidade 0,8), área de 4.096 pixels e escala entre 0,5 e 1. Na variante com CutLER, limiar 0,1 e mistura de fontes: ambas 50% das iterações, somente CutLER 20%, somente Diffuse2Seg 30%.

Round 2 inicia do checkpoint correspondente de Round 1. O modelo faz inferência global e sobre recortes de pais previstos. Reamostrar crops a 1024 × 1024 permite descobrir partes pequenas. Pais têm fração de área entre 0,003 e 0,30; padding 0,05; limite de IoU entre pais 0,30. Mantêm-se filhos com contenção `area(filho ∩ pai)/area(filho) ≥ 0,70`. Substituem-se máscaras originais com IoU > 0,5, acrescentando candidatos restantes.

Sem CutLER: θ = 0,55, k = 12, NMS de crops 0,75 e intra-crop 0,40. Com CutLER: θ = 0,65, k = 9, NMS 0,85/0,50; CutLER é adicionado após o enriquecimento. Round 2 usa 100 mil imagens com rótulos enriquecidos, independentemente do orçamento de Round 1. O mecanismo de crops não depende do detector.

## Dados e protocolos

| Papel | Dados | Leitura correta |
|---|---|---|
| Gerar rótulos e treinar | SA-1B: 100k / 200k / 400k imagens | Aproximadamente 1% / 2% / 4%; não é custo total do pré-treinamento SD2/DINO |
| Fixar parâmetros/validar | Holdout SA-1B de 1.000 imagens não sobrepostas | Parâmetros congelados antes de transferir aos demais domínios |
| Avaliar gerador | SA-1B, ADE20K, EntitySeg, UVO, PACO | Respectivamente entidades, cenas, entidades, vídeo e partes |
| Avaliar estudante | COCO, LVIS, ADE20K, EntitySeg, SA-1B, PACO | Inferência zero-shot em domínio alvo; UVO aparece no gerador |
| Fine-tuning semi-supervisionado | 100, 500, 1k, 2k, 5k, 10k imagens SA-1B com máscaras | Backbone congelado; ajusta-se o decoder |

PartImageNet é excluído porque compartilha imagens ImageNet com o pré-treinamento DINO/CutLER, prejudicando a comparação com SD2. Isso não elimina todos os possíveis vazamentos: o corpus de pré-treinamento de difusão e a sobreposição com benchmarks não são auditados em detalhe no PDF.

## Métricas e resultados — números publicados, não reprodução

AR1000 é recall médio nos limiares de IoU 0,50 a 0,95 com no máximo 1.000 propostas por imagem. A sigla mAR é usada pelos autores para esse recall; nos agrupamentos de datasets calculam médias dos valores por domínio. ARS/ARM/ARL separam escalas COCO. mAP é usada em análises selecionadas e considera precisão/ranking; não é intercambiável com recall. A taxonomia dos rótulos varia entre datasets, motivo dado pelos autores para priorizar AR.

Na Tabela 1, sem refinamento de borda, o gerador Diffuse2Seg alcança AR1000 = 20,7 contra UnSAM 16,4 (+4,3 p.p.) e DiffSeg 16,0 (+4,7 p.p.). Entretanto ARS = 2,6 é inferior ao UnSAM 5,7. O ganho agregado vem de escalas médias/grandes, não de uma vitória uniforme.

Na Tabela 2, os ganhos frente a UnSAM são 4,3 / 6,5 / 4,5 / 7,1 / 4,3 p.p. em SA-1B / ADE20K / EntitySeg / UVO / PACO. Diffuse2Seg não supera a referência treinada CutLER em ADE20K (22,5 vs. 24,8) e UVO (29,9 vs. 32,3). A afirmação de liderança deve restringir-se ao conjunto de geradores comparados sem treinamento adicional específico.

Na Tabela 3, sem detector, Diffuse2Seg usa 100k imagens e atinge 37,2 em things, 40,3 em stuff+things e 27,1 em partes, versus SOHES com 200k: 29,8 / 32,6 / 17,1. Os ganhos são 7,4 / 7,7 / 10,0 p.p. Não confundir esse resultado do estudante com AR20,7 do gerador.

Com CutLER e 400k imagens, a média stuff+things é 43,8 vs. UnSAM 41,7 (+2,1 p.p.); things é 41,1 vs. 41,3 (ligeiramente menor). Em EntitySeg, 43,2 fica 2,7 p.p. abaixo de SAM 45,9. As comparações usam a mesma tarefa de propostas automáticas; não são comparações universais de segmentação interativa.

Na extensão semi-supervisionada, a referência treinada do zero com 100k imagens atinge média 51,1; 100 imagens de ajuste recuperam 93,5% dessa referência. Isso não significa treinar desde zero com 100 imagens: o estudante já foi treinado em centenas de milhares de imagens pseudo-rotuladas e usa modelos pré-treinados. O texto diz que 5k alcançam a referência; abstract sugere superar. Evitar aumentar a precisão dessa afirmação. Com 10k, ganho reportado é 1,4 p.p. Não foram digitizados pontos intermediários da Figura 4.

### Ablações publicadas

- p próximo de 1,2 maximiza recall; p = 1,6 favorece mAP. A escolha prioriza precisão dos pseudo-rótulos, não o máximo recall.
- t = 150 preserva bom recall e favorece mAP; a Figura 5b normaliza cada métrica pelo seu próprio máximo. Não apresentar os valores do eixo como recall absoluto.
- Tabela 4a: três → seis níveis, AR16,3 → 18,1 (+1,8 p.p.). Tabela 4b: NMS0,5 → 0,9, AR15,4 → 18,5 (+3,1 p.p.). Não misturar esses controles com o resultado final 20,7 da Tabela 1: o artigo não estabelece que todas as condições são idênticas.
- Tabela 5a: crops melhoram SA-1B de 39,8 → 43,5 (+3,7 p.p.) e PACO de 25,8 → 27,1 (+1,3 p.p.); CutLER aumenta COCO de 37,4 → 41,0 (+3,6 p.p.).
- Tabela 5b: a segunda rodada global melhora stuff+things 40,3 → 40,7; com crops chega a 42,5. O ganho de 2,2 p.p. usa Round 1 como referência, enquanto 1,8 p.p. compara às predições globais de Round 2.

## Crítica, limitações e custo

Os autores reconhecem falhas em objetos pequenos, limitados pela resolução latente, e sobresegmentação por textura, por exemplo no céu. Sugerem coarse-to-fine e ranking de objectness. Um sistema multigranular pode cobrir densamente uma cena sem produzir a definição semântica desejada por uma aplicação.

Como avaliação própria, AR alto com 1.000 propostas favorece cobertura, mas não assegura precisão, calibração, tempo de inferência ou utilidade de cada máscara. O PDF não fornece uma tabela sistemática de latência, VRAM, energia ou dispersão entre sementes. Ganhos não vêm acompanhados de intervalos de confiança. Pré-treinamentos, arquiteturas, orçamentos e refinadores diferem em algumas comparações.

N = 140² dá N² = 384.160.000 elementos: uma única afinidade densa FP32 precisa de 1,54 GB decimais (1,43 GiB), antes de cabeças, camadas, buffers e U-Net. Esse é um cálculo teórico, não medição de VRAM. A propagação depende de K prompts e envolve multiplicações com A; fusão adiciona distâncias K². Destilar o estudante reduz a necessidade de repetir essa geração na inferência final. “Leve” é relativo aos backbones e não significa execução barata do sistema inteiro.

Não há repositório/checkpoints oficiais identificados no PDF nem na consulta pública registrada em 03/10/2026. Isso é ausência de localização verificada, não prova de inexistência. Identificador preciso SD2, implementação de atenção/temperatura, tratamento de zeros e detalhes de scoring para mAP precisam ser confirmados para reprodução numérica fiel.

## O que é essencial explicar oralmente

1. Qual é a unidade de saída: conjunto de máscaras de entidades, inclusive partes/fundo.
2. De onde vem a supervisão e por que “sem supervisão” tem um escopo específico.
3. Por que usar autoatenção de geração; por que threshold de afinidade sozinho é insuficiente.
4. Papel separado de p, λ e critério de parada; fusão KL e níveis hierárquicos.
5. Gerador congelado versus estudante treinado; crops na segunda rodada.
6. AR do gerador versus AR do estudante, com budgets e variante CutLER explícitos.
7. Um resultado desfavorável (objetos pequenos) e uma limitação da reprodução própria.
8. Transferência a SAR como hipótese experimental, não resultado demonstrado.

## Contribuições e aplicações

A contribuição relatada é converter autoatenção generativa em pseudo-rótulos multigranulares automáticos, demonstrar a utilidade desses rótulos no estudante e aproveitar a inicialização em ajuste semi-supervisionado. A prioridade “primeiro método” é uma afirmação dos autores e não uma revisão independente exaustiva. Aplicações plausíveis são anotação assistida, robótica e preparação de dados fora de vocabulários fechados. O artigo menciona relevância médica, mas não apresenta benchmark médico ou SAR.
