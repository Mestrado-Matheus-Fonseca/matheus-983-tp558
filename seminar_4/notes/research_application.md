# Relação com a pesquisa: segmentação de pistas em SAR Sentinel-1

**Contexto local:** `project/initial_project.pdf`, “Avaliação comparativa e otimização de modelos de aprendizado profundo para segmentação de pistas de pouso em imagens SAR Sentinel-1”. A pesquisa compara SAM ajustado, YOLOv8x e YOLO11x, com avaliação por IoU/Dice, otimização e ablações de pré-processamento/aumento. Não foram alterados datasets, splits ou experimentos dessa pesquisa.

## Compatibilidade e limites

O gerador Diffuse2Seg produz entidades genéricas e não identifica a classe pista. A tarefa de pesquisa requer uma máscara semântica específica, fina e alongada; portanto pseudo-máscaras precisam de seleção e validação. SD2 foi pré-treinado para imagens naturais RGB, ao passo que Sentinel-1 traz intensidades/polarizações, speckle e geometria de radar. Replicar uma banda em três canais satisfaz formato, mas não transforma SAR em RGB nem garante afinidades úteis.

Na resolução latente, uma pista estreita pode ocupar poucos tokens ou desaparecer; a fraqueza publicada em ARS é particularmente relevante. A agregação multigranular também pode fundir pista, estrada e clareira, ou dividir pista por textura. O artigo não demonstra resultado SAR. O experimento sintético RGB deste seminário tampouco fornece evidência de desempenho em SAR.

| Componente | Possível reutilização | Adaptação necessária | Evidência atual |
|---|---|---|---|
| Pseudo-máscaras sem classes | Pré-anotar regiões em imagens não rotuladas | Selecionar pistas por especialista/validação | Hipótese; não testada em SAR |
| Propagação p-Laplaciana | Regularização espacial de afinidades de features | Afinidade de encoder apropriado ao radar; estabilidade de contornos | Algoritmo demonstrado com proxy RGB |
| Multigranularidade | Propor pista inteira e regiões ao redor | Mapear candidatos à classe pista; preservar largura | Mecanismo presente no artigo |
| Predição global + crops | Recuperar regiões estreitas usando detalhe local | Janelas, padding e remapeamento geográfico consistentes | Ganhos publicados em imagens naturais |
| Treino com pseudo-rótulos | Usar SAR não rotulado para inicializar SAM/segmentador | Filtrar erros e separar perda supervisionada/pseudo-supervisionada | Evidência indireta do artigo |
| Ajuste só do decoder | Comparar eficiência de rótulos | Não assumir que congelar encoder RGB serve para SAR | Precisa de ablação no domínio |

Não é necessário substituir o pipeline atual por Mask2Former para testar a ideia. O primeiro candidato prático é verificar inferência por crops e enriquecimento de rótulos com o SAM já utilizado. Isso aproveita código e checkpoints existentes e evita introduzir simultaneamente novas arquitetura e supervisão.

## Protocolo futuro proposto — nenhuma execução ou promessa de ganho

1. Fixar o split existente e acrescentar controle de separação por local/geografia quando possível. Nada de ajustar filtros, thresholds ou pseudo-rótulos no teste. Preservar o conjunto comum necessário às comparações históricas.
2. Reavaliar um checkpoint SAM com pipeline de referência congelado; registrar inicialização, resolução, normalização SAR, sementes e orçamento. Testar crops como ablação de inferência, com o mesmo checkpoint e remapeamento de probabilidades/máscaras ao referencial original.
3. Em validação apenas, avaliar se afinidades de features SAM já existentes preservam limites das pistas. Só depois avaliar extração SD2 com checkpoint verificável e hardware apropriado. Comparar à mesma propagação linear p = 2; medir custo real.
4. Auditar um conjunto pequeno de pseudo-máscaras sem olhar o teste. Medir aceitação/rejeição, pista ausente, fusão com estradas e falhas de largura. Seleção por alongamento ou contraste sozinha não distingue semanticamente pista.
5. Se a qualidade justificar, treinar uma configuração pseudo-supervisionada e uma supervisionada, com mesma arquitetura, inicialização e número de updates. Ablação separada de filtragem, crops, p e proporção de rótulos; preservar humanos como referência. Comparar budgets de anotação de forma explícita.
6. Teste final com macro IoU/Dice, recall por instância e taxa de falsos positivos, incluindo imagens sem pista quando disponíveis. Acrescentar análise de largura/contorno e medidas reais de latência/VRAM. Repetições pareadas e intervalos por amostragem de locais, sem tratar pixels correlacionados como observações independentes.

O tamanho e disponibilidade de dados não rotulados precisam ser verificados na pesquisa; o seminário não fez um inventário desse acervo. Uma grade de orçamento pode usar proporções do treino disponível em vez de copiar os 100k–400k exemplos do paper. Pseudo-rótulos usam imagens reais, enquanto geração de SAR sintético é outra hipótese e exige checagens próprias de fidelidade e pareamento imagem/máscara.

**Conclusão técnica:** crops e supervisão com pseudo-rótulos são conceitos reutilizáveis; transferência direta da autoatenção SD2 para pistas SAR é a parte mais incerta e deve ser tratada como experimento exploratório. A comparação quantitativa deste seminário não valida essa transferência.
