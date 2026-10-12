# Relatório dos experimentos executados

Tipo: **reprodução parcial dos algoritmos com afinidade proxy**. Nenhuma extração SD2, inferência oficial Diffuse2Seg ou treinamento Mask2Former foi executado.

| p | Recall toy médio (%) | Std amostral (p.p.) | Mean best IoU |
|---|---:|---:|---:|
| 2.0 | 56.11 | 9.62 | 0.7348 |
| 1.6 | 25.00 | 0.00 | 0.3568 |
| 1.2 | 45.00 | 0.00 | 0.5269 |

As sementes 7/23/42 modificam somente ruído RGB numa mesma cena/GT; não são três cenas independentes. O melhor p neste proxy é 2. Não houve reprodução do ganho não linear publicado. Não ajustar parâmetros pelo melhor resultado desta demonstração e depois tratá-la como teste independente.

## Configuração e métrica

Entrada 24×24, 576 nós, 16 prompts automáticos. Afinidade simétrica é kernel Gaussiano de diferença RGB e distância espacial, sem GT. Normalização global pelo maior grau preserva simetria. λ=0,03; epsilon de gradiente=1e−8; tolerância de atualização quadrática=1e−8; máximo400 iterações; mínimo3 pixels; NMS0,9. Seis thresholds fixos aproximadamente log-espaçados entre0,186 e2,99. Dados e configurações estão em protocol.json e nos NPZ.

A cena tem cinco regiões disjuntas; a referência também inclui a máscara inteira do retângulo com sua parte, totalizando seis referências multigranulares. O GT entra apenas na avaliação. Recall toy é média de recall nos dez limiares IoU0,50…0,95, com matching greedy 1:1 por IoU, sem scores de objectness, API COCO ou categoria. Cada referência e cada proposta só casa uma vez. Mean best IoU permite a melhor proposta por referência e é uma medida complementar. Nenhuma dessas métricas é o AR1000 oficial do artigo.

## Verificação e convergência

O caso p=2 foi comparado à solução fechada `[2L+λI]f=λf⁰`: erro máximo 1.903e-09. Também são verificados simetria KL, mapas iguais, IoU perfeito, ausência de propostas e não duplicação de matching. As nove execuções atingiram o critério de atualização; isso é diagnóstico numérico, não prova de segmentação correta. O epsilon evita potência negativa de gradiente zero. O solver usa grafo simétrico, enquanto o tratamento de simetria da atenção original precisaria ser confirmado.

## Ablação local

| Níveis | NMS | Candidatos | Propostas | Recall toy (%) |
|---:|---:|---:|---:|---:|
| 3 | 0.5 | 5 | 2 | 16.67 |
| 3 | 0.9 | 5 | 3 | 25.00 |
| 6 | 0.5 | 11 | 2 | 16.67 |
| 6 | 0.9 | 11 | 3 | 25.00 |

p=1,6, semente7. Mais níveis aumentam candidatos sem aumentar recall neste caso. NMS0,9 preserva propostas úteis que NMS0,5 rejeita.

## Diferenças para o artigo

| Elemento | Artigo | Execução local |
|---|---|---|
| Representação | SD2 VAE/U-Net congelados | RGB/posição, sem features aprendidas |
| Resolução | RGB1120²; latent140² | 24², sem VAE |
| Ancoragem/parada | λ1e−5; update²1e−4 | λ0,03; update²1e−8 |
| Saída | Upsampling, componentes, NMS, CascadePSP opcional | Componentes4-conexos e NMS na grade original |
| Avaliação | Benchmarks naturais, AR1000 | Uma cena toy, métrica própria |
| Treinamento | Duas rodadas Mask2Former | Não realizado |

As diferenças de λ, escala de A, epsilon e parada alteram a geometria dos mapas e as distâncias KL. Manter os thresholds do artigo não calibra automaticamente um grafo RGB. No p=1,6, grupos colapsam e entidades desaparecem; o mapa suave não é suficiente para preservar a hierarquia.

## Demonstração em fotografia

Imagem COCO39769 existente no ciclo3, com grafo reduzido a24²: p=1,6 gera 1 proposta final; p=2 gera 16 propostas. A figura apresenta a fotografia original e as duas partições finas calculadas na grade. Não se usam anotações e não se mediu recall COCO. As partições mostram agrupamento por aparência, sem demonstrar descoberta semântica de gatos ou controles remotos.

## Por que o pipeline integral não foi executado

O ambiente disponível registra PyTorch CPU, sem CUDA e sem diffusers/checkpoints SD2. Não foi localizado código/checkpoint oficial verificável do Diffuse2Seg na consulta. O experimento original usa 100k–400k imagens e treinamento em quatro A100; a afinidade densa de19600tokens sozinha tem1,43GiB emFP32. CPU não torna toda inferência SD2 impossível, mas uma configuração arbitrária não validaria o resultado oficial. Priorizou-se a alternativa parcial solicitada, sem download de pesos e sem prometer resultados de benchmarks.

## Conclusão sustentada

A implementação demonstra operações e reproduz o caso linear verificável. Os resultados medidos deixam claro que a afinidade é central para a hipótese Diffuse2Seg. Não validam nem refutam seus números originais, e não constituem evidência SAR.
