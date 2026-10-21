# DDPM para aumento SAR e comparação futura com LCM

## Evidência atual e nova direção

As entregas `project/initial_project.pdf` e, principalmente,
`project/second_part_project.pdf` descrevem segmentação de pistas de pouso em
SAR Sentinel-1. A entrega mais recente compara YOLOv8s-seg, YOLO11s-seg,
YOLOv8x-seg, YOLO11x-seg e SAM ajustado. Usa otimização Ray Tune para YOLO,
um estudo Optuna de 150 tentativas para SAM e ablações de pré-processamento e
aumento. O manuscrito destaca que diferenças de dados, transformações e
inferência limitam comparações causais entre arquiteturas.

Na entrega, o teste YOLO tem 115 imagens. IoU macro reportada é 22,33% para
YOLOv8x-seg e 24,56% para YOLO11x-seg após otimização. As variantes Small têm
26,61% e 28,24%, respectivamente. O valor histórico SAM é IoU 0,3769, com
protocolo distinto. A avaliação final no teste da nova configuração SAM escolhida
pela validação ainda é descrita como pendente no documento. Esses valores são
**reportados na entrega**, não recalculados nem validados nesta atividade.
Não os usar como um ranking conclusivo de arquiteturas.

**Orientação do usuário nesta sessão:** acrescentar DDPM para aumentar os dados
de treino, seguindo o princípio dos experimentos SAM, e posteriormente usar
LCM para comparação. Esta é uma nova direção planejada da pesquisa, não um
resultado já presente nos documentos. Não foi implementado nem treinado um
gerador SAR nesta atividade.

## Aplicabilidade

LCM é diretamente relevante como alternativa para acelerar um **gerador** de
dados sintéticos já treinado. Não substitui SAM/YOLO e não segmenta pistas por
si só. A cadeia proposta é:

```text
Treino SAR real + máscaras
          ↓
Professor DDPM condicionado em máscara
          ↓
Dados sintéticos pareados ──────────────────┐
          ↓                                │
Destilação por consistência                │ comparação
          ↓                                │ controlada
Gerador de consistência em 1–4 passos       │
          ↓                                │
Dados sintéticos pareados ──────────────────┘
          ↓
Mesmo segmentador SAM ou YOLO, mesmo protocolo
          ↓
Teste SAR real intocado: IoU, Dice, falsos positivos/negativos
```

É necessário primeiro definir como o DDPM fornecerá supervisão utilizável.
O caminho recomendado para o primeiro experimento é gerar a imagem SAR
**condicionada numa máscara conhecida**. A máscara permanece o rótulo somente
se a imagem gerada respeitar sua geometria. É preciso auditar esse alinhamento.
Gerar imagens sem condição e inventar uma máscara correspondente não constitui
um par supervisionado. Usar o próprio segmentador para pseudo-rotular sintéticos
introduz outra intervenção e possível reforço dos seus erros.

### Decisão arquitetural essencial

| Professor planejado | Comparador coerente | O que evitar |
|---|---|---|
| DDPM em pixels SAR | Consistency Distillation em pixels, com a mesma condição espacial | Chamar esse aluno de LCM, pois não há espaço latente |
| Difusão em latente SAR com autoencoder validado | LCD/LCM com mesmo encoder, decoder e condição | Comparar professores/dados distintos e atribuir toda diferença ao sampler |
| DDPM de pixels comparado a um LCM latente novo | Estudo de sistemas completos, com ablação do autoencoder | Tratar o estudo como mudança exclusiva de número de passos |

Portanto, DDPM não se converte automaticamente no LCM do artigo. A expressão
DDPM pode descrever a família de treinamento, mas é preciso registrar se ele
opera em pixels ou latentes. Para uma comparação limpa, destilar do **mesmo
professor adaptado ao domínio** é preferível a usar diretamente Dreamshaper.
O LCF do artigo é uma alternativa de adaptação futura, mas não resolve sozinho
a necessidade de condicionamento espacial por máscara e fidelidade SAR.

## Adaptações necessárias

- Preservar a representação SAR: polarização, escala de intensidade, recorte,
  resolução e normalização precisam ser iguais entre braços experimentais.
  Replicar uma banda em RGB não garante adequação física do modelo.
- Incorporar a máscara espacial ao gerador, por concatenação ou condicionador
  compatível. Essa extensão não é demonstrada pelo artigo LCM para SAR.
- Validar o autoencoder antes da destilação. Medir reconstrução na região da
  pista e preservação de largura/contorno, além de erro global.
- Usar apenas dados de treino no gerador, no autoencoder, na seleção de
  exemplos e na elaboração de máscaras condicionantes.
- Preservar máscaras com interpolação adequada, evitando que transforms de
  imagem e rótulo divirjam. Considerar speckle e estatísticas de fundo.
- Estabelecer política de aceitação dos pares em validação e mantê-la fixa.
  Registrar rejeições, pistas ausentes, alterações de largura e repetição.

## Experimento futuro A: validar aumento com DDPM

**Hipótese:** dados SAR sintéticos condicionados em máscara melhoram a
generalização do segmentador em imagens reais, sob orçamento de treino fixo.
O ganho não é assumido.

**Baseline:** primeiro, a configuração SAM escolhida pela validação, após
resolver a avaliação e congelar o protocolo. Um experimento separado pode usar
YOLO11s-seg ou outro modelo já estabelecido. Não selecionar o segmentador pelo
resultado deste experimento no teste.

**Braços:** real apenas, real com aumento convencional já estudado, e real com
dados DDPM. Definir previamente uma proporção, por exemplo 25% de sintéticos
por batch, como **configuração proposta**, não valor extraído do artigo.
Manter inicialização, número de atualizações, batches, transforms compartilhadas,
tratamento de máscaras e política de early stopping iguais.

**Dados:** auditar sobreposição geográfica e de cenas entre splits. Usar a
validação para selecionar gerador, filtros e proporção. O teste real de 115
imagens pode preservar a comparação histórica, desde que sua independência
seja auditada. Se houver imagens da mesma pista ou cena em splits diferentes,
é necessário definir uma avaliação por local adicional.

**Métricas:** IoU macro e Dice macro no teste real, métricas de contorno/largura,
FP/FN, resultados por pista/local. Registrar também custo para gerar cada par
aceito e taxa de rejeição. Não converter média IoU em média Dice pela fórmula
não linear aplicada à média.

**Critério de sucesso proposto:** ganho de pelo menos 1 ponto percentual de
IoU macro frente ao aumento convencional, sem queda material de Dice/contorno,
com intervalo de confiança do ganho acima de zero quando o tamanho amostral
permitir. O limiar é uma proposta a pré-registrar, não uma previsão.
Usar pelo menos três sementes de treino pareadas e bootstrap por pista/local,
não tratar pixels como observações independentes.

## Experimento futuro B: DDPM versus consistência/LCM

**Hipótese:** a destilação reduz o custo por par sintético aceito e preserva
o benefício downstream do DDPM dentro de uma margem predefinida.

**Método:** congelar o professor DDPM aprovado no experimento A. Caso ele opere
em latentes, destilar um LCM com as mesmas condições espaciais. Caso opere em
pixels, o primeiro comparador será um modelo de consistência em pixels.
Medir 1, 2 e 4 passos do aluno, um DDIM de poucos passos do mesmo professor e o
sampler DDPM de referência. O DDIM controla quanto do ganho vem simplesmente
de reduzir passos sem destilação.

**Dois controles complementares:**

1. Mesmo número de pares sintéticos aceitos, máscaras condicionantes, proporção
   real/sintético e número de updates do segmentador. Isola qualidade/utilidade.
2. Mesmo tempo disponível para geração. Permite comparar o uso de maior volume,
   mas constitui um segundo experimento e deve ser relatado separadamente.

**Métricas:** mesmas métricas downstream do experimento A, latência total
incluindo encoder/decoder e filtragem, throughput, pico de memória, tempo e
custo de destilação, taxa de aceitação e diversidade de pares.
Usar aquecimento e repetições, mesmo hardware e precisão numérica.
Contar chamadas condicionadas e não condicionadas explicitamente.

**Critério de sucesso proposto:** o limite inferior do intervalo pareado
LCM−DDPM em IoU macro deve ficar acima de −1 ponto percentual, com redução de
pelo menos 2× no custo de geração por par aceito. A margem de 1 p.p. e o fator
2× são escolhas de protocolo a discutir, não achados do seminário.
Contabilizar o custo de destilação:
\[
N_{\rm equilíbrio}=
\frac{C_{\rm destilação}}{c_{\rm DDPM}-c_{\rm aluno}},
\]
válido quando o denominador é positivo e os custos são definidos na mesma
unidade por par aceito. Se o volume pretendido for menor, o aluno pode não
compensar economicamente.

## Conclusão crítica

A nova direção torna LCM um candidato pertinente, condicionado ao sucesso do
gerador DDPM e à preservação das máscaras. O primeiro problema científico é a
utilidade de dados SAR sintéticos. O segundo é acelerar a geração preservando
essa utilidade. Resultados em MNIST ou FID/CLIP em imagens naturais não
respondem a nenhuma dessas perguntas em SAR. A demonstração deste seminário
fornece uma implementação verificável de LCD e um protocolo para investigá-las.
