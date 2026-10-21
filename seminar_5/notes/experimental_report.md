# Relatório experimental executado

**Classificação: demonstração conceitual de LCD, não reprodução quantitativa LAION.**

Dados: 6000 treino, 1000 validação e 1000 teste. PCA e classificador ajustados
somente no treino. Professor com 5000 updates e dois alunos com 5000 cada.
Seed de treino 42. Seeds de geração 101, 202 e 303, 300 amostras por seed.

| Método | Passos | Aderência à classe (%) ± DP | Amostragem (ms/imagem) ± DP |
|---|---:|---:|---:|
| ddim | 1 | 76.78 ± 2.22 | 0.00216 ± 0.00004 |
| ddim | 2 | 96.00 ± 0.88 | 0.00358 ± 0.00011 |
| ddim | 4 | 97.56 ± 0.69 | 0.00677 ± 0.00034 |
| ddim | 8 | 97.00 ± 0.00 | 0.01244 ± 0.00031 |
| ddim | 50 | 96.89 ± 0.51 | 0.07783 ± 0.00057 |
| lcm_k1 | 1 | 66.22 ± 3.02 | 0.00169 ± 0.00011 |
| lcm_k1 | 2 | 88.67 ± 2.52 | 0.00266 ± 0.00001 |
| lcm_k1 | 4 | 95.11 ± 0.69 | 0.00495 ± 0.00005 |
| lcm_k1 | 8 | 94.89 ± 0.51 | 0.00938 ± 0.00015 |
| lcm_k20 | 1 | 81.44 ± 3.01 | 0.00151 ± 0.00003 |
| lcm_k20 | 2 | 95.00 ± 0.58 | 0.00262 ± 0.00006 |
| lcm_k20 | 4 | 97.67 ± 0.33 | 0.00504 ± 0.00017 |
| lcm_k20 | 8 | 98.56 ± 1.02 | 0.00951 ± 0.00032 |

DP é desvio padrão amostral entre três sementes de geração, não intervalo
de confiança nem variabilidade de três treinamentos independentes.

Classificador local em reais: validação 92.700%, teste 92.000%.
Aderência de sintéticos pode ser maior por serem mais fáceis ou menos diversos.
Não é acurácia do gerador em imagens reais, FID, CLIP ou medida de utilidade SAR.

A latência exclui PCA decode, classificador e I/O. É uma medida curta em CPU,
após warmup. Não representa latência ponta a ponta ou desempenho de Stable Diffusion.

## Artigo versus demonstração

| Experimento | Artigo | Local | Comparação direta |
|---|---|---|---|
| FID LCM 512, 4 passos | 11,10 | Não calculado | N/A — configuração experimental diferente |
| CLIP LCM 512, 4 passos | 28,69 | Não calculado | N/A — configuração experimental diferente |
| Aderência de classe MNIST | Não reportada | Tabela acima | N/A — configuração experimental diferente |
| IoU/Dice SAR após aumento | Não reportado | Não executado | Hipótese para pesquisa futura |

## Interpretação

O experimento demonstra treino DDPM, orientação CFG, DDIM, destilação com EMA
e inferência LCM. A ablação k=1/20 tem inicialização e orçamento iguais.
Perda menor em k=1 não assegura melhor geração. Não há inferência de
significância estatística nem seleção de hiperparâmetros pelo teste.

MSE latente contra DDIM-50 está no CSV, com limite de interpretação na auditoria.
Os exemplos visuais usam a primeira amostra de cada classe da seed 101, sem
seleção manual. As imagens reais são referências não pareadas, não ground truth
da geração. O baseline local usa DDIM, apesar de o professor ter perda DDPM.

## Tempo de treinamento observado

- classifier_seconds: 0.472 s.
- teacher_seconds: 4.904 s.
- lcm_k1_seconds: 9.386 s.
- lcm_k20_seconds: 9.060 s.

Configuração integral, versões e hashes: `experiments/results/protocol.json`.
Perdas: `training.csv`. Métricas por semente: `metrics.csv`.
Checkpoints: `experiments/checkpoints/`. Nenhum checkpoint SAR foi criado.
