# Auditoria metodológica

## Versão e fontes

O objeto de estudo é o PDF v1 de 06/10/2023, não uma revisão posterior. O
repositório oficial foi consultado na revisão
`a9ad79587cc8bd1e404ccd1a3056a3da969b2f62`. Também foram inspecionados o script
de treino e o scheduler LCM do Diffusers **v0.35.1**. Esse código posterior é
referência de implementação, não prova da configuração que produziu as tabelas
do preprint.

## Diferenças verificadas

| Tema | Paper v1 | Código/artefato público consultado | Consequência |
|---|---|---|---|
| Professor principal | SD2.1-Base/SD2.1 | README promove LCM_Dreamshaper_v7 e versões posteriores | Checkpoints distintos não reproduzem automaticamente Tabelas 1/2 |
| Orçamento | 100k iterações nas tabelas | Divulgação de 4k e 32 horas-GPU | Não atribuir todo resultado ao orçamento menor |
| Nome visual | “Dreamer-V7” no PDF | “Dreamshaper_v7” no README | Preservar grafias e registrar discrepância |
| EMA | 0,999943 | Default 0,95 no script Diffusers consultado | Defaults não são a receita do PDF |
| Otimizador | RAdam no Apêndice F | AdamW/8-bit AdamW no script consultado | Diferença de treinamento |
| Guidance | ω∈[2,14], combinação (1+ω)ε_c−ωε_∅ | Defaults w_min=5, w_max=15 e comentários com convenções a conferir | Inspecionar cálculo e embedding, não só ajuda de CLI |
| Perda | Distância d, incluindo quadrática | l2 ou Huber, default l2 no arquivo consultado | Registrar opção efetivamente usada |
| Boundary | Condições c_skip(0)=1, c_out(0)=0 | σ_data=0,5 e tempo discreto×10 no scheduler | Não assumir mesma escala com tempo normalizado |
| Avaliação | 30k imagens, ViT-g/14 | API de inferência não implementa todo protocolo de avaliação | Usar a pipeline não reproduz FID/CLIP |

O comentário de argumentos de guidance no script Diffusers afirma uma mudança
de convenção, enquanto o bloco de cálculo da predição guiada menciona a convenção
do paper. Isso exige atenção na integração. Neste seminário, a fórmula efetiva
é declarada explicitamente em todos os experimentos locais e não depende dessa API.
Não foi auditado o conteúdo numérico de pesos remotos, pois não foram baixados.

## Desvios deliberados da demonstração

| Componente | Paper | Demonstração executada |
|---|---|---|
| Imagem | RGB 512/768 | MNIST cinza 14×14 |
| Latente | VAE de SD | Autoencoder linear PCA, 32 dimensões, whitening |
| Condição | Texto CLIP | Classe 0–9 e classe nula |
| Backbone | U-Net de SD | MLP, duas camadas ocultas de 128 |
| Professor | Modelo pré-treinado de grande escala | Treinado localmente, 5000 updates DDPM |
| LCD | 100k updates | 5000 updates por k |
| Batch | 72/16 | 128 |
| Learning rate aluno | 8e-6 | 2e-4 |
| EMA | 0,999943 | 0,99 |
| Guidance treino | [2,14] | [0,2], avaliação ω=1 |
| Solver | DDIM, k=20 | DDIM, ablação k=1/20 |
| Tempos | Agenda de 1000, redução descrita no artigo | Índices inteiros com salto fixo, inclui zero exato |
| Coeficientes boundary | Formulação geral | Tempo normalizado×10, σ_data=0,5 |
| Métricas | FID/CLIP | Aderência à classe local, desvio latente, latência |
| Hardware | A100 | CPU, duas threads |

Os hiperparâmetros locais são escolhas didáticas fixadas na configuração, não
valores omitidos que teriam sido “recuperados” do paper.

## Comparabilidade e limites

O classificador de avaliação é treinado apenas no split de treino. Seu desempenho
no teste real é registrado no protocolo. Aderência de sintéticos pode exceder a
acurácia em reais porque os geradores produzem exemplos mais fáceis ou menos
diversos. Isso não prova fidelidade nem cobertura da distribuição.

O mesmo ruído inicial é usado entre métodos. LCM com vários passos acrescenta
novos ruídos, então a MSE contra DDIM-50 não compara trajetórias idênticas.
Ela quantifica desvio de um professor, não distância a um ground truth de geração.
MNIST não tem imagem real única associada a cada ruído inicial.

Há uma semente de treino por modelo e três sementes de geração. A média e o
desvio padrão entre gerações não representam incerteza de treinamento.
A latência mede apenas amostragem latente em batch de 300, sem decoder,
classificador e disco. Tempos curtos são sensíveis a ruído do sistema.
As medições não demonstram aceleração em GPU nem em Stable Diffusion.

Uma perda menor para k=1 não torna esse modelo melhor: pares próximos facilitam
a tarefa de consistência local sem assegurar boa previsão de longa distância.
Não comparar as escalas de perda de k diferentes como ranking de geradores.

## Ambiente e escopo de mudanças

O Python do `.venv` antigo aponta para uma instalação removida do VS Code.
O launcher identifica explicitamente essa condição e usa o Python 3.13 instalado
pelo uv com os pacotes existentes. Não corrige nem modifica o ambiente antigo.
`python-pptx` foi instalado isoladamente em `seminar_5/.deps`.
Um ambiente independente com `requirements.txt` é a opção portátil documentada.

A reprodução integral não foi tentada: o ambiente observado é CPU, sem CUDA,
e não há os checkpoints SD nem a coleta LAION necessários nesta pasta. O acesso
ao código oficial foi verificado. Não há alegação de indisponibilidade universal
de dados/pesos, apenas de ausência no experimento executado.
