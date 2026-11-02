# Plano e contrato experimental

Pedido: material completo do Seminário 5, em português, baseado exclusivamente na
versão arXiv:2310.04378v1 para os resultados publicados. A estrutura solicitada
define o escopo. Todos os novos arquivos ficam em `seminar_5/`. Seminários anteriores
e entregas da pesquisa são somente fontes de leitura.

## Decisões

1. Preservar `articles/`, nome já utilizado no repositório.
2. Demonstração conceitual: MNIST local, redução 28→14 pixels, codificador/decodificador
   PCA de 32 componentes ajustado somente no treino, professor DDPM condicional MLP,
   destilação LCD com DDIM, alvo EMA e condicionamento de guidance, amostragem LCM.
   PCA é um autoencoder linear, não o VAE do artigo. Classes substituem texto.
3. Comparar DDIM e LCM no mesmo latente e com o mesmo professor. Ablacionar k=1/20
   com inicialização e orçamento iguais. Três sementes de amostragem não são três
   treinamentos independentes. Nenhuma seleção de hiperparâmetros pelo teste.
4. Medir aderência à classe por classificador local, MSE para a saída do professor
   DDIM-50 com o mesmo ruído inicial e latência de amostragem. Não denominar essas
   medidas FID/CLIP. Salvar checkpoints, índices, configurações e ambiente.
5. Sete notebooks: artigo, dados, método, matemática, treino, inferência, resultados.
   Treino longo fica em script e em célula explicitamente opt-in. Os notebooks
   carregam resultados executados e fazem verificações/cálculos próprios.
6. Manim: pipeline, arquitetura, consistência/EMA, comparação de chamadas, resultados.
   MP4 e GIF. Usar o fundo e as cores dos seminários 3/4. Gráficos científicos
   em fundo branco, tipografia serifada e cores discretas.
7. Apresentação PPTX editável e PDF, com identidade dos arquivos anteriores:
   16:9, fundo branco, Verdana conforme o Seminário 4, títulos escuros e rodapé Inatel/TP558.
   Vídeos/GIFs disponíveis e notas de fala detalhadas.
8. Pesquisa: a entrega `project/second_part_project.pdf` é a fonte histórica mais
   recente. A orientação do usuário nesta sessão acrescenta DDPM para aumento de
   treino e comparação futura com LCM. Esta orientação é proposta, não experimento
   já executado. Discutir condicionamento por máscara, SAR, vazamento e controle de
   orçamento antes de qualquer alegação de ganho.

## Verificação prevista

- Identidade DDIM com ruído conhecido, boundary f(z,0)=z, guidance linear,
  EMA sem gradiente, splits disjuntos e reprodutibilidade de inferência.
- Treinar efetivamente professor e alunos, salvar perdas e checkpoints.
- Executar os sete notebooks sem erros, validar JSON/CSV, finitude e metadados.
- Renderizar MP4/GIF e slides, inspecionar imagens e conferir duração/contagem.
- Conferir transcrição das duas tabelas contra o PDF e distinguir as três fontes:
  artigo, demonstração executada e hipóteses para a pesquisa.

## Sequência

- [x] Ler artigo, apêndices, seminários e entregas da pesquisa.
- [x] Verificar viabilidade local e consultar fontes oficiais.
- [x] Implementar e validar os componentes matemáticos.
- [x] Executar treino, inferência e ablação.
- [x] Redigir análises e produzir notebooks.
- [x] Gerar gráficos, animações e slides.
- [x] Revisar documentação, apresentação e artefatos.
