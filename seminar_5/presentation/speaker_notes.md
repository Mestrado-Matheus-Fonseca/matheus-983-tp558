# Roteiro do Seminário 5

Duração sugerida: 30–35 minutos. Vídeos incorporados por clique.
Texto detalhado nas notas do PPTX e nos documentos científicos.

## 01. Latent Consistency Models

Apresentar o objetivo: entender o mecanismo, executar uma demonstração verificável e avaliar a relação com a pesquisa. A reprodução é conceitual em MNIST. A pesquisa SAR acrescentará DDPM para aumentar o treino e depois comparará consistência/LCM. Nenhum resultado SAR foi produzido aqui.

Fonte: Luo et al. · arXiv:2310.04378v1 · 6 de outubro de 2023

## 02. Três perguntas para conduzir a apresentação

Roteiro sugerido para 30–35 minutos, adaptável. Aproximadamente 12 minutos para método, 10 para evidências e demonstração e 8 para pesquisa e discussão. Os detalhes matemáticos e a auditoria estão nos notebooks e notas.



## 03. A geração por difusão exige repetir a rede

Conectar ao seminário de DDPM. Difusão latente reduz resolução interna, enquanto o LCM reduz avaliações necessárias. Dois ramos podem ser agrupados num batch, mas isso não significa custo equivalente a um ramo. Não converter número de passos diretamente em aceleração real.

Fonte: Paper, §§1–3

## 04. Solvers e destilação atacam custos diferentes

Explicar que o LCM não inventa difusão latente, guidance ou consistência separadamente. Sua contribuição integra esses elementos para texto e alta resolução. A comparação com Guided-Distill no paper usa uma reimplementação e recursos alterados.

Fonte: Paper, §§2–3 e Apêndice G

## 05. LCM aprende o destino da trajetória guiada

O ponto final é o da PF-ODE guiada. Não é possível recuperar uma imagem original única a partir de qualquer ruído. LCF é outra etapa proposta pelos autores, não o treinamento executado nesta demonstração.

Fonte: Paper, §4

## 06. Professor e alvo ajudam no treino

Reproduzir o vídeo por clique. Separar professor congelado, aluno treinável e alvo EMA. Na inferência, professor e solver de treino são removidos. A estimativa limpa passa pelo decoder.

Fonte: Diagrama próprio · Algoritmos 1 e 3

## 07. O aluno recebe tempo, condição e guidance

O paper inicializa o aluno a partir do professor e acrescenta projeção do embedding de guidance. O espaço latente do paper vem do VAE e o backbone é U-Net. O esquema funcional é comum, mas a demonstração usa PCA e MLP. Não apresentar o MLP como arquitetura do artigo.

Fonte: Paper, §4.2 e Apêndice F

## 08. Ruído conhecido permite treinar o professor

O professor aprende a prever o ruído sorteado por MSE. A previsão ideal é uma expectativa condicional. Relacionar score e epsilon: epsilon aproxima menos sigma vezes o score. Na demonstração, a agenda VP tem alpha²+sigma²=1.

Fonte: Paper, Eq. 22 · exemplo numérico didático

## 09. Guidance altera a trajetória que será destilada

Com epsilon condicional 0,2, vazio 0,5 e omega 2, a previsão guiada é −0,4. Uma API pode usar outra convenção. Não copiar valores sem verificar a fórmula. Guidance maior não garante melhor FID ou maior diversidade.

Fonte: Paper, Eqs. 12–15

## 10. A consistência une dois tempos

Os dois pontos devem estar na mesma trajetória e compartilhar condição e guidance. O alvo no tempo menor é calculado sem gradiente. A condição de contorno f(z,0)=z ancora a função. A linha do vídeo é uma ilustração, não uma integração numérica da PF-ODE.

Fonte: Esquema conceitual · Eqs. 4, 9 e 16

## 11. O aluno deve concordar com o alvo EMA

Explicar c_skip e c_out e a recuperação z0=(zt−sigma epsilon)/alpha. O alvo EMA acompanha o aluno sem receber gradiente. Exemplo de EMA: 0,9×2+0,1×4=2,2. A implementação verifica a condição de contorno e o passo DDIM contra identidades numéricas.

Fonte: Paper, Eq. 9 e Algoritmo 1

## 12. Pares próximos demais podem ensinar pouco

Não ler uma perda menor entre k diferentes como uma comparação de qualidade. O paper explora k=1,5,10,20 e 50. DPM/DPM++ lidam melhor com saltos grandes em algumas condições. Os valores completos das curvas não são tabulados, portanto não foram digitalizados ou inventados.

Fonte: Paper, §4.3 e Figura 3

## 13. Poucos passos, sem consultar o professor

LCM alterna previsão limpa e reinjeção de ruído num tempo menor. No último passo retorna a previsão limpa. A contagem de 100 versus 4 avaliações não é uma medição de speedup. O decoder e os custos de infraestrutura também existem.

Fonte: Algoritmo 3 · contagem teórica de avaliações

## 14. As tabelas usam 100 mil iterações

Os dois modelos usam learning rate 8e−6, RAdam, EMA 0,999943, DDIM e k20. O paper informa oito A100. Não confundir o experimento visual de 4k iterações/32 horas-GPU com as tabelas principais.

Fonte: Paper, §5.1 e Apêndice F

## 15. O ganho de FID se concentra em poucos passos

O eixo menor é melhor. Em quatro passos e 512 pixels, a diferença é 11,28 pontos de FID. LCM não ganha em toda configuração. Em 768 e oito passos, DPM tem FID 14,08 contra 14,97 do LCM. A comparação não informa desempenho de segmentação.

Fonte: Transcrição das Tabelas 1/2 · não executado localmente

## 16. Alinhamento e distribuição não são a mesma métrica

CLIP maior e FID menor representam objetivos diferentes. Guidance pode melhorar CLIP e prejudicar FID. Nenhuma dessas medidas garante que uma máscara de pista esteja correta. Os números são os reportados no artigo, sem recálculo local da avaliação.

Fonte: Transcrição das Tabelas 1/2 · ViT-g/14

## 17. Exemplos selecionados pelos autores

A figura original mostra exemplos selecionados pelos autores. O LCM é comparado com Guided-Distill e DPM-Solver++ em dois e quatro passos. Não atribuir essas imagens à demonstração local. A página original completa permanece nos assets para consulta.

Fonte: Luo et al., v1, Figura 2, p. 8 · recorte da página original

## 18. LCF ajusta um LCM sem um professor novo

A operação de ruído compartilhado aparece no notebook matemático. O mecanismo difere de usar um professor para estimar outro ponto da PF-ODE. O paper não demonstra adaptação LCF para SAR ou condicionamento espacial por máscara.

Fonte: Paper, §5.3 e Apêndice C

## 19. Uma demonstração conceitual executada em CPU

A classificação é demonstração conceitual, não reprodução completa nem quantitativa parcial das tabelas. Os componentes matemáticos foram implementados e testados. As escolhas locais estão registradas, em vez de atribuídas aos autores.

Fonte: experiments/configs/demo.json · protocol.json

## 20. A compressão também impõe um limite

PCA é um autoencoder linear. A dimensão 32 foi fixada antes da avaliação. Mostramos reais e reconstruções para revelar perda de informação. Em SAR, uma região fina pode ser importante apesar de ocupar poucos pixels. MSE global de reconstrução não é suficiente para validar pistas.

Fonte: MNIST local · exemplos de teste · imagens 14×14

## 21. Professor e dois alunos foram treinados

O professor aprende previsão de ruído com perda DDPM. O baseline de inferência é DDIM. Os alunos começam dos mesmos pesos do professor e usam o mesmo orçamento. A perda de k1 é menor, mas isso não assegura melhor geração. Três seeds posteriores são de amostragem.

Fonte: Perdas efetivamente registradas · escalas logarítmicas separadas

## 22. Saídas salvas, sem seleção manual

Comparar a aparência de um passo com quatro passos. Todas as colunas são escolhidas por regra fixa. O real da primeira linha é somente referência da classe. Não há imagem verdadeira única associada ao ruído inicial.

Fonte: Demonstração MNIST · a linha real não é ground truth pareado

## 23. Quatro previsões limpas durante a geração

O vídeo mostra estimativas limpas intermediárias, não os latentes ruidosos. DDIM e LCM usam o mesmo ruído inicial para esse batch, mas LCM reinjeta novos ruídos entre passos. O contraste é didático e não prova qualidade SAR.

Fonte: Históricos salvos dos modelos executados · seed 101

## 24. Aderência de classe e custo de amostragem

Os números exatos ficam no relatório gerado do CSV. O classificador acerta 92% do teste real. Sintéticos podem ser mais fáceis ou menos diversos e ter aderência maior. A latência mede somente o sampler latente, não o pipeline inteiro. Não inferir significância de pequenas diferenças.

Fonte: metrics.csv · batch 300 · CPU · não é FID/CLIP

## 25. O que cada resultado permite afirmar

A comparação correta precisa alinhar dataset, modelo e avaliação. O valor FID 11,10 é do artigo. Não há FID local equivalente. Os resultados MNIST verificam o funcionamento da implementação. IoU/Dice SAR depois de aumento são objetivos futuros, não resultados desta apresentação.

Fonte: Tabelas 1/2, protocolo local e proposta de pesquisa

## 26. Paper, defaults de código e checkpoint diferem

A auditoria separa evidência do preprint e implementação posterior. O script Diffusers consultado usa defaults diferentes, incluindo EMA 0,95 e AdamW. Não foram baixados pesos remotos, portanto não há auditoria de seu conteúdo numérico.

Fonte: Paper v1 · README oficial fixado · Diffusers v0.35.1

## 27. Nova direção: DDPM para aumentar o treino SAR

Distinguir o estado documentado do novo plano. A entrega descreve otimização e ablações, com limites de comparação entre modelos. O usuário acrescentou DDPM e comparação futura com LCM. Não afirmar que a geração SAR já foi realizada.

Fonte: project/second_part_project.pdf + orientação do usuário nesta sessão

## 28. A máscara precisa continuar sendo verdadeira

Recomendar gerar a imagem condicionada em máscara conhecida, validando alinhamento. Só gerar imagens não fornece supervisão de segmentação. Separar a geração do treinamento do SAM/YOLO e usar teste real intocado. Condicionar por máscara é adaptação proposta, não resultado do artigo LCM.

Fonte: Proposta de experimento · nenhuma execução SAR

## 29. DDPM em pixels não vira LCM automaticamente

O termo latent é parte da definição. Se a pesquisa começar com DDPM em pixels, chamar seu aluno de LCM seria impreciso. Pode-se comparar sistemas completos em outro estudo, mas não atribuir o resultado apenas ao sampler.

Fonte: Análise de aplicabilidade · notes/research_application.md

## 30. Primeiro utilidade, depois aceleração

Propor pelo menos três sementes pareadas de treino e avaliação por pista/local. Primeiro usar mesmo número de pares aceitos. Depois fazer um experimento separado com mesmo tempo de geração. Incluir DDIM como controle de redução de passos sem destilação.

Fonte: Hipóteses e protocolo propostos, não resultados

## 31. Velocidade só compensa se preservar utilidade

Os limiares são propostas de protocolo, não valores do paper. Contabilizar custo da destilação e número de imagens necessário para amortizar. Aumento não cria informação automaticamente. Critério downstream e auditoria das máscaras são decisivos.

Fonte: Limiares propostos para pré-registro · não são ganhos esperados medidos

## 32. Aprender o destino pode reduzir a geração

Fechar com a evidência central e o próximo passo concreto. LCM é pertinente à nova direção da pesquisa, mas o benefício deve ser demonstrado com segmentação em SAR real. Disponibilizar README, notebooks, vídeos, código e relatórios.



## 33. Fontes e trilha de reprodução

Referências conceituais adicionais são listadas no resumo técnico, com a origem de cada afirmação. O README contém os comandos para executar todas as etapas. Esta apresentação contém vídeos MP4 incorporados, reproduzidos por clique.


