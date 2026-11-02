# Matemática, interpretação e implementação

Convenção: \(z_0\) é o latente limpo, \(z_t\) o ruidoso, \(c\) a condição e
\(\omega\) a intensidade CFG na convenção do artigo. \(\phi\) denota o professor
congelado, \(\theta\) o aluno e \(\theta^-\) o alvo EMA. O artigo às vezes reutiliza
\(\theta\) para redes diferentes. Aqui separam-se os papéis para evitar ambiguidade.
Os exemplos abaixo são escalares e didáticos, não métricas publicadas.

## 1. Difusão direta e PF-ODE (Eqs. 1–3, 8, 19–24)

\[
z_t=\alpha_tz_0+\sigma_t\epsilon,\qquad\epsilon\sim\mathcal N(0,I).
\]

\(\alpha_t\) reduz o sinal e \(\sigma_t\) aumenta o ruído. Na agenda VP adotada
localmente, \(\alpha_t^2+\sigma_t^2=1\). Com \(z_0=2,\alpha=0.8,\sigma=0.6,
\epsilon=-1\), obtém-se \(z_t=1\). Essa transformação está no loop de treino.

A formulação contínua é
\[
dz_t=f(t)z_tdt+g(t)dW_t,\quad
f(t)=\frac{d\log\alpha_t}{dt},\quad
g^2(t)=\frac{d\sigma_t^2}{dt}-2f(t)\sigma_t^2.
\]
Sua PF-ODE tem as mesmas distribuições marginais da SDE sob as hipóteses do
modelo, mas não as mesmas trajetórias estocásticas:
\[
\frac{dz_t}{dt}=f(t)z_t-\frac{g^2(t)}2\nabla_z\log q_t(z_t)
\simeq f(t)z_t+\frac{g^2(t)}{2\sigma_t}\epsilon_\phi(z_t,c,t).
\]
O score é \(\nabla_z\log q_t\), e a previsão de ruído aproxima
\(-\sigma_t\nabla_z\log q_t\). A redação do PDF é inconsistente ao chamar
ocasionalmente o score negativo de score. A relação com a previsão de ruído
resolve o sinal. O fator \(1/2\) diferencia o drift da PF-ODE do drift da SDE
reversa. A implementação usa a atualização DDIM discreta e não integra
diretamente uma SDE contínua.

Exemplo VP com \(\beta=1\): \(f=-1/2,\ g^2=1\). Se \(z=1,\sigma=0.6,
\epsilon_\phi=0.3\), o drift é \(-0.5+0.3/1.2=-0.25\). No sentido de
geração, \(dt<0\). Não se deve interpretar o sinal sem considerar essa direção.

## 2. Perda do professor DDPM (Eq. 22)

\[
\mathcal L_\epsilon=\mathbb E_{z_0,t,\epsilon,c}
\|\epsilon_\phi(\alpha_tz_0+\sigma_t\epsilon,c,t)-\epsilon\|_2^2.
\]
O ruído sorteado é o alvo conhecido. O professor aprende uma expectativa
condicional, não memoriza o ruído de cada par individual. A demonstração usa
MSE média, Adam e dropout de condição de 10%, escolhas locais.
Para alvo \(-1\) e previsão \(-0.8\), o erro quadrático é \(0.04\).
Código: `train`, bloco `teacher`, `F.mse_loss`.

## 3. Guidance e PF-ODE aumentada (Eqs. 12–15)

\[
\widetilde\epsilon_\phi=(1+\omega)\epsilon_\phi(z_t,c,t)
-\omega\epsilon_\phi(z_t,\varnothing,t).
\]

A condição vazia estabelece a direção sem prompt. A diferença entre os ramos
acentua a direção associada à condição. Com \(\epsilon_c=0.2,\epsilon_\varnothing
=0.5,\omega=2\), o resultado é \(-0.4\).
Na convenção \(\epsilon_\varnothing+s(\epsilon_c-\epsilon_\varnothing)\),
o mesmo valor exige \(s=1+\omega=3\). Não copiar o valor de uma API sem verificar
a convenção. No LCM, \(\omega\) entra como embedding na rede, de modo que a
inferência não consulta os dois ramos do professor.
Código: `NoiseMLP.forward` e a expressão `epsilon` no loop LCD.

Substituir \(\epsilon_\phi\) por \(\widetilde\epsilon_\phi\) na PF-ODE define a
trajetória guiada. O termo “aumentada” refere-se à incorporação de guidance,
não a um aumento da resolução ou do dataset.

## 4. Autoconsistência (Eq. 4)

\[
f_\theta(z_t,\omega,c,t)=f_\theta(z_s,\omega,c,s)
\]

Essa igualdade vale idealmente para pontos da **mesma trajetória** e as mesmas
condições. Dois pontos quaisquer com níveis de ruído diferentes não constituem
um par de treinamento válido por esse motivo.
Intuição: vários pontos de uma estrada apontam para o mesmo destino.
Exemplo de uma trajetória com destino 2: estimativas 1.9 e 2.1 têm desacordo
quadrático 0.04. A perda reduz o desacordo, mas precisa da condição de contorno
para ancorar a solução e evitar a função constante degenerada.

## 5. Condição de contorno e parametrização (Eqs. 5, 9, 27–30)

\[
\widehat z_0=\frac{z_t-\sigma_t\widehat\epsilon_\theta}{\alpha_t},\quad
f_\theta=c_{\rm skip}(t)z_t+c_{\rm out}(t)\widehat z_0 .
\]
\(c_{\rm skip}(0)=1\) e \(c_{\rm out}(0)=0\) impõem \(f_\theta(z,0)=z\).
Isso é uma identidade estrutural, não um resultado aprendido.
Com \(z_t=1,\alpha=0.8,\sigma=0.6,\widehat\epsilon=-1\), \(\widehat z_0=2\).
Se \(c_{\rm skip}=0.2,c_{\rm out}=0.8\), a saída é 1.8. Esses coeficientes são
apenas um exemplo aritmético, não a regra usada no código.

Na demonstração:
\[
u=10\,t/1000,\quad
c_{\rm skip}=\frac{0.5^2}{u^2+0.5^2},\quad
c_{\rm out}=\frac{u}{\sqrt{u^2+0.5^2}}.
\]
O tempo normalizado é uma escolha local, explicitamente diferente do tempo
discreto multiplicado por 10 do scheduler Diffusers consultado.
Código: `consistency`. O teste verifica erro exatamente zero em t=0.

Outras parametrizações do artigo:
\[
\widehat z_0=x_\theta(z_t,c,t),\qquad
\widehat z_0=\alpha_tz_t-\sigma_tv_\theta(z_t,c,t).
\]
A identidade de v assume a convenção VP, com
\(v=\alpha_t\epsilon-\sigma_tz_0\). No exemplo anterior, \(v=-2\), então
\(0.8(1)-0.6(-2)=2\). A demonstração usa somente predição ε, não v.

## 6. Passo DDIM e o significado de Ψ (Eqs. 11, 15, 17, 18, 25, 31)

\[
S_{t\to s}(z_t)=\alpha_s
\frac{z_t-\sigma_t\epsilon_\phi}{\alpha_t}
+\sigma_s\epsilon_\phi,\qquad
\Psi(z_t,t,s,c)=S_{t\to s}(z_t)-z_t.
\]

No artigo, Ψ representa o **incremento**, não o estado final. Adicionar \(z_t\)
duas vezes é um erro de implementação.
Com o exemplo anterior e \(\alpha_s=1,\sigma_s=0\), o estado final é 2 e o
incremento é 1.

No LCD guiado:
\[
\widehat z_s=z_t+(1+\omega)\Psi(z_t,t,s,c)-\omega\Psi(z_t,t,s,\varnothing).
\]
A linearidade do passo DDIM em ε permite calcular uma única atualização com
\(\widetilde\epsilon_\phi\). Não implica que fluxos exatos de ODEs não lineares
possam ser combinados dessa maneira sem aproximação.
Código: `ddim_step`, teste `guidance_max_error`.

## 7. Destilação, saltos e EMA (Eqs. 6, 7, 10, 14, 16, 26, Algoritmo 1)

\[
\mathcal L_{\rm LCD}=
\mathbb E\,d\left[f_\theta(z_{t_{n+k}},\omega,c,t_{n+k}),
\operatorname{stopgrad}
f_{\theta^-}(\widehat z_{t_n},\omega,c,t_n)\right].
\]

O primeiro termo recebe gradiente. O segundo é um alvo móvel calculado pela EMA:
\[
\theta^-\leftarrow\mu\theta^-+(1-\mu)\theta.
\]
Se o peso alvo era 2, o novo peso aluno é 4 e \(\mu=0.9\), o novo alvo é 2.2.
EMA é feita nos pesos, não nas imagens. Código: `ema_update`.

O salto k aumenta a distância temporal do par. Com uma agenda de 1000 índices,
k=20 corresponde a cerca de 50 intervalos na agenda reduzida. A demonstração
sorteia tempos finais inteiros e usa distância fixa k, incluindo o endpoint zero.
Esse detalhe é diferente de restringir todos os pares à mesma grade de 50 pontos.
Um k pequeno pode produzir uma perda minúscula porque os alvos são parecidos,
sem aprender a transportar uma amostra distante até a origem.

## 8. Inferência de poucos passos (Algoritmo 3)

\[
z\leftarrow f_\theta(z_T,\omega,c,T),\qquad z_T\sim\mathcal N(0,I),
\]
\[
\widetilde z_{\tau_i}=\alpha_{\tau_i}z+\sigma_{\tau_i}\epsilon_i,\quad
z\leftarrow f_\theta(\widetilde z_{\tau_i},\omega,c,\tau_i).
\]

\(\tau_i\) decresce. O último resultado é a previsão limpa, sem injeção final de
ruído. Se \(z=2,\alpha=0.8,\sigma=0.6,\epsilon_i=-1\), a nova entrada é 1.
O aluno produz outra estimativa limpa a partir desse nível.
Código: `sample`, ramo `lcm`. Não é uma sequência de Euler ou DDIM na inferência.
Não usar a mesma interpretação de trajetórias para a MSE contra DDIM-50 depois
da reinjeção de ruído. Essa MSE é apenas desvio entre amostras acopladas pelo ruído
inicial, não uma medida universal de qualidade.

## 9. LCF sem professor (Algoritmo 4)

\[
z_t=\alpha_tz_0+\sigma_t\epsilon,\quad
z_s=\alpha_sz_0+\sigma_s\epsilon.
\]
Usa-se o **mesmo** ε para os dois estados. Com \(z_0=2,\epsilon=-1\),
os pares \((\alpha_t,\sigma_t)=(0.6,0.8)\) e
\((\alpha_s,\sigma_s)=(0.8,0.6)\) produzem 0.4 e 1.
Aplica-se uma perda de consistência entre aluno e alvo nos dois tempos.
O par de ruído compartilhado não é, em geral, uma trajetória exata da PF-ODE
guiada para salto finito. O artigo o propõe como mecanismo de adaptação.
O notebook matemático demonstra o pareamento. **Não foi treinado LCF localmente.**

## 10. Solvers alternativos e baseline (Eqs. 32–36)

O Apêndice E escreve DPM-Solver e DPM-Solver++ de segunda ordem em termos de
\(\lambda_t=\log(\alpha_t/\sigma_t)\), intervalo
\(h=\lambda_s-\lambda_t\) e um ponto intermediário. DPM corrige a estimativa a
partir da diferença entre predições de ruído no ponto intermediário e inicial.
DPM++ faz uma correção análoga em predições de dados. Por exemplo, para
\(\alpha=0.8,\sigma=0.6\), \(\lambda=\log(4/3)\approx0.28768\).
Esses solvers necessitam predição intermediária e não são implementados nesta
demonstração, que fixa DDIM para isolar LCD. As Eqs. 32–35 devem ser consultadas
no PDF antes de substituir o solver.

A Eq. 36 descreve a primeira etapa do baseline Guided-Distill: uma rede imita
a combinação guiada de predições do professor. A etapa seguinte faz destilação
progressiva. Não confundir essa perda de imitação direta com a perda de
consistência entre tempos do LCM. O baseline não foi reimplementado aqui.

## Mapa de cobertura das equações numeradas

| Equações do PDF | Tema | Tratamento local |
|---|---|---|
| 1–3, 8, 19–24 | SDE, PF-ODE, score e treino ε | Derivação acima, agenda e professor discretos |
| 4–7, 25–26 | Consistência, boundary e CD | Implementado e verificado |
| 9–11, 27–28 | LCD com predição ε | Implementado |
| 12–15 | Guidance e ODE aumentada | Implementado e identidade DDIM verificada |
| 16–18, 31 | Skipping e solver DDIM | Implementado, ablação k=1/20 |
| 29–30 | Predição x e v | Explicadas e exemplo numérico, sem treino |
| 32–35 | DPM/DPM++ de segunda ordem | Contextualizadas, não reproduzidas |
| 36 | Baseline Guided-Distill | Contextualizada, não reproduzida |
| Algoritmos 1/3 | LCD e amostragem LCM | Implementados com adaptações registradas |
| Algoritmos 2/4 | CD original e LCF | Explicados, LCF somente exemplo de pareamento |
