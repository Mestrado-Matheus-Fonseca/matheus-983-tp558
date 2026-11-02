# Fontes, proveniência e padrão visual

Data da sessão: **20/10/2026**, fuso informado pelo usuário America/Sao_Paulo.
A data de consulta não altera a data de publicação do artigo.

| Fonte | Uso | Localização e limite |
|---|---|---|
| Luo et al., v1, 06/10/2023 | Fonte primária, texto completo e apêndices | `articles/2310.04378v1.pdf`, SHA no protocolo |
| [Página do projeto](https://latent-consistency-models.github.io/) | URL indicada no próprio paper | Link de referência, não tratada como fonte numérica adicional |
| [Repositório oficial fixado](https://github.com/luosiallen/latent-consistency-model/tree/a9ad79587cc8bd1e404ccd1a3056a3da969b2f62) | Código/checkpoints anunciados e cronologia | README consultado e preservado em `articles/sources/` |
| [Treino Diffusers v0.35.1](https://github.com/huggingface/diffusers/blob/v0.35.1/examples/consistency_distillation/train_lcm_distill_sd_wds.py) | Auditoria de implementação | Snapshot de texto, sem execução do código remoto |
| [Scheduler Diffusers v0.35.1](https://github.com/huggingface/diffusers/blob/v0.35.1/src/diffusers/schedulers/scheduling_lcm.py) | Boundary e amostragem | Snapshot de texto, não representa necessariamente o código do preprint |
| `project/initial_project.pdf` | Objetivo inicial do mestrado | Fonte local, somente leitura |
| `project/second_part_project.pdf` | Estado mais recente documentado da pesquisa | Fonte principal para SAM/YOLO, protocolos e limites históricos |
| Mensagem do usuário nesta sessão | Inclusão de DDPM para aumento e comparação futura com LCM | Orientação de pesquisa, não evidência de execução |
| MNIST local do Seminário 1 | Dados da demonstração | IDX originais somente lidos, hashes e índices preservados |

As fontes remotas foram recuperadas por HTTPS após liberação de acesso de rede.
A ferramenta de busca não retornou conteúdo utilizável. A consulta foi feita
diretamente às fontes oficiais, sem apoiar conclusões em snippets de busca.
Os snapshots têm URL e SHA-256 em `articles/sources/manifest.json`.
Não foram obtidos modelos remotos nem executados programas contidos nas fontes.

## Localização de afirmações no paper

- Problema e contribuições: pp. 1–3.
- Difusão, score e consistência: pp. 3–4, Apêndice A pp. 12–13.
- LCD, guidance e skipping: pp. 4–7.
- Tabelas 1/2 e Algoritmo 1: p. 7.
- Protocolos e ablações: pp. 7–9, Apêndices F/G p. 16.
- LCF: pp. 7/9–10 e Apêndice C p. 14.
- Amostragem: Apêndice B pp. 13–14.
- Parametrizações e solvers: Apêndices D/E pp. 14–15.
- Showcase de 4k iterações/32 horas-GPU: Fig. 1 p. 2, Apêndice H pp. 17–18.

## Figuras e tabelas

`paper_results.json` é transcrição manual das Tabelas 1/2, conferida no PDF
renderizado. `paper_fid` e `paper_clip` redesenham esses dados com atribuição.
`paper_page2.png` e `paper_page8.png` são páginas do PDF original renderizadas
com Poppler, sem alegação de autoria local. Todos os demais gráficos decorrem
dos CSVs/NPZs locais. Diagramas e cenas Manim são explicações próprias.
`paper_figure2.png` e `paper_figure3.png` são recortes renderizados diretamente
da página 8 para manter as figuras legíveis nos slides.
Os números não tabulados das ablações do paper não foram inventados.

## Referências visuais inspecionadas

Foram analisados o template e as apresentações dos Seminários 1, 3 e 4,
os notebooks dos ciclos anteriores como organização de arquivos e os scripts
Manim dos Seminários 3/4 como padrão visual.

- Slides: 12192000×6858000 EMU, proporção 16:9.
- Seminário 3: Arial e cores azul `#004F8B`/`#0066B3`, fundo branco.
- Seminário 4: títulos/textos em Verdana, capa em Arial, texto `#172033`,
  azul `#004F8B`, preenchimento `#EFF6FF`.
- Novo deck: fundo branco, Verdana, azul institucional, cabeçalho de seção,
  rodapé Inatel/TP558, conteúdo curto e explicação nas notas.
- Manim: fundo `#101827`, azul `#60a5fa`, verde `#34d399`,
  amarelo `#fbbf24`, cinza `#b9c7db`, fonte DejaVu Sans.
- Gráficos: fundo branco, fonte DejaVu Serif, paleta discreta, eixos limpos,
  exportação PNG e PDF vetorial com o estilo Scientific Plot Maker.

Nenhum material anterior foi alterado. A pasta existente se chama `articles/`,
e não `article/`, portanto foi preservada a convenção do repositório.
