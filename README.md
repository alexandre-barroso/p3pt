# Controles terminais na recuperação de rótulos acentuais

Código original do estudo que compara atributos de toda a palavra com atributos restritos às terminações gráficas. O único dado necessário é a versão pública do Portuguese Stress Lexicon, obtida separadamente. O código não depende do antigo inventário de transcrições nem de arquivos privados do projeto que motivou a pesquisa. O CSV não é distribuído aqui. **Observação importante: para revisão anônima de pares, CITATION.cff foi removido, o que afeta os hashes do repositório, sendo necessário (na rodada de revisão) fazer uma pequena alteração nos arquivos para rodar o repositório sem que ele sinta falta falta desse arquivo.**

## Ambiente e verificação

Ambiente numérico: Python 3.12.3, NumPy 2.4.4, SciPy 1.17.1 e Unicode 15.0.0. As figuras usam Matplotlib 3.10.9. A verificação formal exige uma instalação existente do Lean 4.34.0 por Elan, com o `leanchecker` correspondente. Nenhum programa instala ferramentas automaticamente. Use um ambiente local dedicado e um diretório de resultados fora deste repositório.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -B verify.py --psl /caminho/autorizado/psl.csv --work-dir /caminho/resultados
```

O comando termina com código diferente de zero se uma etapa falhar. Confere os hashes do código e do CSV, executa testes sintéticos, compila as cinco declarações formais, audita suas dependências axiomáticas, faz uma nova verificação de kernel e refaz os 27 ajustes. Compara todos os agregados, sem escolher execuções ou parâmetros depois de observar o resultado. O arquivo de referência contém agregados e oito ilustrações lexicais selecionadas por uma regra explícita; não contém a base, os vocabulários ou os parâmetros ajustados.

Os modos abaixo têm alcance menor e são nomeados como tal no resultado:

```sh
python3.12 -B verify.py --synthetic-only --work-dir /caminho/testes
python3.12 -B verify.py --synthetic-only --python-only --work-dir /caminho/testes-python
python3.12 -B verify.py --existing-results /caminho/results.json --work-dir /caminho/conferencia
python3.12 -B draw_figures.py --results /caminho/results.json --output /caminho/figuras --anonymous
```

`--existing-results` verifica os agregados e sua consistência com o código registrado; não refaz os ajustes nem autentica a história de execução de um JSON. `--python-only` omite os resultados empíricos e as provas. O fluxo de CI usa apenas esse último modo e não foi executado em serviço hospedado. Não execute com `-O`.

## Entrada e proveniência

O PSL é documentado por Guilherme D. Garcia em https://gdgarcia.ca/psl; a versão estudada foi obtida pelo arquivo público https://osf.io/xmzbu/. É necessário obter o CSV separadamente e respeitar seus termos de acesso. SHA-256 exigido:

```text
96e083977e093a9e3550d2feee2ec725ac0768d4bff386a295480bafdeabe0e6
```

O programa rejeita outra versão, mesmo que tenha o mesmo nome. Lê a entrada sem a modificar. Os caminhos são fornecidos pelo usuário; não há caminhos pessoais fixados no repositório. `PROTOCOL.md` descreve população, cronologia, decisões e limites. `claim_map.json` relaciona as afirmações aos cálculos; `verification_inputs.json` vincula o conteúdo do repositório por hash, sem constituir assinatura ou comprovação externa de autoria.
