# Estudo de robustez com checkpoint congelado

O comando `evaluate_research.py` amplia a avaliação da pesquisa existente em
três frentes: robustez, condições retidas e comparação com métodos simples.
Ele executa fora da interface gráfica, no mesmo ambiente Python, e produz um
relatório HTML local. Não exige dependências novas.

O estudo carrega `weights/classifier.pt` e mantém seus pesos e sua normalização.
Não treina, calibra ou seleciona novamente a CNN. Os resultados originais de
treino/teste continuam nos seus arquivos; cada estudo recebe uma pasta nova.
O comando verifica que o checkpoint e o banco de treino não mudaram durante a
execução. A configuração e o hash SHA-256 de ambos entram no relatório.

## Executar

Na raiz do repositório, com o ambiente Python do projeto ativado:

```powershell
python evaluate_research.py
```

O padrão usa **100 neurônios por classe**, sementes **101, 202 e 303** e as 23
condições abaixo, com os metadados padrão do treinamento original: 69 execuções
e 20.700 observações de neurônios. Cada condição/semente gera uma população;
o relatório agrega as sementes. Em uma rede, as observações compartilham
conexões e não são 300 réplicas experimentais independentes.

São necessários o checkpoint já treinado e o **banco original** que forneceu
seu conjunto de treino. Se estiverem em outros locais:

```powershell
python evaluate_research.py --checkpoint weights/classifier.pt --database data/neural_patterns.sqlite
```

O script usa somente o split `train` do SQLite para ajustar os baselines.
As configurações registradas no banco devem coincidir com as do checkpoint.
O checkpoint legado não contém o hash de seu banco de origem: a correspondência
de configurações não prova que o operador forneceu exatamente o banco original.
Também há uma verificação de traços de avaliação idênticos aos traços carregados
do treino. Essa verificação não certifica todas as formas possíveis de vazamento.

Se o banco original não estiver disponível, esta alternativa deve ser escolhida
explicitamente:

```powershell
python evaluate_research.py --baseline-source synthetic --synthetic-fit-per-class 500
```

Ela gera um novo conjunto de ajuste com corrente constante, semente 17 e fluxo
aleatório separado da avaliação. Filtra exemplos sem atividade suficiente apenas
nesse conjunto de ajuste. O relatório identifica `explicit_synthetic_fit`:
o baseline não viu as mesmas linhas de treino da CNN, portanto essa comparação
é exploratória. O programa nunca substitui um checkpoint ausente por pesos
aleatórios e nunca regenera o banco original automaticamente.

Para um ensaio menor, sem confundir seu resultado com o protocolo completo:

```powershell
python evaluate_research.py --samples-per-class 20 --seeds 101 202 303
```

Para aumentar a população e manter as réplicas:

```powershell
python evaluate_research.py --samples-per-class 1000 --seeds 101 202 303 --out-degree 100 --threads 8
```

Esse exemplo usa 3.000 neurônios por execução e 300.000 sinapses nas condições
conectadas. O tamanho da população altera a experiência; compare estudos com
o mesmo tamanho. O limite do estudo é 10.000 neurônios por classe e a rede
mantém o orçamento de 10 milhões de sinapses. Esse estudo mede classificação,
não o limite de desempenho do computador.

O comando imprime o andamento e o caminho de `report.html`. Abra esse arquivo
no navegador. Os caminhos informados explicitamente são relativos à pasta
atual; os caminhos padrão são relativos à raiz do projeto.

## 1. Varreduras de robustez

| Eixo | Valores do protocolo | O que fica fixo |
|---|---|---|
| Referência | corrente constante, neurônios independentes | metadados do checkpoint |
| Ruído de observação | 0, valor do treino, 1, 2, 5, 10 mV | parâmetros e corrente por neurônio |
| Variação paramétrica | 0, valor do treino, 0,06, 0,10, 0,15 | sorteios latentes, corrente e ruído |
| Força sináptica | 0, 0,1, 0,5, 1 | topologia, parâmetros, corrente e ruído |
| Início de pulso | 10%, 30%, 50% da janela | família de pulsos e sorteios dos neurônios |

Valores repetidos em um eixo são removidos. Assim, a quantidade de condições
pode mudar se o checkpoint usar outros valores de ruído ou variação.
O controle de força zero reproduz os traços da referência bit a bit.

O pulso dura 50% da janela. Comparações entre seus instantes de início isolam
esse deslocamento dentro da família de pulsos. Compará-lo com a referência de
corrente constante também altera a forma e a carga total do estímulo.

A rede usa conexões dirigidas aleatórias distintas, sem autoconexões, 20% de
fontes inibitórias, ganho inibitório 4, atraso de 1 ms e decaimento de 5 ms.
`--out-degree` define as saídas por neurônio, limitado a `N-1`. A força é a
amplitude excitatória; a inibitória recebe sinal negativo e o ganho configurado.
Os papéis excitatório/inibitório são independentes dos rótulos RS/CH/FS.

As condições de um mesmo eixo reutilizam sorteios latentes por semente.
Mudar a variação modifica sua escala, sem sortear uma nova população. O ruído
é acrescentado depois da simulação, sem criar disparos físicos ou sinápticos.
O ruído segue o mesmo recorte de tensão do gerador original; com ruído zero,
esse recorte não é aplicado. Portanto, esse eixo avalia o canal de observação
completo do projeto, não apenas uma perturbação gaussiana sem recorte.

## 2. Condições retidas

O conjunto de ajuste usa exclusivamente entradas constantes dentro do intervalo
registrado. A avaliação inclui estas famílias/ranges que ele não recebe:

| Condição | Definição |
|---|---|
| Pulsos | amplitude sorteada no intervalo do treino, zero fora do pulso |
| Rampa | amplitude multiplicada por uma rampa linear de 0,25 a 1,75 |
| Senoide | amplitude multiplicada por `1 + 0,5 sin(2π · 20 Hz · t)` |
| Corrente inferior | de `mínimo - metade da largura do intervalo` até um float32 estritamente menor que o mínimo |
| Corrente superior | de um float32 estritamente maior que o máximo até `máximo + metade da largura do intervalo` |

Para o intervalo padrão 5–30, as novas faixas são aproximadamente -7,5–5 e
30–42,5, excluindo as fronteiras 5 e 30. A amostragem e a conversão para float32
preservam essa exclusão. Rampa/senoide podem também extrapolar a amplitude
instantânea do domínio original; o CSV sinaliza isso por neurônio.

“Retido” significa excluído do protocolo de treinamento de corrente constante
implementado neste repositório, com a limitação de proveniência do checkpoint
legado. Não constitui evidência de desempenho em sinais biológicos.

A distribuição gaussiana de variação paramétrica tem suporte ilimitado.
Aumentar sua dispersão constitui mudança de distribuição, sem criar uma faixa
paramétrica rigorosamente nunca vista. O relatório faz essa distinção.

## 3. Baselines de atributos temporais

Dois classificadores de centróide mais próximo são ajustados nas linhas de
treino. Cada atributo é padronizado usando **somente** sua média e seu desvio
no treino; desvios praticamente nulos recebem escala 1. Depois, a classe com
menor distância euclidiana ao centróide padronizado é escolhida. Empates usam
a primeira classe na ordem RS, CH, FS. Não há busca de hiperparâmetros no teste.

| Modelo | Atributos observados |
|---|---|
| `rate_centroid` | taxa de picos |
| `timing_centroid` | taxa, ISI mediano, CV dos ISIs, fração de ISIs curtos, razão último/primeiro ISI |

Um pico observado começa na borda de uma região contígua com tensão ≥29 mV;
uma região com várias amostras conta uma única vez. A taxa usa a duração
`número de amostras × dt`. Os intervalos são diferenças entre os instantes
desses picos. “Curto” significa ≤5 ms, fixado antes da avaliação e registrado
por `--burst-isi-ms`. Trata-se de um atributo temporal, não de diagnóstico de
burst biológico. CV e razão de adaptação exigem três picos; os outros atributos
de ISI exigem dois. Atributos sem intervalos suficientes recebem zero.

Ambos os baselines recebem apenas atributos extraídos dos **mesmos sinais
observados** enviados à CNN. Parâmetros, classes esperadas, entradas e disparos
limpos não entram em nenhum preditor. As distâncias dos baselines não são
probabilidades calibradas.

## Métricas e interpretação

- **Acurácia total:** corretos / todos os neurônios solicitados. Exemplos
  silenciosos ou ambíguos permanecem no denominador. Falhas de predição contam
  como erros e têm coluna própria na matriz de confusão.
- **Macro F1:** média de F1 das três classes, com valor zero para uma classe
  sem positivos verdadeiros. Há matriz de confusão por condição/semente/modelo.
- **Acurácia entre ativos:** análise secundária com pelo menos o número mínimo
  de picos observados registrado no checkpoint; aplica o mesmo subconjunto
  aos três modelos.
- **Cobertura da CNN:** fração com sinal finito, atividade observável suficiente
  e confiança acima do limiar já registrado. Acurácia entre aceitos sempre
  acompanha a cobertura; 100% com poucas previsões não equivale a 100% do total.
- **NLL, Brier multiclasses e ECE:** avaliam probabilidades da CNN sem ajustar
  um calibrador. Brier soma os três erros quadráticos por exemplo; ECE usa
  dez bins de confiança de largura igual e não prova calibração universal.
- **Variação entre sementes:** desvio amostral das acurácias por execução,
  indisponível com uma única semente. Não é um intervalo de confiança que
  considere os neurônios de uma rede como observações independentes.
- **Tempo de inferência:** CNN inclui cópia/normalização/lotes; cada baseline
  inclui extração dos cinco atributos e sua própria predição. Exclui geração
  da população, ajuste e métricas. São tempos diagnósticos, sem aquecimento,
  não um benchmark controlado de desempenho.

O caminho diagnóstico da CNN faz a predição mesmo em correntes fora do intervalo
original; não usa a informação da corrente para rejeitá-las antecipadamente.
Isso permite medir falhas nesses exemplos. A política de rejeição do aplicativo
continua sendo uma política operacional diferente. A cobertura secundária do
estudo usa atividade/confiança, sem esse filtro prévio de corrente.

Os rótulos de avaliação indicam a **família paramétrica de origem** RS/CH/FS.
Um neurônio pode mudar sua resposta sob outro estímulo ou recorrência: acertar
essa origem não identifica necessariamente seu padrão atual de disparo.

As contagens físicas usam os eventos de limiar ≥30 mV **antes do ruído**, tanto
nas populações independentes quanto nas conectadas. O estudo independente passa
pelo mesmo contador do simulador conectado, com grau zero e traços equivalentes.
Essas contagens servem como diagnóstico e não como entrada dos classificadores.
O contador legado da interface para neurônios independentes permanece separado;
não compare sua contagem de amostras com esses eventos como se fossem idênticas.

Se o baseline de taxa acompanhar a CNN, a separação pode depender principalmente
da frequência. Se os atributos temporais também mantiverem alta acurácia, eles
fornecem uma explicação simples da separação nessa simulação. Uma vantagem da
CNN em condições retidas precisa aparecer com denominadores e replicação
comparáveis. Acurácia baixa, por si só, não demonstra maior realismo.

## Saídas e reprodução

Cada execução cria `artifacts/research/run_<data_hora>/`:

| Arquivo | Conteúdo |
|---|---|
| `report.html` | mapa comparativo, cobertura, contagens, variação e condições retidas; funciona offline |
| `report.md` | versão textual do relatório |
| `study.json` | protocolo completo, versões, hashes de código/dados/checkpoint, sementes, normalizações, métricas e agregações |
| `summary.csv` | uma linha por condição/semente/modelo, incluindo cobertura, calibração e tempos |
| `predictions.csv` | parâmetros e predições por neurônio para auditoria |
| `status.json` | estado `running`, `complete`, `failed` ou `interrupted` |

Os traços não são salvos integralmente para evitar arquivos enormes; o protocolo,
a versão do código, os metadados e as sementes permitem regenerá-los no mesmo
ambiente. As previsões dependem também do checkpoint identificado por hash.
Kernels/versões diferentes podem produzir pequenas diferenças numéricas.

O script recusa uma pasta de saída não vazia. Interrupções deixam um CSV com
sufixo `.partial` e estado incompleto; esse material não é um estudo concluído.
Por padrão, as saídas são ignoradas pelo Git, assim como o checkpoint e o banco.

Para listar ou selecionar condições:

```powershell
python evaluate_research.py --list-conditions
python evaluate_research.py --conditions reference noise_5 network_0.5 ramp --seeds 101 202 303
```

Selecionar um subconjunto fica explícito nos resultados; ele não deve ser
apresentado como a avaliação do protocolo inteiro. Para repetir o estudo, use
o mesmo commit, checkpoint, banco (ou escolha sintética explícita), argumentos
e ambiente, com outra pasta de saída vazia. A configuração padrão inclui uma
referência pareada; sem selecioná-la, a diferença relativa aparece indisponível.
