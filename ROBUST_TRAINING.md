# Treinar e comparar um candidato robusto

Este documento descreve o protocolo 1, cuja confirmação já foi inspecionada.
Reexecutá-lo reproduz o estudo anterior; não constitui outra confirmação
independente. O protocolo 2 é descrito em
[EXPANDED_ROBUST_TRAINING.md](EXPANDED_ROBUST_TRAINING.md).

`train_robust.py` treina um checkpoint separado, compara os dois modelos em
populações pareadas e pode adotar o candidato somente se todos os critérios
predefinidos forem atendidos. O checkpoint ativo é necessário para a comparação;
um CSV de resultados anteriores não o substitui.

Na raiz do projeto, com o ambiente Python ativado:

```powershell
python train_robust.py --seed 17029 --adopt-if-superior
```

O comando usa o checkpoint atual em `weights/classifier.pt` para obter a
configuração de simulação e como referência da comparação. O candidato é
treinado do zero. O banco SQLite original não é necessário: os novos sinais
são gerados pelo simulador do projeto, com os mesmos parâmetros de origem
RS/CH/FS. Nenhum dado biológico é usado.

Cada execução recebe uma pasta nova em `artifacts/research/`. Os arquivos
originais de resultados da pesquisa não são sobrescritos. Há suporte a CPU;
a RX 6600 não é usada por este treinamento. `--threads 8` limita as threads
de cálculo, e `--epochs`/`--patience` controlam a duração máxima.

## Ajuste, seleção e calibração

Com os valores padrão, são 25 configurações em cada split:

| Split | Neurônios por classe/configuração | Total | Uso |
|---|---:|---:|---|
| `train` | 200 | 15.000 | otimizar os pesos |
| `selection` | 50 | 3.750 | selecionar época e ajustar taxa de aprendizado |
| `calibration` | 50 | 3.750 | ajustar uma temperatura de confiança |

Os splits têm sementes e namespaces distintos. Verificações de identidade de
traços impedem sobreposição exata. A normalização usa somente o treino.
Exemplos silenciosos continuam presentes, com sua família paramétrica de
origem; o programa não assume que a família seja observável em todo sinal.

O treino contém:

- correntes constantes com ruído de observação e variação paramétrica;
- pulsos com vários inícios e durações, inclusive pulsos que terminam na borda
  da janela e pulsos curtos tardios;
- rampas e senoides, com versões ruidosas;
- redes com forças 0,15, 0,4 e 0,8, e combinações de rede com pulso/rampa.

Os detalhes exatos de cada configuração, suas sementes, a topologia solicitada
e os hashes do código são registrados no checkpoint e em `training.json`.
A arquitetura continua sendo `NeuronClassifier1D`, com a mesma interface de
entrada da aplicação. AdamW usa taxa inicial 0,001 e regularização 0,0001;
o treino usa label smoothing 0,03. A seleção usa NLL sem smoothing.

A melhor época é escolhida pelo NLL no split `selection`, com redução da taxa
de aprendizado e early stopping. O limite padrão é 40 épocas, com paciência 8.
Somente depois da escolha dos pesos é ajustada uma temperatura positiva usando
o split `calibration`, em uma grade fixa de 0,1 a 10. O valor 1 está incluído.
Esse ajuste não troca a classe vencedora: modifica as probabilidades.

O limiar de confiança do candidato permanece em 0,70; não é escolhido no teste.
Temperatura, normalização, classes e protocolo acompanham os pesos. A inferência
da interface, do worker, do benchmark e do estudo de robustez aplica essa
temperatura; checkpoints anteriores usam o padrão 1.

## Comparação final

O teste usa namespaces distintos dos três splits de ajuste e as sementes
**6101, 6203, 6301, 6407 e 6503**, diferentes do estudo diagnóstico anterior.
São 20 condições, 100 neurônios por classe/semente por padrão: **30.000 sinais
por modelo**, sempre com os mesmos sinais para os dois modelos.

| Grupo | Condições |
|---|---|
| Referência | neurônios independentes nas configurações originais |
| Ruído | 2, 5 e 10 mV |
| Variação paramétrica | 0,06, 0,10 e 0,15 |
| Rede | forças 0,1, 0,5 e 1,0 |
| Pulso | inícios 10%, 30%, 50%; pulso adicional com início 55% e duração 35% |
| Famílias já usadas no treino | rampa e senoide com novas populações |
| Famílias ausentes do ajuste | pulso duplo e envelope triangular |
| Amplitude constante externa | abaixo e acima da faixa base registrada |

O pulso duplo tem duas ativações, em 10–30% e 60–80% da janela. O envelope
triangular vai de 0,25 a 1,75 vezes a amplitude e retorna. Nenhum deles entra
no treino, seleção ou calibração.

Rampas e senoides passam a integrar o treino do candidato. Seus resultados
com novas sementes são testes em novas populações, **não** generalização para
famílias de entrada inteiramente ausentes. A variação gaussiana também não
cria faixas paramétricas rigorosamente disjuntas.

As faixas constantes externas são aproximadamente -7,5–5 e 30–42,5 no protocolo
padrão. Os limites 5 e 30 são excluídos por float32. Essa retenção se refere
à amplitude base de correntes constantes: valores instantâneos maiores que 30
podem ocorrer nas rampas usadas no treino.

Todos os neurônios entram na acurácia principal; silêncio não é removido.
Matrizes de confusão, macro F1, NLL, Brier, ECE, cobertura e erros aceitos são
calculados separadamente. A cobertura usa o mesmo contador legado da aplicação
(amostras observadas ≥29 mV) e o limiar de cada checkpoint. Ela não equivale à
contagem de eventos físicos nem ao contador de bordas do estudo anterior.

A rejeição prévia pela faixa de corrente é ignorada nessa análise diagnóstica
para permitir medir erros fora do domínio. A interface continua usando sua
política operacional de domínio. Os dois modelos recebem somente os sinais;
rótulos, parâmetros e informações da corrente não entram nos preditores.

## Critérios de adoção

Os critérios são definidos antes da geração do teste final:

| Critério | Exigência |
|---|---|
| Acurácia média nas 20 condições | ganho de pelo menos 2 pontos percentuais |
| Acurácia média nos pulsos | ganho de pelo menos 10 pontos percentuais |
| Acurácia na referência | pelo menos 99%, com queda máxima de 1 ponto percentual |
| Queda em qualquer condição | no máximo 3 pontos percentuais |
| Cobertura média | queda máxima de 5 pontos percentuais |
| Erros aceitos por neurônio solicitado | redução média de pelo menos 15% |
| NLL médio | menor que o da referência |
| Brier médio | não pior que o da referência |
| Ganho por semente | ganho médio positivo em cada uma das cinco sementes |

Esses limites constituem uma política de engenharia, não um teste formal de
significância estatística. Neurônios conectados são correlacionados; as cinco
sementes não devem ser interpretadas como milhares de redes independentes.
Cada condição recebe o mesmo peso na média, e as comparações são pareadas.

Se qualquer critério falhar, o checkpoint ativo permanece igual. Não há
reajuste automático do treino depois de observar o teste. Se houver outra
rodada de desenvolvimento, seus testes de confirmação precisarão de condições
e sementes ainda não usadas para decisões de desenvolvimento.

## Usar um candidato já treinado

É possível avaliar o candidato distribuído sem treinar novamente. Mantenha-o
com um nome diferente do checkpoint ativo:

```powershell
python train_robust.py --candidate weights/classifier_robust_candidate.pt --adopt-if-superior
```

Sem `--adopt-if-superior`, o programa apenas compara. Com essa opção e todos
os critérios atendidos, ele verifica os hashes dos arquivos comparados, mantém
uma cópia exata do checkpoint anterior e substitui o ativo de forma atômica.
A cópia recebe o nome `classifier.original_<hash>.pt` na mesma pasta.
O programa registra os caminhos e hashes em `adoption.json`.

Reinicie `python main_app.py` para carregar o modelo adotado. Uma distribuição
`.exe` anterior contém seus próprios arquivos; depois de adotar e verificar
o candidato, reconstrua-a no Windows:

```powershell
.\build_windows.ps1 -SkipDependencyInstall
```

Não copie o candidato diretamente sobre `weights/classifier.pt` antes da
comparação. Para voltar ao modelo anterior, feche a aplicação e restaure a
cópia cujo caminho foi registrado em `adoption.json`.

## Evidências produzidas

- `training/candidate.pt`: pesos treinados, normalização, temperatura e
  proveniência, incluindo hashes dos traços usados nos três splits;
- `training/training.json`: protocolo, sementes, versões, época escolhida e
  resultado de calibração;
- `training/training_history.csv`: curva de treino e seleção;
- `comparison/comparison.json`: resultados completos dos dois modelos por
  condição/semente, hashes e decisão de cada critério;
- `comparison/comparison.csv`: tabela por condição/semente/modelo;
- `comparison/comparison.md`: comparação legível e critérios de adoção;
- `adoption.json`: registro da adoção, quando realizada;
- `status.json`: estado da execução; resultados incompletos não autorizam adoção.

O comando recusa diretórios de saída não vazios. Uma execução sem o modelo de
referência pode usar `--train-only --context-study caminho/study.json` somente
para construir um candidato a partir da configuração registrada. Ela termina
como `trained_awaiting_comparison`, sem declarar superioridade nem adotar pesos.
