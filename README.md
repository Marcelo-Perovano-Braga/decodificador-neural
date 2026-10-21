# Classificador de padrões neuronais Izhikevich

Prova de conceito computacional para classificar três padrões simulados de
atividade de um único neurônio: **Regular Spiking (RS)**, **Chattering (CH)** e
**Fast Spiking (FS)**.

O projeto não utiliza amostras biológicas e não constitui uma interface
cérebro-computador. Cada linha do dataset corresponde a uma instância
independente de um neurônio digital. A interface também oferece um modo
exploratório de rede sináptica esparsa, separado do treinamento original.

## O que foi corrigido

- Dataset persistente em SQLite, com sinal, rótulo, parâmetros e metadados.
- Divisões estratificadas e independentes de treino, validação e teste.
- Pequenas variações paramétricas e ruído de observação para evitar memorização
  de três traços determinísticos.
- Treinamento em mini-batches, normalização calculada apenas no treino, early
  stopping e melhor checkpoint selecionado pela validação.
- Acurácia, precisão, recall, F1 e matriz de confusão no conjunto de teste.
- Comparação reproduzível de custo computacional entre LIF e Izhikevich.
- Interface com classe esperada, classe prevista, confiança e estado
  **indeterminado** para sinais silenciosos, fora do domínio ou de baixa confiança.
- Remoção do antigo “preditor de V(t+1)”, cujo alvo já estava presente na entrada.

## Requisitos

- Python 3.10–3.14; Python 3.11 continua sendo a opção recomendada.
- Windows, Linux ou macOS.
- GPU CUDA é opcional. O pipeline utiliza CPU quando CUDA não está disponível.

A interface utiliza **PyQt6-Fluent-Widgets** sobre o PyQt6. O pacote fornece os
cartões, controles, indicadores de estado, barra de confiança e alternância de
tema. O gráfico continua sendo renderizado pelo `pyqtgraph`.

### Instalação no Windows

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Instalação no Linux/macOS

```bash
python3.11 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Treinamento e validação

Na raiz do projeto:

```bash
python train_pipeline.py --regenerate
```

O comando cria:

- `data/neural_patterns.sqlite`: banco de dados reproduzível;
- `weights/classifier.pt`: checkpoint selecionado pela validação;
- `artifacts/test_metrics.json`: métricas no teste retido;
- `artifacts/confusion_matrix.csv`: matriz de confusão;
- `artifacts/training_history.csv`: histórico de treinamento;
- `artifacts/model_benchmark.json`: custo LIF versus Izhikevich.
- `artifacts/results_summary.md`: resumo rastreável para o relatório final.

O banco incluído no pacote pode ser verificado sem PyTorch:

```bash
python validate_dataset.py
```

O resultado somente atende à meta quando `target_met` for `true` em
`test_metrics.json`. A interface funcionar não substitui essa evidência.

Para um ensaio menor de infraestrutura:

```bash
python train_pipeline.py --regenerate --samples-per-class 60 --epochs 3 --patience 2
```

Esse ensaio não deve ser utilizado como resultado científico.

## Interface

Após gerar o checkpoint:

```bash
python main_app.py
```

### Tela cheia e fundo neural

- **F11** ou o botão de expandir no canto superior direito alterna a tela cheia.
- **Esc** sai da tela cheia. O tamanho anterior é recuperado; uma janela que
  estava maximizada volta ao estado maximizado.
- Para iniciar diretamente em tela cheia: `python main_app.py --fullscreen`.
  O executável também aceita `--fullscreen` após ser reconstruído.
- O botão com ícone de imagem, ao lado do controle de tema, ativa ou desativa
  o fundo neural. O fundo se adapta aos temas escuro e claro.

A ilustração é estática e decorativa, sem relação com os disparos da simulação.
É incluída no aplicativo e não exige conexão com a internet. A imagem mantém
suas proporções e recebe um recorte central para preencher a janela. A versão
ajustada é armazenada em cache e reutilizada; não há animação nem timer extra.
Redimensionamento, escala de tela, tema ou ativação do fundo atualizam o cache.
Gráficos e tabelas conservam suas próprias superfícies para leitura dos dados.
Sem o arquivo de imagem, a interface utiliza um fundo de cor uniforme.

O arquivo está em `assets/neural_background.png`; a origem, o prompt e a
descrição de uso estão em [assets/neural_background.md](assets/neural_background.md).

### Atualização de uma instalação já treinada

O checkpoint anterior continua compatível; não é necessário treinar novamente.
Instale a nova dependência e execute a interface:

```powershell
python -m pip install -r requirements.txt
python main_app.py
```

Ao extrair o projeto em uma pasta nova, copie `weights/classifier.pt` da pasta
anterior. Também é possível copiar `artifacts/` para manter as métricas já
produzidas.

### Populações independentes

Defina separadamente as quantidades de neurônios **RS**, **CH** e **FS**, de zero
a 100.000 por classe (300.000 no total), com pelo menos um neurônio no total.
A população inicial contém dez de cada padrão. O controle de estímulo aplica a mesma intensidade a
todos os neurônios; ele não representa a quantidade de neurônios nem a quantidade
de disparos.

Os neurônios continuam sendo instâncias independentes, sem conexões sinápticas.
A opção **Variação entre neurônios** utiliza a variação paramétrica e o ruído de
observação registrados no checkpoint. Desmarque-a para visualizar as
parametrizações centrais. A mesma semente e as mesmas configurações reproduzem
a população; **Nova variação** troca a semente.

A IA recebe apenas os traços de potencial, em lotes limitados a 64 neurônios,
e produz uma classificação independente para cada um. Os rótulos esperados
são usados somente para comparar os resultados. A arquitetura, os pesos e o
pipeline de treinamento permanecem compatíveis com `weights/classifier.pt`.
Os arquivos `.pth` da implementação antiga não substituem esse checkpoint.

### Redes conectadas (exploratório)

No painel esquerdo, **Conexões sinápticas → Conectar neurônios** ativa a rede.
O botão **Preset: 10.000 neurônios** configura 3.334 RS, 3.333 CH e 3.333 FS,
com 100 saídas por neurônio: **1.000.000 de sinapses dirigidas**. Ele preserva o
estímulo, a semente e a opção de variação selecionados.

- Cada neurônio envia para destinos aleatórios distintos, sem autoconexões.
  Em populações pequenas, a quantidade de saídas é limitada a `N − 1`.
- Cada disparo envia uma corrente assinada após o atraso configurado. O padrão
  usa força excitatória 0,5, 20% de neurônios inibitórios, ganho inibitório 4,
  atraso de 1 ms e decaimento exponencial de 5 ms, em unidades do modelo.
- Os papéis excitatório/inibitório são sorteados separadamente dos parâmetros
  RS/CH/FS. Eles não representam uma identificação de tipos celulares reais.
- A mesma semente reproduz parâmetros, conexões e papéis. Alterar apenas a
  força ou o atraso mantém a topologia, permitindo comparar os efeitos.
- Há um limite de **10.000.000 de sinapses** e até 1.000 saídas por neurônio.
  Configurações acima do orçamento são rejeitadas antes de alocar a rede.
  O modo independente mantém seu limite de 300.000 neurônios.

A aba **Rede** mostra a taxa de disparo de toda a população em janelas de até
5 ms, a corrente sináptica média, as contagens de conexões entre grupos RS/CH/FS
e o papel, entradas, saídas e uma amostra de destinos do neurônio selecionado.
Todos os destinos são mantidos em memória. O raster continua limitado a 200
neurônios visíveis, com acesso ao traço completo de qualquer neurônio.

No modo conectado, os disparos e as taxas são medidos antes do ruído de
observação; esse ruído não produz eventos sinápticos. As previsões do checkpoint
continuam disponíveis e são identificadas como **experimentais**. A concordância
com os parâmetros de origem não valida o classificador em redes: ele foi treinado
somente com neurônios independentes. Não é necessário retreinar para abrir a
atualização, mas uma avaliação científica da nova condição exige dados e testes
próprios, mantendo os resultados originais separados.

As equações, a ordem de atualização e as limitações estão em
[NETWORKS.md](NETWORKS.md). Este modo usa sinapses de corrente; não implementa
junções elétricas nem plasticidade.

### Visualização e resultados

- **Sinais:** raster dos instantes de disparo e traço completo do neurônio
  selecionado. As cores do raster representam classes previstas; cinza indica
  ausência de classificação determinada. Acima de 200 neurônios, o raster mostra
  uma amostra visual uniformemente distribuída, identificada no subtítulo.
- **Classificações individuais:** uma linha por neurônio, com classe esperada,
  previsão, confiança, disparos e estado. Clique na linha para inspecionar seu
  sinal. Uma hipótese marcada com `(?)` permanece indeterminada.
- **Estatísticas:** contagens esperadas e previstas por classe, matriz de
  confusão com coluna de indeterminados, cobertura e acerto entre classificados.
  A cobertura é `classificados / total`; o acerto é `corretos / classificados`.
  Se nenhum neurônio for classificado, o acerto aparece como indisponível.

Todos os neurônios entram na inferência e nas estatísticas, inclusive os que
não aparecem na amostra visual do raster. A seleção individual permite acessar
o traço de qualquer neurônio. A redução de pontos para desenho preserva picos
e não altera o sinal recebido pelo classificador.

Essas estatísticas descrevem a simulação interativa e não substituem as métricas
do teste retido. Sem checkpoint, a simulação continua disponível, mas previsões
e métricas de classificação aparecem como indisponíveis.

### Responsividade

A simulação e a inferência executam em um único worker, separado da interface.
Alterações rápidas dos controles são agrupadas; solicitações antigas são
canceladas ou descartadas, mantendo apenas a configuração pendente mais recente.
Enquanto o cálculo termina, gráficos e tabelas permanecem identificados como os
resultados da última simulação concluída. Selecionar outro neurônio utiliza os
resultados em memória, sem simular ou classificar a população novamente.

O simulador organiza os valores de um mesmo instante de forma contígua na
memória. Ruído, contagem de disparos e verificação de sinais usam buffers
limitados; a interface aplica o ruído no próprio armazenamento da população.
A simulação verifica cancelamento a cada 32 passos e entre blocos de ruído,
evitando esperar por uma simulação inteira que já ficou obsoleta.

A interface funde as normalizações BatchNorm nas convoluções de uma cópia
exclusiva para inferência, sem modificar o modelo treinável nem o checkpoint.
Ela utiliza até oito threads de inferência na CPU, reservando pelo
menos um processador lógico para outras tarefas quando disponível. A equação,
o passo temporal, os traços completos, a sequência de ruído para uma mesma
semente e o checkpoint foram preservados. A inferência mantém precisão float32;
pequenas diferenças de arredondamento entre kernels paralelos são possíveis.

### Medir capacidade no seu computador

No mesmo ambiente da interface, utilizando seu checkpoint já treinado:

```powershell
python benchmark_population.py --neurons 3000 10000 100000 --threads 8 --output artifacts/population_benchmark.json
```

Para medir a rede com o checkpoint local:

```powershell
python benchmark_population.py --connected --neurons 1000 10000 --out-degree 100 --threads 8 --output artifacts/network_benchmark.json
```

O JSON registra as configurações sinápticas, sinapses efetivas, armazenamento da
topologia, eventos entregues, taxas de disparo e tempos de cada repetição. Use
`--simulation-only` para medir a rede sem incluir a classificação.

O benchmark realiza três medições por tamanho, após aquecimento, e inclui
simulação, ruído, inferência, estatísticas e preparação do raster. Não inclui
desenho da interface nem manter a população anterior em memória. Para comparar
o uso da CPU, execute novamente com `--threads 1` ou `--threads 4`. Sem checkpoint,
`--simulation-only` mede apenas a simulação e preparação dos resultados; esse
tempo não representa o custo completo de classificação.

Após comparar os resultados, a interface também aceita `python main_app.py
--threads 4` (ou o executável com `--threads 4`) para utilizar o número de threads
mais adequado ao computador. Sem esse argumento, a seleção é automática.

Com a janela padrão de 200 ms e passo de 0,1 ms, cada neurônio ocupa 8.000 bytes
somente em potencial armazenado. Durante uma atualização, a interface pode
manter a população anterior e a nova: aproximadamente `16.000 × neurônios`
bytes, além de parâmetros, previsões, buffers do modelo e da interface. O limite
configurado não garante atualização instantânea nem execução em tempo real.

As fórmulas de memória, hipóteses sobre conexões esparsas e densas e estimativas
para um Ryzen 7 5700X3D com 32 GiB estão em [CAPACITY.md](CAPACITY.md).

A interface mantém alternância entre temas escuro e claro, indicação do
checkpoint carregado e os estados **correto**, **divergente** e **indeterminado**.

Os arquivos `assets/app_icon.png` e `assets/app_icon.ico` são incluídos no
projeto. No Windows, a aplicação também define uma identidade própria para não
ser agrupada visualmente sob o ícone genérico do interpretador Python.

## Gerar o aplicativo Windows

O projeto inclui uma configuração do PyInstaller para gerar um aplicativo
Windows de 64 bits. A distribuição utiliza o formato **onedir**, mais confiável
para PyTorch, PyQt6 e seus arquivos binários do que concentrar tudo em um único
executável temporariamente extraído.

No Windows, ative o mesmo ambiente virtual usado no treinamento e execute:

```powershell
.\build_windows.ps1
```

Se a política de execução do PowerShell bloquear o arquivo:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\build_windows.ps1
```

O script:

1. confirma que está sendo executado no Windows;
2. verifica a presença de `weights/classifier.pt`;
3. instala as dependências de aplicação e empacotamento;
4. incorpora o checkpoint, o ícone e os metadados do executável;
5. produz `dist/ClassificadorIzhikevich/ClassificadorIzhikevich.exe`;
6. cria `dist/ClassificadorIzhikevich-Windows-x64.zip` para distribuição.

O arquivo `.exe` deve permanecer junto ao subdiretório `_internal`. O computador
que executar essa distribuição não precisa ter Python ou as bibliotecas do
projeto instaladas.

Como o executável não possui assinatura digital, o Windows SmartScreen ou um
antivírus pode solicitar confirmação na primeira execução. Isso não é removido
por configuração de código; distribuição pública sem esse aviso requer um
certificado de assinatura de código.

Estímulos sem disparos suficientes são exibidos como indeterminados. Isso é uma
restrição de observabilidade: sinais vazios não contêm informação suficiente
para distinguir todas as parametrizações.

## Testes

```bash
python -m unittest discover -s tests -v
```

Os testes de simulação, banco de dados, população e métricas usam NumPy e a
biblioteca padrão. Os testes da arquitetura e da inferência em lote exigem
PyTorch. Os testes de integração da interface e do worker exigem as dependências
gráficas e executam com a plataforma Qt `offscreen`, sem abrir janelas. Eles
verificam cancelamento, descarte de resultados antigos, seleção individual e
contabilização dos resultados. Checkpoints de teste são fixtures não treinadas
e não constituem validação da acurácia científica.

## Benchmark isolado

```bash
python benchmark_models.py --samples 512 --repeats 5
```

O benchmark compara custo de simulação, não equivalência biológica. LIF e
Izhikevich possuem equações, estados e escalas de entrada diferentes.

## Limitações que devem constar no relatório final

1. Os sinais são sintéticos e os rótulos derivam de parametrizações conhecidas.
2. A acurácia mede generalização dentro da distribuição simulada, não em tecido
   neural real.
3. RS, CH e FS são padrões fenomenológicos simulados, não diagnósticos.
4. A classificação requer atividade observável; silêncio neural pode ser
   intrinsecamente ambíguo.
5. A comparação LIF–Izhikevich não transforma os dois modelos em representações
   biologicamente equivalentes.
