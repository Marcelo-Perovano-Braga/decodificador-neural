# Classificador de padrões neuronais Izhikevich

Prova de conceito computacional para classificar três padrões simulados de
atividade de um único neurônio: **Regular Spiking (RS)**, **Chattering (CH)** e
**Fast Spiking (FS)**.

O projeto não utiliza amostras biológicas, não simula uma rede sináptica e não
constitui uma interface cérebro-computador. Cada linha do dataset corresponde a
uma instância independente de um neurônio digital.

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

A lista seleciona o padrão usado pelo simulador. A IA recebe somente o traço de
potencial e deve identificar esse padrão. A interface compara a previsão com o
rótulo esperado sem fornecer o rótulo como entrada ao modelo.

A tela reformulada apresenta:

- cartões separados para configuração, classificação e métricas da simulação;
- indicador visual do carregamento do checkpoint;
- estados **correto**, **divergente** e **indeterminado**;
- barra animada de confiança;
- alternância entre temas escuro e claro;
- ícone próprio na janela, no Alt+Tab e na barra de tarefas do Windows;
- traçado responsivo preservado com `pyqtgraph`.

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

Os testes de simulação, banco de dados e métricas usam NumPy e a biblioteca
padrão. Os testes da arquitetura também exigem PyTorch.

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
