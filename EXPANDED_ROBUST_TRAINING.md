# Protocolo 2: envelopes variados e seleção por grupos

O primeiro candidato ampliado falhou no estímulo triangular. Esse resultado
passou a ser informação de desenvolvimento. Triângulos e pulsos duplos não
são mais famílias inéditas neste protocolo.

```bash
python train_robust.py --protocol 2 --seed 27149 --epochs 50 --patience 12 --train-per-class 200 --validation-per-class 50 --test-per-class 100 --threads 4 --adopt-if-superior
```

`--protocol 1` continua disponível para reproduzir o estudo anterior. Ao
comparar `--candidate`, a versão é lida do próprio checkpoint.

## Ajuste do candidato

A arquitetura `NeuronClassifier1D` e o limiar de confiança de 0,7 são mantidos.
O modelo é inicializado do zero. Os 25 controles anteriores são preservados,
com 31 condições adicionais: rampas descendentes, triângulos, pulsos duplos,
pulsos aleatórios, degraus aleatórios, envelopes interpolados, rampas
aleatórias, triângulos assimétricos aleatórios e senoides aleatórias. Quatro
dessas famílias também são simuladas em redes conectadas. Os envelopes
aleatórios variam por neurônio, usando um fluxo de RNG separado dos rótulos,
parâmetros, amplitudes e ruído de observação.

Com os parâmetros acima: 33.600 traços de treino, 8.400 de seleção e 8.400 de
calibração. Classes equilibradas em cada condição. Todos os neurônios são
retidos, inclusive os silenciosos. Namespaces de ajuste: 30, 31 e 32.

Uma época com acurácia de referência de pelo menos 99% na seleção tem
prioridade sobre épocas abaixo desse piso. Entre épocas elegíveis, minimizar:

`NLL média + 0,5 × pior NLL por condição + 0,25 × pior NLL por condição/classe`.

Quando nenhuma época é elegível, acrescenta-se `10 × déficit de acurácia de
referência` ao escore e registra-se o melhor resultado provisório. A comparação
final continua exigindo o piso original de 99%. O conjunto de calibração é
usado apenas para ajustar uma temperatura escalar, após selecionar os pesos.

## Confirmação reservada antes do treino

`training/reserved_comparison.json` registra condições, política, namespace e
seeds antes de gerar os dados de ajuste. Seu SHA-256 é incluído no checkpoint.
A comparação recusa mudanças nessa reserva e traços que coincidam exatamente
com qualquer divisão usada no ajuste ou na calibração.

São 30 condições, cinco seeds novas (`7109, 7211, 7307, 7411, 7523`), namespace
33 e 100 neurônios por classe/condição/seed: 45.000 sinais por modelo. Ambos
recebem os mesmos sinais. Três famílias são excluídas de todas as divisões de
ajuste: chirp de 5 a 65 Hz, envelope exponencial descendente e surto de cosseno
elevado. Cada uma é testada com ruído de 0,2 e 5 mV. Essas são famílias retidas
do gerador; não há alegação de independência matemática em relação a qualquer
combinação das outras ondas.

Os dez critérios de adoção de `ROBUST_TRAINING.md` permanecem inalterados,
inclusive o limite de três pontos percentuais de regressão em qualquer condição.
Não há ajuste de limiar nem nova escolha de pesos após observar a confirmação.
Se qualquer critério falhar, o checkpoint ativo permanece intacto. Após
inspecionar essa confirmação, ela passa a ser desenvolvimento para futuras
iterações; uma nova confirmação precisa ser reservada.

As cinco seeds são as réplicas de simulação. Neurônios conectados não são
réplicas estatisticamente independentes. Os rótulos são famílias de parâmetros
RS/CH/FS, não diagnósticos biológicos. Critérios de engenharia em simulação
sintética não demonstram funcionamento com sinais neurais reais.
