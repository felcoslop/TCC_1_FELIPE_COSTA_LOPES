# Projeto de Detecção de Estados Operacionais de Equipamentos Industriais

## Visão Geral

Sistema totalmente automático para detecção do estado operacional (LIGADO ou DESLIGADO)
de equipamentos industriais a partir de dados de sensores IoT consultados no InfluxDB.
O motor de classificação utiliza K-Means (aprendizado não supervisionado) combinado a um
sistema de regras e pontuações lógicas baseadas em thresholds dinâmicos por equipamento.

A classificação é binária: cada amostra é rotulada como LIGADO ou DESLIGADO. Não há
modelo CNN, autoencoder ou estado de TRANSIÇÃO no projeto final; essas abordagens foram
avaliadas e descartadas por adicionarem complexidade sem ganho de qualidade frente ao
K-Means (ver discussão na monografia).

## Tipos de Equipamento

O sistema possui dois fluxos distintos, escolhidos automaticamente conforme a
disponibilidade de sensores elétricos.

### Equipamentos Elétricos (ex.: c_636, c_637, c_640)

- Possuem sensores de corrente e RPM (presença de arquivos `estimated`).
- Score: `(vel_rms x 1,0) + (current x 2,0) + (rpm x 2,0) + (temperatura x 0,5)`.
- Validação física: corrente < 15% e RPM < 20% (após normalização) forçam DESLIGADO.
- Orquestrador: `pipeline_deteccao_estados.py`.

### Equipamentos Mecânicos (ex.: c_1518, c_670)

- Sem sensores elétricos; dependem de variações térmicas e de vibração.
- Score: `(temperatura x 1,5) + (vel_rms x 1,0) + (magnetômetro x 0,5)`.
- Validação física: temperatura e vibração próximas ao valor residual forçam DESLIGADO.
- Orquestrador: `pipeline_deteccao_estados_mecanico.py`.

## Pipeline de Processamento

1. Ingestão: scripts em `scripts/baixar_*.py` conectam ao InfluxDB e baixam
   `validated_default`, `estimated` e `validated_slip`.
2. Sincronização: as tabelas têm frequências diferentes (20 s, ~1 min, 2 min). É feito um
   alinhamento por timestamp via backward-fill, exigindo sobreposição mínima de 70% entre
   séries de vibração e corrente.
3. Pré-processamento:
   - Segmentação: gaps maiores que 3 horas separam os dados em períodos independentes.
   - Interpolação adaptativa: PCHIP para lacunas de até 30 min; KNN temporal multivariado
     para lacunas entre 30 min e 3 horas; sem interpolação acima de 3 horas.
   - Outliers (IQR, multiplicador 3,0): sequências de 10 ou mais amostras são preservadas
     (inércia mecânica legítima, não ruído).
   - Filtro EMI: se RPM > 100 mas corrente abaixo do limiar dinâmico `max(P90 x 5%, 2,0 A)`,
     os valores são zerados para evitar falsos positivos por interferência eletromagnética.
   - Clipping no percentil 99,5% e normalização MinMax em [0, 1].
4. Clusterização K-Means (k = 6), com verificação de confiabilidade que exige mínimo
   absoluto de 1.000 registros válidos de corrente/RPM para prosseguir com a classificação
   elétrica (evita fallback mecânico incorreto).
5. Classificação inteligente: cálculo do score ponderado de cada centróide, comparado a
   limites dinâmicos (ex.: percentil 95% do período) para definir LIGADO/DESLIGADO. Os dois
   clusters de menor score são candidatos a DESLIGADO; o segundo só é confirmado se passar
   pela validação (diferença de score < 0,3, ambos < 0,5 e vibração residual).
   Há ainda a trava Vibration Safety Floor: vibração normalizada acima de um piso fixo
   (0,06 elétrico / 0,08 mecânico) força LIGADO.

## Execução

### Treinar todos os equipamentos de uma vez

O script `treinar_todos.py` (na raiz de `code/`) executa o pipeline completo para todos os
equipamentos listados, detectando automaticamente se cada um é elétrico ou mecânico.

```bash
cd code

# Treina todos os mpoints suportados
python treinar_todos.py

# Treina apenas mpoints específicos
python treinar_todos.py c_636 c_1518
```

### Treinar um equipamento individualmente

```bash
cd code

# Equipamento elétrico
python pipeline_deteccao_estados.py --mpoint c_636

# Equipamento mecânico
python pipeline_deteccao_estados_mecanico.py --mpoint c_1518
```

### Interface gráfica

```bash
cd code
python gui_pipeline.py
```

A GUI unifica treino, análise de intervalos e visualização 3D.

## Equipamentos de Referência

| Equipamento | Tipo     | Descrição        |
|-------------|----------|------------------|
| c_636       | Elétrico | Motobomba        |
| c_637       | Elétrico | Motobomba        |
| c_640       | Elétrico | Motor REC        |
| c_1518      | Mecânico | Mancal           |
| c_670       | Mecânico | Redutora bridle  |

## Estrutura do Projeto

```
code/
├── gui_pipeline.py                        # Interface gráfica unificada
├── treinar_todos.py                       # Executa o pipeline para todos os mpoints
├── pipeline_deteccao_estados.py           # Orquestrador ELÉTRICO
├── pipeline_deteccao_estados_mecanico.py  # Orquestrador MECÂNICO
├── data/
│   ├── raw/                               # Dados brutos
│   ├── raw_preenchido/                    # Dados por período (segmentados/preenchidos)
│   ├── processed/                         # Dados unificados e classificados
│   └── normalized/                        # Dados normalizados para K-Means
├── models/
│   └── c_<mpoint>/                        # Modelos, scaler e configs por equipamento
├── results/
│   └── c_<mpoint>/                        # Visualizações e relatórios
├── logs/                                  # Logs de execução
├── scripts/                               # Módulos passo-a-passo do pipeline
├── utils/                                 # Utilitários (artifact_paths, logging_utils)
└── docs/                                  # Esta documentação
```

## Requisitos

- Python 3.11+
- pandas 2.x (o pipeline elétrico depende da frequência `20S`, removida no pandas 3.x)
- scikit-learn, scipy, numpy
- customtkinter (GUI), matplotlib, plotly
