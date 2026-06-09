# Índice da Documentação

Índice da documentação do sistema de detecção de estados operacionais (LIGADO/DESLIGADO)
de equipamentos industriais.

## Visão Geral do Projeto

Objetivo: classificar automaticamente equipamentos industriais como LIGADO ou DESLIGADO a
partir de dados de sensores IoT, usando K-Means (não supervisionado) com regras e
pontuações baseadas em thresholds dinâmicos por equipamento. A classificação é binária e o
sistema trata tanto equipamentos elétricos (com corrente e RPM) quanto mecânicos (apenas
temperatura e vibração), sem intervenção manual.

Características principais:

- Motor único de K-Means (k = 6) com classificação por score ponderado.
- Filtro EMI antialucinação com limiar dinâmico `max(P90 x 5%, 2,0 A)`.
- Trava física Vibration Safety Floor e limite absoluto de 1.000 registros para a rota
  elétrica.
- Interpolação adaptativa (PCHIP e KNN temporal) e segmentação por lacunas maiores que 3 h.

## Documentação por Categoria

### Documentação Geral

- [README_PROJETO_FINAL.md](README_PROJETO_FINAL.md): documentação principal do projeto
  final (visão geral, tipos de equipamento, pipeline, execução e estrutura).
- [README_SCRIPTS_PROCESSAMENTO.md](README_SCRIPTS_PROCESSAMENTO.md): scripts do pipeline
  de processamento, fluxo passo-a-passo e parâmetros.

### Modelos e Algoritmos

- [README_KMEANS_CLASSIFICACAO_MODERADO.md](README_KMEANS_CLASSIFICACAO_MODERADO.md):
  K-Means com k = 6 e a lógica de classificação por score (rota elétrica).

### Scripts Específicos

- [README_NORMALIZAR_DADOS_KMEANS.md](README_NORMALIZAR_DADOS_KMEANS.md): normalização e
  preparo dos dados para o K-Means.

## Guia de Início Rápido

Treinar todos os equipamentos de uma vez (detecta elétrico/mecânico automaticamente):

```bash
cd code
python treinar_todos.py
# ou apenas alguns:
python treinar_todos.py c_636 c_1518
```

Treinar um equipamento individualmente:

```bash
cd code
python pipeline_deteccao_estados.py --mpoint c_636            # elétrico
python pipeline_deteccao_estados_mecanico.py --mpoint c_1518  # mecânico
```

Interface gráfica (treino, análise de intervalos e visualização 3D):

```bash
cd code
python gui_pipeline.py
```

## Estrutura do Projeto

```
code/
├── gui_pipeline.py                        # Interface gráfica unificada
├── treinar_todos.py                       # Executa o pipeline para todos os mpoints
├── pipeline_deteccao_estados.py           # Orquestrador ELÉTRICO
├── pipeline_deteccao_estados_mecanico.py  # Orquestrador MECÂNICO
├── data/                                  # raw, raw_preenchido, processed, normalized
├── models/c_<mpoint>/                     # Modelos, scaler e configs por equipamento
├── results/c_<mpoint>/                    # Visualizações e relatórios
├── logs/                                  # Logs de execução
├── scripts/                               # Módulos do pipeline
├── utils/                                 # Utilitários
└── docs/                                  # Esta documentação
```

## Suporte

1. Consulte a documentação específica de cada etapa.
2. Verifique os logs em `code/logs/`.
3. Confirme os pré-requisitos (Python 3.11+, pandas 2.x, scikit-learn).
4. Execute as etapas na ordem do pipeline, ou use `treinar_todos.py`.
