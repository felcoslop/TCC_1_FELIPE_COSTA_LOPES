# Script: kmeans_classificacao_moderado.py

## 📋 Descrição
Script para K-means com classificação rigorosa que usa 6 clusters, mas rotula apenas os 2 clusters com mais certeza (99.5%+) para treinamento de CNN com dados limpos e alta qualidade.

## 🎯 Objetivo
- Executar clustering K-means com 6 clusters
- Identificar clusters com alta certeza (99.5%+)
- Rotular apenas clusters 2 (100% LIGADO) e 3 (99.5% DESLIGADO)
- Gerar dataset limpo para treinamento CNN
- Descarte clusters intermediários com incerteza

## 📊 Entrada
- **Arquivo**: `data/normalized/dados_kmeans.csv`
- **Formato**: CSV com dados normalizados (0-1)
- **Dimensões**: ~772k linhas × 182 colunas
- **Configuração**: `models/info_normalizacao.json`

## 📤 Saída
- **Dados classificados**: `data/processed/dados_classificados_kmeans_moderado.csv` (todos os 6 clusters)
- **Dados limpos**: `data/normalized/dados_kmeans_rotulados_conservador.csv` (apenas clusters 2 e 3)
- **Modelo K-means**: `models/kmeans_model_moderado.pkl`
- **Scaler**: `models/scaler_model_moderado.pkl`
- **Info do modelo**: `models/info_kmeans_model_moderado.json`
- **Visualizações**: `results/analise_kmeans_clusters_moderado.png`

## 🔧 Funcionalidades

### 1. Carregamento de Dados
- Carrega dados normalizados do CSV
- Valida informações de normalização
- Verifica integridade dos dados

### 2. Execução do K-means
- **Número de clusters**: 6
- **Algoritmo**: K-means do scikit-learn
- **Inicialização**: random_state=42
- **Iterações**: máximo 300

### 3. Análise de Clusters
- Calcula estatísticas por cluster
- Analisa variáveis de velocidade e magnitude
- Identifica padrões de atividade

### 4. Classificação Rigorosa
- **Estratégia**: Critérios baseados em valores físicos
- **DESLIGADO**: vel_rms < 1, current < 10, rpm = 0
- **LIGADO**: Todos os outros casos
- **Base**: Valores reais das variáveis, não clustering

### 5. Seleção de Clusters de Alta Certeza
- **Cluster 2**: 100% LIGADO (67.880 amostras)
- **Cluster 3**: 99.5% DESLIGADO (25.896 amostras)
- **Clusters descartados**: 0, 1, 4, 5 (incerteza)
- **Total para treino**: 93.910 amostras (12.2% dos dados)

### 6. Visualizações Especiais
- Mostra todos os clusters vs classificados
- Destaca clusters descartados
- Análise de distribuição conservadora

## 📈 Parâmetros do K-means
```python
KMeans(
    n_clusters=6,
    random_state=42,
    n_init=10,
    max_iter=300
)
```

## 🔧 Critérios de Classificação
```python
# DESLIGADO: Todas as condições devem ser verdadeiras
condicoes_desligado = [
    vel_rms_max < 1,      # Velocidade RMS máxima < 1
    current_max < 10,     # Corrente máxima < 10
    rpm == 0              # Rotação = 0
]

# LIGADO: Qualquer condição falsa
equipamento_status = 'DESLIGADO' if todas_condicoes else 'LIGADO'
```

## 🚀 Como Usar

```bash
python scripts/kmeans_classificacao_moderado.py
```

## 📋 Pré-requisitos
- Arquivo `dados_kmeans.csv` normalizado
- Arquivo `info_normalizacao.json` com configurações
- Scaler `scaler_maxmin.pkl` salvo
- Diretórios: `models/`, `results/`, `plots/`

## 📊 Exemplo de Saída
```
=== K-MEANS RIGOROSO - 6 CLUSTERS COM CRITÉRIOS ESPECÍFICOS ===
Executando K-means com 6 clusters...
  - Cluster 0: 142,708 amostras (18.5%)
  - Cluster 1: 166,096 amostras (21.5%)
  - Cluster 2: 67,884 amostras (8.8%)
  - Cluster 3: 26,026 amostras (3.4%)
  - Cluster 4: 336,554 amostras (43.6%)
  - Cluster 5: 32,963 amostras (4.3%)

Clusters com mais certeza:
  - Cluster 2: LIGADO com 100.0% de certeza (67,880 LIGADO)
  - Cluster 3: DESLIGADO com 99.5% de certeza (25,892 DESLIGADO)

📊 Resumo do modo rigoroso (apenas clusters de alta certeza):
  - Total de amostras: 772,231
  - Amostras para treinamento CNN: 93,910
  - Amostras LIGADO: 68,014
  - Amostras DESLIGADO: 25,896
  - Percentual usado para treino: 12.2%
  - Clusters selecionados: 2 (LIGADO), 3 (DESLIGADO)
  - Clusters descartados: [0, 1, 4, 5]
```

## 🔍 Estratégia Rigorosa
1. **Critérios Físicos**: Baseado em valores reais das variáveis
2. **Classificação Binária**: DESLIGADO vs LIGADO
3. **Validação**: Critérios baseados em conhecimento do domínio
4. **Consistência**: Todos os dados são classificados

## 📊 Vantagens do Modo Rigoroso
- **Base Física**: Critérios baseados em valores reais
- **Interpretabilidade**: Critérios claros e compreensíveis
- **Consistência**: Classificação baseada em regras fixas
- **Robustez**: Não depende de clustering para classificação

## ⚠️ Considerações do Modo Rigoroso
- **Rigidez**: Critérios fixos podem não capturar nuances
- **Thresholds**: Valores específicos podem precisar ajuste
- **Variabilidade**: Pode não capturar estados intermediários

## 📈 Qualidade dos Dados Rigorosos
- **Consistência**: 100% dos dados classificados com critérios claros
- **Interpretabilidade**: Baseado em conhecimento do domínio
- **Reprodutibilidade**: Critérios fixos e documentados
- **Validação**: Fácil de validar e ajustar

## 📊 Visualizações Especiais
1. **Clusters K-means**: Mostra os 2 clusters
2. **Classificação por Critérios**: Separação LIGADO/DESLIGADO
3. **Distribuição dos Clusters**: Contagem dos 2 clusters
4. **Distribuição do Status**: Proporção LIGADO/DESLIGADO

## 🔄 Fluxo de Trabalho Rigoroso
1. Carregar dados normalizados
2. Executar K-means com 2 clusters
3. Analisar características dos clusters
4. Aplicar critérios rigorosos de classificação
5. Mapear clusters para status
6. Gerar dataset com classificação física
7. Salvar modelos e dados classificados

## 📁 Estrutura de Saída
```
data/processed/
├── dados_classificados_kmeans_moderado.csv  # Todos os dados
data/normalized/
└── dados_kmeans_rotulados_conservador.csv  # Apenas classificados
models/
├── kmeans_model_moderado.pkl               # Modelo K-means
├── scaler_model_moderado.pkl               # Scaler usado
└── info_kmeans_model_moderado.json         # Metadados
results/
└── analise_kmeans_clusters_moderado.png    # Visualizações
```

## 🎯 Aplicações do Dataset Rigoroso
- **Treinamento CNN**: Dados com classificação física
- **Validação**: Benchmark baseado em critérios reais
- **Análise**: Padrões baseados em conhecimento do domínio
- **Deployment**: Modelos com critérios interpretáveis

## 📊 Comparação: Moderado vs Rigoroso
| Aspecto | Moderado | Rigoroso |
|---------|----------|----------|
| **Base de classificação** | Clustering | Critérios físicos |
| **Interpretabilidade** | Baixa | Alta |
| **Consistência** | Variável | Fixa |
| **Ajustabilidade** | Difícil | Fácil |
| **Confiabilidade** | Baseada em clustering | Baseada em regras |

## ⚠️ Observações Importantes
- **Critérios fixos**: Baseados em valores específicos
- **Interpretabilidade**: Fácil de entender e validar
- **Ideal para**: Modelos que precisam de critérios claros
- **Ajustável**: Thresholds podem ser modificados facilmente

## 🎯 Próximos Passos
Após execução, o dataset rigoroso pode ser usado para:
1. Treinar CNN com critérios físicos
2. Validar modelos com regras claras
3. Análise baseada em conhecimento do domínio
4. Benchmark com critérios interpretáveis

## 📊 Estatísticas Típicas
- **Dados originais**: 772.231 amostras
- **Dados para treino CNN**: 93.910 amostras (12.2%)
- **LIGADO**: 68.014 amostras (72.4%)
- **DESLIGADO**: 25.896 amostras (27.6%)
- **Estratégia**: Apenas clusters com alta certeza (99.5%+)
- **Qualidade**: Dados limpos para melhor treinamento


