# Script: normalizar_dados_kmeans.py

## 📋 Descrição
Script para normalizar dados completos (772k linhas) removendo colunas `m_point` e preparando dados para algoritmos de machine learning, especialmente K-means.

## 🎯 Objetivo
- Carregar dados unificados completos (772k linhas)
- Remover colunas relacionadas a `m_point`
- Normalizar dados com pipeline configurável (scikit-learn)
- Preparar dados para K-means e outros algoritmos ML
- Manter máximo de dados possível (removendo apenas o mínimo necessário)

## 📊 Entrada
- **Arquivo**: `data/processed/dados_unificados_final.csv`
- **Dimensões**: ~772k linhas × 187 colunas
- **Conteúdo**: Dados unificados com todas as features
- **Colunas m_point**: `m_point`, `fft_acc_m_point`, `fft_mag_m_point`, `slip_m_point`

## 📤 Saída
- **Dados K-means**: `data/normalized/dados_kmeans.csv`
- **Dados completos**: `data/normalized/dados_normalizados_completos.npy`
- **Scaler**: `models/scaler_maxmin.pkl`
- **Pipeline**: `models/preprocess_pipeline.pkl`
- **Configuração**: `models/info_normalizacao.json`

## 🔧 Funcionalidades

### 1. Remoção de Colunas m_point
- Identifica automaticamente colunas com `m_point`
- Remove 4 colunas: `m_point`, `fft_acc_m_point`, `fft_mag_m_point`, `slip_m_point`
- Mantém 182 colunas de features válidas

### 2. Análise de Dados
- Verifica valores nulos por coluna
- Identifica colunas com muitos valores nulos (>50%)
- Analisa estatísticas básicas das features

### 3. Limpeza Conservadora
- **Estratégia**: `SimpleImputer(strategy='median')` no Pipeline
- Remove apenas colunas com >50% de valores nulos
- Mantém timestamps intactos; valida monotonicidade e duplicatas

### 4. Pré-processamento (configurável)
- **Scaler**: `minmax` (padrão), `standard` ou `robust`
- **Power Transform**: `yeo-johnson` (opcional)
- **Quantile Transform**: `normal` ou `uniform` (opcional; exclusivo com power)
- **VarianceThreshold**: remove features de variância zero
- **Filtro de Correlação**: remove features com |corr| ≥ threshold (opcional)
- **PCA**: por número de componentes ou variância explicada (opcional)

### 5. Preparação para ML
- Cria DataFrame para K-means
- Divide dados em treino/teste (80/20)
- Salva dados em formatos otimizados (.npy, .csv)

## 📈 Parâmetros de Normalização
Flags CLI do script:
```bash
--scaler {minmax,standard,robust}
--power {none,yeo-johnson}
--quantile {none,normal,uniform}
--variance-threshold FLOAT
--corr-threshold FLOAT
--pca-components INT
--pca-variance FLOAT
```

## 🚀 Como Usar

```bash
# Padrão (MinMax + VarianceThreshold 0.0)
python scripts/normalizar_dados_kmeans.py

# Robust a outliers
python scripts/normalizar_dados_kmeans.py --scaler robust

# Gaussianizar distribuição e padronizar
python scripts/normalizar_dados_kmeans.py --power yeo-johnson --scaler standard

# Quantile para distribuição normal + PCA por variância
python scripts/normalizar_dados_kmeans.py --quantile normal --pca-variance 0.95

# Remover colinearidade forte antes do pipeline
python scripts/normalizar_dados_kmeans.py --corr-threshold 0.95
```

## 📋 Pré-requisitos
- Arquivo `dados_unificados_final.csv` deve existir
- Diretórios: `data/normalized/`, `models/`, `plots/`
- Bibliotecas: pandas, numpy, scikit-learn, matplotlib

## 📊 Exemplo de Saída
```
=== NORMALIZAÇÃO DE DADOS PARA K-MEANS ===
Carregando dados unificados...
  - Shape: (772238, 187)
  - Período: 2025-02-18 até 2025-04-30

Removendo colunas m_point...
  - Colunas m_point encontradas: 4
  - Colunas após remoção: 183

Preparando dados para normalização...
  - Colunas selecionadas: 182
  - Valores nulos nas features (antes do pipeline): 12,345
  - Linhas consideradas: 772,238

Normalizando dados com Max-Min Scaler...
  - Shape features (após seleção): (772238, 176)
  - Range: [0.000000, 1.000000]
  - Features removidas (variância zero): 6
```

## 🔍 Colunas Removidas
- `m_point`: Ponto de medição
- `fft_acc_m_point`: FFT aceleração m_point
- `fft_mag_m_point`: FFT magnitude m_point
- `slip_m_point`: Slip m_point

## 📊 Estratégia de Limpeza
1. **Análise de nulos**: Identifica colunas problemáticas
2. **Remoção conservadora**: Remove apenas colunas com >50% nulos
3. **Preenchimento inteligente**: Usa mediana para preencher nulos
4. **Validação final**: Remove apenas linhas com nulos restantes

## 📈 Qualidade dos Dados
- **Dados mantidos**: 100% das linhas originais
- **Features válidas**: 182 colunas
- **Normalização**: Range [0, 1] uniforme
- **Distribuição**: Preservada após normalização

## 📊 Visualizações Geradas
- **Distribuição Original vs Normalizada**: Box plots comparativos
- **Histogramas**: Antes e depois da normalização
- **Análise de Features**: Primeiras 20 colunas

## ⚠️ Observações Importantes
- Script mantém TODAS as 772k linhas originais
- Remove apenas colunas `m_point` (não necessárias para ML)
- Usa preenchimento com mediana (mais conservador que dropna)
- Gera dados prontos para K-means e outros algoritmos

## 🔄 Fluxo de Trabalho
1. Carregar dados unificados completos
2. Remover colunas m_point
3. Analisar qualidade dos dados
4. Limpar dados conservadoramente
5. Normalizar com MinMaxScaler
6. Preparar formatos para ML
7. Salvar dados e modelos
8. Gerar visualizações

## 📁 Estrutura de Saída
```
data/normalized/
├── dados_kmeans.csv                    # Dados para K-means
├── dados_normalizados_completos.npy    # Dados completos
models/
├── scaler_maxmin.pkl                   # Scaler escolhido
├── preprocess_pipeline.pkl             # Pipeline completo (imputer, scaler, etc.)
└── info_normalizacao.json              # Metadados
plots/
└── dados_normalizados_analise.png      # Visualizações
```

## 🎯 Próximos Passos
Após normalização, os dados estão prontos para:
1. Executar K-means (`kmeans_classificacao.py`)
2. Treinar CNN/ConvAE
3. Análise exploratória
4. Modelagem preditiva

## 📊 Estatísticas Finais
- **Linhas processadas**: 772,238
- **Features normalizadas**: 182
- **Dados mantidos**: 100%
- **Range normalizado**: [0, 1]
- **Tempo de processamento**: ~2-5 minutos


