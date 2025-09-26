# Script: normalizar_dados_kmeans.py

## 📋 Descrição
Script para normalizar dados completos (772k linhas) removendo colunas `m_point` e preparando dados para algoritmos de machine learning, especialmente K-means.

## 🎯 Objetivo
- Carregar dados unificados completos (772k linhas)
- Remover colunas relacionadas a `m_point`
- Normalizar dados usando MinMaxScaler (0-1)
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
- **Treino/Teste**: `data/normalized/X_train.npy`, `X_test.npy`
- **Scaler**: `models/scaler_maxmin.pkl`
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
- **Estratégia**: Preenchimento com mediana (não remoção)
- Remove apenas colunas com >50% de valores nulos
- Preenche valores nulos com mediana da coluna
- Remove apenas linhas que ainda tenham nulos após preenchimento

### 4. Normalização MinMax
- **Range**: [0, 1]
- **Scaler**: MinMaxScaler do scikit-learn
- **Aplicação**: Todas as features numéricas
- **Preservação**: Mantém distribuição relativa dos dados

### 5. Preparação para ML
- Cria DataFrame para K-means
- Divide dados em treino/teste (80/20)
- Salva dados em formatos otimizados (.npy, .csv)

## 📈 Parâmetros de Normalização
```python
MinMaxScaler(feature_range=(0, 1))
```

## 🚀 Como Usar

```bash
python scripts/normalizar_dados_kmeans.py
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
  - Colunas removidas: 0
  - Preenchendo valores nulos com mediana...
  - Linhas após limpeza: 772,238
  - Percentual de dados mantidos: 100.0%

Normalizando dados com Max-Min Scaler...
  - Shape: (772238, 182)
  - Range: [0.000000, 1.000000]
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
├── X_train.npy                         # Dados de treino
└── X_test.npy                          # Dados de teste
models/
├── scaler_maxmin.pkl                   # Scaler MinMax
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


