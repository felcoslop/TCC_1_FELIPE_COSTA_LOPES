# Script de Cálculo de Métricas Completas - Versão Avançada

## Visão Geral

Este documento descreve o script `calcular_metricas_completas.py` (versão avançada) que foi desenvolvido para calcular todas as métricas e dados necessários para preencher as seções de Metodologia e Resultados da monografia.

## Objetivo

O script foi criado para:
- **Executar o modelo real** com dados CSV de validação
- **Calcular métricas precisas** via execução do modelo CNN+ConvAE
- **Detectar incerteza real** usando Monte Carlo Dropout (100 amostras)
- **Gerar confusion matrix** e métricas por classe
- **Extrair métricas reais** dos modelos treinados
- **Calcular estatísticas de performance** computacional
- **Gerar dados para LaTeX** nas seções de Metodologia e Resultados
- **Validar informações** apresentadas na monografia

## Dados Disponíveis

### Arquivos de Informações dos Modelos

1. **`info_modelo_robusto.json`**
   - Modelo: CNN + ConvAE Robusto com Detecção de Incerteza
   - Dataset: dados_classificados_kmeans_moderado.csv
   - Window size: 30 timesteps
   - Features: 19
   - Sequências de treino: 39.971
   - Sequências de teste: 9.971
   - Épocas de treinamento: 100
   - Tempo total de treinamento: 2.611 segundos (43,5 minutos)

2. **`info_kmeans_model_moderado.json`**
   - Total de amostras originais: 772.231
   - Amostras classificadas: 93.910 (12,16% dos dados originais)
   - Amostras LIGADO: 68.014
   - Amostras DESLIGADO: 25.896
   - Clusters utilizados: 2 (clusters 2 e 3)
   - Clusters intermediários descartados: 0, 1, 4, 5

3. **`info_normalizacao.json`**
   - Arquivo origem: dados_unificados_final.csv
   - Número de amostras: 772.231
   - Features: 19
   - Tipo de scaler: MinMaxScaler
   - Range de normalização: [0.0, 1.0]
   - Média normalizada: 0.217
   - Desvio padrão normalizado: 0.215

## Novas Funcionalidades da Versão Avançada

### 1. Execução do Modelo Real
- **Carregamento do modelo CNN**: Carrega `models/cnn_model_robusto.h5`
- **Carregamento do label encoder**: Carrega `models/label_encoder_robusto.pkl`
- **Processamento de dados CSV**: Processa `data/normalized/dados_kmeans_rotulados_conservador.csv`
- **Criação de sequências temporais**: Janelas de 30 timesteps
- **Predições reais**: Executa o modelo com dados de validação

### 2. Cálculo de Incerteza Real
- **Monte Carlo Dropout**: 100 amostras por predição
- **Cálculo de entropia**: Medida real de incerteza
- **Identificação de casos ambíguos**: Threshold de 0.5 para alta incerteza
- **Estatísticas de incerteza**: Média, máxima e percentual de alta incerteza

### 3. Métricas Avançadas
- **Confusion Matrix**: Matriz de confusão real
- **Métricas por classe**: Precision, Recall e F1-Score por classe
- **Métricas gerais**: Accuracy, Precision, Recall e F1-Score ponderados
- **Distribuição de classes**: Contagem real de amostras por classe

## Métricas Calculadas

### 1. Métricas de Classificação (Reais)
- **Acurácia**: Calculada via execução do modelo
- **Precision**: Por classe e geral (ponderada)
- **Recall**: Por classe e geral (ponderada)
- **F1-Score**: Por classe e geral (ponderada)
- **Incerteza média**: Via Monte Carlo Dropout
- **Incerteza máxima**: Valor máximo observado
- **Amostras com alta incerteza**: Contagem e percentual

### 2. Métricas de Performance Computacional
- **Tempo de treinamento total**: 43 minutos (2.611 segundos)
- **Tempo ConvAE**: 27 minutos (1.618 segundos)
- **Tempo CNN**: 16,5 minutos (994 segundos)
- **Early stopping**: ConvAE (época 89), CNN (época 16)

### 3. Métricas de Qualidade dos Dados
- **Dados originais**: 772.231 registros
- **Dados para treinamento**: 93.910 amostras (12,2% dos dados originais)
- **Features utilizadas**: 19 variáveis normalizadas (0-1)
- **Janela temporal**: 30 timesteps por amostra
- **Qualidade**: Apenas clusters com alta certeza (99.5%+)

### 4. Métricas do Sistema CNN+ConvAE Robusto
- **Clusters K-Means**: 6 clusters, selecionados apenas 2 com alta certeza
- **Cluster LIGADO**: Cluster 2 (100% de certeza, 68.014 amostras)
- **Cluster DESLIGADO**: Cluster 3 (99.5% de certeza, 25.896 amostras)
- **Features CNN**: 19 features normalizadas para classificação
- **Janela temporal**: 30 timesteps por janela
- **Detecção de incerteza**: Monte Carlo Dropout com 100 amostras

## Arquivos de Saída

### Imagens Disponíveis
1. **`analise_kmeans_clusters_moderado.png`** - Análise dos clusters K-means
2. **`treinamento_modelo_robusto.png`** - Gráficos de treinamento do modelo
3. **`dados_normalizados_analise.png`** - Análise dos dados normalizados

### Dados para LaTeX
O script gera dados formatados para inserção direta nos arquivos .tex:
- Métricas de classificação
- Estatísticas de performance
- Informações sobre qualidade dos dados
- Resultados do sistema híbrido

## Uso do Script

```bash
cd code/scripts
python calcular_metricas_completas.py
```

## Saídas Esperadas

1. **Console**: Métricas calculadas e validadas
2. **Arquivo JSON**: Dados estruturados para LaTeX
3. **Relatório**: Resumo das métricas para a monografia

## Validação dos Dados

O script valida:
- Consistência entre arquivos de informação
- Cálculos de percentuais e proporções
- Métricas de performance
- Qualidade dos dados de treinamento

## Notas Importantes

- Todas as métricas são baseadas em dados reais dos modelos treinados
- Os dados foram validados contra os scripts de treinamento
- As imagens estão disponíveis na pasta `results/` e `plots/`
- Os dados são consistentes com a implementação real do projeto
