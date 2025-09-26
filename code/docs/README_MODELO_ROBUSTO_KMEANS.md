# Modelo Robusto CNN + ConvAE para Classificação LIGADO/DESLIGADO

## 📋 Descrição
Modelo robusto que combina CNN (Convolutional Neural Network) e ConvAE (Convolutional Autoencoder) para classificar o status de equipamentos industriais como LIGADO ou DESLIGADO, com capacidade de detectar incerteza nas predições.

## 🎯 Objetivo
- Classificar equipamentos como LIGADO ou DESLIGADO com alta precisão
- Detectar quando o modelo não tem certeza sobre a classificação
- Usar dados rotulados do K-means para treinamento supervisionado
- Aplicar em contexto real com confiança

## 🧠 Arquitetura do Modelo

### 1. ConvAE (Convolutional Autoencoder)
- **Função**: Extração de features e redução de dimensionalidade
- **Encoder**: 3 camadas convolucionais + 2 camadas densas
- **Decoder**: 3 camadas convolucionais + upsampling
- **Dimensão de encoding**: 64 features

### 2. CNN (Convolutional Neural Network)
- **Função**: Classificação LIGADO/DESLIGADO
- **Arquitetura**: 3 camadas convolucionais + 3 camadas densas
- **Dropout**: Para regularização e detecção de incerteza
- **Ativação**: ReLU + Softmax para classificação

### 3. Detecção de Incerteza
- **Método**: Monte Carlo Dropout
- **Processo**: Múltiplas predições com dropout ativo
- **Métrica**: Entropia das predições
- **Threshold**: Incerteza > 0.5 = alta incerteza

## 📊 Dados de Entrada
- **Fonte**: `data/normalized/dados_kmeans_rotulados_conservador.csv`
- **Features**: 19 variáveis normalizadas (0-1)
- **Sequências**: 30 timesteps por amostra
- **Classes**: LIGADO (72.4%) vs DESLIGADO (27.6%)
- **Total**: 93.910 amostras de alta qualidade
- **Estratégia**: Apenas clusters com alta certeza (99.5%+)

### Features Utilizadas
1. `mag_x`, `mag_y`, `mag_z` - Magnitude magnética
2. `object_temp` - Temperatura do objeto
3. `vel_max_x`, `vel_max_y`, `vel_max_z` - Velocidade máxima
4. `vel_rms_x`, `vel_rms_y`, `vel_rms_z` - Velocidade RMS
5. `estimated_current` - Corrente estimada
6. `estimated_rotational_speed` - Velocidade rotacional estimada
7. `estimated_vel_rms` - Velocidade RMS estimada
8. `slip_fe_frequency` - Frequência de escorregamento
9. `slip_fe_magnitude_*` - Magnitude de escorregamento
10. `slip_fr_frequency` - Frequência de escorregamento
11. `slip_rms` - RMS de escorregamento

## 🚀 Como Usar

### Treinamento
```bash
python scripts/treinar_modelo_robusto_kmeans.py
```

### Classificação com Incerteza
```python
import tensorflow as tf
import numpy as np

# Carregar modelo
model = tf.keras.models.load_model('models/cnn_model_robusto.h5')

# Predição com incerteza
def predict_with_uncertainty(model, X, n_samples=100):
    predictions = []
    for _ in range(n_samples):
        pred = model(X, training=True)  # Dropout ativo
        predictions.append(pred)
    
    predictions = np.array(predictions)
    mean_pred = np.mean(predictions, axis=0)
    uncertainty = -np.sum(mean_pred * np.log(mean_pred + 1e-8), axis=1)
    
    return mean_pred, uncertainty

# Exemplo de uso
X_new = np.random.random((1, 30, 19))  # 1 amostra, 30 timesteps, 19 features
mean_pred, uncertainty = predict_with_uncertainty(model, X_new)

if uncertainty[0] > 0.5:
    print("⚠️ Alta incerteza na predição")
else:
    print(f"✅ Predição confiável: {mean_pred[0]}")
```

## 📁 Arquivos Gerados

### Modelos
- `models/convae_model_robusto.h5` - ConvAE completo
- `models/convae_encoder_robusto.h5` - Encoder do ConvAE
- `models/convae_decoder_robusto.h5` - Decoder do ConvAE
- `models/cnn_model_robusto.h5` - CNN para classificação
- `models/cnn_best_robusto.h5` - Melhor CNN (checkpoint)
- `models/label_encoder_robusto.pkl` - Encoder de labels

### Visualizações
- `results/treinamento_modelo_robusto.png` - Gráficos de treinamento

### Metadados
- `models/info_modelo_robusto.json` - Informações do treinamento

## 📈 Performance

### Métricas de Treinamento (Dados Limpos - 100 épocas - FINAL)
- **ConvAE Loss**: 0.0040 (treino) / 0.0040 (validação)
- **CNN Accuracy**: 99.92% (teste)
- **Tempo total**: 2611 segundos (~43 minutos)
- **Estratégia**: Apenas clusters com alta certeza (99.5%+)

### Detecção de Incerteza (FINAL)
- **Amostras com alta incerteza**: 0% (100 amostras testadas)
- **Incerteza média**: 0.0003
- **Incerteza máxima**: 0.0018

### Relatório de Classificação
```
              precision    recall  f1-score   support
   DESLIGADO       1.00      1.00      1.00        43
      LIGADO       1.00      1.00      1.00        57
    accuracy                           1.00       100
```

## 🔧 Configurações

### Parâmetros de Treinamento (FINAL)
- **Épocas**: 100 (completo - ConvAE parou em 99, CNN em 16)
- **Batch Size**: 32
- **Window Size**: 30 timesteps
- **Max Samples per Class**: 25,000 (dados limpos)
- **Test Size**: 20%
- **Dados**: Apenas clusters com alta certeza (99.5%+)
- **Early Stopping**: ConvAE (época 89), CNN (época 1)

### Arquitetura CNN
```python
Conv1D(64, 3, padding='same') + BatchNorm + MaxPool + Dropout(0.2)
Conv1D(128, 3, padding='same') + BatchNorm + MaxPool + Dropout(0.2)
Conv1D(256, 3, padding='same') + BatchNorm + GlobalMaxPool + Dropout(0.3)
Dense(512) + BatchNorm + Dropout(0.4)
Dense(256) + Dropout(0.3)
Dense(128) + Dropout(0.2)
Dense(2, activation='softmax')  # LIGADO/DESLIGADO
```

### Callbacks
- **Early Stopping**: Paciência 15 épocas
- **Reduce LR on Plateau**: Fator 0.5, paciência 8
- **Model Checkpoint**: Salva melhor modelo
- **Terminate on NaN**: Para em caso de erro

## 🎯 Características Especiais

### 1. Detecção de Incerteza
- **Monte Carlo Dropout**: 100 amostras por predição
- **Entropia**: Medida de incerteza
- **Threshold**: 0.5 para alta incerteza
- **Aplicação**: Identificar casos ambíguos

### 2. Arquitetura Robusta
- **Padding 'same'**: Preserva dimensões
- **Batch Normalization**: Estabiliza treinamento
- **Dropout**: Regularização e incerteza
- **Global Max Pooling**: Reduz overfitting

### 3. Dados Balanceados e Limpos
- **Estratégia**: Máximo 25k amostras por classe (dados limpos)
- **Distribuição**: 50% LIGADO / 50% DESLIGADO
- **Sequências**: 30 timesteps por amostra
- **Validação**: 20% para teste
- **Qualidade**: Apenas clusters com alta certeza (99.5%+)

## 🔄 Fluxo de Trabalho

1. **Carregamento**: Dados limpos (apenas clusters de alta certeza)
2. **Balanceamento**: 25k amostras por classe
3. **Sequências**: 30 timesteps por amostra
4. **Treinamento ConvAE**: 100 épocas
5. **Treinamento CNN**: 100 épocas
6. **Validação**: Teste com detecção de incerteza
7. **Salvamento**: Modelos e metadados

## 📊 Vantagens

### 1. Alta Precisão
- **Accuracy**: 99.92% no conjunto de teste
- **Precision/Recall**: 1.00 para ambas as classes
- **F1-Score**: 1.00 para ambas as classes
- **Dados Limpos**: Apenas clusters com alta certeza

### 2. Detecção de Incerteza
- **Monte Carlo Dropout**: Quantifica incerteza
- **Threshold**: Identifica casos ambíguos
- **Aplicação**: Melhora confiabilidade

### 3. Robustez
- **Arquitetura**: CNN + ConvAE combinados
- **Regularização**: Dropout e BatchNorm
- **Dados**: Balanceados e normalizados

### 4. Aplicação Real
- **Dados rotulados**: K-means como ground truth
- **Features relevantes**: Velocidade, corrente, vibração
- **Sequências temporais**: 30 timesteps
- **Produção**: Pronto para deployment

## ⚠️ Limitações

### 1. Dependência de Dados
- **K-means**: Requer dados pré-classificados
- **Normalização**: Dados devem estar normalizados
- **Sequências**: Requer 30 timesteps

### 2. Computacional
- **Treinamento**: ~9 minutos (5 épocas)
- **Inferência**: 100 amostras para incerteza
- **Memória**: Modelo relativamente grande

### 3. Interpretabilidade
- **Black box**: Difícil interpretar decisões
- **Features**: 19 variáveis complexas
- **Sequências**: Padrões temporais

## 🚀 Aplicações

### 1. Monitoramento Industrial
- **Status**: LIGADO/DESLIGADO em tempo real
- **Incerteza**: Identificar casos problemáticos
- **Manutenção**: Predição de falhas

### 2. Controle de Qualidade
- **Classificação**: Automática e precisa
- **Validação**: Detecção de incerteza
- **Relatórios**: Métricas de confiabilidade

### 3. Análise de Dados
- **Padrões**: Identificar comportamentos
- **Tendências**: Análise temporal
- **Alertas**: Casos de alta incerteza

## 📈 Próximos Passos

### 1. Otimizações
- **Quantização**: Reduzir tamanho do modelo
- **Pruning**: Remover pesos desnecessários
- **Distillation**: Modelo menor e mais rápido

### 2. Expansões
- **Mais classes**: Estados intermediários
- **Features**: Adicionar mais variáveis
- **Temporal**: LSTM/GRU para sequências

### 3. Deploy
- **API**: Serviço de classificação
- **Streaming**: Dados em tempo real
- **Dashboard**: Interface visual

## 🔧 Manutenção

### 1. Retreinamento
- **Frequência**: Mensal ou conforme necessário
- **Dados**: Novos dados rotulados
- **Validação**: Teste de performance

### 2. Monitoramento
- **Accuracy**: Acompanhar degradação
- **Incerteza**: Monitorar casos ambíguos
- **Drift**: Detectar mudanças nos dados

### 3. Atualizações
- **Modelo**: Versões mais recentes
- **Features**: Novas variáveis
- **Arquitetura**: Melhorias estruturais

## 📊 Estatísticas do Dataset

- **Total de amostras**: 93,910 (dados limpos)
- **Amostras classificadas**: 93,910 (100%)
- **LIGADO**: 68,014 (72.4%)
- **DESLIGADO**: 25,896 (27.6%)
- **Features**: 19 variáveis
- **Sequências**: 30 timesteps
- **Normalização**: MinMax (0-1)
- **Estratégia**: Apenas clusters com alta certeza (99.5%+)

## 🎯 Conclusão

O modelo robusto CNN + ConvAE oferece uma solução completa para classificação de status de equipamentos industriais, combinando alta precisão com detecção de incerteza. A arquitetura híbrida permite extração eficiente de features e classificação confiável, sendo ideal para aplicações em produção onde a confiabilidade é crítica.

A capacidade de detectar incerteza é particularmente valiosa, permitindo identificar casos onde o modelo não tem confiança suficiente, melhorando a robustez geral do sistema e a confiança dos usuários.
