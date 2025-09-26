# Classificador de Produção com Filtro de Data/Hora

## Visão Geral

O `classificador_producao.py` é um script para classificação em tempo real de dados de sensores, utilizando um modelo CNN robusto treinado com dados rotulados do K-means para determinar se um equipamento está **LIGADO** ou **DESLIGADO**. O script inclui detecção de incerteza usando Monte Carlo Dropout e permite filtragem de dados por range de data e hora, facilitando análises específicas de períodos de interesse.

## Funcionalidades

### ✅ Principais Características

- **Classificação em tempo real** usando modelo CNN robusto pré-treinado
- **Detecção de incerteza** usando Monte Carlo Dropout (100 amostras)
- **Filtro por range de data/hora** para análise de períodos específicos
- **Interface de linha de comando** com argumentos flexíveis
- **Salvamento automático de resultados** em formato CSV
- **Estatísticas detalhadas** da classificação incluindo incerteza
- **Suporte a janelas temporais personalizáveis**
- **Janela padrão de 30 timesteps (10 minutos) para corresponder ao modelo treinado**
- **Suporte a dados já normalizados (dados_kmeans.csv) para melhor performance**
- **Modelo treinado com dados rotulados do K-means** para maior confiabilidade

### 📊 Dados de Entrada

O classificador trabalha com dados normalizados do arquivo `dados_kmeans.csv` que contém:

| Tipo | Descrição | Quantidade |
|------|-----------|------------|
| **Dados Básicos** | `mag_x`, `mag_y`, `mag_z`, `object_temp`, velocidades | 10 features |
| **Features Estimadas** | Corrente, velocidade rotacional, RMS estimados | 3 features |
| **Features Slip** | Frequência e magnitude do slip | 5 features |
| **Metadados** | `time`, `cluster`, `equipamento_status` | 3 colunas |
| **Total** | 19 features + metadados (já normalizados) | 22 colunas |

### 🎯 Classes de Saída

- **LIGADO**: Equipamento em funcionamento
- **DESLIGADO**: Equipamento parado

### ⏱️ Janela Temporal

O classificador usa janelas temporais para analisar sequências de dados:

- **Janela padrão**: 30 timesteps = 10 minutos (assumindo 20 segundos por timestamp)
- **Finalidade**: Capturar padrões temporais que indicam o estado do equipamento
- **Personalização**: Pode ser ajustada via parâmetro `--janela`
- **Importante**: Deve corresponder ao tamanho usado no treinamento do modelo (30 timesteps)

### 🔄 Dados Normalizados

O classificador foi otimizado para trabalhar com dados já normalizados:

- **Arquivo padrão**: `dados_kmeans.csv` (dados normalizados entre 0 e 1)
- **Vantagens**: 
  - Não precisa aplicar normalização durante a classificação
  - Performance melhorada (processamento mais rápido)
  - Consistência com o processo de treinamento
- **Compatibilidade**: Ainda suporta dados não normalizados via método `classificar_arquivo()`

## Instalação e Dependências

### 📋 Pré-requisitos

```bash
# Dependências Python necessárias
pip install pandas numpy tensorflow scikit-learn joblib
```

### 📁 Estrutura de Arquivos Necessários

```
projeto/
├── scripts/
│   └── classificador_producao.py
├── data/normalized/
│   └── dados_kmeans.csv
├── models/
│   ├── cnn_model_robusto.h5                  # Modelo CNN robusto treinado
│   ├── label_encoder_robusto.pkl             # Encoder de labels
│   ├── scaler_maxmin.pkl                     # Normalizador de dados (não usado com dados normalizados)
│   └── info_modelo_robusto.json              # Metadados do modelo
└── results/                      # Diretório para resultados
```

## Uso

### 🚀 Execução Básica

```bash
# Classificar arquivo completo
python scripts/classificador_producao.py

# Mostrar ajuda
python scripts/classificador_producao.py --help
```

### 📅 Filtro por Data/Hora

```bash
# Classificar período específico (formato: "YYYY-MM-DD HH:MM:SS")
python scripts/classificador_producao.py \
    --inicio "2025-02-18 16:30:00" \
    --fim "2025-02-18 17:00:00"

# Exemplo: Analisar período de 30 minutos
python scripts/classificador_producao.py \
    --inicio "2025-02-18 16:30:00" \
    --fim "2025-02-18 17:00:00" \
    --saida results/analise_30min.csv
```

### ⚙️ Configurações Avançadas

```bash
# Janela temporal personalizada (padrão: 30)
python scripts/classificador_producao.py \
    --janela 60 \
    --inicio "2025-02-18 16:30:00" \
    --fim "2025-02-18 16:45:00"

# Arquivo de entrada personalizado
python scripts/classificador_producao.py \
    --arquivo data/raw/meus_dados.csv \
    --inicio "2025-02-18 16:30:00" \
    --fim "2025-02-18 17:00:00"

# Modelos personalizados
python scripts/classificador_producao.py \
    --modelo models/meu_modelo.h5 \
    --label-encoder models/meu_encoder.pkl \
    --scaler models/meu_scaler.pkl
```

## Argumentos da Linha de Comando

| Argumento | Tipo | Padrão | Descrição |
|-----------|------|--------|-----------|
| `--arquivo` | String | `data/normalized/dados_kmeans.csv` | Caminho para o arquivo de dados |
| `--inicio` | String | - | Data/hora de início (formato: "YYYY-MM-DD HH:MM:SS") |
| `--fim` | String | - | Data/hora de fim (formato: "YYYY-MM-DD HH:MM:SS") |
| `--janela` | Integer | 30 | Tamanho da janela temporal (10 minutos) |
| `--saida` | String | Auto-gerado | Caminho para salvar resultados |
| `--modelo` | String | `models/cnn_model_robusto.h5` | Caminho do modelo CNN |
| `--label-encoder` | String | `models/label_encoder_robusto.pkl` | Caminho do label encoder |
| `--scaler` | String | `models/scaler_maxmin.pkl` | Caminho do scaler |

## Saída e Resultados

### 📄 Formato do Arquivo de Saída

O arquivo CSV gerado contém as seguintes colunas:

| Coluna | Descrição |
|--------|-----------|
| `timestamp` | Timestamp da predição |
| `predicao` | Classe predita (LIGADO/DESLIGADO) |
| `prob_ligado` | Probabilidade de estar LIGADO |
| `prob_desligado` | Probabilidade de estar DESLIGADO |
| `incerteza` | Nível de incerteza da predição (0-1) |
| `alta_incerteza` | Boolean indicando se incerteza > 0.5 |

### 📊 Estatísticas Exibidas

Durante a execução, o script exibe:

- Número de linhas carregadas e filtradas
- Percentual de dados mantidos após filtro
- Número de sequências criadas
- Distribuição das predições
- Probabilidades médias
- Estatísticas de incerteza (média, máxima, alta incerteza)
- Confiança da classificação

### 💾 Exemplo de Saída

```csv
timestamp,predicao,prob_ligado,prob_desligado,incerteza,alta_incerteza
2025-02-18T16:32:20Z,LIGADO,0.892,0.108,0.123,False
2025-02-18T16:32:40Z,LIGADO,0.756,0.244,0.456,False
2025-02-18T16:33:00Z,DESLIGADO,0.234,0.766,0.678,True
```

## Exemplos Práticos

### 🔍 Análise de Período de Manutenção

```bash
# Analisar 2 horas antes e depois de uma manutenção
python scripts/classificador_producao.py \
    --inicio "2025-02-18 14:00:00" \
    --fim "2025-02-18 18:00:00" \
    --saida results/analise_manutencao.csv
```

### ⏰ Análise por Turno de Trabalho

```bash
# Turno da manhã (6h às 14h)
python scripts/classificador_producao.py \
    --inicio "2025-02-18 06:00:00" \
    --fim "2025-02-18 14:00:00" \
    --saida results/turno_manha.csv

# Turno da noite (14h às 22h)
python scripts/classificador_producao.py \
    --inicio "2025-02-18 14:00:00" \
    --fim "2025-02-18 22:00:00" \
    --saida results/turno_noite.csv
```

### 📈 Análise de Evento Específico

```bash
# Analisar 30 minutos antes e depois de um evento
python scripts/classificador_producao.py \
    --inicio "2025-02-18 16:15:00" \
    --fim "2025-02-18 17:15:00" \
    --janela 30 \
    --saida results/evento_especifico.csv
```

## Tratamento de Erros

### ⚠️ Possíveis Problemas

1. **Dados insuficientes**: Se o range especificado contém menos dados que a janela temporal
2. **Formato de data inválido**: Data/hora em formato incorreto
3. **Arquivo não encontrado**: Caminho para dados ou modelos incorreto
4. **Features ausentes**: Colunas necessárias não presentes no arquivo

### 🔧 Soluções

- Verifique o formato da data: `"YYYY-MM-DD HH:MM:SS"`
- Certifique-se que o range contém dados suficientes
- Valide se todos os arquivos de modelo existem
- Confirme se o arquivo CSV tem as colunas necessárias

## Integração com Outros Scripts

### 🔗 Fluxo de Trabalho

```bash
# 1. Processar dados brutos
python scripts/unificar_dados_final.py

# 2. Normalizar dados
python scripts/normalizar_dados_kmeans.py

# 3. Treinar modelo
python scripts/treinar_cnn_convae_rotulados.py

# 4. Classificar em produção
python scripts/classificador_producao.py \
    --inicio "2025-02-18 16:30:00" \
    --fim "2025-02-18 17:00:00"
```

## Performance e Otimização

### ⚡ Considerações de Performance

- **Janela temporal**: Valores menores = mais predições, maior processamento
- **Range de dados**: Períodos maiores = mais tempo de processamento
- **Memória**: Arquivos grandes podem exigir mais RAM

### 🎯 Recomendações

- Use janelas de 30 timesteps (padrão) para corresponder ao modelo treinado
- Para análises longas, processe em chunks menores
- Monitore uso de memória com arquivos grandes

## Logs e Debugging

### 📝 Informações de Log

O script fornece logs detalhados incluindo:

- Status do carregamento dos modelos
- Número de dados processados
- Estatísticas de filtragem
- Resultados da classificação
- Tempo de processamento

### 🐛 Modo Debug

Para debug mais detalhado, adicione prints adicionais no código ou use:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contribuição e Desenvolvimento

### 🔄 Próximas Melhorias

- [ ] Suporte a múltiplos arquivos
- [ ] Interface web
- [ ] Visualizações em tempo real
- [ ] Alertas automáticos
- [ ] Integração com bancos de dados

### 📞 Suporte

Para dúvidas ou problemas:
1. Verifique os logs de erro
2. Confirme se todos os arquivos existem
3. Valide o formato dos dados de entrada
4. Teste com ranges menores primeiro

---

**Versão**: 3.0  
**Última atualização**: 2025-09-24  
**Compatibilidade**: Python 3.7+, TensorFlow 2.x

## Histórico de Versões

### v3.0 (2025-09-24)
- ✅ **Corrigido tamanho da janela**: Alterado de 50 para 30 timesteps (10 minutos)
- ✅ **Suporte a dados normalizados**: Novo método `preparar_dados_normalizados()`
- ✅ **Modelos conservadores**: Usa `cnn_model_conservador.h5` por padrão
- ✅ **Detecção automática**: Escolhe método baseado no tipo de dados
- ✅ **Performance melhorada**: Dados já normalizados = processamento mais rápido
- ✅ **Compatibilidade mantida**: Ainda funciona com dados não normalizados

### v2.0 (2025-02-18)
- ✅ Filtro por range de data/hora
- ✅ Interface de linha de comando
- ✅ Salvamento automático de resultados

### v1.0 (2025-02-18)
- ✅ Classificação básica com modelo CNN
