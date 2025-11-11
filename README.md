# Sistema de Detecção de Estados Operacionais de Equipamentos Industriais

## Visão Geral

Sistema completo para detecção automática de estados operacionais (LIGADO/DESLIGADO) de equipamentos industriais utilizando Machine Learning. Suporta dois tipos de equipamentos:

- **ELÉTRICOS**: Current + RPM + Vibração (motobombas com sensores elétricos)
- **MECÂNICOS**: Temperatura + Vibração (equipamentos sem sensores elétricos)

Combina clustering K-means com processamento avançado de dados de sensores para classificação robusta e confiável.

## Início Rápido com GUI

Para usuários que preferem interface gráfica:

```bash
# Executar GUI (Windows)
double-click no arquivo GUI.bat

# Ou via linha de comando
cd code
python gui_pipeline.py
```

**Funcionalidades da GUI:**
- **Treino**: Detecção automática do tipo de equipamento (ELÉTRICO/MECÂNICO) e treinamento
- **Análise por Intervalo**: Seleção visual de datas com calendário (restrito a 2025+), configuração InfluxDB
- **Visualização 3D**: Gráficos interativos adaptados ao tipo de equipamento

## Índice

- [Funcionalidades Principais](#funcionalidades-principais)
- [Tipos de Equipamentos](#tipos-de-equipamentos)
- [Arquitetura do Sistema](#arquitetura-do-sistema)
- [Fundamentação Teórica](#fundamentação-teórica)
- [Instalação e Configuração](#instalação-e-configuração)
- [Modo de Uso](#modo-de-uso)
- [Interface Gráfica (GUI)](#interface-gráfica-gui)
- [Estrutura de Dados](#estrutura-de-dados)
- [Resultados Obtidos](#resultados-obtidos)
- [Limitações e Considerações](#limitações-e-considerações)
- [Referências Técnicas](#referências-técnicas)

---

## Funcionalidades Principais

### 1. Processamento de Dados Industriais
- **Coleta de Dados**: Integração com InfluxDB para dados de sensores IoT
- **Detecção Automática**: Sistema identifica tipo de equipamento (ELÉTRICO/MECÂNICO)
- **Segmentação Temporal**: Identificação automática de períodos contínuos (gap máximo: 3 horas)
- **Interpolação Adaptativa**: 
  - Gaps < 1h: Spline cúbica (qualidade 0.95)
  - Gaps 1-3h: KNN temporal (qualidade 0.75)
  - Gaps > 3h: Segmentação em períodos distintos
- **Tratamento de Outliers**: Método IQR (multiplicador 3.0) + Clipping percentil 99.5%
- **Normalização**: MinMaxScaler [0,1] unificado para todas as features

### 2. Classificação por Clustering K-Means Versátil
- **Algoritmo**: K-means com k=6 clusters (k-means++, 10 inicializações)
- **Classificação Dinâmica Inteligente**: Sistema de pontuação PONDERADO adaptado ao tipo

#### Equipamentos ELÉTRICOS
  - **Score** = (vel_rms × 1.0) + (current × 2.0) + (rpm × 2.0) + (temperatura × 0.5)
  - **Preferência**: 1 cluster DESLIGADO (conservador)
  - **Verificação Física**: Current < 15% E RPM < 20% → DESLIGADO automático
  - **Thresholds Dinâmicos**: vel_rms (p95×1.2), current (p95×1.3), rpm (p75×1.1 ou 100 fixo)

#### Equipamentos MECÂNICOS
  - **Score** = (temperatura × 1.5) + (vel_rms × 1.0) + (magnetômetro × 0.5)
  - **Preferência**: 1-2 clusters DESLIGADO
  - **Verificação Física**: Temperatura ambiente E vibração residual → DESLIGADO
  - **Thresholds Dinâmicos**: temperatura (p95×1.05), vel_rms (p95×1.3), mag (p95×1.2)
  - **Lógica de % de Tempo**: Intervalo 3D selecionado com 10-90% do tempo desligado

### 3. Análise por Intervalos
- **Validação de Dados**: Verifica continuidade temporal (lacunas < 10 minutos)
- **Requisito Mínimo**: 24 horas de dados
- **Conversão de Timezone**: Suporte GMT-3 → UTC automático
- **Restrições de Data**: Apenas datas entre 01/01/2025 e hoje
- **Cache Local**: Otimização para queries grandes
- **Tratamento de Erros**: Sistema robusto para problemas de conexão

### 4. Visualizações Avançadas
#### Equipamentos ELÉTRICOS
- **Gráficos 3D**: Corrente × Vibração × Tempo, RPM × Vibração × Tempo
- **Seleção Inteligente**: Intervalo de 3 dias com transições de estado

#### Equipamentos MECÂNICOS
- **Gráficos 3D**: Temperatura × Vibração × Tempo
- **Seleção com % de Tempo**: Intervalo de 3 dias baseado em % do cluster desligado
- **Análise de Clusters**: Visualização de centróides e distribuição
- **Relatórios Temporais**: Evolução dos estados operacionais

---

## Tipos de Equipamentos

### Equipamentos ELÉTRICOS

**Identificação**: Possui arquivo `dados_estimated_{mpoint}.csv`

**Sensores Utilizados**:
- Corrente elétrica (current) - Peso 2.0
- Velocidade rotacional (RPM) - Peso 2.0
- Vibração RMS (vel_rms) - Peso 1.0
- Temperatura (object_temp) - Peso 0.5
- Magnetômetro (mag_x/y/z)
- Análise de escorregamento (slip)

**Pipeline**:
```
pipeline_deteccao_estados.py (ELÉTRICO)
├── segmentar_preencher_dados.py
├── processar_dados_simples.py
├── unir_sincronizar_periodos.py
├── normalizar_dados_kmeans.py
├── kmeans_classificacao_moderado.py
├── analise_intervalo_completa.py
└── visualizar_clusters_3d_simples.py
```

**Critério DESLIGADO**:
- Current < 15% normalizado E RPM < 20% normalizado
- OU: Todos abaixo dos thresholds dinâmicos específicos

### Equipamentos MECÂNICOS

**Identificação**: NÃO possui arquivo `dados_estimated_{mpoint}.csv`

**Sensores Utilizados**:
- Temperatura (object_temp) - Peso 1.5
- Vibração RMS (vel_rms) - Peso 1.0
- Magnetômetro (mag_x/y/z) - Peso 0.5
- Análise de escorregamento (slip)

**Pipeline**:
```
pipeline_deteccao_estados_mecanico.py (MECÂNICO)
├── processar_dados_simples_mecanico.py
├── unir_sincronizar_periodos_mecanico.py
├── normalizar_dados_kmeans_mecanico.py
├── kmeans_classificacao_mecanico.py
├── analise_intervalo_completa_mecanico.py
└── visualizar_clusters_3d_mecanico.py
```

**Critério DESLIGADO**:
- Temperatura próxima ao ambiente (< 15% normalizado)
- E Vibração próxima de zero (< 10% normalizado)
- OU: Todos abaixo dos thresholds dinâmicos específicos

---

## Arquitetura do Sistema

### Pipeline de Processamento

```
Dados Brutos → Detecção Tipo → Segmentação → Interpolação → União → 
Normalização → K-means → Classificação → Visualização 3D
```

### Estrutura de Diretórios

```
code/
├── gui_pipeline.py                      # Interface gráfica unificada
├── pipeline_deteccao_estados.py         # Pipeline ELÉTRICO
├── pipeline_deteccao_estados_mecanico.py # Pipeline MECÂNICO
├── scripts/
│   # Scripts ELÉTRICOS
│   ├── baixar_estimated_intervalo.py
│   ├── baixar_validated_default_intervalo.py
│   ├── baixar_validated_slip_intervalo.py
│   ├── segmentar_preencher_dados.py
│   ├── processar_dados_simples.py
│   ├── unir_sincronizar_periodos.py
│   ├── normalizar_dados_kmeans.py
│   ├── kmeans_classificacao_moderado.py
│   ├── analise_intervalo_completa.py
│   ├── visualizar_clusters_3d_simples.py
│   # Scripts MECÂNICOS
│   ├── processar_dados_simples_mecanico.py
│   ├── unir_sincronizar_periodos_mecanico.py
│   ├── normalizar_dados_kmeans_mecanico.py
│   ├── kmeans_classificacao_mecanico.py
│   ├── analise_intervalo_completa_mecanico.py
│   └── visualizar_clusters_3d_mecanico.py
├── utils/
│   ├── artifact_paths.py
│   └── logging_utils.py
├── data/
│   ├── raw/                    # Dados brutos do InfluxDB
│   ├── raw_preenchido/         # Dados processados por período
│   ├── processed/              # Dados unificados
│   └── normalized/             # Dados normalizados
├── models/
│   └── {mpoint}/              # Modelos treinados por equipamento
│       ├── config_{mpoint}.json  (equipment_type: MECHANICAL ou padrão)
│       ├── kmeans_model_moderado_{mpoint}.pkl
│       ├── scaler_model_moderado_{mpoint}.pkl
│       ├── scaler_maxmin_{mpoint}.pkl
│       └── info_kmeans_model_moderado_{mpoint}.json
├── results/
│   └── {mpoint}/              # Resultados, relatórios e visualizações
└── plots/                     # Gráficos de normalização
```

---

## Fundamentação Teórica

### 1. Aprendizado Não Supervisionado (K-means)

O sistema utiliza K-means para agrupar dados sem necessidade de rotulação manual prévia:

- **Vantagem**: Não requer dados históricos rotulados
- **Interpretabilidade**: Identificação de padrões naturais nos dados
- **Escalabilidade**: Processa milhões de registros eficientemente
- **Reprodutibilidade**: Random state fixo (42) garante resultados consistentes

**Configuração do K-means**:
- Número de clusters: 6
- Inicialização: k-means++
- Número de inicializações: 10
- Máximo de iterações: 300

### 2. Classificação Automática de Clusters (Versão Inteligente e Versátil)

Após o clustering, calcula-se um **score combinado PONDERADO** adaptado ao tipo de equipamento:

#### Equipamentos ELÉTRICOS
```python
score = (vel_rms_normalizado × 1.0) + (current_normalizado × 2.0) + 
        (rpm_normalizado × 2.0) + (temperatura_normalizado × 0.5)
```

**Justificativa da Ponderação**:
- **Current e RPM (peso 2.0)**: Sinais mais confiáveis para identificar operação
- **Vibração (peso 1.0)**: Pode ter componentes residuais em equipamento desligado
- **Temperatura (peso 0.5)**: Varia com ambiente, menos confiável

**Critérios de Classificação Dinâmica**:
- Clusters ordenados por score crescente
- **PREFERÊNCIA: 1 cluster DESLIGADO** (mais conservador)
- **Cluster adicional DESLIGADO apenas se**:
  1. Diferença de score < 0.3 E ambos scores < 0.5 E vibrações < 1.5 (residual)
  2. OU 3+ clusters com diferença < 0.2 E todos scores < 0.4

**Verificação Física Obrigatória**:
```python
if current_normalizado < 0.15 AND rpm_normalizado < 0.20:
    cluster → DESLIGADO  # Independente da vibração
```

#### Equipamentos MECÂNICOS
```python
score = (temperatura_normalizado × 1.5) + (vel_rms_normalizado × 1.0) + 
        (magnetometro_normalizado × 0.5)
```

**Justificativa da Ponderação**:
- **Temperatura (peso 1.5)**: Indicador mais confiável para equipamentos mecânicos
- **Vibração (peso 1.0)**: Importante mas pode ter componentes residuais
- **Magnetômetro (peso 0.5)**: Detecta vibrações mínimas

**Critérios de Classificação Dinâmica**:
- Preferência: 1-2 clusters DESLIGADO
- Considera % de tempo desligado para validação

**Verificação Física Obrigatória**:
```python
if temperatura_normalizado < 0.15 AND vel_rms_normalizado < 0.10:
    cluster → DESLIGADO  # Equipamento FISICAMENTE DESLIGADO
```

### 3. Thresholds Dinâmicos por Equipamento

Cada mpoint tem thresholds específicos calculados dinamicamente:

#### ELÉTRICO
- **Vel_RMS**: Percentil 95% × 1.2 (margem 20%)
- **Current**: Percentil 95% × 1.3 (margem 30%)
- **RPM**: Se mediana < 50 → threshold fixo 100 RPM, senão percentil 75% × 1.1
- Salvos como **valores REAIS** (não normalizados) no JSON do equipamento

#### MECÂNICO
- **Temperatura**: Percentil 95% × 1.05 (margem 5%)
- **Vel_RMS**: Percentil 95% × 1.3 (margem 30%)
- **Magnetômetro**: Percentil 95% × 1.2 (margem 20%)
- Salvos como **valores REAIS** no JSON do equipamento

### 4. Tratamento de Dados Temporais

#### Segmentação por Gaps
- **Threshold**: 3 horas
- **Duração mínima**: 24 horas (modo análise)
- Períodos separados por gaps > 3h são processados independentemente

#### Interpolação Adaptativa
- **Gaps < 1 hora**: Interpolação spline cúbica (qualidade 1.00 → 0.95)
- **Gaps 1-3 horas**: KNN temporal + interpolação linear (qualidade 0.75)
- **Gaps > 3 horas**: Segmentação (sem interpolação)

#### Marcação de Qualidade
Cada dado interpolado recebe indicador de confiabilidade:
- **1.00**: Dado original (100% confiável)
- **0.95**: Interpolação simples (<1h)
- **0.75**: Interpolação avançada (1-3h)

### 5. Normalização MinMax com Outlier Clipping

Equaliza escalas de diferentes sensores:

```python
# Passo 1: Clipping de outliers (percentil 99.5%)
dados_clipped = clip_outliers(dados, percentile=99.5)

# Passo 2: Normalização MinMax
x_normalizado = (x - min) / (max - min)
```

**Benefícios do Clipping**:
- Remove valores extremos antes da normalização
- Evita compressão excessiva dos dados normais
- Melhora separação dos clusters

---

## Instalação e Configuração

### Pré-requisitos

```bash
Python 3.11+
pip install -r requirements.txt
```

### Dependências Principais

```bash
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
scipy>=1.11.0
influxdb>=5.3.0
joblib>=1.3.0
customtkinter>=5.2.0  # GUI
tkcalendar>=1.6.0     # GUI
Pillow>=10.0.0        # GUI
```

### Configuração Inicial

1. **Criar estrutura de diretórios** (criada automaticamente ao executar):
```bash
mkdir -p code/data/{raw,raw_preenchido,processed,normalized}
mkdir -p code/models
mkdir -p code/results
mkdir -p code/plots
```

2. **Configurar acesso ao InfluxDB**:
   - IP do servidor InfluxDB (ex: 10.8.0.121)
   - Porta: 8086 (API do InfluxDB)
   - Banco de dados: aihub

3. **Requisitos de Dados**:

   **IMPORTANTE**: Os arquivos de dados para treinamento estarão disponíveis na pasta raw:
   [OneDrive - Pasta Raw](https://1drv.ms/f/c/5f1c3aa3f12af1a2/IgAXz-STR6zmRrccv2fN6JW9Aa4Qy9Yz6RLAjaxAfnqGuj8?e=b2pwI4)

   Baixe os arquivos necessários e coloque na pasta `code/data/raw/` antes de executar o treinamento.

   **Equipamento ELÉTRICO** (3 arquivos):
   - `dados_c_{mpoint}.csv` (validated_default)
   - `dados_estimated_{mpoint}.csv` (current + RPM)
   - `dados_slip_{mpoint}.csv`

   **Equipamento MECÂNICO** (2 arquivos):
   - `dados_c_{mpoint}.csv` (validated_default)
   - `dados_slip_{mpoint}.csv`
   - **NÃO TEM** `dados_estimated_{mpoint}.csv`

---

## Modo de Uso

### Modo Interativo - ELÉTRICO

```bash
cd code
python pipeline_deteccao_estados.py
```

### Modo Interativo - MECÂNICO

```bash
cd code
python pipeline_deteccao_estados_mecanico.py
```

### Modo Linha de Comando

**Treino (detecta tipo automaticamente)**:
```bash
python pipeline_deteccao_estados.py --mpoint c_636 --modo treino --auto
python pipeline_deteccao_estados_mecanico.py --mpoint c_1518 --modo treino --auto
```

**Análise**:
```bash
python pipeline_deteccao_estados.py --mpoint c_636 --modo analise \
  --ip 10.8.0.121 --inicio "2025-01-15 00:00:00" --fim "2025-01-16 00:00:00"
```

---

## Interface Gráfica (GUI)

### Visão Geral

Interface moderna e intuitiva que **detecta automaticamente** o tipo de equipamento e ajusta o processamento.

### Funcionalidades

#### Aba Treino
- **Detecção Automática**: Identifica ELÉTRICO/MECÂNICO pelos arquivos
- **Lista Inteligente**: Mostra tipo [ELÉTRICO] ou [MECÂNICO]
- **Validação Automática**: Verifica arquivos necessários
- **Console Unificado**: Logs em tempo real
- **Abertura Automática**: Abre PNG + TXT ao finalizar

#### Aba Análise por Intervalo
- **Configuração InfluxDB**: IP e porta (padrão: 10.8.0.121:8086)
- **Calendário Restrito**: Apenas datas entre 01/01/2025 e hoje
- **Validações**: Data inicial < data final, não permite futuro
- **Script Automático**: Chama analise_intervalo_completa.py ou _mecanico.py
- **Thresholds Dinâmicos**: Carrega automaticamente do equipamento

#### Aba Visualização 3D
- **Detecção Automática**: Chama script correto (simples.py ou mecanico.py)
- **Gráficos Adaptativos**:
  - ELÉTRICO: Current/RPM × Vibração × Tempo
  - MECÂNICO: Temperatura × Vibração × Tempo
- **Seleção Inteligente**: 3 dias com transições de estado

### Atalhos
- **F11**: Alternar fullscreen
- **ESC**: Sair do fullscreen
- **Fechar janela**: Mata todos os processos filhos automaticamente

---

## Estrutura de Dados

### Entrada (InfluxDB)

#### validated_default (frequência: 20s) - AMBOS
```csv
time,mag_x,mag_y,mag_z,object_temp,vel_max_x,vel_max_y,vel_rms_x,vel_max_z,vel_rms_y,vel_rms_z,m_point
2025-01-15T03:00:00Z,0.123,0.456,0.789,45.2,2.1,1.8,0.95,1.5,0.87,0.92,c_636
```

#### estimated (frequência: 1-2 min) - APENAS ELÉTRICO
```csv
time,rotational_speed,vel_rms,current
2025-01-15T03:00:00Z,1190.5,5.2,350.8
```

#### validated_slip (frequência: 2 min) - AMBOS
```csv
time,fe_frequency,fe_magnitude_-_1,fe_magnitude_0,fe_magnitude_1,fr_frequency,rms
2025-01-15T03:00:00Z,19.8,0.15,0.25,0.18,19.9,0.82
```

### Saída Processada

#### config_{mpoint}.json
```json
{
  "mpoint": "c_1518",
  "equipment_type": "MECHANICAL",  // ou omitido para ELÉTRICO
  "data_sources": ["temperature", "vibration"],  // ou ["current", "rpm", "vibration"]
  "no_current_rpm": true,  // apenas MECÂNICO
  "data_treino": "2025-01-15T10:30:00"
}
```

#### analise_completa_{mpoint}_{timestamp}_resultados.csv
```csv
time,cluster,estado,rotational_speed,current,vel_rms,object_temp,...
2025-01-15T03:00:00Z,0,DESLIGADO,0.0,2.5,0.3,28.5,...
2025-01-15T06:15:00Z,4,LIGADO,1190.2,380.5,8.5,52.3,...
```

---

## Resultados Obtidos

### Equipamentos Testados

**ELÉTRICOS**: c_636, c_637  
**MECÂNICOS**: c_640, c_1518  
**Período de análise**: Janeiro-Outubro 2025  
**Total de registros processados**: > 4 milhões  
**Acurácia de classificação**: 100% (baseado em centróides)

### Distribuição de Estados

#### Equipamento c_636 (ELÉTRICO)
- **DESLIGADO**: 18% do tempo
- **LIGADO**: 82% do tempo
- **Thresholds**: RPM ≤ 100, Current ≤ 30.7A, Vibração ≤ 1.76 mm/s

#### Equipamento c_1518 (MECÂNICO)
- **DESLIGADO**: ~35% do tempo
- **LIGADO**: ~65% do tempo
- **Thresholds**: Temp ≤ 32°C, Vibração ≤ 2.1 mm/s

### Tempo de Processamento

Tempos médios por etapa (Intel i7, 16GB RAM):
- **ELÉTRICO**: 8-12 minutos (treino completo)
- **MECÂNICO**: 6-10 minutos (treino completo)
- Análise de intervalo: 3-10 minutos

---

## Limitações e Considerações

### Limitações Técnicas
- Requer VPN corporativa para acesso a dados em tempo real
- Treinamento específico por equipamento
- Períodos com gaps > 3h são descartados
- Intervalo mínimo de 24 horas para análise
- Datas restritas: 01/01/2025 até hoje

### Limitações por Tipo

#### ELÉTRICO
- Requer dados de corrente e RPM preenchidos
- Não detecta estados intermediários (carga parcial)

#### MECÂNICO
- Depende mais de temperatura ambiente
- Vibrações residuais podem causar ambiguidade
- Menos features disponíveis (sem current/RPM)

---

## Trabalhos Futuros

### Melhorias Técnicas
- Implementação de aprendizado incremental
- Detecção de múltiplos estados (standby, carga parcial)
- Integração com sistemas SCADA
- Otimização para dados em tempo real

### Expansões Funcionais
- Interface web para visualização
- Alertas automáticos (Telegram/email)
- API REST para integração
- Suporte a novos tipos de sensores

---

## Referências Técnicas

### Algoritmos e Métodos
- **K-means clustering** (scikit-learn implementation)
- **MinMax normalization** with outlier clipping
- **Interquartile Range (IQR)** outlier detection
- **Cubic spline interpolation** (scipy)
- **K-Nearest Neighbors temporal interpolation**

### Tecnologias
- **Python 3.11**: Linguagem de programação
- **pandas 2.0+**: Manipulação de dados
- **scikit-learn 1.3+**: Machine Learning
- **matplotlib 3.7+**: Visualização
- **scipy 1.11+**: Computação científica
- **influxdb 5.3+**: Banco de dados temporal
- **customtkinter 5.2+**: Interface gráfica moderna

### Requisitos de Sistema

**Hardware Mínimo**:
- Processador: Intel Core i5 ou equivalente
- Memória RAM: 8GB
- Armazenamento: 10GB livres

**Hardware Recomendado**:
- Processador: Intel Core i7 ou equivalente
- Memória RAM: 16GB ou superior
- Armazenamento: 50GB livres (SSD)

**Software**:
- Sistema Operacional: Windows 10/11, Linux, ou macOS
- Python: 3.11 ou superior
- InfluxDB: Acesso a servidor (leitura)

---

## Aplicações Práticas

### Gestão Energética
- Identificação de padrões de consumo
- Cálculo preciso de tempo de operação
- Otimização de turnos de produção
- Planejamento de demanda energética

### Manutenção Preditiva
- Contabilização precisa de horas de operação
- Identificação de padrões anormais
- Agendamento baseado em tempo real de uso
- Análise de degradação por tempo de operação

### Análise de Eficiência
- Taxa de utilização de equipamentos
- Identificação de períodos ociosos
- Otimização de processos
- Análise comparativa entre equipamentos

---

## Glossário

**Cluster**: Grupo de dados similares identificado pelo algoritmo K-means.

**ELÉTRICO**: Equipamento com sensores de corrente e RPM (possui dados_estimated).

**InfluxDB**: Banco de dados otimizado para séries temporais.

**Interpolação**: Técnica de estimação de valores entre pontos conhecidos.

**K-means**: Algoritmo de clustering não supervisionado.

**MECÂNICO**: Equipamento sem sensores elétricos (NÃO possui dados_estimated).

**MinMax**: Técnica de normalização que escala dados para intervalo [0,1].

**Mpoint**: Identificador único de ponto de monitoramento (equipamento).

**Outlier**: Valor atípico que se afasta significativamente do padrão.

**Pipeline**: Sequência automatizada de etapas de processamento.

**RMS**: Root Mean Square, medida de magnitude de vibração.

**Scaler**: Objeto que armazena parâmetros de normalização.

**Score**: Valor calculado para classificação automática de clusters.

**Segmentação**: Divisão de dados em períodos baseado em gaps temporais.

**Threshold**: Limite dinâmico calculado para classificação de estados.

**Timestamp**: Marca temporal que identifica momento de aquisição de dado.

**UTC**: Tempo Universal Coordenado (referência temporal padrão).

---

## Autor

**Felipe Costa Lopes**  
Matrícula: 2018019648  
Engenharia de Sistemas - UFMG

---

## Licença

Este projeto é parte de um Trabalho de Conclusão de Curso (TCC) da UFMG.
