# Documentação Técnica: Pipeline de Processamento e Classificação de Dados Industriais

Este documento resume o funcionamento do sistema de detecção de estados operacional (LIGADO/DESLIGADO) para equipamentos industriais (Elétricos e Mecânicos).

## 1. Arquitetura do Pipeline de Dados (End-to-End)

O sistema segue um fluxo contínuo dividido em cinco etapas principais:

1.  **Ingestão de Dados**: Coleta automática via InfluxDB das tabelas `dados_slip`, `dados_validated_default` e `dados_estimated` (corrente/RPM).
2.  **Sincronização Multi-Fonte e Alinhamento**:
    *   Como as tabelas possuem taxas de amostragem distintas, o sistema realiza um **Outer Merge** (união externa) baseado no timestamp (coluna `time`).
    *   **Identificação de Encaixe**: O algoritmo verifica se há um overlap temporal (sobreposição) de pelo menos 70% entre os arquivos para garantir que os dados de vibração e corrente pertençam ao mesmo período operacional.
    *   **Resampling via Interpolação**: Após a união, os "buracos" gerados pelas diferentes frequências de sensores são preenchidos pela lógica de interpolação adaptativa, criando uma linha do tempo unificada onde todas as features estão presentes para cada instante.
3.  **Pré-processamento e Segmentação**:

    *   **Segmentação Temporal**: Identificação automática de gaps. Interrupções > 3h criam novos períodos de análise.
    *   **Tratamento de Outliers**: Aplicação do método IQR (multiplicador 3.0) com **proteção de sequências operacionais** (sequências >= 10 amostras são mantidas para evitar perda de picos reais).
    *   **Clipping**: Limitação de valores extremos no percentil 99.5%.
3.  **Interpolação Adaptativa**:
    *   Gaps < 1h: Interpolados via **Spline Cúbica** (preservação de tendência).
    *   Gaps 1-3h: Preenchidos via **KNN Temporal** (K-Nearest Neighbors com pesos de tempo).
4.  **Normalização**: Utilização de `MinMaxScaler` para converter todas as features (vibração, temperatura, corrente, RPM) para o intervalo [0, 1], garantindo peso uniforme no algoritmo de clustering.
5.  **Clusterização e Classificação**:
    *   Algoritmo **K-Means** (k=6) identifica padrões naturais de funcionamento.
    *   Os clusters são classificados como LIGADO ou DESLIGADO através de um **sistema de pontuação ponderada**.

## 2. Especialização por Tipo de Equipamento

O sistema detecta automaticamente o tipo de ativo e aplica lógicas distintas:

*   **Equipamentos ELÉTRICOS**:
    *   **Features Críticas**: Corrente Elétrica e RPM.
    *   **Score**: `(vel_rms × 1.0) + (corrente × 2.0) + (RPM × 2.0) + (temp × 0.5)`.
    *   **Validação Física**: Mesmo que o K-Means falhe, se `corrente < 15%` e `RPM < 20%` do nominal, o estado é forçado para DESLIGADO.
*   **Equipamentos MECÂNICOS**:
    *   **Features Críticas**: Velocidade RMS (Vibração) e Temperatura.
    *   **Score**: `(temp × 1.5) + (vel_rms × 1.0) + (magnitude × 0.5)`.
    *   **Validação Física**: Baseada na variação de temperatura e níveis de vibração residual.

## 3. Interface de Operação (GUI)

A interface construída em `customtkinter` organiza as funcionalidades em três frentes:

1.  **Aba de Treino**: Configura parâmetros do InfluxDB, seleciona o equipamento e treina o modelo K-Means gerando arquivos de configuração (`.json`) e modelos salvos (`.pkl`).
2.  **Análise de Intervalo**: Permite processar um período histórico específico, aplicando o modelo treinado para gerar relatórios de disponibilidade e gráficos de estado.
3.  **Visualização 3D**: Renderização interativa da distribuição dos clusters em 3D, facilitando a análise visual da separação entre os estados de operação.

## 4. Status do Projeto (80/20)

*   **Concluído (80%)**: Pipeline de dados completo, limpeza robusta de outliers, interpolação inteligente, interface funcional e classificação dinâmica validada para motores elétricos e equipamentos mecânicos.
*   **Em Desenvolvimento (20%)**:
    *   Integração para monitoramento em tempo real (dashboard web).
    *   Refinamento de detecção de estados intermediários (ex: standby ou carga parcial).
    *   Otimização de performance para grandes volumes históricos no InfluxDB.
