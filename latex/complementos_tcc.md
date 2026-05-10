# Complementos para o TCC (Arquivos LaTeX)

O usuário solicitou complementar os arquivos LaTeX sem editá-los diretamente, gerando um documento com o que deve ser adicionado. A seguir estão os conteúdos em Markdown formatados para o LaTeX, correspondentes a cada seção do seu TCC.

## 1. Complemento para `Metodologia.tex`
**Objetivo:** Adicionar a estrutura dos dados (tabelas e frequências) e um fluxograma em formato de lista/texto do Pipeline completo.

**Onde inserir:** No final da seção "Visão Geral dos Pipelines" ou "Coleta de Dados e Processamento".

**Texto a complementar (em LaTeX):**

```latex
\subsection{Estrutura e Granularidade dos Dados}
A arquitetura do sistema baseia-se na ingestão de dados provenientes do InfluxDB, originados de sensores IoT com diferentes taxas de amostragem. A estrutura dos dados é categorizada em três fontes principais:

\begin{itemize}
    \item \textbf{Tabela \texttt{validated\_default}:} Fornece a base de alta frequência (amostragem a cada 20 segundos). Contém as features mecânicas fundamentais: vibração máxima e RMS em três eixos (X, Y, Z), temperatura do equipamento e magnitude magnética.
    \item \textbf{Tabela \texttt{estimated} (Exclusivo para Equipamentos Elétricos):} Amostrada a cada 1 a 2 minutos. Fornece features críticas para detecção operacional como Corrente Elétrica (\textit{current}) e Velocidade Rotacional (\textit{rpm}).
    \item \textbf{Tabela \texttt{validated\_slip}:} Amostrada a cada 2 minutos, contendo dados espectrais como frequências de escorregamento mecânico.
\end{itemize}

Devido a essa disparidade nas taxas de amostragem, o sistema executa um \textit{Outer Merge} temporal (baseado na coluna de \textit{timestamp}), alinhando os dados e preparando-os para as etapas subsequentes de tratamento.

\subsection{Fluxo Completo do Pipeline (End-to-End)}
O pipeline completo pode ser visualizado através das seguintes etapas sequenciais operacionais:

\begin{enumerate}
    \item \textbf{Ingestão e Merge Temporal:} Download em lote, conversão de timezone (GMT-3 para UTC) e alinhamento multi-frequência.
    \item \textbf{Segmentação e Isolamento:} Detecção de inatividade de rede. Lacunas maiores que 3 horas quebram a série em múltiplos "períodos contínuos", garantindo que não existam falsas ligações lógicas entre dias distintos.
    \item \textbf{Limpeza de Outliers:} Aplicação do método IQR (multiplicador 3.0). Destaca-se a inovação de "proteção da inércia mecânica": anomalias que ocorrem repetidamente por mais de 10 amostras consecutivas não são filtradas, sendo tratadas como mudança real de estado do equipamento.
    \item \textbf{Interpolação Adaptativa:} Lacunas curtas ($< 1h$) sofrem interpolação \textit{Spline Cúbica}. Lacunas médias (1h a 3h) utilizam uma regressão por \textit{K-Nearest Neighbors (KNN)} ponderado pelo tempo.
    \item \textbf{MinMax Scaler e Clipping:} Normalização de todas as variáveis físicas (RPM, graus Celsius, Amperes, mm/s) para a escala $[0, 1]$, aplicando \textit{clipping} no percentil 99.5\% para evitar a compressão dos dados devido a picos.
    \item \textbf{Classificação Dinâmica (K-Means):} Clusterização em $k=6$. Os centróides gerados recebem um \textit{Score Ponderado} dependente do tipo do equipamento (ex: elétrico prioriza Corrente e RPM com peso 2.0). 
    \item \textbf{Hard Rules de Segurança:} Se a corrente $< 15\%$ nominal e RPM $< 20\%$, força o estado DESLIGADO, anulando falsos positivos da vibração residual.
\end{enumerate}
```

---

## 2. Complemento para `Resultados.tex`
**Objetivo:** Introduzir as animações/gráficos didáticos de KNN, MinMax e KMeans que foram gerados pelo novo script Python, ilustrando didaticamente a eficácia do método.

**Onde inserir:** Criar uma nova subseção após a "Análise de Dados Normalizados" ou dentro de "Visualizações dos Clusters".

**Texto a complementar (em LaTeX):**

```latex
\section{Ilustração Visual do Processamento Matemático}

Para fins didáticos e comprovação da eficácia metodológica, o sistema gera ilustrações visuais utilizando o comportamento do equipamento elétrico $c_{636}$ (Motobomba). Estas ilustrações demonstram o comportamento interno do pipeline diante de flutuações.

\begin{figure}[H]
    \centering
    % Nota: Referencie o caminho gerado pelo script python (plots/animacoes_tcc/01_minmax_scaler.png)
    \includegraphics[width=0.85\textwidth]{plots/animacoes_tcc/01_minmax_scaler.png}
    \caption{Efeito do Scaler MinMax com Clipping. À esquerda, os dados originais mostram a discrepância de escala entre Corrente (0-40A) e Vibração (0-6 mm/s). À direita, as variáveis são unificadas no intervalo [0,1], evitando viés de magnitude no K-Means.}
    \label{fig:minmax_scaler_effect}
\end{figure}

A correção de perdas temporais se provou essencial para evitar ruídos de classificação. A Figura \ref{fig:knn_interpolation} ilustra a atuação do KNN preenchendo lacunas de telemetria baseando-se em vizinhos temporais próximos, respeitando a tendência de subida/descida do estado do equipamento.

\begin{figure}[H]
    \centering
    % Nota: Referencie o caminho gerado pelo script python (plots/animacoes_tcc/02_knn_interpolacao.png)
    \includegraphics[width=0.85\textwidth]{plots/animacoes_tcc/02_knn_interpolacao.png}
    \caption{Ilustração da regressão KNN aplicada a uma falha temporal (gap). A linha de reconstrução acompanha a estabilidade da vibração RMS sem introduzir picos artificiais.}
    \label{fig:knn_interpolation}
\end{figure}

Para garantir a confiabilidade do sinal antes da interpolação e clusterização, o método IQR remove anomalias instantâneas (outliers), conforme detalhado na Figura \ref{fig:remocao_outliers}.

\begin{figure}[H]
    \centering
    % Nota: Referencie o caminho gerado pelo script python (plots/animacoes_tcc/04_remocao_outliers.png)
    \includegraphics[width=0.85\textwidth]{plots/animacoes_tcc/04_remocao_outliers.png}
    \caption{Tratamento de anomalias pelo método IQR. Picos fisicamente impossíveis causados por erros de rede ou de hardware são identificados e removidos antes da etapa de Machine Learning, preservando o envelope real da vibração.}
    \label{fig:remocao_outliers}
\end{figure}

Por fim, o processo de clusterização do K-Means é capaz de separar eficientemente a nuvem de dados. O algoritmo converge os centróides iterativamente em direção aos polos de alta concentração: um polo próximo a zero (Equipamento Desligado) e um polo de alta corrente/vibração (Equipamento Ligado), justificando o sucesso do método não supervisionado.
```
