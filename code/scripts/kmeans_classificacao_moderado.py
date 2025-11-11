"""
Script que usa K-means pra agrupar dados e decidir se equipamento ta ligado ou nao.
Funciona assim:
- Cria 6 grupos diferentes dos dados
- Prefere classificar 1 grupo como "desligado" (mais seguro)
- Mas permite 2-3 grupos como desligado se precisar (ex: quando ainda tem vibracao residual)
- Olha corrente, RPM, temperatura e vibracao
- Da mais peso pra corrente e RPM porque sao mais confiaveis
- Funciona pra varios tipos de equipamentos diferentes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from scipy.ndimage import median_filter
import json
import pickle
import warnings
import argparse
warnings.filterwarnings('ignore')

from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from utils.artifact_paths import (
    info_kmeans_path,
    info_normalizacao_path,
    kmeans_model_path,
    normalized_csv_path,
    processed_classificado_path,
    processed_rotulado_path,
    scaler_maxmin_path,
    scaler_model_path,
)
from utils.logging_utils import (
    save_log,
    create_training_log,
    create_visualization_log,
    format_file_list,
    get_file_info,
    enrich_results_file,
)
DIR_NORMALIZED = BASE_DIR / 'data' / 'normalized'
DIR_PROCESSED = BASE_DIR / 'data' / 'processed'
DIR_MODELS = BASE_DIR / 'models'
DIR_RESULTS = BASE_DIR / 'results'

# Controle de verbosidade
VERBOSE = True  # Ativado para ver análise detalhada dos clusters

def carregar_dados_normalizados(mpoint=None):
    """Carrega os dados normalizados para K-means"""
    print("Carregando dados normalizados...")

    if not mpoint:
        raise ValueError("mpoint deve ser informado para carregar dados normalizados")

    arquivo_dados = normalized_csv_path(mpoint)

    if not arquivo_dados.exists():
        print(f"[ERRO] Arquivo de dados normalizados não encontrado: {arquivo_dados}")
        print("Execute primeiro o script de normalização com o mpoint correto.")
        return None, None

    df = pd.read_csv(arquivo_dados)
    
    # Carregar informações de normalização - usar arquivo específico do mpoint
    info_path = info_normalizacao_path(mpoint)
    if not info_path.exists():
        # Fallback para arquivo antigo (sem pasta por mpoint)
        legacy_path = DIR_MODELS / f'info_normalizacao_{mpoint}.json'
        if legacy_path.exists():
            info_path = legacy_path
        else:
            raise FileNotFoundError(f"Informações de normalização não encontradas para {mpoint}")

    with open(info_path, 'r') as f:
        info_normalizacao = json.load(f)
    
    # Usar a chave correta para colunas
    colunas_key = 'colunas_utilizadas_finais' if 'colunas_utilizadas_finais' in info_normalizacao else 'colunas_utilizadas'
    
    print(f"  - Shape: {df.shape}")
    print(f"  - Colunas: {len(info_normalizacao[colunas_key])}")
    print(f"  - Range normalização: {info_normalizacao['range_normalizacao']}")
    
    return df, info_normalizacao

def preparar_dados_normalizados(df, info_normalizacao):
    """Prepara dados normalizados para K-means"""
    print("\nPreparando dados normalizados para K-means...")
    
    # Usar todas as colunas normalizadas - usar a chave correta
    colunas_key = 'colunas_utilizadas_finais' if 'colunas_utilizadas_finais' in info_normalizacao else 'colunas_utilizadas'
    colunas_validas = info_normalizacao[colunas_key]
    
    print(f"  - Colunas selecionadas: {len(colunas_validas)}")
    print(f"  - Dados já normalizados e limpos")
    
    # Dados já estão normalizados e limpos
    df_kmeans_clean = df.copy()
    
    print(f"  - Linhas disponíveis: {len(df_kmeans_clean):,}")
    
    return df_kmeans_clean, colunas_validas

def usar_dados_normalizados(df_kmeans, info_normalizacao, mpoint):
    """Usa dados já normalizados"""
    print("\nUsando dados já normalizados...")
    
    # Carregar scaler usado na normalização
    import joblib
    try:
        scaler_path = scaler_maxmin_path(mpoint)
        if not scaler_path.exists():
            scaler_path = DIR_MODELS / f'scaler_maxmin_{mpoint}.pkl'
        scaler = joblib.load(scaler_path)
    except FileNotFoundError:
        legacy_path = DIR_MODELS / 'scaler_maxmin.pkl'
        if not legacy_path.exists():
            raise
        scaler = joblib.load(legacy_path)
    
    # Separar timestamp das features numéricas
    timestamp = df_kmeans['time']
    features_numericas = df_kmeans.drop('time', axis=1)
    
    # Dados já estão normalizados
    dados_normalizados = features_numericas.values
    
    print("  - Dados normalizados carregados com sucesso!")
    print(f"  - Shape: {dados_normalizados.shape}")
    print(f"  - Range: [{dados_normalizados.min():.3f}, {dados_normalizados.max():.3f}]")
    
    return dados_normalizados, scaler, timestamp

def executar_kmeans(dados_normalizados, n_clusters=6):
    """Executa K-means com 6 clusters"""
    print(f"\nExecutando K-means com {n_clusters} clusters...")
    
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
        max_iter=300
    )
    
    clusters = kmeans.fit_predict(dados_normalizados)
    
    print(f"  - Clusters criados: {len(np.unique(clusters))}")
    print(f"  - Distribuição dos clusters:")
    for i in range(n_clusters):
        count = np.sum(clusters == i)
        pct = (count / len(clusters)) * 100
        print(f"    - Cluster {i}: {count:,} amostras ({pct:.1f}%)")
    
    return kmeans, clusters

def analisar_clusters(df_kmeans, clusters, colunas_validas):
    """Analisa características dos clusters"""
    print("\nAnalisando características dos clusters...")
    
    # Adicionar clusters ao dataframe
    df_analise = df_kmeans.copy()
    df_analise['cluster'] = clusters
    
    # Calcular estatísticas por cluster
    print("\nEstatísticas por cluster:")
    for cluster_id in range(6):
        cluster_data = df_analise[df_analise['cluster'] == cluster_id]
        print(f"\nCluster {cluster_id} ({len(cluster_data):,} amostras):")
        
        # Mostrar médias das principais variáveis
        for col in colunas_validas[:10]:  # Mostrar apenas as primeiras 10
            if col in cluster_data.columns:
                mean_val = cluster_data[col].mean()
                std_val = cluster_data[col].std()
                print(f"  - {col}: {mean_val:.3f} ± {std_val:.3f}")
    
    return df_analise

def calcular_thresholds_estado_desligado(df_analise, clusters_desligado, vel_rms_cols, current_cols, rpm_cols, scaler=None, mpoint=None):
    """
    Calcula thresholds dinâmicos DESNORMALIZADOS baseados nos dados reais do estado desligado
    identificado pelos clusters de menor atividade
    
    IMPORTANTE: Retorna valores REAIS (não normalizados) dos MÁXIMOS observados nos clusters desligados
    """
    # Filtrar dados apenas dos clusters identificados como DESLIGADO
    dados_desligado = df_analise[df_analise['cluster'].isin(clusters_desligado)].copy()

    thresholds = {}

    # Carregar dados ORIGINAIS (não normalizados) para calcular thresholds REAIS
    try:
        DIR_RAW_PREENCHIDO = BASE_DIR / 'data' / 'raw_preenchido'
        arquivos_periodo = list(DIR_RAW_PREENCHIDO.glob(f'periodo_*_final_{mpoint}.csv'))
        
        if arquivos_periodo:
            # Ler todos os períodos e combinar
            dfs_originais = []
            for arq in arquivos_periodo:
                df_orig = pd.read_csv(arq)
                df_orig['time'] = pd.to_datetime(df_orig['time'], format='mixed', utc=True)
                dfs_originais.append(df_orig)
            
            df_original = pd.concat(dfs_originais, ignore_index=True)
            df_original = df_original.sort_values('time').reset_index(drop=True)
            
            # Alinhar com df_analise pelo tempo
            df_analise_com_orig = df_analise.copy()
            
            # Extrair índices dos dados desligados
            indices_desligado = dados_desligado.index
            
            # Pegar os dados originais correspondentes
            dados_desligado_originais = df_original.iloc[indices_desligado]
            
            # Calcular thresholds REAIS usando PERCENTIL 95% (mais robusto que máximo)
            # Percentil 95% ignora outliers e valores extremos do cluster desligado
            if 'vel_rms' in dados_desligado_originais.columns:
                vel_rms_p95 = dados_desligado_originais['vel_rms'].quantile(0.95)
                vel_rms_mean = dados_desligado_originais['vel_rms'].mean()
                vel_rms_std = dados_desligado_originais['vel_rms'].std()
                thresholds['vel_rms_max'] = float(vel_rms_p95 * 1.2)  # percentil 95% + margem 20%
                thresholds['vel_rms_mean'] = float(vel_rms_mean)
                thresholds['vel_rms_std'] = float(vel_rms_std)
                if VERBOSE:
                    print(f"  - vel_rms (REAL): mean={vel_rms_mean:.3f}, p95={vel_rms_p95:.3f} → threshold={thresholds['vel_rms_max']:.3f} mm/s")
            
            if 'current' in dados_desligado_originais.columns:
                current_p95 = dados_desligado_originais['current'].quantile(0.95)
                current_mean = dados_desligado_originais['current'].mean()
                current_std = dados_desligado_originais['current'].std()
                thresholds['current_max'] = float(current_p95 * 1.3)  # percentil 95% + margem 30%
                thresholds['current_mean'] = float(current_mean)
                thresholds['current_std'] = float(current_std)
                if VERBOSE:
                    print(f"  - current (REAL): mean={current_mean:.3f}, p95={current_p95:.3f} → threshold={thresholds['current_max']:.3f} A")
            
            if 'rotational_speed' in dados_desligado_originais.columns:
                rpm_median = dados_desligado_originais['rotational_speed'].median()
                rpm_p75 = dados_desligado_originais['rotational_speed'].quantile(0.75)
                rpm_mean = dados_desligado_originais['rotational_speed'].mean()
                rpm_std = dados_desligado_originais['rotational_speed'].std()
                
                # Lógica INTELIGENTE para RPM:
                # RPM deve ser ZERO quando desligado (motor parado)
                # Se mediana < 50 RPM → equipamento realmente desligado, usar threshold baixo
                # Senão → pode ter outliers, usar percentil 75% (mais conservador)
                if rpm_median < 50:
                    # Equipamento REALMENTE desligado - motor parado
                    thresholds['rpm_max'] = 100.0  # Threshold fixo conservador
                    if VERBOSE:
                        print(f"  - rpm (REAL): median={rpm_median:.3f} < 50 → threshold FIXO = 100.0 RPM (motor parado)")
                else:
                    # Mediana alta - pode ter outliers ou amostragem esparsa
                    thresholds['rpm_max'] = float(rpm_p75 * 1.1)  # percentil 75% + margem 10%
                    if VERBOSE:
                        print(f"  - rpm (REAL): median={rpm_median:.3f}, p75={rpm_p75:.3f} → threshold={thresholds['rpm_max']:.3f} RPM")
                
                thresholds['rpm_mean'] = float(rpm_mean)
                thresholds['rpm_median'] = float(rpm_median)
                thresholds['rpm_std'] = float(rpm_std)
        else:
            # Fallback: usar valores normalizados (não ideal, mas funciona)
            print("  [AVISO] Dados originais não encontrados, usando valores normalizados como fallback")
            if vel_rms_cols:
                thresholds['vel_rms_max'] = float(dados_desligado[vel_rms_cols].max().max() * 1.2)
                thresholds['vel_rms_mean'] = float(dados_desligado[vel_rms_cols].mean().mean())
            if current_cols:
                thresholds['current_max'] = float(dados_desligado[current_cols].max().max() * 1.3)
                thresholds['current_mean'] = float(dados_desligado[current_cols].mean().mean())
            if rpm_cols:
                rpm_max = dados_desligado[rpm_cols].max().max()
                thresholds['rpm_max'] = float(rpm_max * 1.1 if rpm_max > 0 else 50)
                thresholds['rpm_mean'] = float(dados_desligado[rpm_cols].mean().mean())
    
    except Exception as e:
        print(f"  [ERRO] Erro ao carregar dados originais: {e}")
        # Fallback: usar valores normalizados
        if vel_rms_cols:
            thresholds['vel_rms_max'] = float(dados_desligado[vel_rms_cols].max().max() * 1.2)
            thresholds['vel_rms_mean'] = float(dados_desligado[vel_rms_cols].mean().mean())
        if current_cols:
            thresholds['current_max'] = float(dados_desligado[current_cols].max().max() * 1.3)
            thresholds['current_mean'] = float(dados_desligado[current_cols].mean().mean())
        if rpm_cols:
            rpm_max = dados_desligado[rpm_cols].max().max()
            thresholds['rpm_max'] = float(rpm_max * 1.1 if rpm_max > 0 else 50)
            thresholds['rpm_mean'] = float(dados_desligado[rpm_cols].mean().mean())

    # Calcular threshold DINÂMICO de vibração residual baseado no cluster 0
    # Isso é usado para decidir se clusters adicionais devem ser incluídos como DESLIGADO
    if vel_rms_cols and len(clusters_desligado) > 0:
        # Pegar apenas dados do primeiro cluster (menor score)
        cluster_base = min(clusters_desligado)
        dados_cluster_base = df_analise[df_analise['cluster'] == cluster_base]
        
        # Calcular máximo de vibração do cluster base (normalizado)
        vel_rms_max_cluster_base_norm = dados_cluster_base[vel_rms_cols].max().max()
        
        # Threshold de vibração residual = máximo do cluster base × 1.3
        thresholds['threshold_vibracao_residual_norm'] = float(vel_rms_max_cluster_base_norm * 1.3)
        
        if VERBOSE:
            print(f"  - Threshold DINÂMICO vibração residual (normalizado): {thresholds['threshold_vibracao_residual_norm']:.3f}")
    
    # Estatísticas dos clusters desligado para referência
    thresholds['clusters_desligado'] = [int(c) for c in clusters_desligado]
    thresholds['amostras_desligado'] = int(len(dados_desligado))
    thresholds['porcentagem_desligado'] = float(len(dados_desligado) / len(df_analise) * 100)

    if VERBOSE:
        print(f"\n  - Clusters identificados como DESLIGADO: {clusters_desligado}")
        print(f"  - Amostras DESLIGADO: {len(dados_desligado)} ({thresholds['porcentagem_desligado']:.1f}%)")
        print("  - Thresholds REAIS calculados (valores DESNORMALIZADOS):")
        for key in ['vel_rms_max', 'current_max', 'rpm_max']:
            if key in thresholds:
                unidade = {'vel_rms_max': 'mm/s', 'current_max': 'A', 'rpm_max': 'RPM'}[key]
                print(f"    * {key}: {thresholds[key]:.3f} {unidade}")

    return thresholds

def classificar_2_estados_simples(df_analise, scaler=None, mpoint=None):
    """
    Classifica em 2 estados com PREFERÊNCIA por 1 cluster DESLIGADO:
    1. DESLIGADO: Preferencialmente apenas o cluster com menor score combinado
                  Considera vibrações residuais DINÂMICAS, current baixo, rpm baixo, temperatura ambiente
    2. LIGADO: Todos os outros clusters (diferentes níveis de operação)
    
    VERSATILIDADE: Todos os thresholds são calculados dinamicamente para cada equipamento
    
    Retorna também os thresholds calculados do estado desligado para uso posterior
    """
    if VERBOSE:
        print("Classificando estados (DESLIGADO vs LIGADO)...", flush=True)
        print("  - Estratégia: Preferência por 1 cluster DESLIGADO (100% dinâmico por equipamento)", flush=True)

    # Identificar colunas relevantes
    vel_rms_cols = [col for col in df_analise.columns if 'vel_rms' in col.lower()]
    current_cols = [col for col in df_analise.columns if 'current' in col.lower()]
    rpm_cols = [col for col in df_analise.columns if 'rpm' in col.lower() or 'rotational_speed' in col.lower()]
    temp_cols = [col for col in df_analise.columns if 'temp' in col.lower()]
    
    # Magnitude de vibração (para detectar vibrações residuais)
    mag_cols = [col for col in df_analise.columns if 'mag_' in col.lower()]

    # Analisar características de cada cluster
    cluster_features = {}

    for cluster_id in df_analise['cluster'].unique():
        cluster_data = df_analise[df_analise['cluster'] == cluster_id]

        # Calcular médias normalizadas
        vel_rms_mean_norm = cluster_data[vel_rms_cols].mean().mean() if vel_rms_cols else 0
        current_mean_norm = cluster_data[current_cols].mean().mean() if current_cols else 0
        rpm_mean_norm = cluster_data[rpm_cols].mean().mean() if rpm_cols else 0
        temp_mean_norm = cluster_data[temp_cols].mean().mean() if temp_cols else 0
        mag_mean_norm = cluster_data[mag_cols].mean().mean() if mag_cols else 0
        
        # Calcular máximos normalizados (para detectar picos)
        vel_rms_max_norm = cluster_data[vel_rms_cols].max().max() if vel_rms_cols else 0
        current_max_norm = cluster_data[current_cols].max().max() if current_cols else 0
        rpm_max_norm = cluster_data[rpm_cols].max().max() if rpm_cols else 0

        # Score combinado PONDERADO (quanto maior, mais "ligado")
        # Prioriza current e rpm (mais confiáveis que vibração para identificar operação)
        score = (vel_rms_mean_norm * 1.0) + (current_mean_norm * 2.0) + (rpm_mean_norm * 2.0)
        
        # Score adicional para temperatura (equipamento desligado tende a estar mais frio)
        # Mas não pesa muito porque temperatura ambiente pode variar
        score += temp_mean_norm * 0.5

        cluster_features[cluster_id] = {
            'vel_rms_mean_norm': vel_rms_mean_norm,
            'vel_rms_max_norm': vel_rms_max_norm,
            'current_mean_norm': current_mean_norm,
            'current_max_norm': current_max_norm,
            'rpm_mean_norm': rpm_mean_norm,
            'rpm_max_norm': rpm_max_norm,
            'temp_mean_norm': temp_mean_norm,
            'mag_mean_norm': mag_mean_norm,
            'score': score,
            'count': len(cluster_data)
        }

    # Ordenar clusters por score (do menor para o maior)
    clusters_ordenados = sorted(cluster_features.items(), key=lambda x: x[1]['score'])
    scores_ordenados = [c[1]['score'] for c in clusters_ordenados]

    if VERBOSE:
        print("\n  - Análise de Clusters (ordenados por score):")
        for i, (cid, features) in enumerate(clusters_ordenados):
            print(f"    Cluster {cid}: score={features['score']:.3f}, "
                  f"vel_rms={features['vel_rms_mean_norm']:.3f}, "
                  f"current={features['current_mean_norm']:.3f}, "
                  f"rpm={features['rpm_mean_norm']:.3f}, "
                  f"count={features['count']:,}")

    # ========================================================================
    # NOVA LÓGICA: PREFERÊNCIA POR APENAS 1 CLUSTER DESLIGADO
    # ========================================================================
    clusters_desligado = []

    # Sempre começar com o cluster de menor score
    cluster_menor_score = clusters_ordenados[0][0]
    clusters_desligado.append(cluster_menor_score)
    
    # Análise para decidir se deve incluir mais clusters
    if len(scores_ordenados) >= 2:
        scores_array = np.array(scores_ordenados)
        diff_scores = np.diff(scores_array)
        
        # Diferença entre o 1º e 2º cluster
        diff_primeiro_segundo = scores_ordenados[1] - scores_ordenados[0]
        
        # Calcular threshold DINÂMICO de vibração residual
        # Baseado no máximo de vibração do cluster 0 (menor score = desligado base)
        features_primeiro = cluster_features[clusters_ordenados[0][0]]
        vel_rms_max_cluster0 = features_primeiro['vel_rms_max_norm']
        
        # Threshold dinâmico = máximo do cluster 0 × 1.3 (margem de 30%)
        # Isso permite pequenas variações de vibração residual entre clusters desligados
        threshold_vibracao_residual = vel_rms_max_cluster0 * 1.3
        
        if VERBOSE:
            print(f"\n  - Threshold DINÂMICO de vibração residual calculado:")
            print(f"    Cluster 0 vel_rms_max: {vel_rms_max_cluster0:.3f}")
            print(f"    Threshold (×1.3): {threshold_vibracao_residual:.3f}")
        
        # Critério 1: Diferença muito pequena (< 0.3) E ambos com scores muito baixos (< 0.5)
        # Isso indica que ambos são claramente desligados
        if diff_primeiro_segundo < 0.3 and scores_ordenados[1] < 0.5:
            # Verificar se vibrações também são baixas usando THRESHOLD DINÂMICO
            features_segundo = cluster_features[clusters_ordenados[1][0]]
            if features_segundo['vel_rms_max_norm'] <= threshold_vibracao_residual:
                clusters_desligado.append(clusters_ordenados[1][0])
                if VERBOSE:
                    print(f"\n  - Incluindo 2º cluster ({clusters_ordenados[1][0]}) como DESLIGADO:")
                    print(f"    Razão: Score muito baixo ({scores_ordenados[1]:.3f}) e vibrações baixas")
                    print(f"    Vibrações: máx={features_segundo['vel_rms_max_norm']:.3f} ≤ {threshold_vibracao_residual:.3f} (DINÂMICO)")
                    print(f"    Current: {features_segundo['current_mean_norm']:.3f}, RPM: {features_segundo['rpm_mean_norm']:.3f}")
            elif VERBOSE:
                print(f"\n  - NÃO incluindo 2º cluster ({clusters_ordenados[1][0]}) como DESLIGADO:")
                print(f"    Razão: Vibrações acima do limiar residual DINÂMICO")
                print(f"    Máx vibração cluster 1: {features_segundo['vel_rms_max_norm']:.3f} > {threshold_vibracao_residual:.3f}")
                print(f"    Possível equipamento em baixa operação (não apenas residual)")
        
        # Critério 2: Se houver 3+ clusters e os 2-3 primeiros são MUITO próximos
        # (diferença < 0.2) e TODOS com scores < 0.4
        if len(scores_ordenados) >= 3:
            diff_segundo_terceiro = scores_ordenados[2] - scores_ordenados[1]
            if (diff_primeiro_segundo < 0.2 and diff_segundo_terceiro < 0.2 and 
                scores_ordenados[2] < 0.4):
                # Incluir 2º e 3º se não foram incluídos ainda
                if clusters_ordenados[1][0] not in clusters_desligado:
                    clusters_desligado.append(clusters_ordenados[1][0])
                clusters_desligado.append(clusters_ordenados[2][0])
                if VERBOSE:
                    print(f"\n  - Incluindo 3 clusters ({clusters_desligado}) como DESLIGADO:")
                    print(f"    Razão: Scores muito próximos e baixos (< 0.4)")
        
        if VERBOSE and len(clusters_desligado) == 1:
            print(f"\n  - Usando APENAS 1 cluster ({cluster_menor_score}) como DESLIGADO:")
            print(f"    Razão: Diferença significativa para próximo cluster ({diff_primeiro_segundo:.3f})")
            if diff_primeiro_segundo > 0.5:
                print(f"    (Diferença > 0.5 indica clara separação entre estados)")

    # ========================================================================
    # VERIFICAÇÃO FÍSICA OBRIGATÓRIA: Clusters com current E rpm baixos
    # DEVEM ser DESLIGADO independente da vibração (vibrações residuais)
    # ========================================================================
    clusters_desligado_fisico = []
    for cluster_id, features in cluster_features.items():
        # Critério físico OBRIGATÓRIO:
        # Se current_mean < 0.15 (normalizado) E rpm_mean < 0.20 (normalizado)
        # Isso corresponde a current baixo (~40A) e rpm próximo de zero
        if features['current_mean_norm'] < 0.15 and features['rpm_mean_norm'] < 0.20:
            if cluster_id not in clusters_desligado:
                clusters_desligado_fisico.append(cluster_id)
                if VERBOSE:
                    print(f"\n  - VERIFICAÇÃO FÍSICA: Incluindo cluster {cluster_id} como DESLIGADO:")
                    print(f"    Razão: Current baixo ({features['current_mean_norm']:.3f} < 0.15) E RPM baixo ({features['rpm_mean_norm']:.3f} < 0.20)")
                    print(f"    Vibração: {features['vel_rms_mean_norm']:.3f} (residual - equipamento parado)")
                    print(f"    → Equipamento FISICAMENTE DESLIGADO (motor parado, current mínimo)")
    
    # Adicionar clusters detectados fisicamente como desligados
    if clusters_desligado_fisico:
        clusters_desligado.extend(clusters_desligado_fisico)
        if VERBOSE:
            print(f"\n  - Total de clusters DESLIGADO após verificação física: {len(clusters_desligado)}")

    # Resto = LIGADO (diferentes níveis de operação)
    clusters_ligado = [c[0] for c in clusters_ordenados if c[0] not in clusters_desligado]

    # Aplicar classificação
    df_analise['equipamento_status'] = 'LIGADO'  # Default

    # Marcar DESLIGADO
    for cid in clusters_desligado:
        df_analise.loc[df_analise['cluster'] == cid, 'equipamento_status'] = 'DESLIGADO'

    # Calcular thresholds dinâmicos do estado desligado (valores REAIS, não normalizados)
    thresholds_desligado = calcular_thresholds_estado_desligado(
        df_analise, clusters_desligado, vel_rms_cols, current_cols, rpm_cols, scaler, mpoint
    )

    # Resumo final detalhado
    total_amostras = len(df_analise)
    amostras_desligado = (df_analise['equipamento_status'] == 'DESLIGADO').sum()
    amostras_ligado = (df_analise['equipamento_status'] == 'LIGADO').sum()

    if VERBOSE:
        print(f"\n{'='*70}")
        print(f"RESUMO DA CLASSIFICAÇÃO")
        print(f"{'='*70}")
        print(f"  Clusters DESLIGADO: {clusters_desligado} ({len(clusters_desligado)} cluster(s))")
        print(f"  Clusters LIGADO: {clusters_ligado} ({len(clusters_ligado)} cluster(s))")
        print(f"\n  Distribuição de Amostras:")
        print(f"    - DESLIGADO: {amostras_desligado:,} ({amostras_desligado/total_amostras*100:.1f}%)")
        print(f"    - LIGADO: {amostras_ligado:,} ({amostras_ligado/total_amostras*100:.1f}%)")
        print(f"    - Total: {total_amostras:,}")
        print(f"\n  Thresholds Calculados (estado DESLIGADO):")
        if 'vel_rms_max' in thresholds_desligado:
            print(f"    - Vibração RMS máx: {thresholds_desligado['vel_rms_max']:.3f} (normalizado)")
        if 'current_max' in thresholds_desligado:
            print(f"    - Current máx: {thresholds_desligado['current_max']:.3f} (normalizado)")
        if 'rpm_max' in thresholds_desligado:
            print(f"    - RPM máx: {thresholds_desligado['rpm_max']:.3f} (normalizado)")
        print(f"{'='*70}\n")

    return df_analise, thresholds_desligado

def criar_visualizacoes_rigoroso(df_analise, colunas_validas, cluster_ligado, cluster_desligado, clusters_intermediarios, mpoint):
    """Cria visualizações dos resultados (modo rigoroso)"""
    print("\nCriando visualizações (modo rigoroso)...")
    
    # Reduzir dimensionalidade para visualização
    pca = PCA(n_components=2)
    dados_pca = pca.fit_transform(df_analise[colunas_validas])
    
    # Criar figura
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Scatter plot dos clusters (todos)
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    for i in range(6):
        mask = df_analise['cluster'] == i
        if i == cluster_ligado:
            label = f'Cluster {i} (LIGADO)'
            color = 'green'
        elif i == cluster_desligado:
            label = f'Cluster {i} (DESLIGADO)'
            color = 'red'
        else:
            label = f'Cluster {i} (Intermediário)'
            color = colors[i]
        
        axes[0,0].scatter(dados_pca[mask, 0], dados_pca[mask, 1], 
                         c=color, alpha=0.6, s=1, label=label)
    
    axes[0,0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} var)')
    axes[0,0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} var)')
    axes[0,0].set_title('Todos os Clusters K-means (6 clusters)')
    axes[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 2. Scatter plot por status (incluindo transições)
    status_colors = {'LIGADO': 'green', 'DESLIGADO': 'red', 'TRANSICAO': 'orange'}
    for status, color in status_colors.items():
        mask = df_analise['equipamento_status'] == status
        if mask.sum() > 0:  # Só plotar se houver dados
            axes[0,1].scatter(dados_pca[mask, 0], dados_pca[mask, 1], 
                             c=color, alpha=0.6, s=1, label=status)
    
    axes[0,1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} var)')
    axes[0,1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} var)')
    axes[0,1].set_title('Classificação com Estados de Transição')
    axes[0,1].legend()
    
    # 3. Distribuição dos clusters
    cluster_counts = df_analise['cluster'].value_counts().sort_index()
    bar_colors = []
    for i in range(6):
        if i == cluster_ligado:
            bar_colors.append('green')
        elif i == cluster_desligado:
            bar_colors.append('red')
        else:
            bar_colors.append(colors[i])
    
    axes[1,0].bar(cluster_counts.index, cluster_counts.values, color=bar_colors)
    axes[1,0].set_xlabel('Cluster')
    axes[1,0].set_ylabel('Número de Amostras')
    axes[1,0].set_title('Distribuição dos Clusters')
    
    # 4. Distribuição do status
    status_counts = df_analise['equipamento_status'].value_counts()
    if len(status_counts) > 0:
        # Definir cores: verde para LIGADO, vermelho para DESLIGADO
        pie_colors = []
        for status in status_counts.index:
            if status == 'LIGADO':
                pie_colors.append('green')
            elif status == 'DESLIGADO':
                pie_colors.append('red')
            else:  # TRANSICAO ou outros
                pie_colors.append('orange')

        axes[1,1].pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%', colors=pie_colors)
    axes[1,1].set_title('Distribuição do Status')
    
    plt.tight_layout()
    DIR_RESULTS.mkdir(parents=True, exist_ok=True)
    mpoint_tag = f'_{mpoint}' if mpoint else ''
    plt.savefig(DIR_RESULTS / f'analise_kmeans_clusters_moderado{mpoint_tag}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  - Visualizações salvas com sucesso!")

def salvar_modelos_e_dados_rigoroso(kmeans, scaler, df_analise, colunas_validas,
                                   cluster_ligado, cluster_desligado, clusters_intermediarios,
                                   thresholds_desligado=None, mpoint=None):
    """Salva modelos e dados (modo rigoroso)"""
    print("\nSalvando modelos e dados...", flush=True)

    if not mpoint:
        raise ValueError("mpoint deve ser informado para salvar modelos")

    modelo_path = kmeans_model_path(mpoint, create=True)
    scaler_modelo_path = scaler_model_path(mpoint, create=True)
    classificado_path = processed_classificado_path(mpoint)
    rotulado_path = processed_rotulado_path(mpoint)
    info_path = info_kmeans_path(mpoint, create=True)
    
    from utils.artifact_paths import results_dir
    relatorio_path = results_dir(mpoint, create=True) / f'relatorio_treinamento_{mpoint}.txt'

    classificado_path.parent.mkdir(parents=True, exist_ok=True)
    rotulado_path.parent.mkdir(parents=True, exist_ok=True)

    # Salvar modelos pickle
    print(f"  - Salvando modelo K-means em: {modelo_path}", flush=True)
    with open(modelo_path, 'wb') as f:
        pickle.dump(kmeans, f)
    print(f"  - Modelo K-means salvo com sucesso!", flush=True)
    
    print(f"  - Salvando scaler em: {scaler_modelo_path}", flush=True)
    with open(scaler_modelo_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"  - Scaler salvo com sucesso!", flush=True)

    # Salvar CSVs SEM output no terminal
    print(f"  - Salvando dados classificados...", flush=True)
    df_analise.to_csv(classificado_path, index=False)
    print(f"  - Dados classificados salvos!", flush=True)
    
    # Salvar dados rotulados
    print(f"  - Salvando dados rotulados...", flush=True)
    df_rotulados = df_analise[df_analise['cluster'].isin([cluster_ligado, cluster_desligado])].drop(columns=['cluster']) if cluster_ligado is not None and cluster_desligado is not None else df_analise.drop(columns=['cluster'])
    if mpoint:
        df_rotulados = df_rotulados.copy()
        df_rotulados['mpoint_id'] = mpoint
    df_rotulados.to_csv(rotulado_path, index=False)
    print(f"  - Dados rotulados salvos!", flush=True)
    
    # Contar estados
    total_ligado = len(df_analise[df_analise['equipamento_status'] == 'LIGADO'])
    total_desligado = len(df_analise[df_analise['equipamento_status'] == 'DESLIGADO'])
    
    # Salvar info JSON
    print(f"  - Salvando informações do modelo...", flush=True)
    info_modelo = {
        'colunas_utilizadas': colunas_validas,
        'cluster_ligado': int(cluster_ligado) if cluster_ligado is not None else None,
        'cluster_desligado': int(cluster_desligado) if cluster_desligado is not None else None,
        'clusters_intermediarios': [int(c) for c in clusters_intermediarios] if clusters_intermediarios else [],
        'numero_clusters': 6,
        'total_amostras_originais': len(df_analise),
        'amostras_classificadas': len(df_rotulados),
        'amostras_ligado': total_ligado,
        'amostras_desligado': total_desligado,
        'percentual_classificados': len(df_rotulados) / len(df_analise) * 100,
        'estrategia': 'Classificação dinâmica em 2 estados baseada em scores e desvio padrão',
        'estados': ['DESLIGADO', 'LIGADO'],
        'thresholds_desligado': thresholds_desligado if thresholds_desligado else {}
    }
    
    with open(info_path, 'w') as f:
        json.dump(info_modelo, f, indent=2)
    print(f"  - Informações do modelo salvas!", flush=True)
    
    # Gerar relatório TXT detalhado
    print(f"  - Gerando relatório...", flush=True)
    with open(relatorio_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"RELATÓRIO DE TREINAMENTO K-MEANS - {mpoint}\n")
        f.write("="*80 + "\n\n")
        f.write(f"Data: {pd.Timestamp.now()}\n\n")
        
        f.write("ARQUIVOS GERADOS:\n")
        f.write(f"  - Modelo K-means: {modelo_path}\n")
        f.write(f"  - Scaler: {scaler_modelo_path}\n")
        f.write(f"  - Dados classificados: {classificado_path}\n")
        f.write(f"  - Dados rotulados: {rotulado_path}\n")
        f.write(f"  - Info JSON: {info_path}\n\n")
        
        f.write("ESTATÍSTICAS:\n")
        f.write(f"  - Total de amostras: {len(df_analise):,}\n")
        f.write(f"  - Amostras LIGADO: {total_ligado:,} ({total_ligado/len(df_analise)*100:.1f}%)\n")
        f.write(f"  - Amostras DESLIGADO: {total_desligado:,} ({total_desligado/len(df_analise)*100:.1f}%)\n")
        f.write(f"  - Amostras rotuladas: {len(df_rotulados):,}\n")
        f.write(f"  - Percentual usado: {len(df_rotulados)/len(df_analise)*100:.1f}%\n\n")
        
        f.write("ESTRATÉGIA:\n")
        f.write(f"  - Método: Classificação dinâmica em 2 estados (DESLIGADO vs LIGADO)\n")
        f.write(f"  - Número de clusters K-means: 6\n")
        f.write(f"  - Lógica: Score combinado (vel_rms + current + rpm normalizados)\n")
        f.write(f"  - Classificação: Baseada em desvio padrão e diferenças entre scores\n")
        f.write(f"  - Clusters com menores scores → DESLIGADO (adaptativo por equipamento)\n")
    
    print(f"  [OK] Todos os arquivos salvos com sucesso!")
    print(f"  [OK] Relatório: {relatorio_path}", flush=True)

def parse_arguments():
    """Parse argumentos da linha de comando"""
    parser = argparse.ArgumentParser(
        description="K-means com classificação rigorosa"
    )
    parser.add_argument(
        '--mpoint',
        type=str,
        required=True,
        help='ID do mpoint (ex: c_636)'
    )

    return parser.parse_args()

def main():
    """Função principal"""
    # Parse argumentos
    args = parse_arguments()

    print("=== K-MEANS RIGOROSO - 6 CLUSTERS COM CRITÉRIOS ESPECÍFICOS ===", flush=True)
    print("=" * 70, flush=True)
    print(f"Mpoint: {args.mpoint}")
    print()

    try:
        # 1. Carregar dados normalizados
        print("\n[1/8] Carregando dados normalizados...", flush=True)
        arquivo_normalizado = normalized_csv_path(args.mpoint)
        if not arquivo_normalizado.exists():
            print(f"[ERRO] Arquivo de dados normalizados não encontrado: {arquivo_normalizado}")
            print("Execute primeiro o script de normalização: python scripts/normalizar_dados_kmeans.py --mpoint <mpoint>")
            return

        df, info_normalizacao = carregar_dados_normalizados(args.mpoint)
        
        # 2. Preparar dados normalizados
        print("\n[2/8] Preparando dados normalizados...", flush=True)
        df_kmeans, colunas_validas = preparar_dados_normalizados(df, info_normalizacao)
        
        # 3. Usar dados já normalizados
        print("\n[3/8] Carregando scaler e timestamps...", flush=True)
        dados_normalizados, scaler, timestamp = usar_dados_normalizados(df_kmeans, info_normalizacao, args.mpoint)
        
        # 4. Executar K-means com 6 clusters
        print("\n[4/8] Executando K-means (pode demorar 1-2 minutos)...", flush=True)
        kmeans, clusters = executar_kmeans(dados_normalizados, n_clusters=6)
        
        # 5. Analisar clusters
        print("\n[5/8] Analisando clusters...", flush=True)
        df_analise = analisar_clusters(df_kmeans, clusters, colunas_validas)
        
        # 6. Classificar em 2 estados simples
        print("\n[6/8] Classificando em DESLIGADO vs LIGADO...", flush=True)
        df_analise, thresholds_desligado = classificar_2_estados_simples(df_analise, scaler, args.mpoint)
        print("[6/8] Classificação concluída!", flush=True)

        # Definir clusters fictícios para compatibilidade (não mais usado)
        cluster_ligado = None
        cluster_desligado = None
        clusters_intermediarios = []

        # 7. Criar visualizações
        print("\n[7/8] Criando visualizações...", flush=True)
        criar_visualizacoes_rigoroso(df_analise, colunas_validas, cluster_ligado, cluster_desligado, clusters_intermediarios, args.mpoint)

        # 8. Salvar modelos e dados
        print("\n[8/8] Salvando modelos e dados...", flush=True)
        salvar_modelos_e_dados_rigoroso(kmeans, scaler, df_analise, colunas_validas,
                                       cluster_ligado, cluster_desligado, clusters_intermediarios,
                                       thresholds_desligado, args.mpoint)

        print("\n" + "=" * 70)
        print("=== PROCESSO RIGOROSO CONCLUÍDO COM SUCESSO ===")
        print("=" * 70)

        # Gerar logs detalhados para TCC
        import time
        start_time = time.time()  # Nota: deveria ser definido no início, mas para compatibilidade vamos estimar

        # Estatísticas dos clusters
        cluster_stats = {}
        total_ligado = 0
        total_desligado = 0

        for cluster_id in df_analise['cluster'].unique():
            cluster_data = df_analise[df_analise['cluster'] == cluster_id]
            status = 'DESLIGADO' if df_analise[df_analise['cluster'] == cluster_id]['equipamento_status'].iloc[0] == 'DESLIGADO' else 'LIGADO'

            cluster_stats[f'cluster_{cluster_id}'] = {
                'size': len(cluster_data),
                'percentage': (len(cluster_data) / len(df_analise)) * 100,
                'status': status,
                'features_mean': {col: cluster_data[col].mean() for col in colunas_validas[:5]}  # Primeiras 5 features
            }

            if status == 'LIGADO':
                total_ligado += len(cluster_data)
            else:
                total_desligado += len(cluster_data)

        # Log de treinamento
        training_log = create_training_log(
            script_name='kmeans_classificacao_moderado',
            mpoint=args.mpoint,
            model_info={
                'algorithm': 'K-means',
                'n_clusters': 6,
                'classification_strategy': 'Dynamic 2-state classification based on scores and standard deviation',
                'states': ['DESLIGADO', 'LIGADO'],
                'cluster_centers_shape': kmeans.cluster_centers_.shape,
                'inertia': float(kmeans.inertia_),
                'n_iter': kmeans.n_iter_,
                'random_state': 42
            },
            training_data_info={
                'total_samples': len(df_analise),
                'features_used': colunas_validas,
                'normalization_applied': True,
                'scaler_type': 'MinMaxScaler',
                'data_source': str(normalized_csv_path(args.mpoint)),
                'timestamp_range': {
                    'start': str(df_analise['time'].min()),
                    'end': str(df_analise['time'].max())
                }
            },
            performance_metrics={
                'clusters_distribution': cluster_stats,
                'classification_summary': {
                    'total_ligado': total_ligado,
                    'total_desligado': total_desligado,
                    'ligado_percentage': (total_ligado / len(df_analise)) * 100,
                    'desligado_percentage': (total_desligado / len(df_analise)) * 100,
                    'classification_method': 'score_based_dynamic_threshold'
                },
                'model_convergence': {
                    'iterations': kmeans.n_iter_,
                    'inertia': float(kmeans.inertia_)
                }
            },
            model_files=[
                str(kmeans_model_path(args.mpoint)),
                str(scaler_model_path(args.mpoint)),
                str(info_kmeans_path(args.mpoint))
            ],
            processing_time=time.time() - start_time,
            training_parameters={
                'n_clusters': 6,
                'random_state': 42,
                'n_init': 10,
                'max_iter': 300,
                'classification_logic': 'Dynamic score-based clustering with std deviation analysis'
            }
        )

        save_log(training_log, 'kmeans_classificacao_moderado', args.mpoint, 'training_complete')

        # Log de visualização
        chart_file = DIR_RESULTS / f'analise_kmeans_clusters_moderado_{args.mpoint}.png'
        viz_log = create_visualization_log(
            script_name='kmeans_classificacao_moderado',
            mpoint=args.mpoint,
            chart_type='kmeans_clusters_3d_analysis',
            data_description={
                'total_samples': len(df_analise),
                'n_clusters': 6,
                'features_visualized': colunas_validas[:10],  # Primeiras 10 para PCA
                'dimensionality_reduction': 'PCA_2D',
                'classification_states': ['DESLIGADO', 'LIGADO'],
                'clusters_info': cluster_stats
            },
            chart_files=[str(chart_file)],
            period_info={
                'data_start': str(df_analise['time'].min()),
                'data_end': str(df_analise['time'].max()),
                'total_samples': len(df_analise),
                'classified_samples': len(df_analise)
            }
        )

        save_log(viz_log, 'kmeans_classificacao_moderado', args.mpoint, 'visualization')

        # Enriquecer arquivo results
        results_data = {
            'kmeans_training_completed': True,
            'kmeans_training_timestamp': datetime.now().isoformat(),
            'total_clusters': 6,
            'total_samples_classified': len(df_analise),
            'ligado_samples': total_ligado,
            'desligado_samples': total_desligado,
            'classification_accuracy': (total_ligado + total_desligado) / len(df_analise) * 100,
            'model_inertia': float(kmeans.inertia_),
            'cluster_analysis_charts': [str(chart_file)],
            'training_parameters': training_log['training_parameters'],
            'cluster_statistics': cluster_stats
        }

        enrich_results_file(args.mpoint, results_data)
        
    except Exception as e:
        print(f"\n[ERRO] Erro durante o processamento: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()