"""
K-means para equipamentos MECÂNICOS (sem current, sem RPM).
Classificação baseada em:
- TEMPERATURA (object_temp): Equipamento desligado = temperatura ambiente
- VIBRAÇÃO (vel_rms, mag_x/y/z): Equipamento desligado = vibrações próximas de zero ou residuais

Estratégia:
- 6 clusters K-means
- Preferência por 1-2 clusters como DESLIGADO
- Score baseado em: temperatura + vibração (sem current/RPM)
- Thresholds dinâmicos calculados automaticamente
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
    enrich_results_file,
)

DIR_NORMALIZED = BASE_DIR / 'data' / 'normalized'
DIR_PROCESSED = BASE_DIR / 'data' / 'processed'
DIR_MODELS = BASE_DIR / 'models'
DIR_RESULTS = BASE_DIR / 'results'

VERBOSE = True

def carregar_dados_normalizados(mpoint=None):
    """Carrega os dados normalizados para K-means"""
    print("Carregando dados normalizados...")

    if not mpoint:
        raise ValueError("mpoint deve ser informado")

    arquivo_dados = normalized_csv_path(mpoint)

    if not arquivo_dados.exists():
        print(f"[ERRO] Arquivo de dados normalizados não encontrado: {arquivo_dados}")
        print("Execute primeiro: python scripts/normalizar_dados_kmeans_mecanico.py --mpoint <mpoint>")
        return None, None

    df = pd.read_csv(arquivo_dados)
    
    info_path = info_normalizacao_path(mpoint)
    if not info_path.exists():
        raise FileNotFoundError(f"Informações de normalização não encontradas para {mpoint}")

    with open(info_path, 'r') as f:
        info_normalizacao = json.load(f)
    
    colunas_key = 'colunas_utilizadas_finais' if 'colunas_utilizadas_finais' in info_normalizacao else 'colunas_utilizadas'
    
    print(f"  - Shape: {df.shape}")
    print(f"  - Colunas: {len(info_normalizacao[colunas_key])}")
    print(f"  - Tipo: EQUIPAMENTO MECÂNICO (temperatura + vibração)")
    
    return df, info_normalizacao

def preparar_dados_normalizados(df, info_normalizacao):
    """Prepara dados normalizados para K-means"""
    print("\nPreparando dados normalizados para K-means...")
    
    colunas_key = 'colunas_utilizadas_finais' if 'colunas_utilizadas_finais' in info_normalizacao else 'colunas_utilizadas'
    colunas_validas = info_normalizacao[colunas_key]
    
    print(f"  - Colunas selecionadas: {len(colunas_validas)}")
    print(f"  - Dados já normalizados e limpos")
    
    df_kmeans_clean = df.copy()
    
    print(f"  - Linhas disponíveis: {len(df_kmeans_clean):,}")
    
    return df_kmeans_clean, colunas_validas

def usar_dados_normalizados(df_kmeans, info_normalizacao, mpoint):
    """Usa dados já normalizados"""
    print("\nUsando dados já normalizados...")
    
    import joblib
    try:
        scaler_path = scaler_maxmin_path(mpoint)
        scaler = joblib.load(scaler_path)
    except FileNotFoundError:
        raise
    
    timestamp = df_kmeans['time']
    features_numericas = df_kmeans.drop('time', axis=1)
    
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
    
    df_analise = df_kmeans.copy()
    df_analise['cluster'] = clusters
    
    print("\nEstatísticas por cluster:")
    for cluster_id in range(6):
        cluster_data = df_analise[df_analise['cluster'] == cluster_id]
        print(f"\nCluster {cluster_id} ({len(cluster_data):,} amostras):")
        
        for col in colunas_validas[:10]:  # Mostrar apenas as primeiras 10
            if col in cluster_data.columns:
                mean_val = cluster_data[col].mean()
                std_val = cluster_data[col].std()
                print(f"  - {col}: {mean_val:.3f} ± {std_val:.3f}")
    
    return df_analise

def calcular_thresholds_estado_desligado_mecanico(df_analise, clusters_desligado, temp_cols, vel_rms_cols, mag_cols, mpoint=None):
    """
    Calcula thresholds dinâmicos para equipamento MECÂNICO.
    Foco: Temperatura e Vibração (sem current, sem RPM).
    
    DESLIGADO = Temperatura ambiente + Vibrações próximas de zero/residuais
    """
    dados_desligado = df_analise[df_analise['cluster'].isin(clusters_desligado)].copy()
    
    thresholds = {}
    
    # Carregar dados ORIGINAIS para calcular thresholds REAIS
    try:
        DIR_RAW_PREENCHIDO = BASE_DIR / 'data' / 'raw_preenchido'
        arquivos_periodo = list(DIR_RAW_PREENCHIDO.glob(f'periodo_*_final_{mpoint}.csv'))
        
        if arquivos_periodo:
            dfs_originais = []
            for arq in arquivos_periodo:
                df_orig = pd.read_csv(arq)
                df_orig['time'] = pd.to_datetime(df_orig['time'], format='mixed', utc=True)
                dfs_originais.append(df_orig)
            
            df_original = pd.concat(dfs_originais, ignore_index=True)
            df_original = df_original.sort_values('time').reset_index(drop=True)
            
            indices_desligado = dados_desligado.index
            dados_desligado_originais = df_original.iloc[indices_desligado]
            
            # TEMPERATURA: Equipamento desligado = temperatura ambiente
            if 'object_temp' in dados_desligado_originais.columns:
                temp_p95 = dados_desligado_originais['object_temp'].quantile(0.95)
                temp_mean = dados_desligado_originais['object_temp'].mean()
                temp_std = dados_desligado_originais['object_temp'].std()
                temp_median = dados_desligado_originais['object_temp'].median()
                
                # Threshold de temperatura = percentil 95% + margem 5% (temperatura ambiente pode variar)
                thresholds['temp_max'] = float(temp_p95 * 1.05)
                thresholds['temp_mean'] = float(temp_mean)
                thresholds['temp_median'] = float(temp_median)
                thresholds['temp_std'] = float(temp_std)
                
                if VERBOSE:
                    print(f"  - object_temp (REAL): mean={temp_mean:.2f}°C, median={temp_median:.2f}°C, p95={temp_p95:.2f}°C")
                    print(f"    → threshold={thresholds['temp_max']:.2f}°C (temperatura ambiente)")
            
            # VIBRAÇÃO: Equipamento desligado = vibrações próximas de zero ou residuais
            colunas_vibracao = [col for col in dados_desligado_originais.columns if 'vel_rms' in col.lower()]
            if colunas_vibracao:
                # Calcular média de todas as colunas de vibração RMS
                vibracao_mean = dados_desligado_originais[colunas_vibracao].mean().mean()
                vibracao_p95 = dados_desligado_originais[colunas_vibracao].quantile(0.95).mean()
                vibracao_max = dados_desligado_originais[colunas_vibracao].max().max()
                
                # Threshold de vibração = percentil 95% + margem 30% (vibrações residuais podem variar)
                thresholds['vibracao_rms_max'] = float(vibracao_p95 * 1.3)
                thresholds['vibracao_rms_mean'] = float(vibracao_mean)
                
                if VERBOSE:
                    print(f"  - vibração RMS (REAL): mean={vibracao_mean:.4f} mm/s, p95={vibracao_p95:.4f} mm/s")
                    print(f"    → threshold={thresholds['vibracao_rms_max']:.4f} mm/s (vibrações residuais)")
            
            # MAGNETÔMETRO: Para detectar vibrações muito pequenas
            colunas_mag = [col for col in dados_desligado_originais.columns if 'mag_' in col.lower()]
            if colunas_mag:
                mag_mean = dados_desligado_originais[colunas_mag].mean().mean()
                mag_p95 = dados_desligado_originais[colunas_mag].quantile(0.95).mean()
                
                thresholds['mag_max'] = float(mag_p95 * 1.2)
                thresholds['mag_mean'] = float(mag_mean)
                
                if VERBOSE:
                    print(f"  - magnetômetro (REAL): mean={mag_mean:.4f}, p95={mag_p95:.4f}")
                    print(f"    → threshold={thresholds['mag_max']:.4f} (vibrações mínimas)")
        else:
            print("  [AVISO] Dados originais não encontrados, usando valores normalizados")
            # Fallback: usar valores normalizados
            if temp_cols:
                thresholds['temp_max'] = float(dados_desligado[temp_cols].quantile(0.95).mean() * 1.05)
                thresholds['temp_mean'] = float(dados_desligado[temp_cols].mean().mean())
            if vel_rms_cols:
                thresholds['vibracao_rms_max'] = float(dados_desligado[vel_rms_cols].quantile(0.95).mean() * 1.3)
                thresholds['vibracao_rms_mean'] = float(dados_desligado[vel_rms_cols].mean().mean())
            if mag_cols:
                thresholds['mag_max'] = float(dados_desligado[mag_cols].quantile(0.95).mean() * 1.2)
                thresholds['mag_mean'] = float(dados_desligado[mag_cols].mean().mean())
    
    except Exception as e:
        print(f"  [ERRO] Erro ao carregar dados originais: {e}")
        # Fallback
        if temp_cols:
            thresholds['temp_max'] = float(dados_desligado[temp_cols].quantile(0.95).mean() * 1.05)
            thresholds['temp_mean'] = float(dados_desligado[temp_cols].mean().mean())
        if vel_rms_cols:
            thresholds['vibracao_rms_max'] = float(dados_desligado[vel_rms_cols].quantile(0.95).mean() * 1.3)
            thresholds['vibracao_rms_mean'] = float(dados_desligado[vel_rms_cols].mean().mean())
        if mag_cols:
            thresholds['mag_max'] = float(dados_desligado[mag_cols].quantile(0.95).mean() * 1.2)
            thresholds['mag_mean'] = float(dados_desligado[mag_cols].mean().mean())
    
    # Calcular threshold DINÂMICO de vibração residual
    if vel_rms_cols and len(clusters_desligado) > 0:
        cluster_base = min(clusters_desligado)
        dados_cluster_base = df_analise[df_analise['cluster'] == cluster_base]
        
        vibracao_max_cluster_base_norm = dados_cluster_base[vel_rms_cols].max().max()
        
        thresholds['threshold_vibracao_residual_norm'] = float(vibracao_max_cluster_base_norm * 1.3)
        
        if VERBOSE:
            print(f"  - Threshold DINÂMICO vibração residual (normalizado): {thresholds['threshold_vibracao_residual_norm']:.3f}")
    
    thresholds['clusters_desligado'] = [int(c) for c in clusters_desligado]
    thresholds['amostras_desligado'] = int(len(dados_desligado))
    thresholds['porcentagem_desligado'] = float(len(dados_desligado) / len(df_analise) * 100)
    
    if VERBOSE:
        print(f"\n  - Clusters identificados como DESLIGADO: {clusters_desligado}")
        print(f"  - Amostras DESLIGADO: {len(dados_desligado)} ({thresholds['porcentagem_desligado']:.1f}%)")
        print("  - Thresholds REAIS calculados (equipamento MECÂNICO):")
        if 'temp_max' in thresholds:
            print(f"    * temperatura_max: {thresholds['temp_max']:.2f} °C")
        if 'vibracao_rms_max' in thresholds:
            print(f"    * vibracao_rms_max: {thresholds['vibracao_rms_max']:.4f} mm/s")
    
    return thresholds

def classificar_2_estados_mecanico(df_analise, mpoint=None):
    """
    Classificação para equipamentos MECÂNICOS (sem current, sem RPM).
    
    Estados:
    - DESLIGADO: Temperatura ambiente + Vibrações próximas de zero/residuais
    - LIGADO: Aumento de temperatura + Vibrações significativas
    
    Score baseado em: temperatura (peso 1.5) + vibração (peso 1.0) + magnetômetro (peso 0.5)
    """
    if VERBOSE:
        print("Classificando estados MECÂNICOS (DESLIGADO vs LIGADO)...", flush=True)
        print("  - Estratégia: Score baseado em Temperatura + Vibração (sem current/RPM)", flush=True)
    
    # Identificar colunas relevantes para equipamento MECÂNICO
    temp_cols = [col for col in df_analise.columns if 'temp' in col.lower()]
    vel_rms_cols = [col for col in df_analise.columns if 'vel_rms' in col.lower()]
    mag_cols = [col for col in df_analise.columns if 'mag_' in col.lower()]
    
    # Analisar características de cada cluster
    cluster_features = {}
    
    for cluster_id in df_analise['cluster'].unique():
        cluster_data = df_analise[df_analise['cluster'] == cluster_id]
        
        # Calcular médias normalizadas
        temp_mean_norm = cluster_data[temp_cols].mean().mean() if temp_cols else 0
        vel_rms_mean_norm = cluster_data[vel_rms_cols].mean().mean() if vel_rms_cols else 0
        mag_mean_norm = cluster_data[mag_cols].mean().mean() if mag_cols else 0
        
        # Calcular máximos normalizados
        temp_max_norm = cluster_data[temp_cols].max().max() if temp_cols else 0
        vel_rms_max_norm = cluster_data[vel_rms_cols].max().max() if vel_rms_cols else 0
        
        # Score combinado PONDERADO para equipamento MECÂNICO
        # Temperatura tem peso maior (1.5) porque é indicador mais confiável
        # Vibração tem peso 1.0 (pode ter vibrações residuais mesmo desligado)
        # Magnetômetro tem peso 0.5 (detecta vibrações mínimas)
        score = (temp_mean_norm * 1.5) + (vel_rms_mean_norm * 1.0) + (mag_mean_norm * 0.5)
        
        cluster_features[cluster_id] = {
            'temp_mean_norm': temp_mean_norm,
            'temp_max_norm': temp_max_norm,
            'vel_rms_mean_norm': vel_rms_mean_norm,
            'vel_rms_max_norm': vel_rms_max_norm,
            'mag_mean_norm': mag_mean_norm,
            'score': score,
            'count': len(cluster_data)
        }
    
    # Ordenar clusters por score
    clusters_ordenados = sorted(cluster_features.items(), key=lambda x: x[1]['score'])
    scores_ordenados = [c[1]['score'] for c in clusters_ordenados]
    
    if VERBOSE:
        print("\n  - Análise de Clusters MECÂNICOS (ordenados por score):")
        for i, (cid, features) in enumerate(clusters_ordenados):
            print(f"    Cluster {cid}: score={features['score']:.3f}, "
                  f"temp={features['temp_mean_norm']:.3f}, "
                  f"vel_rms={features['vel_rms_mean_norm']:.3f}, "
                  f"mag={features['mag_mean_norm']:.3f}, "
                  f"count={features['count']:,}")
    
    # PREFERÊNCIA POR 1 CLUSTER DESLIGADO
    clusters_desligado = []
    
    # Sempre começar com o cluster de menor score
    cluster_menor_score = clusters_ordenados[0][0]
    clusters_desligado.append(cluster_menor_score)
    
    # Análise para decidir se incluir mais clusters
    if len(scores_ordenados) >= 2:
        scores_array = np.array(scores_ordenados)
        diff_primeiro_segundo = scores_ordenados[1] - scores_ordenados[0]
        
        # Threshold DINÂMICO de vibração
        features_primeiro = cluster_features[clusters_ordenados[0][0]]
        vel_rms_max_cluster0 = features_primeiro['vel_rms_max_norm']
        threshold_vibracao_residual = vel_rms_max_cluster0 * 1.3
        
        # Threshold DINÂMICO de temperatura
        temp_max_cluster0 = features_primeiro['temp_max_norm']
        threshold_temp_ambiente = temp_max_cluster0 * 1.1  # +10% margem
        
        if VERBOSE:
            print(f"\n  - Thresholds DINÂMICOS calculados:")
            print(f"    Temperatura ambiente: {threshold_temp_ambiente:.3f} (normalizado)")
            print(f"    Vibração residual: {threshold_vibracao_residual:.3f} (normalizado)")
        
        # Critério: Scores muito próximos E ambos com temperatura/vibração baixas
        if diff_primeiro_segundo < 0.3 and scores_ordenados[1] < 0.5:
            features_segundo = cluster_features[clusters_ordenados[1][0]]
            
            # Verificar se 2º cluster também tem temperatura ambiente e vibrações baixas
            if (features_segundo['temp_max_norm'] <= threshold_temp_ambiente and 
                features_segundo['vel_rms_max_norm'] <= threshold_vibracao_residual):
                clusters_desligado.append(clusters_ordenados[1][0])
                if VERBOSE:
                    print(f"\n  - Incluindo 2º cluster ({clusters_ordenados[1][0]}) como DESLIGADO:")
                    print(f"    Razão: Temperatura e vibrações baixas (dentro dos thresholds)")
                    print(f"    Temp: {features_segundo['temp_max_norm']:.3f} ≤ {threshold_temp_ambiente:.3f}")
                    print(f"    Vibr: {features_segundo['vel_rms_max_norm']:.3f} ≤ {threshold_vibracao_residual:.3f}")
        
        if VERBOSE and len(clusters_desligado) == 1:
            print(f"\n  - Usando APENAS 1 cluster ({cluster_menor_score}) como DESLIGADO:")
            print(f"    Razão: Diferença significativa para próximo cluster ({diff_primeiro_segundo:.3f})")
    
    # VERIFICAÇÃO FÍSICA: Clusters com temperatura E vibração muito baixas
    clusters_desligado_fisico = []
    for cluster_id, features in cluster_features.items():
        # Temperatura próxima de ambiente (< 0.15 normalizado) E vibração próxima de zero (< 0.10 normalizado)
        if features['temp_mean_norm'] < 0.15 and features['vel_rms_mean_norm'] < 0.10:
            if cluster_id not in clusters_desligado:
                clusters_desligado_fisico.append(cluster_id)
                if VERBOSE:
                    print(f"\n  - VERIFICAÇÃO FÍSICA: Incluindo cluster {cluster_id} como DESLIGADO:")
                    print(f"    Razão: Temperatura ambiente ({features['temp_mean_norm']:.3f} < 0.15) E ")
                    print(f"           Vibração próxima de zero ({features['vel_rms_mean_norm']:.3f} < 0.10)")
                    print(f"    → Equipamento FISICAMENTE DESLIGADO")
    
    if clusters_desligado_fisico:
        clusters_desligado.extend(clusters_desligado_fisico)
        if VERBOSE:
            print(f"\n  - Total de clusters DESLIGADO após verificação física: {len(clusters_desligado)}")
    
    # Resto = LIGADO
    clusters_ligado = [c[0] for c in clusters_ordenados if c[0] not in clusters_desligado]
    
    # Aplicar classificação
    df_analise['equipamento_status'] = 'LIGADO'
    for cid in clusters_desligado:
        df_analise.loc[df_analise['cluster'] == cid, 'equipamento_status'] = 'DESLIGADO'
    
    # Calcular thresholds dinâmicos
    thresholds_desligado = calcular_thresholds_estado_desligado_mecanico(
        df_analise, clusters_desligado, temp_cols, vel_rms_cols, mag_cols, mpoint
    )
    
    # Resumo final
    total_amostras = len(df_analise)
    amostras_desligado = (df_analise['equipamento_status'] == 'DESLIGADO').sum()
    amostras_ligado = (df_analise['equipamento_status'] == 'LIGADO').sum()
    
    if VERBOSE:
        print(f"\n{'='*70}")
        print(f"RESUMO DA CLASSIFICAÇÃO - EQUIPAMENTO MECÂNICO")
        print(f"{'='*70}")
        print(f"  Clusters DESLIGADO: {clusters_desligado} ({len(clusters_desligado)} cluster(s))")
        print(f"  Clusters LIGADO: {clusters_ligado} ({len(clusters_ligado)} cluster(s))")
        print(f"\n  Distribuição de Amostras:")
        print(f"    - DESLIGADO: {amostras_desligado:,} ({amostras_desligado/total_amostras*100:.1f}%)")
        print(f"    - LIGADO: {amostras_ligado:,} ({amostras_ligado/total_amostras*100:.1f}%)")
        print(f"    - Total: {total_amostras:,}")
        print(f"{'='*70}\n")
    
    return df_analise, thresholds_desligado

def criar_visualizacoes(df_analise, colunas_validas, mpoint):
    """Cria visualizações dos resultados"""
    print("\nCriando visualizações...")
    
    pca = PCA(n_components=2)
    dados_pca = pca.fit_transform(df_analise[colunas_validas])
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Scatter plot dos clusters
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    for i in range(6):
        mask = df_analise['cluster'] == i
        if mask.sum() > 0:
            status = df_analise[df_analise['cluster'] == i]['equipamento_status'].iloc[0]
            label = f'Cluster {i} ({status})'
            color = 'red' if status == 'DESLIGADO' else colors[i]
            axes[0,0].scatter(dados_pca[mask, 0], dados_pca[mask, 1], 
                             c=color, alpha=0.6, s=1, label=label)
    
    axes[0,0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} var)')
    axes[0,0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} var)')
    axes[0,0].set_title('Clusters K-means - Equipamento MECÂNICO (6 clusters)')
    axes[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 2. Scatter plot por status
    status_colors = {'LIGADO': 'green', 'DESLIGADO': 'red'}
    for status, color in status_colors.items():
        mask = df_analise['equipamento_status'] == status
        if mask.sum() > 0:
            axes[0,1].scatter(dados_pca[mask, 0], dados_pca[mask, 1], 
                             c=color, alpha=0.6, s=1, label=status)
    
    axes[0,1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} var)')
    axes[0,1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} var)')
    axes[0,1].set_title('Classificação - Temperatura + Vibração')
    axes[0,1].legend()
    
    # 3. Distribuição dos clusters
    cluster_counts = df_analise['cluster'].value_counts().sort_index()
    bar_colors = []
    for i in range(6):
        if i in cluster_counts.index:
            status = df_analise[df_analise['cluster'] == i]['equipamento_status'].iloc[0]
            bar_colors.append('red' if status == 'DESLIGADO' else colors[i])
        else:
            bar_colors.append('gray')
    
    axes[1,0].bar(cluster_counts.index, cluster_counts.values, color=bar_colors[:len(cluster_counts)])
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
            else:  # outros status se houver
                pie_colors.append('orange')

        axes[1,1].pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%', colors=pie_colors)
    axes[1,1].set_title('Distribuição do Status')
    
    plt.tight_layout()
    DIR_RESULTS.mkdir(parents=True, exist_ok=True)
    mpoint_tag = f'_{mpoint}' if mpoint else ''
    plt.savefig(DIR_RESULTS / f'analise_kmeans_clusters_mecanico{mpoint_tag}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  - Visualizações salvas com sucesso!")

def salvar_modelos_e_dados(kmeans, scaler, df_analise, colunas_validas, thresholds_desligado, mpoint):
    """Salva modelos e dados"""
    print("\nSalvando modelos e dados...", flush=True)

    if not mpoint:
        raise ValueError("mpoint deve ser informado")

    modelo_path = kmeans_model_path(mpoint, create=True)
    scaler_modelo_path = scaler_model_path(mpoint, create=True)
    classificado_path = processed_classificado_path(mpoint)
    rotulado_path = processed_rotulado_path(mpoint)
    info_path = info_kmeans_path(mpoint, create=True)
    
    from utils.artifact_paths import results_dir
    relatorio_path = results_dir(mpoint, create=True) / f'relatorio_treinamento_{mpoint}_mecanico.txt'

    classificado_path.parent.mkdir(parents=True, exist_ok=True)
    rotulado_path.parent.mkdir(parents=True, exist_ok=True)

    # Salvar modelos
    with open(modelo_path, 'wb') as f:
        pickle.dump(kmeans, f)
    print(f"  - Modelo K-means salvo!")
    
    with open(scaler_modelo_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"  - Scaler salvo!")

    # Salvar CSVs
    df_analise.to_csv(classificado_path, index=False)
    print(f"  - Dados classificados salvos!")
    
    df_rotulados = df_analise.drop(columns=['cluster'])
    if mpoint:
        df_rotulados = df_rotulados.copy()
        df_rotulados['mpoint_id'] = mpoint
    df_rotulados.to_csv(rotulado_path, index=False)
    print(f"  - Dados rotulados salvos!")
    
    # Contar estados
    total_ligado = len(df_analise[df_analise['equipamento_status'] == 'LIGADO'])
    total_desligado = len(df_analise[df_analise['equipamento_status'] == 'DESLIGADO'])
    
    # Salvar info JSON
    info_modelo = {
        'equipment_type': 'MECHANICAL',
        'data_sources': ['temperature', 'vibration'],
        'no_current_rpm': True,
        'colunas_utilizadas': colunas_validas,
        'numero_clusters': 6,
        'total_amostras_originais': len(df_analise),
        'amostras_classificadas': len(df_rotulados),
        'amostras_ligado': total_ligado,
        'amostras_desligado': total_desligado,
        'percentual_classificados': len(df_rotulados) / len(df_analise) * 100,
        'estrategia': 'Classificação dinâmica MECÂNICA baseada em Temperatura + Vibração',
        'estados': ['DESLIGADO', 'LIGADO'],
        'thresholds_desligado': thresholds_desligado
    }
    
    with open(info_path, 'w') as f:
        json.dump(info_modelo, f, indent=2)
    print(f"  - Informações do modelo salvas!")
    
    # Gerar relatório TXT
    with open(relatorio_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"RELATÓRIO DE TREINAMENTO K-MEANS - {mpoint} - EQUIPAMENTO MECÂNICO\n")
        f.write("="*80 + "\n\n")
        f.write(f"Data: {pd.Timestamp.now()}\n\n")
        
        f.write("TIPO DE EQUIPAMENTO:\n")
        f.write("  - Equipamento MECÂNICO (sem current, sem RPM)\n")
        f.write("  - Análise baseada em: Temperatura + Vibração\n\n")
        
        f.write("ESTATÍSTICAS:\n")
        f.write(f"  - Total de amostras: {len(df_analise):,}\n")
        f.write(f"  - Amostras LIGADO: {total_ligado:,} ({total_ligado/len(df_analise)*100:.1f}%)\n")
        f.write(f"  - Amostras DESLIGADO: {total_desligado:,} ({total_desligado/len(df_analise)*100:.1f}%)\n\n")
        
        f.write("ESTRATÉGIA:\n")
        f.write("  - Método: Classificação dinâmica MECÂNICA (DESLIGADO vs LIGADO)\n")
        f.write("  - Número de clusters K-means: 6\n")
        f.write("  - Score: temperatura (1.5x) + vibração (1.0x) + magnetômetro (0.5x)\n")
        f.write("  - Thresholds dinâmicos calculados automaticamente\n")
    
    print(f"  [OK] Todos os arquivos salvos com sucesso!")
    print(f"  [OK] Relatório: {relatorio_path}")

def parse_arguments():
    """Parse argumentos da linha de comando"""
    parser = argparse.ArgumentParser(description="K-means para equipamento MECÂNICO")
    parser.add_argument('--mpoint', type=str, required=True, help='ID do mpoint (ex: c_640)')
    return parser.parse_args()

def main():
    """Função principal"""
    args = parse_arguments()

    print("=== K-MEANS - EQUIPAMENTO MECÂNICO ===")
    print("=" * 70)
    print(f"Mpoint: {args.mpoint}")
    print("Tipo: MECÂNICO (temperatura + vibração)")
    print()

    try:
        print("\n[1/7] Carregando dados normalizados...")
        df, info_normalizacao = carregar_dados_normalizados(args.mpoint)
        if df is None:
            return
        
        print("\n[2/7] Preparando dados normalizados...")
        df_kmeans, colunas_validas = preparar_dados_normalizados(df, info_normalizacao)
        
        print("\n[3/7] Carregando scaler e timestamps...")
        dados_normalizados, scaler, timestamp = usar_dados_normalizados(df_kmeans, info_normalizacao, args.mpoint)
        
        print("\n[4/7] Executando K-means (pode demorar 1-2 minutos)...")
        kmeans, clusters = executar_kmeans(dados_normalizados, n_clusters=6)
        
        print("\n[5/7] Analisando clusters...")
        df_analise = analisar_clusters(df_kmeans, clusters, colunas_validas)
        
        print("\n[6/7] Classificando em DESLIGADO vs LIGADO (MECÂNICO)...")
        df_analise, thresholds_desligado = classificar_2_estados_mecanico(df_analise, args.mpoint)
        print("[6/7] Classificação concluída!")
        
        print("\n[7/7] Criando visualizações...")
        criar_visualizacoes(df_analise, colunas_validas, args.mpoint)
        
        print("\n[7/7] Salvando modelos e dados...")
        salvar_modelos_e_dados(kmeans, scaler, df_analise, colunas_validas, thresholds_desligado, args.mpoint)

        print("\n" + "=" * 70)
        print("=== PROCESSO CONCLUÍDO COM SUCESSO ===")
        print("=== EQUIPAMENTO MECÂNICO ===")
        print("=" * 70)

        # Gerar logs
        import time
        start_time = time.time()

        cluster_stats = {}
        total_ligado = 0
        total_desligado = 0

        for cluster_id in df_analise['cluster'].unique():
            cluster_data = df_analise[df_analise['cluster'] == cluster_id]
            status = cluster_data['equipamento_status'].iloc[0]

            cluster_stats[f'cluster_{cluster_id}'] = {
                'size': len(cluster_data),
                'percentage': (len(cluster_data) / len(df_analise)) * 100,
                'status': status
            }

            if status == 'LIGADO':
                total_ligado += len(cluster_data)
            else:
                total_desligado += len(cluster_data)

        training_log = create_training_log(
            script_name='kmeans_classificacao_mecanico',
            mpoint=args.mpoint,
            model_info={
                'equipment_type': 'MECHANICAL',
                'data_sources': ['temperature', 'vibration'],
                'algorithm': 'K-means',
                'n_clusters': 6,
                'classification_strategy': 'Dynamic 2-state MECHANICAL (temp + vibration)',
                'states': ['DESLIGADO', 'LIGADO'],
                'no_current_rpm': True
            },
            training_data_info={
                'total_samples': len(df_analise),
                'features_used': colunas_validas,
                'data_source': str(normalized_csv_path(args.mpoint))
            },
            performance_metrics={
                'clusters_distribution': cluster_stats,
                'classification_summary': {
                    'total_ligado': total_ligado,
                    'total_desligado': total_desligado,
                    'ligado_percentage': (total_ligado / len(df_analise)) * 100,
                    'desligado_percentage': (total_desligado / len(df_analise)) * 100
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
                'classification_logic': 'Temperature + Vibration based'
            }
        )

        save_log(training_log, 'kmeans_classificacao_mecanico', args.mpoint, 'training_complete')

        results_data = {
            'kmeans_training_completed': True,
            'kmeans_training_timestamp': datetime.now().isoformat(),
            'equipment_type': 'MECHANICAL',
            'total_clusters': 6,
            'total_samples_classified': len(df_analise),
            'ligado_samples': total_ligado,
            'desligado_samples': total_desligado
        }

        enrich_results_file(args.mpoint, results_data)
        
    except Exception as e:
        print(f"\n[ERRO] Erro durante o processamento: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

