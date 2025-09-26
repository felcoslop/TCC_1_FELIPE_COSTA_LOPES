#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para K-means com classificação rigorosa - usa apenas 2 clusters
Critérios rigorosos: DESLIGADO (vel_rms < 1, current < 10, rpm = 0), resto é LIGADO
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

def carregar_dados_normalizados():
    """Carrega os dados normalizados para K-means"""
    print("Carregando dados normalizados...")
    
    # Carregar dados normalizados para K-means
    df = pd.read_csv('data/normalized/dados_kmeans.csv')
    
    # Carregar informações de normalização
    with open('models/info_normalizacao.json', 'r') as f:
        info_normalizacao = json.load(f)
    
    print(f"  - Shape: {df.shape}")
    print(f"  - Colunas: {len(info_normalizacao['colunas_utilizadas'])}")
    print(f"  - Range normalização: {info_normalizacao['range_normalizacao']}")
    
    return df, info_normalizacao

def preparar_dados_normalizados(df, info_normalizacao):
    """Prepara dados normalizados para K-means"""
    print("\nPreparando dados normalizados para K-means...")
    
    # Usar todas as colunas normalizadas
    colunas_validas = info_normalizacao['colunas_utilizadas']
    
    print(f"  - Colunas selecionadas: {len(colunas_validas)}")
    print(f"  - Dados já normalizados e limpos")
    
    # Dados já estão normalizados e limpos
    df_kmeans_clean = df.copy()
    
    print(f"  - Linhas disponíveis: {len(df_kmeans_clean):,}")
    
    return df_kmeans_clean, colunas_validas

def usar_dados_normalizados(df_kmeans, info_normalizacao):
    """Usa dados já normalizados"""
    print("\nUsando dados já normalizados...")
    
    # Carregar scaler usado na normalização
    import joblib
    scaler = joblib.load('models/scaler_maxmin.pkl')
    
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

def classificar_equipamento_rigoroso(df_analise, colunas_validas):
    """Classifica equipamento usando critérios rigorosos e identifica clusters com mais certeza"""
    print("\nClassificando status do equipamento (CRITÉRIOS RIGOROSOS)...")
    
    # Identificar colunas relevantes
    vel_rms_cols = [col for col in df_analise.columns if 'vel_rms' in col.lower()]
    current_cols = [col for col in df_analise.columns if 'current' in col.lower()]
    rpm_cols = [col for col in df_analise.columns if 'rpm' in col.lower() or 'rotational_speed' in col.lower()]
    
    print(f"  - Colunas vel_rms encontradas: {vel_rms_cols}")
    print(f"  - Colunas current encontradas: {current_cols}")
    print(f"  - Colunas rpm encontradas: {rpm_cols}")
    
    # Aplicar critérios rigorosos
    df_analise['equipamento_status'] = None
    
    # Critérios para DESLIGADO: vel_rms < 1, current < 10, rpm = 0
    condicoes_desligado = []
    
    # Verificar vel_rms < 1 (todas as colunas vel_rms)
    if vel_rms_cols:
        cond_vel_rms = df_analise[vel_rms_cols].max(axis=1) < 1
        condicoes_desligado.append(cond_vel_rms)
        print(f"  - Amostras com vel_rms < 1: {cond_vel_rms.sum():,}")
    
    # Verificar current < 10 (todas as colunas current)
    if current_cols:
        cond_current = df_analise[current_cols].max(axis=1) < 10
        condicoes_desligado.append(cond_current)
        print(f"  - Amostras com current < 10: {cond_current.sum():,}")
    
    # Verificar rpm = 0 (todas as colunas rpm)
    if rpm_cols:
        cond_rpm = (df_analise[rpm_cols] == 0).all(axis=1)
        condicoes_desligado.append(cond_rpm)
        print(f"  - Amostras com rpm = 0: {cond_rpm.sum():,}")
    
    # Aplicar critérios combinados
    if condicoes_desligado:
        condicao_desligado = np.logical_and.reduce(condicoes_desligado)
        df_analise.loc[condicao_desligado, 'equipamento_status'] = 'DESLIGADO'
        
        # Resto é LIGADO
        condicao_ligado = ~condicao_desligado
        df_analise.loc[condicao_ligado, 'equipamento_status'] = 'LIGADO'
        
        print(f"\nClassificação rigorosa aplicada:")
        print(f"  - DESLIGADO: {condicao_desligado.sum():,} amostras ({condicao_desligado.sum()/len(df_analise)*100:.1f}%)")
        print(f"  - LIGADO: {condicao_ligado.sum():,} amostras ({condicao_ligado.sum()/len(df_analise)*100:.1f}%)")
    else:
        print("  - ERRO: Nenhuma coluna relevante encontrada para classificação!")
        return df_analise, None, None
    
    # Analisar clusters em relação à classificação
    print(f"\nAnálise dos clusters vs classificação:")
    cluster_stats = {}
    
    for cluster_id in range(6):
        cluster_data = df_analise[df_analise['cluster'] == cluster_id]
        ligado_count = len(cluster_data[cluster_data['equipamento_status'] == 'LIGADO'])
        desligado_count = len(cluster_data[cluster_data['equipamento_status'] == 'DESLIGADO'])
        total_count = len(cluster_data)
        
        # Calcular percentual de cada status no cluster
        pct_ligado = (ligado_count / total_count * 100) if total_count > 0 else 0
        pct_desligado = (desligado_count / total_count * 100) if total_count > 0 else 0
        
        cluster_stats[cluster_id] = {
            'total': total_count,
            'ligado': ligado_count,
            'desligado': desligado_count,
            'pct_ligado': pct_ligado,
            'pct_desligado': pct_desligado,
            'certeza': max(pct_ligado, pct_desligado)  # Maior percentual = mais certeza
        }
        
        print(f"  - Cluster {cluster_id}:")
        print(f"    - LIGADO: {ligado_count:,} ({pct_ligado:.1f}%)")
        print(f"    - DESLIGADO: {desligado_count:,} ({pct_desligado:.1f}%)")
        print(f"    - Certeza: {cluster_stats[cluster_id]['certeza']:.1f}%")
    
    # Identificar clusters com mais certeza para LIGADO e DESLIGADO
    clusters_ligado = [(cid, stats) for cid, stats in cluster_stats.items() if stats['pct_ligado'] > stats['pct_desligado']]
    clusters_desligado = [(cid, stats) for cid, stats in cluster_stats.items() if stats['pct_desligado'] > stats['pct_ligado']]
    
    # Ordenar por certeza (maior primeiro)
    clusters_ligado.sort(key=lambda x: x[1]['certeza'], reverse=True)
    clusters_desligado.sort(key=lambda x: x[1]['certeza'], reverse=True)
    
    # Pegar o cluster com mais certeza para cada status
    cluster_ligado = clusters_ligado[0][0] if clusters_ligado else None
    cluster_desligado = clusters_desligado[0][0] if clusters_desligado else None
    
    print(f"\nClusters com mais certeza:")
    if cluster_ligado is not None:
        stats_ligado = cluster_stats[cluster_ligado]
        print(f"  - Cluster {cluster_ligado}: LIGADO com {stats_ligado['certeza']:.1f}% de certeza")
        print(f"    - {stats_ligado['ligado']:,} LIGADO de {stats_ligado['total']:,} total")
    
    if cluster_desligado is not None:
        stats_desligado = cluster_stats[cluster_desligado]
        print(f"  - Cluster {cluster_desligado}: DESLIGADO com {stats_desligado['certeza']:.1f}% de certeza")
        print(f"    - {stats_desligado['desligado']:,} DESLIGADO de {stats_desligado['total']:,} total")
    
    # Mostrar clusters intermediários (menos certeza)
    clusters_intermediarios = []
    for cluster_id in range(6):
        if cluster_id not in [cluster_ligado, cluster_desligado]:
            stats = cluster_stats[cluster_id]
            clusters_intermediarios.append(cluster_id)
            print(f"  - Cluster {cluster_id}: Intermediário (certeza: {stats['certeza']:.1f}%)")
    
    return df_analise, cluster_ligado, cluster_desligado, clusters_intermediarios

def criar_visualizacoes_rigoroso(df_analise, colunas_validas, cluster_ligado, cluster_desligado, clusters_intermediarios):
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
    
    # 2. Scatter plot por status
    status_colors = {'LIGADO': 'green', 'DESLIGADO': 'red'}
    for status, color in status_colors.items():
        mask = df_analise['equipamento_status'] == status
        axes[0,1].scatter(dados_pca[mask, 0], dados_pca[mask, 1], 
                         c=color, alpha=0.6, s=1, label=status)
    
    axes[0,1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} var)')
    axes[0,1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} var)')
    axes[0,1].set_title('Classificação por Critérios Rigorosos')
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
        axes[1,1].pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%')
    axes[1,1].set_title('Distribuição do Status')
    
    plt.tight_layout()
    plt.savefig('results/analise_kmeans_clusters_moderado.png', dpi=300, bbox_inches='tight')
    plt.show()

def salvar_modelos_e_dados_rigoroso(kmeans, scaler, df_analise, colunas_validas, 
                                   cluster_ligado, cluster_desligado, clusters_intermediarios):
    """Salva modelos e dados (modo rigoroso)"""
    print("\nSalvando modelos e dados (modo rigoroso)...")
    
    # Salvar modelo K-means
    with open('models/kmeans_model_moderado.pkl', 'wb') as f:
        pickle.dump(kmeans, f)
    
    # Salvar scaler
    with open('models/scaler_model_moderado.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Salvar dados com classificação
    df_analise.to_csv('data/processed/dados_classificados_kmeans_moderado.csv', index=False)
    
    # Salvar dados rotulados (apenas clusters com alta certeza: 1 e 3)
    df_rotulados = df_analise[df_analise['cluster'].isin([cluster_ligado, cluster_desligado])].drop(columns=['cluster'])
    df_rotulados.to_csv('data/normalized/dados_kmeans_rotulados_conservador.csv', index=False)
    
    # Salvar informações do modelo
    info_modelo = {
        'colunas_utilizadas': colunas_validas,
        'cluster_ligado': int(cluster_ligado) if cluster_ligado is not None else None,
        'cluster_desligado': int(cluster_desligado) if cluster_desligado is not None else None,
        'clusters_intermediarios': [int(c) for c in clusters_intermediarios] if clusters_intermediarios else [],
        'numero_clusters': 6,
        'total_amostras_originais': len(df_analise),
        'amostras_classificadas': len(df_rotulados),
        'amostras_ligado': len(df_rotulados[df_rotulados['equipamento_status'] == 'LIGADO']),
        'amostras_desligado': len(df_rotulados[df_rotulados['equipamento_status'] == 'DESLIGADO']),
        'percentual_classificados': len(df_rotulados) / len(df_analise) * 100,
        'estrategia': 'Apenas clusters com alta certeza para treinamento CNN',
        'criterios_desligado': {
            'vel_rms_max': 1,
            'current_max': 10,
            'rpm': 0
        }
    }
    
    with open('models/info_kmeans_model_moderado.json', 'w') as f:
        json.dump(info_modelo, f, indent=2)
    
    print("  - Modelo K-means rigoroso salvo: models/kmeans_model_moderado.pkl")
    print("  - Scaler rigoroso salvo: models/scaler_model_moderado.pkl")
    print("  - Dados classificados: data/processed/dados_classificados_kmeans_moderado.csv")
    print("  - Dados rotulados: data/normalized/dados_kmeans_rotulados_conservador.csv")
    print("  - Info do modelo: models/info_kmeans_model_moderado.json")
    
    print(f"\n📊 Resumo do modo rigoroso (apenas clusters de alta certeza):")
    print(f"  - Total de amostras: {len(df_analise):,}")
    print(f"  - Amostras para treinamento CNN: {len(df_rotulados):,}")
    print(f"  - Amostras LIGADO: {len(df_rotulados[df_rotulados['equipamento_status'] == 'LIGADO']):,}")
    print(f"  - Amostras DESLIGADO: {len(df_rotulados[df_rotulados['equipamento_status'] == 'DESLIGADO']):,}")
    print(f"  - Percentual usado para treino: {len(df_rotulados) / len(df_analise) * 100:.1f}%")
    print(f"  - Clusters selecionados: {cluster_ligado} (LIGADO), {cluster_desligado} (DESLIGADO)")
    print(f"  - Clusters descartados: {clusters_intermediarios}")
    print(f"  - Estratégia: Dados limpos para melhor treinamento CNN")

def main():
    """Função principal"""
    print("=== K-MEANS RIGOROSO - 6 CLUSTERS COM CRITÉRIOS ESPECÍFICOS ===")
    print("=" * 70)
    
    try:
        # 1. Carregar dados normalizados
        df, info_normalizacao = carregar_dados_normalizados()
        
        # 2. Preparar dados normalizados
        df_kmeans, colunas_validas = preparar_dados_normalizados(df, info_normalizacao)
        
        # 3. Usar dados já normalizados
        dados_normalizados, scaler, timestamp = usar_dados_normalizados(df_kmeans, info_normalizacao)
        
        # 4. Executar K-means com 6 clusters
        kmeans, clusters = executar_kmeans(dados_normalizados, n_clusters=6)
        
        # 5. Analisar clusters
        df_analise = analisar_clusters(df_kmeans, clusters, colunas_validas)
        
        # 6. Classificar equipamento (modo rigoroso)
        df_analise, cluster_ligado, cluster_desligado, clusters_intermediarios = classificar_equipamento_rigoroso(df_analise, colunas_validas)
        
        # 7. Criar visualizações
        criar_visualizacoes_rigoroso(df_analise, colunas_validas, cluster_ligado, cluster_desligado, clusters_intermediarios)
        
        # 8. Salvar modelos e dados
        salvar_modelos_e_dados_rigoroso(kmeans, scaler, df_analise, colunas_validas, 
                                       cluster_ligado, cluster_desligado, clusters_intermediarios)
        
        print("\n=== PROCESSO RIGOROSO CONCLUÍDO COM SUCESSO ===")
        
    except Exception as e:
        print(f"\nErro durante o processamento: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


