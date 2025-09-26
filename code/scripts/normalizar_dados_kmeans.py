#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para normalizar dados para K-means usando dados_unificados_final.csv
- Trabalha com dados já unificados (20 colunas)
- Remove colunas m_point automaticamente
- Gera arquivo CSV normalizado pronto para K-means
- K-means é não supervisionado, não requer divisão treino/teste
- Mantém timestamp intacto durante normalização
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import json
import pickle
import warnings
import os
from datetime import datetime
warnings.filterwarnings('ignore')

def criar_diretorios():
    """Cria diretórios necessários"""
    diretorios = ['data/normalized', 'models', 'results', 'plots']
    for diretorio in diretorios:
        os.makedirs(diretorio, exist_ok=True)
    print("Diretórios criados/verificados com sucesso!")

def carregar_dados():
    """Carrega os dados unificados finais"""
    print("Carregando dados unificados finais...")
    
    try:
        # Carregar dados unificados completos
        df = pd.read_csv('data/processed/dados_unificados_final.csv')
        print(f"  - Shape: {df.shape}")
        print(f"  - Colunas: {len(df.columns)}")
        
        # Converter coluna time para datetime
        df['time'] = pd.to_datetime(df['time'])
        print(f"  - Período: {df['time'].min()} até {df['time'].max()}")
        print(f"  - Dados carregados completos do arquivo unificado")
        
        return df
    except Exception as e:
        print(f"Erro ao carregar dados: {str(e)}")
        return None

def remover_colunas_m_point(df):
    """Remove colunas relacionadas a m_point"""
    print("\nRemovendo colunas m_point...")
    
    # Identificar colunas m_point
    colunas_m_point = [col for col in df.columns if 'm_point' in col.lower()]
    print(f"  - Colunas m_point encontradas: {len(colunas_m_point)}")
    for col in colunas_m_point:
        print(f"    - {col}")
    
    # Remover colunas m_point
    df_sem_m_point = df.drop(columns=colunas_m_point)
    
    print(f"  - Colunas após remoção: {len(df_sem_m_point.columns)}")
    print(f"  - Linhas mantidas: {len(df_sem_m_point):,}")
    
    return df_sem_m_point

def analisar_dados(df):
    """Analisa características dos dados"""
    print("\nAnalisando características dos dados...")
    
    # Informações gerais
    print(f"  - Total de linhas: {len(df):,}")
    print(f"  - Total de colunas: {len(df.columns)}")
    
    # Identificar colunas numéricas (excluindo time)
    colunas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'time' in colunas_numericas:
        colunas_numericas.remove('time')
    
    print(f"  - Colunas numéricas: {len(colunas_numericas)}")
    
    # Análise de valores nulos
    print("\nAnálise de valores nulos:")
    colunas_com_nulos = df[colunas_numericas].isnull().sum()
    colunas_com_nulos = colunas_com_nulos[colunas_com_nulos > 0].sort_values(ascending=False)
    
    if len(colunas_com_nulos) > 0:
        print(f"  - Colunas com valores nulos: {len(colunas_com_nulos)}")
        for col, n_nulos in colunas_com_nulos.head(10).items():
            pct = (n_nulos / len(df)) * 100
            print(f"    - {col}: {n_nulos:,} nulos ({pct:.1f}%)")
    else:
        print("  - Nenhuma coluna com valores nulos!")
    
    # Análise de estatísticas básicas
    print("\nEstatísticas básicas das colunas numéricas:")
    stats = df[colunas_numericas].describe()
    print(f"  - Média das médias: {stats.loc['mean'].mean():.3f}")
    print(f"  - Desvio padrão médio: {stats.loc['std'].mean():.3f}")
    print(f"  - Range médio: {(stats.loc['max'] - stats.loc['min']).mean():.3f}")
    
    return colunas_numericas

def preparar_dados_para_normalizacao(df, colunas_numericas):
    """Prepara dados para normalização, mantendo timestamp intacto"""
    print("\nPreparando dados para normalização...")
    
    # Remover colunas com muitos valores nulos (>50%) - mais conservador
    colunas_validas = []
    colunas_removidas = []
    
    for col in colunas_numericas:
        if col != 'time':
            pct_nulos = (df[col].isnull().sum() / len(df)) * 100
            if pct_nulos < 50:  # Manter colunas com menos de 50% nulos
                colunas_validas.append(col)
            else:
                colunas_removidas.append((col, pct_nulos))
    
    print(f"  - Colunas selecionadas: {len(colunas_validas)}")
    print(f"  - Colunas removidas: {len(colunas_removidas)}")
    
    if colunas_removidas:
        print("  - Colunas removidas (com muitos nulos):")
        for col, pct in colunas_removidas[:10]:  # Mostrar apenas as primeiras 10
            print(f"    - {col}: {pct:.1f}% nulos")
    
    # Criar dataset para normalização (incluindo timestamp)
    df_norm = df[['time'] + colunas_validas].copy()
    
    # Preencher valores nulos com a mediana da coluna (mais conservador que dropna)
    print("  - Preenchendo valores nulos com mediana...")
    for col in colunas_validas:
        if df_norm[col].isnull().any():
            mediana = df_norm[col].median()
            df_norm[col].fillna(mediana, inplace=True)
    
    # Apenas remover linhas que ainda tenham valores nulos (muito raro após preenchimento)
    df_norm_clean = df_norm.dropna()
    
    print(f"  - Linhas após limpeza: {len(df_norm_clean):,}")
    print(f"  - Linhas removidas: {len(df_norm) - len(df_norm_clean):,}")
    print(f"  - Percentual de dados mantidos: {(len(df_norm_clean)/len(df))*100:.1f}%")
    print(f"  - Timestamp mantido intacto")
    
    return df_norm_clean, colunas_validas

def normalizar_dados_maxmin(df_norm, colunas_validas):
    """Normaliza dados usando Max-Min Scaler, mantendo timestamp intacto"""
    print("\nNormalizando dados com Max-Min Scaler...")
    
    # Separar timestamp das features numéricas
    timestamp = df_norm['time'].copy()
    df_features = df_norm[colunas_validas].copy()
    
    # Criar scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Normalizar apenas as features numéricas
    dados_normalizados = scaler.fit_transform(df_features)
    
    print("  - Dados normalizados com sucesso!")
    print(f"  - Shape features: {dados_normalizados.shape}")
    print(f"  - Range: [{dados_normalizados.min():.6f}, {dados_normalizados.max():.6f}]")
    print(f"  - Média: {dados_normalizados.mean():.6f}")
    print(f"  - Desvio padrão: {dados_normalizados.std():.6f}")
    print(f"  - Timestamp mantido separadamente: {len(timestamp)} registros")
    
    return dados_normalizados, scaler, timestamp

def preparar_dados_kmeans(dados_normalizados, colunas_validas, timestamp):
    """Prepara dados normalizados para K-means, incluindo timestamp"""
    print("\nPreparando dados para K-means...")
    
    # Criar DataFrame com features normalizadas
    df_features = pd.DataFrame(dados_normalizados, columns=colunas_validas)
    
    # Adicionar timestamp
    df_kmeans = pd.concat([timestamp.reset_index(drop=True), df_features], axis=1)
    
    print(f"  - Shape para K-means: {df_kmeans.shape}")
    print(f"  - Colunas: {len(df_kmeans.columns)}")
    print(f"  - Timestamp incluído: Sim")
    
    return df_kmeans

# Função removida - não é necessária para K-means
# K-means é um algoritmo de aprendizado não supervisionado que não requer divisão treino/teste

def criar_visualizacoes(dados_normalizados, colunas_validas, df_original):
    """Cria visualizações dos dados normalizados"""
    print("\nCriando visualizações...")
    
    # Selecionar algumas colunas para visualização (primeiras 20)
    colunas_viz = colunas_validas[:20]
    
    # Criar figura
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Distribuição antes da normalização
    df_original_viz = df_original[colunas_viz]
    axes[0,0].boxplot([df_original_viz[col].dropna() for col in colunas_viz[:10]], 
                      labels=colunas_viz[:10])
    axes[0,0].set_title('Distribuição Original (primeiras 10 colunas)')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # 2. Distribuição após normalização
    dados_viz = dados_normalizados[:, :10]
    axes[0,1].boxplot([dados_viz[:, i] for i in range(10)], 
                      labels=colunas_viz[:10])
    axes[0,1].set_title('Distribuição Normalizada (primeiras 10 colunas)')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # 3. Histograma de uma coluna específica (antes)
    col_exemplo = colunas_viz[0]
    axes[1,0].hist(df_original[col_exemplo].dropna(), bins=50, alpha=0.7, color='blue')
    axes[1,0].set_title(f'Histograma Original - {col_exemplo}')
    axes[1,0].set_xlabel('Valor')
    axes[1,0].set_ylabel('Frequência')
    
    # 4. Histograma de uma coluna específica (depois)
    col_idx = colunas_validas.index(col_exemplo)
    axes[1,1].hist(dados_normalizados[:, col_idx], bins=50, alpha=0.7, color='red')
    axes[1,1].set_title(f'Histograma Normalizado - {col_exemplo}')
    axes[1,1].set_xlabel('Valor Normalizado')
    axes[1,1].set_ylabel('Frequência')
    
    plt.tight_layout()
    plt.savefig('plots/dados_normalizados_analise.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("  - Visualizações salvas em: plots/dados_normalizados_analise.png")

def salvar_dados_e_modelos(dados_normalizados, scaler, colunas_validas, X_kmeans):
    """Salva dados normalizados e modelos para K-means"""
    print("\nSalvando dados normalizados e modelos...")
    
    # Salvar scaler
    with open('models/scaler_maxmin.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("  - Scaler salvo: models/scaler_maxmin.pkl")
    
    # Salvar dados normalizados completos
    np.save('data/normalized/dados_normalizados_completos.npy', dados_normalizados)
    print("  - Dados normalizados completos: data/normalized/dados_normalizados_completos.npy")
    
    # Salvar dados para K-means (arquivo principal)
    X_kmeans.to_csv('data/normalized/dados_kmeans.csv', index=False)
    print("  - Dados para K-means: data/normalized/dados_kmeans.csv")
    
    # Salvar informações do processamento
    info_processamento = {
        'timestamp': datetime.now().isoformat(),
        'arquivo_origem': 'dados_unificados_final.csv',
        'colunas_utilizadas': colunas_validas,
        'numero_colunas': len(colunas_validas),
        'numero_amostras': len(dados_normalizados),
        'shape_dados_normalizados': list(dados_normalizados.shape),
        'shape_kmeans': list(X_kmeans.shape),
        'range_normalizacao': [float(dados_normalizados.min()), float(dados_normalizados.max())],
        'media_normalizada': float(dados_normalizados.mean()),
        'desvio_padrao_normalizado': float(dados_normalizados.std()),
        'tipo_scaler': 'MinMaxScaler',
        'feature_range': [0, 1],
        'colunas_m_point_removidas': True,
        'observacao': 'Dados carregados do arquivo unificado final (20 colunas)'
    }
    
    with open('models/info_normalizacao.json', 'w') as f:
        json.dump(info_processamento, f, indent=2)
    
    print("  - Informações do processamento: models/info_normalizacao.json")

def main():
    """Função principal"""
    print("=== NORMALIZAÇÃO DE DADOS PARA K-MEANS ===")
    print("=" * 50)
    print("Carregando dados unificados finais (20 colunas)")
    print("K-means é um algoritmo não supervisionado - não requer divisão treino/teste")
    print("=" * 50)
    
    try:
        # 1. Criar diretórios
        criar_diretorios()
        
        # 2. Carregar dados unificados finais
        df = carregar_dados()
        if df is None:
            return
        
        # 3. Remover colunas m_point
        df_sem_m_point = remover_colunas_m_point(df)
        
        # 4. Analisar dados
        colunas_numericas = analisar_dados(df_sem_m_point)
        
        # 5. Preparar dados para normalização
        df_norm, colunas_validas = preparar_dados_para_normalizacao(df_sem_m_point, colunas_numericas)
        
        # 6. Normalizar dados
        dados_normalizados, scaler, timestamp = normalizar_dados_maxmin(df_norm, colunas_validas)
        
        # 7. Preparar dados para K-means
        X_kmeans = preparar_dados_kmeans(dados_normalizados, colunas_validas, timestamp)
        
        # 8. Criar visualizações
        criar_visualizacoes(dados_normalizados, colunas_validas, df_sem_m_point)
        
        # 9. Salvar dados e modelos
        salvar_dados_e_modelos(dados_normalizados, scaler, colunas_validas, X_kmeans)
        
        print("\n=== PROCESSO CONCLUÍDO COM SUCESSO ===")
        print("\nDados preparados para K-means:")
        print("  - Arquivo principal: data/normalized/dados_kmeans.csv")
        print("  - Dados normalizados: data/normalized/dados_normalizados_completos.npy")
        print("  - Scaler: models/scaler_maxmin.pkl")
        print("  - Informações: models/info_normalizacao.json")
        print("\nO arquivo dados_kmeans.csv contém os dados normalizados prontos para K-means")
        
    except Exception as e:
        print(f"\nErro durante o processamento: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()