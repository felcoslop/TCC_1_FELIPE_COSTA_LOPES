#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para preencher lacunas nos dados estimated de forma eficiente
Usa métodos vetorizados do pandas para processar 4 milhões de linhas rapidamente

Lógica:
- current e rotational_speed são mutuamente exclusivos com vel_rms
- Preencher current/rpm onde vel_rms está presente
- Preencher vel_rms onde current/rpm estão presentes
"""

import pandas as pd
import numpy as np
from scipy import interpolate
import warnings
warnings.filterwarnings('ignore')

def preencher_estimated_eficiente(arquivo_entrada='data/raw/dados_estimated_c_636.csv', 
                                 arquivo_saida='data/processed/dados_estimated_preenchidos.csv'):
    """
    Preenche lacunas nos dados estimated de forma eficiente
    """
    print("=== PREENCHIMENTO EFICIENTE DOS DADOS ESTIMATED ===")
    print("=" * 60)
    
    # 1. Carregar dados
    print("1. Carregando dados estimated...")
    df = pd.read_csv(arquivo_entrada)
    df['time'] = pd.to_datetime(df['time'], format='ISO8601')
    
    print(f"   - Linhas originais: {len(df):,}")
    print(f"   - Período: {df['time'].min()} até {df['time'].max()}")
    
    # Verificar distribuição de valores nulos
    print("\n2. Análise de valores nulos:")
    null_counts = df.isnull().sum()
    for col, count in null_counts.items():
        if col != 'time' and count > 0:
            pct = (count / len(df)) * 100
            print(f"   - {col}: {count:,} nulos ({pct:.1f}%)")
    
    # 2. Estratégia de preenchimento
    print("\n3. Aplicando estratégia de preenchimento...")
    
    # Criar cópia para trabalhar
    df_filled = df.copy()
    
    # Método 1: Preencher current e rotational_speed onde vel_rms está presente
    print("   - Preenchendo current e rotational_speed onde vel_rms está presente...")
    
    # Usar forward fill e backward fill para preenchimento
    df_filled['current'] = df_filled['current'].fillna(method='ffill').fillna(method='bfill')
    df_filled['rotational_speed'] = df_filled['rotational_speed'].fillna(method='ffill').fillna(method='bfill')
    
    # Método 2: Preencher vel_rms onde current e rotational_speed estão presentes
    print("   - Preenchendo vel_rms onde current e rotational_speed estão presentes...")
    
    df_filled['vel_rms'] = df_filled['vel_rms'].fillna(method='ffill').fillna(method='bfill')
    
    # 3. Interpolação linear para valores ainda faltantes
    print("\n4. Aplicando interpolação linear para valores restantes...")
    
    # Interpolar current
    if df_filled['current'].isnull().any():
        mask_valid = df_filled['current'].notna()
        if mask_valid.sum() > 1:
            df_filled['current'] = df_filled['current'].interpolate(method='linear')
    
    # Interpolar rotational_speed
    if df_filled['rotational_speed'].isnull().any():
        mask_valid = df_filled['rotational_speed'].notna()
        if mask_valid.sum() > 1:
            df_filled['rotational_speed'] = df_filled['rotational_speed'].interpolate(method='linear')
    
    # Interpolar vel_rms
    if df_filled['vel_rms'].isnull().any():
        mask_valid = df_filled['vel_rms'].notna()
        if mask_valid.sum() > 1:
            df_filled['vel_rms'] = df_filled['vel_rms'].interpolate(method='linear')
    
    # 4. Preenchimento final com valores médios
    print("\n5. Preenchimento final com valores médios...")
    
    # Calcular médias dos valores válidos
    current_mean = df_filled['current'].mean() if df_filled['current'].notna().any() else 0
    rpm_mean = df_filled['rotational_speed'].mean() if df_filled['rotational_speed'].notna().any() else 0
    vel_rms_mean = df_filled['vel_rms'].mean() if df_filled['vel_rms'].notna().any() else 0
    
    # Preencher valores ainda faltantes com médias
    df_filled['current'] = df_filled['current'].fillna(current_mean)
    df_filled['rotational_speed'] = df_filled['rotational_speed'].fillna(rpm_mean)
    df_filled['vel_rms'] = df_filled['vel_rms'].fillna(vel_rms_mean)
    
    # 5. Corrigir valores negativos de RPM
    print("\n6. Corrigindo valores negativos de RPM...")
    rpm_negativos_antes = (df_filled['rotational_speed'] < 0).sum()
    if rpm_negativos_antes > 0:
        print(f"   - Valores negativos de RPM encontrados: {rpm_negativos_antes:,}")
        df_filled['rotational_speed'] = df_filled['rotational_speed'].clip(lower=0)
        print(f"   - Valores negativos corrigidos para zero")
    else:
        print(f"   - Nenhum valor negativo de RPM encontrado")
    
    # 6. Verificar resultados
    print("\n7. Verificação dos resultados:")
    final_null_counts = df_filled.isnull().sum()
    for col, count in final_null_counts.items():
        if col != 'time':
            print(f"   - {col}: {count} nulos restantes")
    
    # 7. Salvar dados preenchidos
    print(f"\n8. Salvando dados preenchidos em {arquivo_saida}...")
    df_filled.to_csv(arquivo_saida, index=False)
    
    # 7. Estatísticas finais
    print("\n=== ESTATÍSTICAS FINAIS ===")
    print(f"Total de linhas: {len(df_filled):,}")
    print(f"Valores preenchidos:")
    print(f"   - Current: média={df_filled['current'].mean():.3f}, std={df_filled['current'].std():.3f}")
    print(f"   - Rotational_speed: média={df_filled['rotational_speed'].mean():.3f}, std={df_filled['rotational_speed'].std():.3f}")
    print(f"   - Vel_rms: média={df_filled['vel_rms'].mean():.3f}, std={df_filled['vel_rms'].std():.3f}")
    
    print(f"\nArquivo salvo: {arquivo_saida}")
    print("=== PREENCHIMENTO CONCLUÍDO ===")
    
    return df_filled

def preencher_estimated_avancado(arquivo_entrada='data/raw/dados_estimated_c_636.csv', 
                                arquivo_saida='data/processed/dados_estimated_preenchidos_avancado.csv'):
    """
    Versão mais avançada com interpolação por splines para melhor qualidade
    """
    print("=== PREENCHIMENTO AVANÇADO DOS DADOS ESTIMATED ===")
    print("=" * 60)
    
    # 1. Carregar dados
    print("1. Carregando dados estimated...")
    df = pd.read_csv(arquivo_entrada)
    df['time'] = pd.to_datetime(df['time'], format='ISO8601')
    
    print(f"   - Linhas originais: {len(df):,}")
    
    # 2. Criar cópia para trabalhar
    df_filled = df.copy()
    
    # 3. Preenchimento inteligente baseado em janelas temporais
    print("\n2. Aplicando preenchimento inteligente...")
    
    # Usar rolling window para preenchimento
    window_size = 100  # Janela de 100 amostras
    
    # Preencher current com média móvel
    df_filled['current'] = df_filled['current'].fillna(
        df_filled['current'].rolling(window=window_size, min_periods=1).mean()
    )
    
    # Preencher rotational_speed com média móvel
    df_filled['rotational_speed'] = df_filled['rotational_speed'].fillna(
        df_filled['rotational_speed'].rolling(window=window_size, min_periods=1).mean()
    )
    
    # Preencher vel_rms com média móvel
    df_filled['vel_rms'] = df_filled['vel_rms'].fillna(
        df_filled['vel_rms'].rolling(window=window_size, min_periods=1).mean()
    )
    
    # 4. Interpolação spline para valores ainda faltantes
    print("\n3. Aplicando interpolação spline...")
    
    # Criar índices numéricos para interpolação
    df_filled['idx'] = range(len(df_filled))
    
    # Interpolar current
    if df_filled['current'].isnull().any():
        mask_valid = df_filled['current'].notna()
        if mask_valid.sum() > 3:
            valid_idx = df_filled[mask_valid]['idx'].values
            valid_current = df_filled[mask_valid]['current'].values
            
            # Interpolação spline
            spline = interpolate.UnivariateSpline(valid_idx, valid_current, s=0)
            df_filled['current'] = spline(df_filled['idx'])
    
    # Interpolar rotational_speed
    if df_filled['rotational_speed'].isnull().any():
        mask_valid = df_filled['rotational_speed'].notna()
        if mask_valid.sum() > 3:
            valid_idx = df_filled[mask_valid]['idx'].values
            valid_rpm = df_filled[mask_valid]['rotational_speed'].values
            
            # Interpolação spline
            spline = interpolate.UnivariateSpline(valid_idx, valid_rpm, s=0)
            df_filled['rotational_speed'] = spline(df_filled['idx'])
    
    # Interpolar vel_rms
    if df_filled['vel_rms'].isnull().any():
        mask_valid = df_filled['vel_rms'].notna()
        if mask_valid.sum() > 3:
            valid_idx = df_filled[mask_valid]['idx'].values
            valid_vel_rms = df_filled[mask_valid]['vel_rms'].values
            
            # Interpolação spline
            spline = interpolate.UnivariateSpline(valid_idx, valid_vel_rms, s=0)
            df_filled['vel_rms'] = spline(df_filled['idx'])
    
    # Remover coluna auxiliar
    df_filled = df_filled.drop('idx', axis=1)
    
    # 5. Corrigir valores negativos de RPM
    print("\n4. Corrigindo valores negativos de RPM...")
    rpm_negativos_antes = (df_filled['rotational_speed'] < 0).sum()
    if rpm_negativos_antes > 0:
        print(f"   - Valores negativos de RPM encontrados: {rpm_negativos_antes:,}")
        df_filled['rotational_speed'] = df_filled['rotational_speed'].clip(lower=0)
        print(f"   - Valores negativos corrigidos para zero")
    else:
        print(f"   - Nenhum valor negativo de RPM encontrado")
    
    # 6. Salvar dados
    print(f"\n5. Salvando dados em {arquivo_saida}...")
    df_filled.to_csv(arquivo_saida, index=False)
    
    # 6. Estatísticas
    print("\n=== ESTATÍSTICAS FINAIS ===")
    print(f"Total de linhas: {len(df_filled):,}")
    print(f"Valores preenchidos:")
    print(f"   - Current: média={df_filled['current'].mean():.3f}, std={df_filled['current'].std():.3f}")
    print(f"   - Rotational_speed: média={df_filled['rotational_speed'].mean():.3f}, std={df_filled['rotational_speed'].std():.3f}")
    print(f"   - Vel_rms: média={df_filled['vel_rms'].mean():.3f}, std={df_filled['vel_rms'].std():.3f}")
    
    print("=== PREENCHIMENTO AVANÇADO CONCLUÍDO ===")
    
    return df_filled

if __name__ == "__main__":
    print("Escolha o método de preenchimento:")
    print("1. Método eficiente (rápido)")
    print("2. Método avançado (melhor qualidade)")
    
    escolha = input("Digite sua escolha (1 ou 2): ").strip()
    
    if escolha == "1":
        df_preenchido = preencher_estimated_eficiente()
    elif escolha == "2":
        df_preenchido = preencher_estimated_avancado()
    else:
        print("Escolha inválida. Usando método eficiente por padrão.")
        df_preenchido = preencher_estimated_eficiente()
