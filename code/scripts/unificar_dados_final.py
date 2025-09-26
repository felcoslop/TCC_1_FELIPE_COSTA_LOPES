#!/usr/bin/env python3
"""
Script final para unificar dados com base no timestamp do dados_c_636.csv
Unifica dados_c_636.csv, dados_estimated_preenchidos_avancado.csv e dados_slip_c_636.csv
Garante exatamente 772,238 linhas (mesmo número do dados_c_636.csv)
Remove coluna m_point e mantém timestamp intacto
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

def load_data():
    """Carrega todos os arquivos de dados"""
    print("Carregando dados...")
    
    # Carrega dados_c_636 (referência principal)
    dados_c = pd.read_csv('data/raw/dados_c_636.csv')
    dados_c['time'] = pd.to_datetime(dados_c['time'], format='mixed')
    print(f"dados_c_636: {len(dados_c)} linhas")
    
    # Carrega dados estimated
    dados_estimated = pd.read_csv('data/processed/dados_estimated_preenchidos_avancado.csv')
    dados_estimated['time'] = pd.to_datetime(dados_estimated['time'], format='mixed')
    print(f"dados_estimated: {len(dados_estimated)} linhas")
    
    # Carrega dados slip
    dados_slip = pd.read_csv('data/raw/dados_slip_c_636.csv')
    dados_slip['time'] = pd.to_datetime(dados_slip['time'], format='mixed')
    print(f"dados_slip: {len(dados_slip)} linhas")
    
    return dados_c, dados_estimated, dados_slip

def sync_data_final(dados_c, dados_other, prefix, tolerance_hours=1):
    """
    Sincroniza dados usando merge_asof e garante exatamente o mesmo número de linhas
    """
    print(f"Sincronizando {prefix}...")
    
    # Remove coluna time dos dados
    other_data = dados_other.drop('time', axis=1)
    
    # Adiciona prefixo às colunas
    other_data_prefixed = other_data.add_prefix(f"{prefix}_")
    
    # Usa merge_asof para encontrar o valor mais próximo
    merged = pd.merge_asof(
        dados_c[['time']].sort_values('time'),
        dados_other[['time']].sort_values('time').join(other_data_prefixed),
        on='time',
        direction='backward',
        tolerance=pd.Timedelta(hours=tolerance_hours)
    )
    
    # Remove coluna time duplicada
    merged = merged.drop('time', axis=1)
    
    # Cria DataFrame com exatamente o mesmo número de linhas que dados_c
    result = pd.DataFrame(index=range(len(dados_c)), columns=merged.columns)
    
    # Copia os dados disponíveis
    if len(merged) > 0:
        result.loc[:min(len(merged)-1, len(dados_c)-1)] = merged.iloc[:min(len(merged), len(dados_c))].values
    
    # Forward fill para preencher valores NaN
    result = result.ffill()
    
    print(f"{prefix} sincronizado: {result.notna().sum().sum()} valores preenchidos")
    return result

def main():
    """Função principal"""
    print("=== UNIFICADOR DE DADOS FINAL ===")
    print("Unificando dados_c_636.csv, dados_estimated_preenchidos_avancado.csv e dados_slip_c_636.csv")
    print("Garantindo exatamente 772,238 linhas")
    print("Removendo coluna m_point e mantendo timestamp intacto")
    print()
    
    # Verifica se é modo teste
    test_mode = len(sys.argv) > 1 and sys.argv[1] == '--test'
    if test_mode:
        print("🧪 MODO TESTE: Processando apenas 1000 linhas")
        print()
    
    # Carrega todos os dados
    dados_c, dados_estimated, dados_slip = load_data()
    
    # Se modo teste, limita os dados
    if test_mode:
        dados_c = dados_c.head(1000)
        print(f"Modo teste: limitando dados_c para {len(dados_c)} linhas")
    
    print(f"\nDados carregados:")
    print(f"- dados_c_636: {len(dados_c)} linhas (referência)")
    print(f"- dados_estimated: {len(dados_estimated)} linhas")
    print(f"- dados_slip: {len(dados_slip)} linhas")
    print()
    
    # Cria DataFrame base com dados_c
    df_unificado = dados_c.copy()
    print(f"DataFrame base criado com {len(df_unificado)} linhas")
    
    # Remove coluna m_point se existir
    if 'm_point' in df_unificado.columns:
        df_unificado = df_unificado.drop('m_point', axis=1)
        print("Coluna 'm_point' removida do DataFrame base")
    
    # Sincroniza dados estimated
    print("\n=== SINCRONIZANDO DADOS ESTIMATED ===")
    estimated_cols = sync_data_final(dados_c, dados_estimated, "estimated", tolerance_hours=0.1)  # 6 minutos
    
    # Adiciona colunas estimated ao DataFrame unificado
    df_unificado = pd.concat([df_unificado, estimated_cols], axis=1)
    
    # Sincroniza dados slip
    print("\n=== SINCRONIZANDO DADOS SLIP ===")
    slip_cols = sync_data_final(dados_c, dados_slip, "slip", tolerance_hours=0.05)  # 3 minutos
    
    # Adiciona colunas slip ao DataFrame unificado
    df_unificado = pd.concat([df_unificado, slip_cols], axis=1)
    
    # Remove todas as colunas que contenham 'm_point' (incluindo slip_m_point)
    colunas_m_point = [col for col in df_unificado.columns if 'm_point' in col]
    if colunas_m_point:
        df_unificado = df_unificado.drop(colunas_m_point, axis=1)
        print(f"Colunas removidas: {colunas_m_point}")
    
    # Salva arquivo unificado
    output_file = 'data/processed/dados_unificados_final.csv'
    if test_mode:
        output_file = 'data/processed/dados_unificados_teste_final.csv'
    
    print(f"\n=== SALVANDO ARQUIVO UNIFICADO ===")
    print(f"Salvando em: {output_file}")
    print(f"Total de colunas: {len(df_unificado.columns)}")
    print(f"Total de linhas: {len(df_unificado)}")
    
    # Cria diretório se não existir
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Salva arquivo
    df_unificado.to_csv(output_file, index=False)
    
    print(f"\n✅ Arquivo unificado salvo com sucesso!")
    print(f"📊 Estatísticas finais:")
    print(f"   - Linhas: {len(df_unificado)}")
    print(f"   - Colunas: {len(df_unificado.columns)}")
    print(f"   - Tamanho: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
    
    # Mostra resumo das colunas
    print(f"\n📋 Colunas incluídas:")
    print(f"   - Dados originais (dados_c): {len(dados_c.columns)} colunas")
    print(f"   - Estimated: {len(estimated_cols.columns)} colunas")
    print(f"   - Slip: {len(slip_cols.columns)} colunas")
    
    # Verifica se o número de linhas está correto
    expected_lines = 772238
    if len(df_unificado) == expected_lines:
        print(f"\n🎉 SUCESSO: Arquivo tem exatamente {expected_lines} linhas!")
    else:
        print(f"\n⚠️  AVISO: Arquivo tem {len(df_unificado)} linhas, esperado {expected_lines}")

if __name__ == "__main__":
    main()


