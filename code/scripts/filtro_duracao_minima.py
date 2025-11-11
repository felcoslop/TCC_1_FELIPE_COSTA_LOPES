"""
Script pra filtrar mudancas de estado muito rapidas que nao fazem sentido.
Por exemplo, se o equipamento aparece como ligado por 2 segundos e depois volta,
provavelmente e ruido e nao uma mudanca real de estado.
"""

import pandas as pd
import numpy as np
import argparse
from datetime import datetime, timedelta

def aplicar_filtro_duracao(df, duracao_minima_minutos=30, threshold_confianca=0.6):
    """
    Aplica filtro de duração mínima para evitar oscilações
    
    Args:
        df (pd.DataFrame): DataFrame com predições (timestamp, predicao, confianca)
        duracao_minima_minutos (int): Duração mínima em minutos para manter um estado
        threshold_confianca (float): Threshold de confiança para aceitar mudanças
        
    Returns:
        pd.DataFrame: DataFrame com predições filtradas
    """
    print(f"🔧 Aplicando filtro de duração mínima...")
    print(f"  - Duração mínima: {duracao_minima_minutos} minutos")
    print(f"  - Threshold de confiança: {threshold_confianca}")
    
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Criar coluna de predição filtrada
    df['predicao_filtrada'] = df['predicao'].copy()
    
    # Estado atual e timestamp de início
    estado_atual = df['predicao'].iloc[0]
    timestamp_inicio = df['timestamp'].iloc[0]
    
    transicoes_removidas = 0
    
    for i in range(1, len(df)):
        predicao_nova = df['predicao'].iloc[i]
        timestamp_atual = df['timestamp'].iloc[i]
        confianca = df['confianca'].iloc[i]
        
        # Se a predição mudou
        if predicao_nova != estado_atual:
            # Calcular quanto tempo passou desde a última mudança de estado
            tempo_decorrido = (timestamp_atual - timestamp_inicio).total_seconds() / 60.0
            
            # Se passou tempo suficiente E a confiança é alta, aceitar mudança
            if tempo_decorrido >= duracao_minima_minutos and confianca >= threshold_confianca:
                # Aceitar mudança de estado
                estado_atual = predicao_nova
                timestamp_inicio = timestamp_atual
            else:
                # Rejeitar mudança - manter estado anterior
                df.loc[i, 'predicao_filtrada'] = estado_atual
                transicoes_removidas += 1
        else:
            # Predição igual ao estado atual, manter
            df.loc[i, 'predicao_filtrada'] = estado_atual
    
    print(f"  - Transições removidas: {transicoes_removidas}")
    
    # Estatísticas antes e depois
    transicoes_antes = 0
    transicoes_depois = 0
    
    for i in range(1, len(df)):
        if df['predicao'].iloc[i] != df['predicao'].iloc[i-1]:
            transicoes_antes += 1
        if df['predicao_filtrada'].iloc[i] != df['predicao_filtrada'].iloc[i-1]:
            transicoes_depois += 1
    
    print(f"\n📊 Estatísticas:")
    print(f"  - Transições ANTES do filtro: {transicoes_antes}")
    print(f"  - Transições DEPOIS do filtro: {transicoes_depois}")
    print(f"  - Redução: {(transicoes_antes - transicoes_depois) / transicoes_antes * 100:.1f}%")
    
    print(f"\n  - Distribuição ANTES:")
    for classe, count in df['predicao'].value_counts().items():
        print(f"    • {classe}: {count} ({count/len(df)*100:.1f}%)")
    
    print(f"\n  - Distribuição DEPOIS:")
    for classe, count in df['predicao_filtrada'].value_counts().items():
        print(f"    • {classe}: {count} ({count/len(df)*100:.1f}%)")
    
    return df

def calcular_duracao_estados(df, coluna_predicao='predicao_filtrada'):
    """
    Calcula a duração de cada período em cada estado
    
    Args:
        df (pd.DataFrame): DataFrame com predições
        coluna_predicao (str): Nome da coluna de predição
        
    Returns:
        pd.DataFrame: DataFrame com períodos e durações
    """
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    periodos = []
    estado_atual = df[coluna_predicao].iloc[0]
    timestamp_inicio = df['timestamp'].iloc[0]
    
    for i in range(1, len(df)):
        if df[coluna_predicao].iloc[i] != estado_atual:
            # Fim do período atual
            timestamp_fim = df['timestamp'].iloc[i-1]
            duracao_minutos = (timestamp_fim - timestamp_inicio).total_seconds() / 60.0
            
            periodos.append({
                'estado': estado_atual,
                'inicio': timestamp_inicio,
                'fim': timestamp_fim,
                'duracao_minutos': duracao_minutos
            })
            
            # Novo período
            estado_atual = df[coluna_predicao].iloc[i]
            timestamp_inicio = df['timestamp'].iloc[i]
    
    # Último período
    timestamp_fim = df['timestamp'].iloc[-1]
    duracao_minutos = (timestamp_fim - timestamp_inicio).total_seconds() / 60.0
    periodos.append({
        'estado': estado_atual,
        'inicio': timestamp_inicio,
        'fim': timestamp_fim,
        'duracao_minutos': duracao_minutos
    })
    
    df_periodos = pd.DataFrame(periodos)
    
    print(f"\n📅 Análise de períodos:")
    print(f"  - Total de períodos: {len(df_periodos)}")
    
    for estado in df_periodos['estado'].unique():
        df_estado = df_periodos[df_periodos['estado'] == estado]
        print(f"\n  - Estado: {estado}")
        print(f"    • Número de períodos: {len(df_estado)}")
        print(f"    • Duração total: {df_estado['duracao_minutos'].sum():.1f} minutos ({df_estado['duracao_minutos'].sum()/60:.1f} horas)")
        print(f"    • Duração média: {df_estado['duracao_minutos'].mean():.1f} minutos")
        print(f"    • Duração mínima: {df_estado['duracao_minutos'].min():.1f} minutos")
        print(f"    • Duração máxima: {df_estado['duracao_minutos'].max():.1f} minutos")
    
    return df_periodos

def parse_arguments():
    """Configura e parseia argumentos da linha de comando"""
    parser = argparse.ArgumentParser(
        description="Filtro de duração mínima para classificações"
    )
    
    parser.add_argument(
        '--arquivo', 
        type=str, 
        required=True,
        help='Caminho para o arquivo CSV com classificações'
    )
    
    parser.add_argument(
        '--duracao', 
        type=int, 
        default=30,
        help='Duração mínima em minutos para aceitar mudança de estado (padrão: 30)'
    )
    
    parser.add_argument(
        '--confianca', 
        type=float, 
        default=0.6,
        help='Threshold de confiança para aceitar mudança (padrão: 0.6)'
    )
    
    parser.add_argument(
        '--saida', 
        type=str,
        help='Caminho para salvar resultados filtrados'
    )
    
    return parser.parse_args()

def main():
    """Função principal"""
    print("=== FILTRO DE DURAÇÃO MÍNIMA ===")
    print("=" * 60)
    
    try:
        args = parse_arguments()
        
        print(f"\n📁 Carregando arquivo: {args.arquivo}")
        df = pd.read_csv(args.arquivo)
        print(f"  - Linhas carregadas: {len(df)}")
        
        # Verificar colunas necessárias
        if 'timestamp' not in df.columns or 'predicao' not in df.columns:
            print("❌ Erro: Arquivo deve ter colunas 'timestamp' e 'predicao'")
            return
        
        if 'confianca' not in df.columns:
            print("⚠️  Aviso: Coluna 'confianca' não encontrada, usando 1.0 para todas")
            df['confianca'] = 1.0
        
        # Aplicar filtro
        df_filtrado = aplicar_filtro_duracao(
            df, 
            duracao_minima_minutos=args.duracao,
            threshold_confianca=args.confianca
        )
        
        # Calcular durações dos estados
        df_periodos = calcular_duracao_estados(df_filtrado)
        
        # Salvar resultados
        if args.saida is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.saida = f"results/classificacao_filtrada_{timestamp}.csv"
        
        import os
        os.makedirs(os.path.dirname(args.saida), exist_ok=True)
        
        df_filtrado.to_csv(args.saida, index=False)
        print(f"\n💾 Resultados salvos em: {args.saida}")
        
        # Salvar também os períodos
        arquivo_periodos = args.saida.replace('.csv', '_periodos.csv')
        df_periodos.to_csv(arquivo_periodos, index=False)
        print(f"💾 Períodos salvos em: {arquivo_periodos}")
        
        print("\n✅ Filtro aplicado com sucesso!")
        
    except Exception as e:
        print(f"\n❌ Erro: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


