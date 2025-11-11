"""
Visualização 3D de intervalo específico para equipamentos MECÂNICOS.
Mostra 3 dias de dados com marcação clara de estados LIGADO/DESLIGADO.
Temperatura x Vibração x Tempo (com datas/horas visíveis)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.dates as mdates
from pathlib import Path
import argparse
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from utils.artifact_paths import (
    processed_classificado_path,
    processed_rotulado_path,
    results_dir,
)
import pickle
from scipy.stats import iqr

def remover_outliers_iqr(series, factor=3.0):
    """Remove outliers usando IQR (Interquartile Range)"""
    if len(series) < 4:
        return series
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    return series.clip(lower=lower, upper=upper)

def criar_visualizacao_3d_intervalo_mecanico(mpoint, dias=3):
    """Cria visualização 3D de intervalo para equipamento MECÂNICO com DADOS REAIS"""
    print("="*80)
    print(f"VISUALIZAÇÃO 3D - INTERVALO DE {dias} DIAS - EQUIPAMENTO MECÂNICO")
    print("="*80)
    print(f"Mpoint: {mpoint}")
    
    # Carregar dados ORIGINAIS (não normalizados)
    print("[INFO] Carregando dados ORIGINAIS (não normalizados)...")
    
    DIR_RAW_PREENCHIDO = BASE_DIR / 'data' / 'raw_preenchido'
    arquivos_periodo = sorted(DIR_RAW_PREENCHIDO.glob(f'periodo_*_final_{mpoint}.csv'))
    
    if not arquivos_periodo:
        print(f"[ERRO] Dados originais não encontrados em {DIR_RAW_PREENCHIDO}")
        return False
    
    # Combinar todos os períodos
    dfs_originais = []
    for arq in arquivos_periodo:
        df_orig = pd.read_csv(arq)
        df_orig['time'] = pd.to_datetime(df_orig['time'], format='mixed', utc=True)
        dfs_originais.append(df_orig)
    
    df_original = pd.concat(dfs_originais, ignore_index=True)
    df_original = df_original.sort_values('time').reset_index(drop=True)
    
    # APLICAR REMOÇÃO DE OUTLIERS nos dados REAIS
    print("[INFO] Aplicando remoção de outliers (IQR)...")
    if 'object_temp' in df_original.columns:
        df_original['object_temp'] = remover_outliers_iqr(df_original['object_temp'], factor=3.0)
    
    # Remover outliers de todas as colunas de vibração
    colunas_vibracao = [col for col in df_original.columns if 'vel_' in col.lower()]
    for col in colunas_vibracao:
        df_original[col] = remover_outliers_iqr(df_original[col], factor=3.0)
    
    print(f"  - Outliers removidos de {1 + len(colunas_vibracao)} colunas")
    
    # Carregar dados classificados (apenas para pegar os estados)
    dados_path = processed_classificado_path(mpoint)
    
    if not dados_path.exists():
        print(f"[ERRO] Dados classificados não encontrados: {dados_path}")
        return False
    
    df_classificado = pd.read_csv(dados_path)
    df_classificado['time'] = pd.to_datetime(df_classificado['time'], format='mixed', utc=True)
    
    # Merge para juntar dados originais com classificação
    df = df_original.copy()
    df = pd.merge(df, df_classificado[['time', 'equipamento_status', 'cluster']], 
                  on='time', how='left')
    
    # Preencher valores nulos de status (se houver)
    df['equipamento_status'] = df['equipamento_status'].fillna('DESCONHECIDO')
    
    print(f"[INFO] Dados REAIS carregados: {len(df):,} registros (SEM outliers)")
    
    # Converter timestamp
    df['time'] = pd.to_datetime(df['time'], format='mixed', utc=True)
    
    # Ordenar por tempo
    df = df.sort_values('time').reset_index(drop=True)
    
    print(f"[INFO] Total de dados: {len(df):,} registros")
    print(f"[INFO] Período completo: {df['time'].min()} até {df['time'].max()}")
    
    # SELEÇÃO INTELIGENTE DE INTERVALO
    # Objetivo: Encontrar período com TEMPO CONSIDERÁVEL de DESLIGADO e LIGADO
    print("\n[INFO] Procurando intervalo balanceado (mínimo 20% cada estado)...")
    
    # Detectar transições
    df['mudanca_estado'] = (df['equipamento_status'] != df['equipamento_status'].shift(1)).astype(int)
    transicoes = df[df['mudanca_estado'] == 1].copy()
    
    print(f"[INFO] Encontradas {len(transicoes)} transições de estado")
    
    # Tentar múltiplos intervalos candidatos e escolher o melhor
    melhor_intervalo = None
    melhor_score = -1
    
    duracao_intervalo = pd.Timedelta(days=dias)
    
    # Estratégia: Verificar múltiplas janelas deslizantes
    tempo_total = (df['time'].max() - df['time'].min()).total_seconds()
    num_janelas = min(30, int(tempo_total / (dias * 24 * 3600)))  # Máximo 30 janelas
    
    for i in range(max(1, num_janelas)):
        # Janela deslizante
        t_offset = i * (tempo_total / max(1, num_janelas))
        inicio_teste = df['time'].min() + pd.Timedelta(seconds=t_offset)
        fim_teste = inicio_teste + duracao_intervalo
        
        if fim_teste > df['time'].max():
            break
        
        # Filtrar intervalo de teste
        mask = (df['time'] >= inicio_teste) & (df['time'] <= fim_teste)
        df_teste = df[mask].copy()
        
        if len(df_teste) < 100:  # Intervalo muito pequeno
            continue
        
        # Calcular métricas do intervalo
        n_ligado = (df_teste['equipamento_status'] == 'LIGADO').sum()
        n_desligado = (df_teste['equipamento_status'] == 'DESLIGADO').sum()
        total = len(df_teste)
        
        pct_ligado = (n_ligado / total) * 100 if total > 0 else 0
        pct_desligado = (n_desligado / total) * 100 if total > 0 else 0
        n_transicoes = df_teste['mudanca_estado'].sum()
        
        # Score: priorizar intervalos com AMBOS os estados bem representados
        min_pct = min(pct_ligado, pct_desligado)
        
        # CRITÉRIO PRINCIPAL: ambos estados presentes (mínimo 20% cada)
        if min_pct >= 20:
            score_balance = 100 - abs(50 - pct_ligado)  # Máximo quando 50/50
            score_minimo = min_pct  # Bônus por ter mais do estado minoritário
            score_transicoes = min(n_transicoes, 20)  # Máximo 20 transições
            
            score_total = (score_balance * 2.0) + (score_minimo * 1.5) + (score_transicoes * 0.5)
        else:
            score_total = 0  # Rejeitar intervalos sem representação adequada
        
        if score_total > melhor_score:
            melhor_score = score_total
            melhor_intervalo = {
                'inicio': inicio_teste,
                'fim': fim_teste,
                'df': df_teste,
                'pct_ligado': pct_ligado,
                'pct_desligado': pct_desligado,
                'n_transicoes': n_transicoes,
                'score': score_total
            }
    
    # Se não encontrou intervalo ideal, usar fallback
    if melhor_intervalo is None or melhor_score <= 0:
        print("  [AVISO] Não encontrou intervalo balanceado, usando início dos dados...")
        inicio = df['time'].min()
        fim = inicio + duracao_intervalo
        mask = (df['time'] >= inicio) & (df['time'] <= fim)
        df_plot = df[mask].copy()
        
        n_ligado = (df_plot['equipamento_status'] == 'LIGADO').sum()
        n_desligado = (df_plot['equipamento_status'] == 'DESLIGADO').sum()
        n_transicoes_intervalo = df_plot['mudanca_estado'].sum()
        
        pct_ligado = (n_ligado / len(df_plot)) * 100
        pct_desligado = (n_desligado / len(df_plot)) * 100
    else:
        inicio = melhor_intervalo['inicio']
        fim = melhor_intervalo['fim']
        df_plot = melhor_intervalo['df']
        pct_ligado = melhor_intervalo['pct_ligado']
        pct_desligado = melhor_intervalo['pct_desligado']
        n_transicoes_intervalo = melhor_intervalo['n_transicoes']
        
        print(f"  [OK] Melhor intervalo encontrado (score: {melhor_score:.1f})")
    
    n_ligado = (df_plot['equipamento_status'] == 'LIGADO').sum()
    n_desligado = (df_plot['equipamento_status'] == 'DESLIGADO').sum()
    
    print(f"\n[OK] Intervalo selecionado: {inicio} até {fim}")
    print(f"     Registros: {len(df_plot):,}")
    print(f"     LIGADO: {n_ligado:,} ({pct_ligado:.1f}%)")
    print(f"     DESLIGADO: {n_desligado:,} ({pct_desligado:.1f}%)")
    print(f"     Transições: {n_transicoes_intervalo}")
    
    # Identificar colunas
    colunas_temp = [col for col in df_plot.columns if 'temp' in col.lower()]
    colunas_vibracao = [col for col in df_plot.columns if 'vel_rms' in col.lower()]
    
    if not colunas_temp:
        print("[AVISO] Nenhuma coluna de temperatura encontrada, usando valor fixo")
        df_plot['temperatura'] = 25.0
        col_temp = 'temperatura'
    else:
        col_temp = colunas_temp[0]
    
    if not colunas_vibracao:
        print("[AVISO] Nenhuma coluna de vibração encontrada, usando valor fixo")
        df_plot['vibracao'] = 0.5
        col_vibracao = 'vibracao'
    else:
        # Calcular média das vibrações
        df_plot['vibracao_media'] = df_plot[colunas_vibracao].mean(axis=1)
        col_vibracao = 'vibracao_media'
    
    # Preparar dados
    tempo = df_plot['time'].values
    temperatura = df_plot[col_temp].values
    vibracao = df_plot[col_vibracao].values
    status = df_plot['equipamento_status'].values
    
    # Cores por estado
    cores = ['red' if s == 'DESLIGADO' else 'green' for s in status]
    
    # Criar figura grande
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Gráfico 3D principal: Temperatura x Vibração x Tempo
    ax1 = fig.add_subplot(221, projection='3d')
    
    # Converter tempo para número para plotar
    tempo_num = mdates.date2num(tempo)
    
    # Plotar por estado para ter legenda
    mask_desligado = status == 'DESLIGADO'
    mask_ligado = status == 'LIGADO'
    
    if mask_desligado.any():
        ax1.scatter(tempo_num[mask_desligado], temperatura[mask_desligado], vibracao[mask_desligado], 
                   c='red', alpha=0.6, s=5, label='DESLIGADO')
    
    if mask_ligado.any():
        ax1.scatter(tempo_num[mask_ligado], temperatura[mask_ligado], vibracao[mask_ligado], 
                   c='green', alpha=0.6, s=5, label='LIGADO')
    
    ax1.set_xlabel('Tempo', fontsize=10)
    ax1.set_ylabel('Temperatura (°C)', fontsize=10)
    ax1.set_zlabel('Vibração RMS (mm/s)', fontsize=10)
    ax1.set_title(f'Temperatura x Vibração x Tempo\n{mpoint} - {dias} dias', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    
    # Formatar eixo de tempo
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m\n%H:%M'))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    
    # 2. Gráfico 2D: Temperatura x Tempo
    ax2 = fig.add_subplot(222)
    for estado, cor in [('DESLIGADO', 'red'), ('LIGADO', 'green')]:
        mask = status == estado
        if mask.any():
            ax2.plot(tempo[mask], temperatura[mask], 'o', color=cor, alpha=0.6, 
                    markersize=2, label=estado)
    
    ax2.set_xlabel('Tempo', fontsize=10)
    ax2.set_ylabel('Temperatura (°C)', fontsize=10)
    ax2.set_title('Temperatura ao longo do Tempo', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m\n%H:%M'))
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 3. Gráfico 2D: Vibração x Tempo
    ax3 = fig.add_subplot(223)
    for estado, cor in [('DESLIGADO', 'red'), ('LIGADO', 'green')]:
        mask = status == estado
        if mask.any():
            ax3.plot(tempo[mask], vibracao[mask], 'o', color=cor, alpha=0.6, 
                    markersize=2, label=estado)
    
    ax3.set_xlabel('Tempo', fontsize=10)
    ax3.set_ylabel('Vibração RMS (mm/s)', fontsize=10)
    ax3.set_title('Vibração ao longo do Tempo', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m\n%H:%M'))
    ax3.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 4. Timeline de Estados
    ax4 = fig.add_subplot(224)
    
    # Criar linha temporal de estados
    estados_num = [1 if s == 'LIGADO' else 0 for s in status]
    ax4.fill_between(tempo, 0, estados_num, where=np.array(estados_num)==1, 
                     color='green', alpha=0.3, label='LIGADO')
    ax4.fill_between(tempo, 0, estados_num, where=np.array(estados_num)==0, 
                     color='red', alpha=0.3, label='DESLIGADO')
    ax4.plot(tempo, estados_num, 'k-', linewidth=0.5, alpha=0.5)
    
    ax4.set_xlabel('Tempo', fontsize=10)
    ax4.set_ylabel('Estado', fontsize=10)
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(['DESLIGADO', 'LIGADO'])
    ax4.set_title('Timeline de Estados do Equipamento', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis='x')
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m\n%H:%M'))
    ax4.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Estatísticas do intervalo
    total = len(df_plot)
    ligado = (status == 'LIGADO').sum()
    desligado = (status == 'DESLIGADO').sum()
    
    texto_stats = (
        f"Período: {inicio.strftime('%Y-%m-%d %H:%M')} até {fim.strftime('%Y-%m-%d %H:%M')}\n"
        f"Total: {total:,} amostras | "
        f"LIGADO: {ligado:,} ({ligado/total*100:.1f}%) | "
        f"DESLIGADO: {desligado:,} ({desligado/total*100:.1f}%)"
    )
    
    fig.text(0.5, 0.98, texto_stats, ha='center', va='top', fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Salvar
    results_path = results_dir(mpoint, create=True)
    inicio_str = inicio.strftime('%Y%m%d_%H%M')
    fim_str = fim.strftime('%Y%m%d_%H%M')
    arquivo_plot = results_path / f'estados_3d_intervalo_{dias}dias_{mpoint}_{inicio_str}_{fim_str}.png'
    plt.savefig(arquivo_plot, dpi=300, bbox_inches='tight')
    
    print(f"\n[OK] Visualização salva: {arquivo_plot}")
    
    # MOSTRAR NA TELA
    print("\n[INFO] Abrindo visualização na tela...")
    plt.show()
    
    # Estatísticas
    print("\n" + "="*80)
    print("ESTATÍSTICAS DO INTERVALO")
    print("="*80)
    print(f"Período: {inicio} até {fim}")
    print(f"Total de amostras: {total:,}")
    print(f"LIGADO: {ligado:,} ({ligado/total*100:.1f}%)")
    print(f"DESLIGADO: {desligado:,} ({desligado/total*100:.1f}%)")
    print(f"Temperatura: {temperatura.min():.2f}°C até {temperatura.max():.2f}°C")
    print(f"Vibração: {vibracao.min():.4f} até {vibracao.max():.4f} mm/s")
    print("="*80)
    
    return True

def main():
    """Função principal"""
    parser = argparse.ArgumentParser(
        description="Visualização 3D de intervalo para equipamento MECÂNICO"
    )
    parser.add_argument('--mpoint', type=str, required=True, help='ID do mpoint')
    parser.add_argument('--dias', type=int, default=3, help='Número de dias para visualizar (padrão: 3)')
    
    args = parser.parse_args()
    
    if not criar_visualizacao_3d_intervalo_mecanico(args.mpoint, args.dias):
        print("\n[ERRO] Falha ao criar visualização")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("VISUALIZAÇÃO CONCLUÍDA COM SUCESSO!")
    print("="*80)

if __name__ == '__main__':
    main()

