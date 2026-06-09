import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import os
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection

# Configurações de cores solicitadas pelo usuário
COLOR_LIGADO = '#2ecc71'           # Verde claro
COLOR_DESLIGADO = '#e74c3c'        # Vermelho
COLOR_KNN_LIGADO = '#ff00ff'       # Rosa Fluorescente
COLOR_KNN_DESLIGADO = '#c71585'    # Rosa Escuro
COLOR_SPLINE_LIGADO = '#85c1e9'    # Azul Claro
COLOR_SPLINE_DESLIGADO = '#2e86c1' # Azul Escuro
COLOR_VAZIO = '#bdc3c7'            # Cinza (Gaps > 3h)

def plot_vibracao_reconstrucao_completa(mpoint='c_637', start_time=None, end_time=None):
    """
    Gera o gráfico de reconstrução temporal otimizado com LineCollection.
    Garante que os dados originais sejam detectados corretamente (removendo timezones para comparação).
    """
    print(f"\n=== GERANDO GRÁFICO DE RECONSTRUÇÃO TEMPORAL - {mpoint} ===", flush=True)
    
    # 1. Caminhos usando utilitários centralizados
    from utils.artifact_paths import processed_classificado_path, processed_unificado_path
    file_processed = processed_classificado_path(mpoint)
    file_real_units = processed_unificado_path(mpoint)
    
    if not file_processed.exists() or not file_real_units.exists():
        print(f" [ERRO] Arquivos não encontrados para {mpoint}")
        print(f"        - Classificado: {file_processed}")
        print(f"        - Unificado: {file_real_units}")
        return

    # 2. Carregar e Merge
    print(" [1/5] Carregando e normalizando timestamps...", flush=True)
    df_values = pd.read_csv(file_real_units)
    df_status = pd.read_csv(file_processed, usecols=['time', 'equipamento_status'])
    
    # REMOÇÃO DE TIMEZONE PARA COMPARAÇÃO EXATA
    df_values['time'] = pd.to_datetime(df_values['time'], format='mixed', utc=True).dt.tz_localize(None)
    df_status['time'] = pd.to_datetime(df_status['time'], format='mixed', utc=True).dt.tz_localize(None)
    
    df = pd.merge(df_values, df_status, on='time', how='inner')
    df = df.sort_values('time').reset_index(drop=True)
    
    if start_time and end_time:
        st, et = pd.to_datetime(start_time, format='mixed'), pd.to_datetime(end_time, format='mixed')
        df = df[(df['time'] >= st) & (df['time'] <= et)].copy()

    # [NOVO] Garantir coluna de vibração (vel_rms)
    if 'vel_rms' not in df.columns:
        print(" [INFO] Coluna 'vel_rms' não encontrada. Calculando média de x, y, z...", flush=True)
        vibration_cols = [c for c in ['vel_rms_x', 'vel_rms_y', 'vel_rms_z'] if c in df.columns]
        if vibration_cols:
            df['vel_rms'] = df[vibration_cols].mean(axis=1)
        elif 'rms' in df.columns:
            df['vel_rms'] = df['rms']
        else:
            # Fallback para qualquer coluna que pareça vibração
            fallback = [c for c in df.columns if 'vel' in c.lower() or 'vibr' in c.lower()]
            if fallback:
                df['vel_rms'] = df[fallback[0]]
            else:
                print(" [ERRO] Nenhuma coluna de vibração encontrada!")
                return

    # 3. Identificar Origem (REAL vs INTERPOLADO) usando a flag do pipeline
    print(" [2/5] Identificando origem dos dados usando flag do pipeline...", flush=True)
    if 'interpolado' in df.columns:
        df['is_original'] = ~df['interpolado'].fillna(False).astype(bool)
        n_orig = df['is_original'].sum()
        print(f"       -> Pontos REAIS (verdes/vermelhos): {n_orig:,}")
        print(f"       -> Pontos RECONSTRUÍDOS (amarelo/azul): {len(df) - n_orig:,}")
    else:
        print("       -> [AVISO] Coluna 'interpolado' não encontrada. Usando apenas cor REAL.")
        df['is_original'] = True
        
    df['metodo'] = 'REAL'
    df['gap_group'] = df['is_original'].cumsum()
    gap_mask = ~df['is_original']
    if gap_mask.any():
        gap_sizes = df[gap_mask].groupby('gap_group')['time'].transform('count')
        df.loc[gap_mask & (gap_sizes <= 90), 'metodo'] = 'SPLINE'
        df.loc[gap_mask & (gap_sizes > 90), 'metodo'] = 'KNN'
    
    # 4. Amostragem (Preservando Transições)
    print(" [3/5] Otimizando amostragem para visualização...", flush=True)
    limit_pts = 350000
    df['status_code'] = df['equipamento_status'].astype('category').cat.codes
    df['metodo_code'] = df['metodo'].astype('category').cat.codes
    df['style_key'] = df['status_code'].astype(str) + "_" + df['metodo_code'].astype(str)
    
    if len(df) > limit_pts:
        step = len(df) // limit_pts
        idx_regular = np.arange(0, len(df), step)
        idx_trans = np.where(df['style_key'].values[:-1] != df['style_key'].values[1:])[0]
        idx_trans = np.unique(np.concatenate([idx_trans, idx_trans + 1]))
        idx_final = np.unique(np.concatenate([idx_regular, idx_trans, [len(df)-1]]))
        idx_final = idx_final[idx_final < len(df)]
        df_plot = df.iloc[idx_final].copy().reset_index(drop=True)
    else:
        df_plot = df.copy().reset_index(drop=True)
        
    # 5. Preparar segmentos para LineCollection
    print(f" [4/5] Renderizando {len(df_plot):,} pontos via LineCollection...", flush=True)
    times_num = mdates.date2num(df_plot['time'])
    vibs = df_plot['vel_rms'].values
    
    color_map = {
        'LIGADO_REAL': COLOR_LIGADO,
        'DESLIGADO_REAL': COLOR_DESLIGADO,
        'LIGADO_KNN': COLOR_KNN_LIGADO,
        'DESLIGADO_KNN': COLOR_KNN_DESLIGADO,
        'LIGADO_SPLINE': COLOR_SPLINE_LIGADO,
        'DESLIGADO_SPLINE': COLOR_SPLINE_DESLIGADO
    }
    
    df_plot['color_val'] = (df_plot['equipamento_status'] + "_" + df_plot['metodo']).map(color_map)
    colors = df_plot['color_val'].values
    
    # Criar segmentos
    points = np.array([times_num, vibs]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # LÓGICA DE CORES E FILTRAGEM DE GAPS:
    orig_mask = df_plot['is_original'].values
    metodo_vals = df_plot['metodo'].values
    status_vals = df_plot['equipamento_status'].values
    times_raw = df_plot['time'].values
    
    seg_colors = []
    seg_is_rec = []
    valid_segments_idx = []
    
    for i in range(len(df_plot) - 1):
        # 1. IGNORAR SEGMENTOS DE LACUNA > 3h (Não desenhar linha colorida cruzando o vazio)
        # Usando cálculo robusto em segundos para evitar bugs do numpy timedelta
        diff_sec = (times_raw[i+1] - times_raw[i]).astype('timedelta64[s]').astype(float)
        if diff_sec > 3 * 3600: # 3 horas = 10800 segundos
            continue
            
        valid_segments_idx.append(i)
        
        # 2. DEFINIR COR: Para evitar que pequenos gaps de 1 ponto dominem o gráfico,
        # um segmento SÓ é considerado reconstrução se AMBOS os pontos forem artificiais.
        # Caso contrário, é a transição de/para a realidade, que deve ser REAL.
        if not orig_mask[i] and not orig_mask[i+1]:
            key = f"{status_vals[i]}_{metodo_vals[i]}"
            seg_is_rec.append(True)
        else:
            # Se a transição envolve um dado real, a prioridade da cor é REAL
            status_to_use = status_vals[i] if orig_mask[i] else status_vals[i+1]
            key = f"{status_to_use}_REAL"
            seg_is_rec.append(False)
            
        seg_colors.append(color_map.get(key, '#000000'))
    
    # Filtrar segmentos e cores
    segments = segments[valid_segments_idx]
    seg_colors = np.array(seg_colors)
    seg_is_rec = np.array(seg_is_rec)
    
    plt.rcParams.update({
        'font.size': 18,
        'axes.titlesize': 20,
        'axes.labelsize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
        'legend.borderpad': 1.2,
        'legend.handlelength': 3.0,
        'legend.handleheight': 1.8,
        'legend.handletextpad': 1.0,
        'legend.labelspacing': 0.8,
        'legend.framealpha': 0.95,
    })
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(df_plot['time'], vibs, color='lightgray', linewidth=0.3, alpha=0.1, zorder=1)
    
    # RECONSTRUCTED (KNN/Spline) - Agora 100% opaco e mais grosso para sobressair
    if seg_is_rec.any():
        lc_rec = LineCollection(segments[seg_is_rec], colors=seg_colors[seg_is_rec], linewidths=2.5, alpha=1.0, zorder=5)
        ax.add_collection(lc_rec)
        
    # REAL (Sensor) - Ligeiramente mais fino e transparente para servir de base
    if (~seg_is_rec).any():
        lc_real = LineCollection(segments[~seg_is_rec], colors=seg_colors[~seg_is_rec], linewidths=1.0, alpha=0.7, zorder=4)
        ax.add_collection(lc_real)
    
    # Gaps longos (> 3h): Apenas a caixa cinza, sem linha colorida cruzando
    large_gaps = df_plot[df_plot['time'].diff() > timedelta(hours=3)]
    for idx in large_gaps.index:
        s, e = df_plot.loc[idx-1, 'time'], df_plot.loc[idx, 'time']
        ax.axvspan(s, e, color=COLOR_VAZIO, alpha=0.2, zorder=2)
        # Linha de base discreta apenas para indicar continuidade do eixo
        ax.plot([s, e], [0, 0], color='gray', linestyle=':', linewidth=0.5, alpha=0.3)

    # Estética
    ax.autoscale_view()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%y'))
    vib_max = max(vibs) * 1.1 if len(vibs) > 0 else 10
    tick_step = 0.5 if vib_max <= 6 else (1.0 if vib_max <= 12 else 2.0)
    ax.yaxis.set_major_locator(plt.MultipleLocator(tick_step))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(tick_step / 2))
    ax.grid(axis='y', color='gray', linestyle='-', alpha=0.2)
    ax.grid(axis='y', which='minor', color='gray', linestyle=':', alpha=0.1)
    plt.xticks(rotation=45)
    # Ticks com fonte grande (acima do rcParams)
    ax.tick_params(axis='both', labelsize=20)

    title = f'Vibração Temporal Reconstruída - {mpoint}'
    if start_time: title += f' ({start_time} a {end_time})'
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Data', fontsize=20, fontweight='bold')
    ax.set_ylabel('Vibração RMS (mm/s)', fontsize=20, fontweight='bold')
    ax.set_ylim(0, vib_max)

    # Legenda externa com círculos (padrão do gráfico 3D: markersize=13, edgecolor k)
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_LIGADO,
               markeredgecolor='k', markeredgewidth=0.5, markersize=13, label='Real: LIGADO'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_DESLIGADO,
               markeredgecolor='k', markeredgewidth=0.5, markersize=13, label='Real: DESLIGADO'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_KNN_LIGADO,
               markeredgecolor='k', markeredgewidth=0.5, markersize=13, label='KNN: LIGADO'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_KNN_DESLIGADO,
               markeredgecolor='k', markeredgewidth=0.5, markersize=13, label='KNN: DESLIGADO'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_SPLINE_LIGADO,
               markeredgecolor='k', markeredgewidth=0.5, markersize=13, label='Spline: LIGADO'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_SPLINE_DESLIGADO,
               markeredgecolor='k', markeredgewidth=0.5, markersize=13, label='Spline: DESLIGADO'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=COLOR_VAZIO,
               markeredgecolor='k', markeredgewidth=0.5, markersize=13,
               alpha=0.6, label='Lacuna (> 3h)'),
    ]
    fig.subplots_adjust(right=0.76)
    ax.legend(handles=legend_elements,
              bbox_to_anchor=(1.01, 1), loc='upper left',
              frameon=True, shadow=True,
              fontsize=18, borderpad=1.0, handletextpad=0.4,
              labelspacing=0.3, borderaxespad=0.0)
    
    plt.tight_layout()
    tag = f"_{str(start_time).replace(':', '-')}" if start_time else ""
    from utils.artifact_paths import results_dir
    dir_resultados = results_dir(mpoint, create=True)
    out = dir_resultados / f'vibracao_reconstrucao_{mpoint}{tag}.png'
    plt.savefig(out, dpi=300)
    print(f" [5/5] [OK] Gráfico salvo em: {out}", flush=True)

if __name__ == "__main__":
    plot_vibracao_reconstrucao_completa('c_637')
