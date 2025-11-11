"""
Script simples pra ver os grupos do K-means num grafico 3D.
Mostra corrente vs vibracao vs tempo (ou temperatura se RPM nao tiver).
Ajuda a entender se os grupos tao bem separados.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import argparse
import warnings
import sys
from datetime import datetime
warnings.filterwarnings('ignore')

from pathlib import Path

# Importações para logging estruturado
BASE_DIR_LOG = Path(__file__).resolve().parent.parent
if str(BASE_DIR_LOG) not in sys.path:
    sys.path.insert(0, str(BASE_DIR_LOG))

from utils.logging_utils import (
    save_log,
    create_visualization_log,
    format_file_list,
    get_file_info,
    enrich_results_file,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DIR_PROCESSED = BASE_DIR / 'data' / 'processed'
DIR_RESULTS = BASE_DIR / 'results'
DIR_MODELS = BASE_DIR / 'models'

def main(mpoint=None, data_inicio=None, data_fim=None, intervalo_arquivo=None):
    print("=" * 80)
    print("[VIZ] VISUALIZAÇÃO 3D - ESTADOS DO EQUIPAMENTO")
    print("=" * 80)
    if mpoint:
        print(f"Mpoint: {mpoint}")
    print("=" * 80)

    # Carregar dados classificados
    mpoint_tag = f"_{mpoint}" if mpoint else ""
    
    # MODO INTERVALO: Carregar arquivo de resultados mais recente
    if intervalo_arquivo:
        dir_results_mpoint = DIR_RESULTS / mpoint
        if not dir_results_mpoint.exists():
            print(f"[ERRO] Pasta de resultados não encontrada: {dir_results_mpoint}")
            return False
        
        # Buscar arquivo de resultados mais recente
        arquivos_resultados = list(dir_results_mpoint.glob('analise_completa_*_resultados.csv'))
        if not arquivos_resultados:
            print(f"[ERRO] Nenhum arquivo de resultados encontrado em: {dir_results_mpoint}")
            return False
        
        # Pegar o mais recente
        arquivo = max(arquivos_resultados, key=lambda p: p.stat().st_mtime)
        print(f"[MODO INTERVALO] Carregando resultados: {arquivo.name}")
    # MODO TREINO: Carregar arquivo de dados classificados
    else:
        arquivo = DIR_PROCESSED / f'dados_classificados_kmeans_moderado{mpoint_tag}.csv'
        print(f"[MODO TREINO] Carregando dados classificados: {arquivo.name}")
    
    if not arquivo.exists():
        print(f"[ERRO] Arquivo não encontrado: {arquivo}")
        return False

    print("\nCarregando dados...")
    df = pd.read_csv(arquivo)
    df['time'] = pd.to_datetime(df['time'], format='mixed', utc=True)

    # Se foi especificado intervalo (modo análise), filtrar apenas dados desse período
    if data_inicio and data_fim:
        print(f"Modo análise - Filtrando dados do intervalo: {data_inicio} até {data_fim}")
        data_inicio_dt = pd.to_datetime(data_inicio, utc=True)
        data_fim_dt = pd.to_datetime(data_fim, utc=True)
        mask = (df['time'] >= data_inicio_dt) & (df['time'] <= data_fim_dt)
        df_filtrado = df[mask].copy()
        print(f"  Dados filtrados: {len(df_filtrado)} de {len(df)} registros")

        if len(df_filtrado) == 0:
            print(f"[ERRO] Nenhum dado encontrado no intervalo {data_inicio} até {data_fim}")
            return False

        df = df_filtrado
    else:
        print("Modo treino - Usando dados completos")
    
    # Carregar scaler
    if mpoint:
        # Tentar primeiro na pasta específica do mpoint
        scaler_path = DIR_MODELS / mpoint / f'scaler_model_moderado{mpoint_tag}.pkl'
        if not scaler_path.exists():
            # Fallback para o diretório raiz de models
            scaler_path = DIR_MODELS / f'scaler_model_moderado{mpoint_tag}.pkl'
    else:
        scaler_path = DIR_MODELS / 'scaler_model_moderado.pkl'
    
    if not scaler_path.exists():
        print(f"[ERRO] Scaler não encontrado em: {scaler_path}")
        return False
    
    print(f"Carregando scaler: {scaler_path.name}")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Desnormalizar
    print("Desnormalizando...")
    colunas_excluir = ['time', 'cluster', 'equipamento_status', 'estado', 'm_point', 'periodo_id', 'interpolado', 'arquivo_origem']
    feature_cols = [col for col in df.columns if col not in colunas_excluir and df[col].dtype in ['float64', 'int64']]
    
    dados_desnorm = scaler.inverse_transform(df[feature_cols].values)
    df_desnorm = pd.DataFrame(dados_desnorm, columns=feature_cols, index=df.index)
    
    # Detectar coluna de status (pode ser 'equipamento_status' ou 'estado')
    coluna_status = 'equipamento_status' if 'equipamento_status' in df.columns else 'estado'
    df_final = pd.concat([df[['time', coluna_status]], df_desnorm], axis=1)
    
    # Selecionar período para visualização
    if data_inicio and data_fim:
        # MODO ANÁLISE: usar todos os dados do intervalo filtrado
        print(f"\nUsando intervalo da análise: {data_inicio} até {data_fim}")
        df_sorted = df_final.sort_values('time').reset_index(drop=True)
        df_viz = df_sorted.copy()
        print(f"Pontos: {len(df_viz)}")
    else:
        # MODO TREINO: procurar intervalo de 3 dias com ambos estados
        print("\nSelecionando intervalo de 3 dias (modo treino)...")
        df_sorted = df_final.sort_values('time').reset_index(drop=True)

        # Procurar intervalo com ambos estados (otimizado)
        melhor_idx = 0
        melhor_score = 0
        
        # Limitar busca para evitar travamento (máximo 10000 iterações ou 10% dos dados)
        max_iters = min(10000, len(df_sorted) // 10)
        step = max(1, (len(df_sorted) - 1000) // max_iters) if len(df_sorted) > 1000 else 1
        
        print(f"Procurando melhor intervalo (verificando {max_iters} posições)...")

        for i in range(0, min(len(df_sorted) - 1000, max_iters * step), step):
            tempo_ini = df_sorted.loc[i, 'time']
            tempo_fim = tempo_ini + pd.Timedelta(hours=72)
            mask = (df_sorted['time'] >= tempo_ini) & (df_sorted['time'] <= tempo_fim)
            df_temp = df_sorted[mask]

            if len(df_temp) > 100 and coluna_status in df_temp.columns:
                estados = df_temp[coluna_status].unique()
                if 'LIGADO' in estados and 'DESLIGADO' in estados:
                    n_lig = (df_temp[coluna_status] == 'LIGADO').sum()
                    n_des = (df_temp[coluna_status] == 'DESLIGADO').sum()
                    score = min(n_lig, n_des) / max(n_lig, n_des)
                    if score > melhor_score:
                        melhor_score = score
                        melhor_idx = i
                        if score > 0.3:  # Bom o suficiente
                            break

        # Selecionar dados dos 3 dias
        tempo_ini = df_sorted.loc[melhor_idx, 'time']
        tempo_fim = tempo_ini + pd.Timedelta(hours=72)
        mask = (df_sorted['time'] >= tempo_ini) & (df_sorted['time'] <= tempo_fim)
        df_3dias = df_sorted[mask].copy()

        print(f"Período: {tempo_ini.strftime('%d/%m/%Y')} a {tempo_fim.strftime('%d/%m/%Y')}")
        print(f"Pontos: {len(df_3dias)}")
        df_viz = df_3dias.copy()
    
    # Amostrar (sempre usar df_viz que já foi definido acima)
    if len(df_viz) > 1000:
        indices = np.linspace(0, len(df_viz) - 1, 1000, dtype=int)
        df_viz = df_viz.iloc[indices].copy()
    
    # Extrair variáveis
    corrente = df_viz['current'].values
    vibracao = df_viz['vel_rms'].values
    rpm = df_viz['rotational_speed'].values
    temp = df_viz['object_temp'].values if 'object_temp' in df_viz.columns else np.zeros(len(df_viz))
    status = df_viz[coluna_status].values
    tempo = df_viz['time']
    
    # Decidir eixo X baseado em variância
    rpm_var = np.var(rpm)
    temp_var = np.var(temp)
    
    if rpm_var > 100:
        eixo_x = rpm
        label_x = 'RPM'
    else:
        eixo_x = temp
        label_x = 'Temperatura (°C)'
    
    print(f"\nUsando {label_x} no eixo X (variância: {rpm_var if rpm_var > 100 else temp_var:.2f})")
    
    # Converter tempo
    t_ini = tempo.min()
    t_horas = [(t - t_ini).total_seconds() / 3600 for t in tempo]
    
    # Labels do eixo Z
    t_meio = t_ini + (tempo.max() - t_ini) / 2
    labels_z = [
        t_ini.strftime('%d/%m %H:%M'),
        t_meio.strftime('%d/%m %H:%M'),
        tempo.max().strftime('%d/%m %H:%M')
    ]
    z_pos = [0, max(t_horas)/2, max(t_horas)]
    
    # Gráfico 1: Corrente x Vibração x Tempo
    print("\nGerando gráfico 1: Corrente x Vibração x Tempo...")
    fig1 = plt.figure(figsize=(16, 12))
    ax1 = fig1.add_subplot(111, projection='3d')
    
    cores = {'DESLIGADO': '#e74c3c', 'LIGADO': '#2ecc71'}
    
    for estado in ['DESLIGADO', 'LIGADO']:
        mask = status == estado
        if mask.sum() > 0:
            ax1.scatter(corrente[mask], vibracao[mask], np.array(t_horas)[mask],
                       c=cores[estado], label=estado, s=50, alpha=0.7, edgecolors='k', linewidths=0.5)
    
    ax1.set_xlabel('Corrente (A)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Vibração (mm/s)', fontsize=12, fontweight='bold')
    ax1.set_zlabel('Tempo', fontsize=12, fontweight='bold')
    ax1.set_zticks(z_pos)
    ax1.set_zticklabels(labels_z, fontsize=10)
    ax1.set_title(f'Corrente x Vibração x Tempo (3 dias)\n{mpoint or ""}', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.view_init(elev=20, azim=45)
    
    DIR_RESULTS.mkdir(parents=True, exist_ok=True)

    # Criar sufixo com intervalo se fornecido
    intervalo_tag = ""
    if data_inicio and data_fim:
        # Formatar datas para nome do arquivo (remover caracteres especiais)
        inicio_fmt = data_inicio.replace("-", "").replace(" ", "_").replace(":", "")
        fim_fmt = data_fim.replace("-", "").replace(" ", "_").replace(":", "")
        intervalo_tag = f"_{inicio_fmt}_to_{fim_fmt}"

    out1 = DIR_RESULTS / f'estados_corrente_vibracao_tempo_3d{"_" + mpoint if mpoint else ""}{intervalo_tag}.png'
    plt.savefig(out1, dpi=300, bbox_inches='tight')
    print(f"Salvo: {out1}")

    # Gráfico 2
    print(f"\nGerando gráfico 2: {label_x} x Vibração x Tempo...")
    fig2 = plt.figure(figsize=(16, 12))
    ax2 = fig2.add_subplot(111, projection='3d')
    
    for estado in ['DESLIGADO', 'LIGADO']:
        mask = status == estado
        if mask.sum() > 0:
            ax2.scatter(eixo_x[mask], vibracao[mask], np.array(t_horas)[mask],
                       c=cores[estado], label=estado, s=50, alpha=0.7, edgecolors='k', linewidths=0.5)
    
    ax2.set_xlabel(label_x, fontsize=12, fontweight='bold')
    ax2.set_ylabel('Vibração (mm/s)', fontsize=12, fontweight='bold')
    ax2.set_zlabel('Tempo', fontsize=12, fontweight='bold')
    ax2.set_zticks(z_pos)
    ax2.set_zticklabels(labels_z, fontsize=10)
    ax2.set_title(f'{label_x} x Vibração x Tempo (3 dias)\n{mpoint or ""}', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.view_init(elev=20, azim=45)
    
    nome = 'temperatura' if label_x == 'Temperatura (°C)' else 'rpm'
    out2 = DIR_RESULTS / f'estados_{nome}_vibracao_tempo_3d{"_" + mpoint if mpoint else ""}{intervalo_tag}.png'
    plt.savefig(out2, dpi=300, bbox_inches='tight')
    print(f"Salvo: {out2}")
    
    # Mostrar janelas interativas
    print("\n[INFO] Abrindo janelas interativas (use mouse para rotacionar)...")
    plt.show()
    
    print("\n[OK] Visualização concluída!")

    # Gerar logs detalhados para TCC
    import time
    start_time = time.time()  # Nota: deveria ser definido no início, mas para compatibilidade vamos estimar

    # Coletar informações dos gráficos gerados
    chart_files = []
    if DIR_RESULTS.exists():
        for file_path in DIR_RESULTS.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.svg']:
                if mpoint and mpoint in file_path.name:
                    chart_files.append(str(file_path))
                elif not mpoint:
                    chart_files.append(str(file_path))

    # Estatísticas da visualização
    viz_stats = {
        'total_samples_visualized': len(df),
        'n_clusters': len(df['cluster'].unique()) if 'cluster' in df.columns else 0,
        'visualization_type': '3d_scatter_plots',
        'charts_generated': len(chart_files),
        'data_period_days': 3,  # Como especificado no código
        'equipment_states': ['LIGADO', 'DESLIGADO'] if coluna_status in df.columns else []
    }

    # Log de visualização 3D
    viz_log = create_visualization_log(
        script_name='visualizar_clusters_3d_simples',
        mpoint=mpoint,
        chart_type='3d_equipment_states_visualization',
        data_description={
            'data_source': str(arquivo),
            'total_samples': len(df),
            'n_clusters': len(df['cluster'].unique()) if 'cluster' in df.columns else 0,
            'visualization_dimensions': ['selected_sensor', 'vibration', 'time'],
            'sensor_used': 'temperature' if 'Temperatura' in str(out1) else 'rpm',
            'time_window': '3_days_sampled',
            'equipment_states_identified': viz_stats['equipment_states']
        },
        chart_files=chart_files,
        period_info={
            'data_start': str(df['time'].min()),
            'data_end': str(df['time'].max()),
            'total_samples': len(df),
            'visualization_period_days': 3,
            'time_sampling': 'sampled_for_visualization'
        },
        visualization_details={
            'plot_type': '3d_scatter_with_time',
            'color_coding': 'equipment_states',
            'interactive_rotation': True,
            'dpi_quality': 300,
            'view_angles': 'multiple_optimized'
        }
    )

    save_log(viz_log, 'visualizar_clusters_3d_simples', mpoint, '3d_visualization_complete')

    # Enriquecer arquivo results
    results_data = {
        'visualization_3d_completed': True,
        'visualization_3d_timestamp': datetime.now().isoformat(),
        'charts_generated': len(chart_files),
        'visualization_type': '3d_states_scatter',
        'data_samples_visualized': len(df),
        'time_window_days': 3,
        'equipment_states_detected': viz_stats['equipment_states'],
        'visualization_charts': chart_files,
        'visualization_parameters': viz_log['visualization_details']
    }

    if mpoint:
        enrich_results_file(mpoint, results_data)

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mpoint', type=str, help='ID do mpoint')
    parser.add_argument('--data-inicio', type=str, help='Data/hora inicial (YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--data-fim', type=str, help='Data/hora final (YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--intervalo-arquivo', type=str, help='Intervalo formatado para modo análise')
    args = parser.parse_args()
    main(mpoint=args.mpoint, data_inicio=args.data_inicio, data_fim=args.data_fim, intervalo_arquivo=args.intervalo_arquivo)

