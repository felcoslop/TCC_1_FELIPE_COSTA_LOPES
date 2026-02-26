"""
Script pra ver os grupos do K-means em 3D com os valores originais.
Mostra corrente vs vibracao vs tempo, separado por ligado/desligado.
Ajuda a ver se a classificacao ta fazendo sentido visualmente.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import warnings
import argparse
import sys
warnings.filterwarnings('ignore')

from pathlib import Path

# Importacoes para logging estruturado
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

def carregar_dados_e_scaler(mpoint=None):
    """Carrega dados classificados e scaler para desnormalização"""
    print("Carregando dados classificados e scaler...")

    mpoint_tag = f"_{mpoint}" if mpoint else ""

    # Carregar dados classificados
    arquivo_classificado = DIR_PROCESSED / f'dados_classificados_kmeans_moderado{mpoint_tag}.csv'
    if not arquivo_classificado.exists():
        print(f"[ERRO] Arquivo não encontrado: {arquivo_classificado}")
        return None, None

    df_classificado = pd.read_csv(arquivo_classificado)
    print(f"  - Dados classificados: {df_classificado.shape}")

    if 'equipamento_status' in df_classificado.columns:
        print(f"  - Estados: {df_classificado['equipamento_status'].unique()}")

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
        return None, None

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    print(f"  - Scaler carregado: {scaler_path.name}")

    # Converter timestamp
    df_classificado['time'] = pd.to_datetime(df_classificado['time'], format='mixed', utc=True)

    return df_classificado, scaler

def desnormalizar_dados(df_classificado, scaler):
    """Desnormaliza os dados usando o scaler"""
    print("\nDesnormalizando dados...")

    # Colunas que NÃO devem ser desnormalizadas
    colunas_excluir = [
        'time', 'cluster', 'equipamento_status',
        'm_point', 'periodo_id', 'interpolado', 'arquivo_origem'
    ]

    # Extrair colunas de features (apenas numéricas)
    feature_cols = [col for col in df_classificado.columns
                   if col not in colunas_excluir and df_classificado[col].dtype in ['float64', 'int64']]

    print(f"  - Features para desnormalizar: {len(feature_cols)}")

    # Desnormalizar
    dados_norm = df_classificado[feature_cols].values
    dados_desnorm = scaler.inverse_transform(dados_norm)

    # Criar DataFrame desnormalizado
    df_desnorm = pd.DataFrame(dados_desnorm, columns=feature_cols, index=df_classificado.index)

    # Combinar com informações de classificação
    df_final = pd.concat([
        df_classificado[['time', 'equipamento_status']],
        df_desnorm
    ], axis=1)

    print(f"  - Dados desnormalizados: {len(df_final)} linhas")

    return df_final

def selecionar_intervalo_3_dias(df_dados):
    """Seleciona intervalo contínuo de 3 dias (72 horas) com melhor visualização de estados"""
    print("\nSelecionando intervalo de 3 dias com distinção clara entre estados...")
    
    df_dados = df_dados.sort_values('time').reset_index(drop=True)
    
    if 'equipamento_status' not in df_dados.columns:
        print("  - Coluna equipamento_status não encontrada, usando todos os dados")
        tempo_inicio = df_dados['time'].min()
        tempo_fim = tempo_inicio + pd.Timedelta(hours=72)
        mask = (df_dados['time'] >= tempo_inicio) & (df_dados['time'] <= tempo_fim)
        return df_dados[mask].copy()
    
    # Procurar intervalo com transições claras e boa separação visual
    horas_desejadas = 72  # 3 dias
    melhor_intervalo = None
    melhor_score = 0
    melhor_info = None
    
    # Testar muitos pontos diferentes
    import random
    random.seed(42)
    n_testes = min(100, len(df_dados) - 1000)
    
    # Criar lista de índices para testar
    indices_teste = []
    
    # 1. Testar pontos aleatórios
    if len(df_dados) > 1000:
        indices_teste.extend(random.sample(range(0, len(df_dados) - 1000), n_testes // 2))
    
    # 2. Testar pontos onde há transições de estado (mais provável ter ambos estados)
    df_dados['estado_anterior'] = df_dados['equipamento_status'].shift(1)
    transicoes = df_dados[df_dados['equipamento_status'] != df_dados['estado_anterior']].index.tolist()
    if len(transicoes) > 0:
        indices_teste.extend(random.sample(transicoes, min(n_testes // 2, len(transicoes))))
    
    for idx in indices_teste:
        if idx >= len(df_dados) - 100:
            continue
            
        tempo_inicio = df_dados.loc[idx, 'time']
        tempo_fim = tempo_inicio + pd.Timedelta(hours=horas_desejadas)
        
        mask = (df_dados['time'] >= tempo_inicio) & (df_dados['time'] <= tempo_fim)
        df_intervalo = df_dados[mask]
        
        if len(df_intervalo) < 100:
            continue
        
        # Verificar se tem ambos estados
        estados = df_intervalo['equipamento_status'].unique()
        if 'LIGADO' not in estados or 'DESLIGADO' not in estados:
            continue
        
        # Calcular métricas de qualidade do intervalo
        n_ligado = (df_intervalo['equipamento_status'] == 'LIGADO').sum()
        n_desligado = (df_intervalo['equipamento_status'] == 'DESLIGADO').sum()
        
        # Evitar intervalos onde um estado domina completamente
        if n_ligado < 50 or n_desligado < 50:
            continue
        
        balanceamento = min(n_ligado, n_desligado) / max(n_ligado, n_desligado)
        transicoes_intervalo = (df_intervalo['equipamento_status'] != df_intervalo['equipamento_status'].shift(1)).sum()
        score = len(df_intervalo) * 0.3 + balanceamento * 1000 + min(transicoes_intervalo, 50) * 10
        
        if score > melhor_score:
            melhor_score = score
            melhor_intervalo = (tempo_inicio, tempo_fim)
            melhor_info = {'ligado': n_ligado, 'desligado': n_desligado, 'transicoes': transicoes_intervalo, 'balanceamento': balanceamento}
    
    if melhor_intervalo and melhor_info:
        tempo_inicio, tempo_fim = melhor_intervalo
        mask = (df_dados['time'] >= tempo_inicio) & (df_dados['time'] <= tempo_fim)
        df_selecionado = df_dados[mask].copy()
        
        print(f"  - Intervalo selecionado: {tempo_inicio.strftime('%d/%m/%Y %H:%M')} a {tempo_fim.strftime('%d/%m/%Y %H:%M')}")
        print(f"  - Total de pontos: {len(df_selecionado)}")
        print(f"  - LIGADO: {melhor_info['ligado']} pontos ({melhor_info['ligado']/len(df_selecionado)*100:.1f}%)")
        print(f"  - DESLIGADO: {melhor_info['desligado']} pontos ({melhor_info['desligado']/len(df_selecionado)*100:.1f}%)")
        print(f"  - Transições de estado: {melhor_info['transicoes']}")
        print(f"  - Balanceamento: {melhor_info['balanceamento']:.2f}")
    else:
        print("  - Procurando intervalo alternativo...")
        for i in range(0, len(df_dados) - 1000, 500):
            tempo_inicio = df_dados.loc[i, 'time']
            tempo_fim = tempo_inicio + pd.Timedelta(hours=horas_desejadas)
            mask = (df_dados['time'] >= tempo_inicio) & (df_dados['time'] <= tempo_fim)
            df_temp = df_dados[mask]
            
            if len(df_temp) > 100:
                estados = df_temp['equipamento_status'].unique()
                if 'LIGADO' in estados and 'DESLIGADO' in estados:
                    df_selecionado = df_temp.copy()
                    print(f"  - Intervalo encontrado: {tempo_inicio.strftime('%d/%m/%Y %H:%M')}")
                    print(f"  - Pontos: {len(df_selecionado)}")
                    return df_selecionado
        
        # Último fallback: usar primeiros 3 dias
        print("  - Usando primeiros 3 dias dos dados")
        tempo_inicio = df_dados['time'].min()
        tempo_fim = tempo_inicio + pd.Timedelta(hours=horas_desejadas)
        mask = (df_dados['time'] >= tempo_inicio) & (df_dados['time'] <= tempo_fim)
        df_selecionado = df_dados[mask].copy()
    
    return df_selecionado

def remover_outliers(df_dados, colunas=['current', 'vel_rms', 'rotational_speed']):
    """Remove outliers usando método IQR, preservando sequencias consecutivas
    (>=10 amostras) que indicam estados operacionais reais (ex: desligamento)"""
    print("\nRemovendo outliers...")
    
    df_limpo = df_dados.copy()
    n_inicial = len(df_limpo)
    
    for col in colunas:
        if col in df_limpo.columns:
            Q1 = df_limpo[col].quantile(0.25)
            Q3 = df_limpo[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Limites para outliers
            limite_inferior = Q1 - 3 * IQR
            limite_superior = Q3 + 3 * IQR
            
            # Identificar outliers
            mask_outlier = (df_limpo[col] < limite_inferior) | (df_limpo[col] > limite_superior)
            
            # Preservar sequencias consecutivas (>=10) = estado real
            indices_outlier = df_limpo.index[mask_outlier].tolist()
            if len(indices_outlier) > 0:
                pos_map = {idx: pos for pos, idx in enumerate(df_limpo.index.tolist())}
                posicoes = [pos_map[idx] for idx in indices_outlier]
                grupos = []
                grupo_atual = [indices_outlier[0]]
                pos_anterior = posicoes[0]
                for i in range(1, len(posicoes)):
                    if posicoes[i] == pos_anterior + 1:
                        grupo_atual.append(indices_outlier[i])
                    else:
                        grupos.append(grupo_atual)
                        grupo_atual = [indices_outlier[i]]
                    pos_anterior = posicoes[i]
                grupos.append(grupo_atual)
                # Des-flagar grupos com >= 10 amostras
                for grupo in grupos:
                    if len(grupo) >= 10:
                        mask_outlier.loc[grupo] = False
            
            # Filtrar apenas outliers pontuais (nao estados reais)
            df_limpo = df_limpo[~mask_outlier]
    
    n_final = len(df_limpo)
    removidos = n_inicial - n_final
    print(f"  - Outliers removidos: {removidos} ({removidos/n_inicial*100:.1f}%)")
    print(f"  - Pontos restantes: {n_final}")
    
    return df_limpo

def amostrar_dados(df_dados, n_amostras=1000):
    """Amostra dados uniformemente ao longo do tempo"""
    print(f"\nAmostrando {n_amostras} pontos para visualização...")

    df_dados = df_dados.sort_values('time').reset_index(drop=True)

    if len(df_dados) > n_amostras:
        indices = np.linspace(0, len(df_dados) - 1, n_amostras, dtype=int)
        df_amostrado = df_dados.iloc[indices].copy()
    else:
        df_amostrado = df_dados.copy()

    print(f"  - Amostras finais: {len(df_amostrado)}")

    return df_amostrado

def plotar_3d(df_dados, mpoint=None):
    """Gera visualizações 3D interativas dos dados classificados"""
    print("\nGerando visualizações 3D interativas...")

    # Extrair dados
    timestamps = df_dados['time']
    corrente = df_dados['current'].values if 'current' in df_dados.columns else np.zeros(len(df_dados))
    vibracao = df_dados['vel_rms'].values if 'vel_rms' in df_dados.columns else np.zeros(len(df_dados))
    rpm = df_dados['rotational_speed'].values if 'rotational_speed' in df_dados.columns else np.zeros(len(df_dados))
    temperatura = df_dados['object_temp'].values if 'object_temp' in df_dados.columns else np.zeros(len(df_dados))
    status = df_dados['equipamento_status'].values
    
    # Verificar variância dos dados para escolher melhor eixo
    rpm_variance = np.var(rpm)
    temp_variance = np.var(temperatura)
    
    # Se RPM tem pouca variação, usar temperatura no lugar
    usar_temperatura = rpm_variance < 100  # threshold para decidir
    
    if usar_temperatura:
        print("  - Usando TEMPERATURA no eixo X (RPM tem pouca variação)")
        eixo_x_valores = temperatura
        eixo_x_label = 'Temperatura (°C)'
    else:
        print("  - Usando RPM no eixo X")
        eixo_x_valores = rpm
        eixo_x_label = 'RPM'

    # Converter timestamps para horas desde o início
    tempo_inicio = timestamps.min()
    tempo_horas = np.array([(t - tempo_inicio).total_seconds() / 3600 for t in timestamps])

    # Criar labels de tempo para o eixo Z (3 datas ao longo do período)
    timestamps_sorted = sorted(timestamps)
    tempo_min = timestamps_sorted[0]
    tempo_max = timestamps_sorted[-1]
    tempo_meio = tempo_min + (tempo_max - tempo_min) / 2
    
    # Posições no eixo Z (horas)
    z_inicio = 0
    z_meio = (tempo_max - tempo_min).total_seconds() / 3600 / 2
    z_fim = (tempo_max - tempo_min).total_seconds() / 3600
    
    # Labels formatados
    label_inicio = tempo_min.strftime('%d/%m %H:%M')
    label_meio = tempo_meio.strftime('%d/%m %H:%M')
    label_fim = tempo_max.strftime('%d/%m %H:%M')
    
    # Estatísticas
    print(f"\n[STATS] Estatísticas dos dados:")
    print(f"  - Corrente (A): {corrente.min():.2f} - {corrente.max():.2f} (média: {corrente.mean():.2f})")
    print(f"  - Vibração (mm/s): {vibracao.min():.4f} - {vibracao.max():.4f} (média: {vibracao.mean():.4f})")
    print(f"  - RPM: {rpm.min():.0f} - {rpm.max():.0f} (média: {rpm.mean():.0f}) [var: {rpm_variance:.2f}]")
    print(f"  - Temperatura (°C): {temperatura.min():.2f} - {temperatura.max():.2f} (média: {temperatura.mean():.2f}) [var: {temp_variance:.2f}]")
    print(f"  - Período: {label_inicio} até {label_fim}")

    # Estatísticas por estado
    print(f"\n[STATS] Por estado:")
    for estado in np.unique(status):
        mask = status == estado
        print(f"  - {estado}: {mask.sum()} pontos ({mask.sum()/len(status)*100:.1f}%)")
        if usar_temperatura:
            print(f"    Corrente: {corrente[mask].mean():.2f}A, Vibração: {vibracao[mask].mean():.4f}mm/s, Temp: {temperatura[mask].mean():.2f}°C")
        else:
            print(f"    Corrente: {corrente[mask].mean():.2f}A, Vibração: {vibracao[mask].mean():.4f}mm/s, RPM: {rpm[mask].mean():.0f}")

    # Cores para os estados
    cores = {
        'DESLIGADO': '#e74c3c',  # Vermelho
        'LIGADO': '#2ecc71'       # Verde
    }

    # Gráfico 1: Corrente x Vibração x Tempo
    print("\n[VIZ 1/2] Corrente x Vibração x Tempo - Janela Interativa...")
    fig1 = plt.figure(figsize=(16, 12))
    ax1 = fig1.add_subplot(111, projection='3d')
    
    for estado in ['DESLIGADO', 'LIGADO']:
        mask = status == estado
        if mask.sum() > 0:
            ax1.scatter(
                corrente[mask], 
                vibracao[mask], 
                tempo_horas[mask],
                c=cores.get(estado, '#95a5a6'),
                label=estado,
                s=50,
                alpha=0.7,
                edgecolors='k',
                linewidths=0.5
            )

    ax1.set_xlabel('Corrente (A)', fontsize=12, fontweight='bold', labelpad=10)
    ax1.set_ylabel('Vibração (mm/s)', fontsize=12, fontweight='bold', labelpad=10)
    ax1.set_zlabel('Tempo', fontsize=12, fontweight='bold', labelpad=10)
    
    # Configurar ticks do eixo Z com 3 datas
    ax1.set_zticks([z_inicio, z_meio, z_fim])
    ax1.set_zticklabels([label_inicio, label_meio, label_fim], fontsize=10)
    
    ax1.set_title(f'Corrente x Vibração x Tempo (3 dias) - Estados do Equipamento\n{mpoint or ""}',
                  fontsize=14, fontweight='bold', pad=20)
    ax1.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    DIR_RESULTS.mkdir(parents=True, exist_ok=True)
    output1 = DIR_RESULTS / f'estados_corrente_vibracao_tempo_3d{"_" + mpoint if mpoint else ""}.png'
    plt.savefig(output1, dpi=300, bbox_inches='tight')
    print(f"  [OK] Salvo: {output1}")
    
    # Gráfico 2: Eixo X adaptativo x Vibração x Tempo
    titulo_g2 = f'{eixo_x_label} x Vibração x Tempo (3 dias) - Estados do Equipamento\n{mpoint or ""}'
    print(f"\n[VIZ 2/2] {eixo_x_label} x Vibração x Tempo - Janela Interativa...")
    fig2 = plt.figure(figsize=(16, 12))
    ax2 = fig2.add_subplot(111, projection='3d')
    
    for estado in ['DESLIGADO', 'LIGADO']:
        mask = status == estado
        if mask.sum() > 0:
            ax2.scatter(
                eixo_x_valores[mask],
                vibracao[mask], 
                tempo_horas[mask],
                c=cores.get(estado, '#95a5a6'),
                label=estado,
                s=50,
                alpha=0.7,
                edgecolors='k',
                linewidths=0.5
            )

    ax2.set_xlabel(eixo_x_label, fontsize=12, fontweight='bold', labelpad=10)
    ax2.set_ylabel('Vibração (mm/s)', fontsize=12, fontweight='bold', labelpad=10)
    ax2.set_zlabel('Tempo', fontsize=12, fontweight='bold', labelpad=10)
    
    # Configurar ticks do eixo Z com 3 datas
    ax2.set_zticks([z_inicio, z_meio, z_fim])
    ax2.set_zticklabels([label_inicio, label_meio, label_fim], fontsize=10)
    
    ax2.set_title(titulo_g2, fontsize=14, fontweight='bold', pad=20)
    ax2.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    
    # Nome do arquivo baseado no eixo usado
    nome_eixo = 'temperatura' if usar_temperatura else 'rpm'
    output2 = DIR_RESULTS / f'estados_{nome_eixo}_vibracao_tempo_3d{"_" + mpoint if mpoint else ""}.png'
    plt.savefig(output2, dpi=300, bbox_inches='tight')
    print(f"  [OK] Salvo: {output2}")
    
    # Mostrar janelas interativas
    print("\n[INFO] Abrindo janelas interativas...")
    print("  - Use o mouse para rotacionar os gráficos")
    print("  - Feche as janelas para continuar")
    plt.show()

    print("\n[OK] Visualizações 3D concluídas!")

def main(mpoint=None):
    """Função principal"""
    print("=" * 80)
    print("[VIZ] VISUALIZAÇÃO 3D INTERATIVA - ESTADOS DO EQUIPAMENTO")
    print("=" * 80)
    if mpoint:
        print(f"Mpoint: {mpoint}")
    print("Período: 3 dias contínuos")
    print("Outliers: Removidos automaticamente (IQR)")
    print("=" * 80)

    # 1. Carregar dados e scaler
    df_classificado, scaler = carregar_dados_e_scaler(mpoint)
    if df_classificado is None or scaler is None:
        print("[ERRO] Falha ao carregar dados")
        return False

    # 2. Desnormalizar dados
    df_desnorm = desnormalizar_dados(df_classificado, scaler)

    # 3. Selecionar intervalo de 3 dias
    df_3dias = selecionar_intervalo_3_dias(df_desnorm)

    # 4. Remover outliers
    df_limpo = remover_outliers(df_3dias, colunas=['current', 'vel_rms', 'rotational_speed'])

    # 5. Amostrar para visualização (1000 pontos)
    df_amostrado = amostrar_dados(df_limpo, n_amostras=1000)

    # 6. Plotar 3D interativo
    plotar_3d(df_amostrado, mpoint)
    
    print("\n" + "=" * 80)
    print("[OK] VISUALIZAÇÃO CONCLUÍDA!")
    print("=" * 80)

    # Gerar logs detalhados para TCC
    import time
    from datetime import datetime
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
        'total_samples_original': len(df_classificado) if 'df_classificado' in locals() else 0,
        'samples_after_outlier_removal': len(df_limpo) if 'df_limpo' in locals() else 0,
        'samples_visualized': len(df_amostrado) if 'df_amostrado' in locals() else 0,
        'sampling_rate': 1000,  # Como especificado no código
        'visualization_type': '3d_interactive_current_vibration_time',
        'charts_generated': len(chart_files),
        'data_period_days': 3,  # Como especificado no código
        'equipment_states': ['LIGADO', 'DESLIGADO'] if df_classificado is not None and 'equipamento_status' in df_classificado.columns else []
    }

    # Log de visualização 3D avançada
    viz_log = create_visualization_log(
        script_name='visualizar_clusters_3d',
        mpoint=mpoint,
        chart_type='3d_interactive_equipment_states_advanced',
        data_description={
            'data_source': str(arquivo_classificado) if 'arquivo_classificado' in locals() else 'unknown',
            'total_samples_original': viz_stats['total_samples_original'],
            'samples_after_preprocessing': viz_stats['samples_after_outlier_removal'],
            'samples_visualized': viz_stats['samples_visualized'],
            'visualization_dimensions': ['current', 'vibration', 'time'],
            'data_denormalized': True,
            'outlier_removal_applied': True,
            'equipment_states_identified': viz_stats['equipment_states']
        },
        chart_files=chart_files,
        period_info={
            'data_start': str(df_3dias['time'].min()) if 'df_3dias' in locals() else 'unknown',
            'data_end': str(df_3dias['time'].max()) if 'df_3dias' in locals() else 'unknown',
            'total_samples': viz_stats['total_samples_original'],
            'visualization_period_days': 3,
            'time_sampling': 'random_sampling_1000_points',
            'data_quality_filters': ['outlier_removal', 'denormalization']
        },
        visualization_details={
            'plot_type': '3d_scatter_interactive',
            'axes': ['current', 'vibration_velocity', 'time'],
            'color_coding': 'equipment_states',
            'interactive_rotation': True,
            'dpi_quality': 300,
            'data_preprocessing': ['denormalization', 'outlier_removal', 'sampling'],
            'state_separation': True,
            'view_angles': 'multiple_optimized'
        }
    )

    save_log(viz_log, 'visualizar_clusters_3d', mpoint, '3d_advanced_visualization_complete')

    # Enriquecer arquivo results
    results_data = {
        'visualization_3d_advanced_completed': True,
        'visualization_3d_advanced_timestamp': datetime.now().isoformat(),
        'charts_generated': len(chart_files),
        'visualization_type': '3d_interactive_states',
        'data_samples_original': viz_stats['total_samples_original'],
        'data_samples_visualized': viz_stats['samples_visualized'],
        'time_window_days': 3,
        'sampling_points': 1000,
        'equipment_states_detected': viz_stats['equipment_states'],
        'visualization_charts': chart_files,
        'visualization_parameters': viz_log['visualization_details'],
        'data_preprocessing_applied': ['denormalization', 'outlier_removal', 'random_sampling']
    }

    if mpoint:
        enrich_results_file(mpoint, results_data)

    return True

def parse_arguments():
    """Parse argumentos da linha de comando"""
    parser = argparse.ArgumentParser(
        description="Visualizador 3D dos estados do equipamento"
    )
    parser.add_argument(
        '--mpoint',
        type=str,
        help='ID do mpoint (ex: c_636)'
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    main(mpoint=args.mpoint)
