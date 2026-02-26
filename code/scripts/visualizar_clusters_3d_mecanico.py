"""
Visualização 3D para equipamentos MECÂNICOS.
Gráficos: Temperatura x Vibração x Tempo
Com seleção inteligente de intervalo de 3 dias mostrando mudanças de estado.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import argparse
import sys
import random

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from utils.artifact_paths import (
    processed_classificado_path,
    results_dir,
    scaler_maxmin_path,
    scaler_model_path,
)

def desnormalizar_dados_mecanico(df_classificado, mpoint):
    """Desnormaliza os dados usando o scaler mecânico"""
    print("[INFO] Desnormalizando dados...")

    # Carregar scaler
    scaler_path = scaler_model_path(mpoint)
    if not scaler_path.exists():
        print(f"[ERRO] Scaler não encontrado: {scaler_path}")
        return df_classificado

    try:
        import pickle
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

        # Colunas que NÃO devem ser desnormalizadas
        colunas_excluir = [
            'time', 'cluster', 'equipamento_status', 'estado',
            'm_point', 'periodo_id', 'interpolado', 'arquivo_origem',
            'mpoint_id'
        ]

        # Extrair colunas de features (apenas numéricas)
        feature_cols = [col for col in df_classificado.columns
                       if col not in colunas_excluir and df_classificado[col].dtype in ['float64', 'int64']]

        print(f"  - Features para desnormalizar: {len(feature_cols)}")

        if len(feature_cols) > 0:
            dados_norm = df_classificado[feature_cols].values
            dados_desnorm = scaler.inverse_transform(dados_norm)

            # Criar DataFrame desnormalizado
            df_desnorm = pd.DataFrame(dados_desnorm, columns=feature_cols, index=df_classificado.index)

            # Combinar com informações de classificação
            df_final = pd.concat([
                df_classificado[['time', 'equipamento_status', 'cluster']],
                df_desnorm
            ], axis=1)

            print("  - Desnormalização concluída!")
            return df_final
        else:
            print("  - Nenhuma feature para desnormalizar")
            return df_classificado

    except Exception as e:
        print(f"[ERRO] Falha na desnormalização: {e}")
        return df_classificado

def selecionar_intervalo_3_dias_mecanico(df_dados, horas_desejadas=72):
    """
    Seleciona intervalo de 3 dias (72h) com mudanças de estado LIGADO↔DESLIGADO.
    
    Estratégia para equipamento MECÂNICO:
    - Prioriza intervalos com transições de estado
    - Evita intervalos dominados por um único estado (< 50 amostras)
    - Calcula score baseado em balanceamento e número de transições
    - Remove intervalos com % excessiva de outliers
    """
    print(f"\nSelecionando intervalo de {horas_desejadas}h (3 dias) com mudanças de estado...")
    
    if 'equipamento_status' not in df_dados.columns:
        print("  [AVISO] Coluna 'equipamento_status' não encontrada")
        tempo_inicio = df_dados['time'].min()
        tempo_fim = tempo_inicio + pd.Timedelta(hours=horas_desejadas)
        mask = (df_dados['time'] >= tempo_inicio) & (df_dados['time'] <= tempo_fim)
        return df_dados[mask].copy()
    
    # Calcular % de tempo em cada estado (global)
    total_amostras = len(df_dados)
    n_desligado_global = (df_dados['equipamento_status'] == 'DESLIGADO').sum()
    n_ligado_global = (df_dados['equipamento_status'] == 'LIGADO').sum()
    pct_desligado_global = (n_desligado_global / total_amostras) * 100
    pct_ligado_global = (n_ligado_global / total_amostras) * 100
    
    print(f"  - Distribuição global de estados:")
    print(f"    DESLIGADO: {n_desligado_global:,} amostras ({pct_desligado_global:.1f}%)")
    print(f"    LIGADO: {n_ligado_global:,} amostras ({pct_ligado_global:.1f}%)")
    
    # Thresholds: usar % mínima baseada no cluster desligado (pelo menos 10% do tempo desligado)
    pct_min_desligado = max(10.0, pct_desligado_global * 0.5)
    pct_max_desligado = min(90.0, pct_desligado_global * 2.0)
    
    print(f"  - Buscando intervalos com {pct_min_desligado:.1f}% a {pct_max_desligado:.1f}% do tempo DESLIGADO")
    
    # Buscar melhor intervalo
    melhor_score = -1
    melhor_intervalo = None
    melhor_info = None
    
    n_testes = 100
    indices_teste = []
    
    # 1. Testar pontos aleatórios
    if len(df_dados) > 1000:
        indices_teste.extend(random.sample(range(0, len(df_dados) - 1000), n_testes // 2))
    
    # 2. Testar pontos onde há transições de estado
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
        
        # Calcular métricas
        n_ligado = (df_intervalo['equipamento_status'] == 'LIGADO').sum()
        n_desligado = (df_intervalo['equipamento_status'] == 'DESLIGADO').sum()
        pct_desligado = (n_desligado / len(df_intervalo)) * 100
        
        # Filtro: % de tempo desligado deve estar dentro do range aceitável
        if pct_desligado < pct_min_desligado or pct_desligado > pct_max_desligado:
            continue
        
        # Evitar intervalos onde um estado domina completamente
        if n_ligado < 50 or n_desligado < 50:
            continue
        
        # Calcular balanceamento e transições
        balanceamento = min(n_ligado, n_desligado) / max(n_ligado, n_desligado)
        transicoes_intervalo = (df_intervalo['equipamento_status'] != df_intervalo['equipamento_status'].shift(1)).sum()
        
        # Score favorece:
        # - Maior número de amostras
        # - Melhor balanceamento entre estados
        # - Mais transições (até um limite)
        score = len(df_intervalo) * 0.3 + balanceamento * 1000 + min(transicoes_intervalo, 50) * 10
        
        if score > melhor_score:
            melhor_score = score
            melhor_intervalo = (tempo_inicio, tempo_fim)
            melhor_info = {
                'ligado': n_ligado,
                'desligado': n_desligado,
                'pct_desligado': pct_desligado,
                'transicoes': transicoes_intervalo,
                'balanceamento': balanceamento
            }
    
    # Retornar melhor intervalo encontrado
    if melhor_intervalo and melhor_info:
        tempo_inicio, tempo_fim = melhor_intervalo
        mask = (df_dados['time'] >= tempo_inicio) & (df_dados['time'] <= tempo_fim)
        df_selecionado = df_dados[mask].copy()
        
        print(f"  - Intervalo selecionado: {tempo_inicio.strftime('%d/%m/%Y %H:%M')} a {tempo_fim.strftime('%d/%m/%Y %H:%M')}")
        print(f"  - Total de pontos: {len(df_selecionado)}")
        print(f"  - LIGADO: {melhor_info['ligado']} pontos ({melhor_info['ligado']/len(df_selecionado)*100:.1f}%)")
        print(f"  - DESLIGADO: {melhor_info['desligado']} pontos ({melhor_info['pct_desligado']:.1f}%)")
        print(f"  - Transições de estado: {melhor_info['transicoes']}")
        print(f"  - Balanceamento: {melhor_info['balanceamento']:.2f}")
    else:
        # Fallback: buscar primeiro intervalo com ambos estados
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

def remover_outliers_mecanico(df_dados, colunas=['object_temp', 'vel_rms_x', 'vel_rms_y', 'vel_rms_z']):
    """Remove outliers usando método IQR (adaptado para equipamentos mecânicos),
    preservando sequencias consecutivas (>=10 amostras) = estados reais"""
    print("\nRemovendo outliers...")
    
    df_limpo = df_dados.copy()
    n_inicial = len(df_limpo)
    
    # Identificar colunas disponíveis
    colunas_disponiveis = [col for col in colunas if col in df_limpo.columns]
    
    if not colunas_disponiveis:
        # Fallback: procurar colunas de temperatura e vibração
        colunas_temp = [col for col in df_limpo.columns if 'temp' in col.lower()]
        colunas_vibracao = [col for col in df_limpo.columns if 'vel_rms' in col.lower()]
        colunas_disponiveis = colunas_temp + colunas_vibracao[:3]
    
    for col in colunas_disponiveis:
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
                for grupo in grupos:
                    if len(grupo) >= 10:
                        mask_outlier.loc[grupo] = False
            
            # Filtrar apenas outliers pontuais
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

def criar_visualizacao_3d_mecanico(mpoint):
    """Cria visualização 3D para equipamento MECÂNICO com intervalo de 3 dias"""
    print("="*80)
    print("VISUALIZAÇÃO 3D - EQUIPAMENTO MECÂNICO")
    print("="*80)
    print(f"Mpoint: {mpoint}")
    print("Período: 3 dias contínuos com mudanças de estado")
    print("Eixos: Temperatura x Vibração x Tempo")
    print("="*80)
    
    # Carregar dados classificados
    dados_path = processed_classificado_path(mpoint)
    
    if not dados_path.exists():
        print(f"[ERRO] Dados classificados não encontrados: {dados_path}")
        print("Execute o treino primeiro.")
        return False
    
    print("[INFO] Carregando dados classificados...")
    df = pd.read_csv(dados_path)

    if 'time' not in df.columns:
        print("[ERRO] Coluna 'time' não encontrada")
        return False

    # Converter timestamp para datetime
    df['time'] = pd.to_datetime(df['time'], format='mixed', utc=True)

    # Desnormalizar dados para mostrar valores reais
    df = desnormalizar_dados_mecanico(df, mpoint)
    
    # 1. Selecionar intervalo de 3 dias com mudanças de estado
    df_3dias = selecionar_intervalo_3_dias_mecanico(df, horas_desejadas=72)
    
    # 2. Identificar colunas de temperatura e vibração
    colunas_temp = [col for col in df_3dias.columns if 'temp' in col.lower()]
    colunas_vibracao = [col for col in df_3dias.columns if 'vel_rms' in col.lower()]
    
    if not colunas_temp:
        print("[AVISO] Nenhuma coluna de temperatura encontrada")
        df_3dias['object_temp'] = 0
        colunas_temp = ['object_temp']
    else:
        print(f"[INFO] Usando coluna de temperatura: {colunas_temp[0]}")
    
    if not colunas_vibracao:
        print("[AVISO] Nenhuma coluna de vibração encontrada")
        df_3dias['vel_rms_media'] = 0
        colunas_vibracao = ['vel_rms_media']
    else:
        # Calcular média das vibrações RMS
        df_3dias['vel_rms_media'] = df_3dias[colunas_vibracao].mean(axis=1)
        print(f"[INFO] Calculando média de {len(colunas_vibracao)} colunas de vibração")
    
    # 3. Remover outliers
    colunas_para_limpar = [colunas_temp[0], 'vel_rms_media']
    df_limpo = remover_outliers_mecanico(df_3dias, colunas=colunas_para_limpar)
    
    # 4. Amostrar para visualização (1000 pontos)
    df_amostrado = amostrar_dados(df_limpo, n_amostras=1000)
    
    # 5. Preparar dados para plot
    timestamps = df_amostrado['time']
    tempo_inicio = timestamps.min()
    tempo_fim = timestamps.max()
    
    # Converter para horas desde o início
    tempo_horas = (timestamps - tempo_inicio).dt.total_seconds() / 3600
    
    temperatura = df_amostrado[colunas_temp[0]].values
    vibracao = df_amostrado['vel_rms_media'].values
    status = df_amostrado['equipamento_status'].values if 'equipamento_status' in df_amostrado.columns else ['LIGADO'] * len(df_amostrado)
    
    # Preparar labels do eixo Z (tempo) com 3 datas
    z_inicio = tempo_horas.min()
    z_meio = (tempo_horas.min() + tempo_horas.max()) / 2
    z_fim = tempo_horas.max()
    
    label_inicio = tempo_inicio.strftime('%d/%m %H:%M')
    label_meio = (tempo_inicio + pd.Timedelta(hours=(z_fim - z_inicio) / 2)).strftime('%d/%m %H:%M')
    label_fim = tempo_fim.strftime('%d/%m %H:%M')
    
    # 6. Criar visualizações
    print("\n[VIZ] Gerando gráficos 3D...")
    
    # Cores por estado
    cores = {'DESLIGADO': '#e74c3c', 'LIGADO': '#2ecc71'}
    
    # Gráfico principal: Temperatura x Vibração x Tempo
    fig = plt.figure(figsize=(16, 12))
    ax1 = fig.add_subplot(111, projection='3d')
    
    for estado in ['DESLIGADO', 'LIGADO']:
        mask = status == estado
        if mask.sum() > 0:
            ax1.scatter(
                temperatura[mask],
                vibracao[mask],
                tempo_horas.values[mask],
                c=cores.get(estado, '#95a5a6'),
                label=estado,
                s=50,
                alpha=0.7,
                edgecolors='k',
                linewidths=0.5
            )
    
    ax1.set_xlabel('Temperatura (°C)', fontsize=12, fontweight='bold', labelpad=10)
    ax1.set_ylabel('Vibração RMS Média (mm/s)', fontsize=12, fontweight='bold', labelpad=10)
    ax1.set_zlabel('Tempo', fontsize=12, fontweight='bold', labelpad=10)
    
    # Configurar ticks do eixo Z com 3 datas
    ax1.set_zticks([z_inicio, z_meio, z_fim])
    ax1.set_zticklabels([label_inicio, label_meio, label_fim], fontsize=10)
    
    ax1.set_title(f'Temperatura x Vibração x Tempo (3 dias) - Estados do Equipamento MECÂNICO\n{mpoint}',
                  fontsize=14, fontweight='bold', pad=20)
    ax1.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    
    # Salvar
    results_path = results_dir(mpoint, create=True)
    arquivo_plot = results_path / f'estados_temperatura_vibracao_tempo_3d_{mpoint}.png'
    plt.savefig(arquivo_plot, dpi=300, bbox_inches='tight')
    
    print(f"\n[OK] Visualização salva: {arquivo_plot}")
    
    # Mostrar gráfico interativo
    print("\n[INFO] Abrindo janela interativa...")
    print("  - Use o mouse para rotacionar o gráfico")
    print("  - Feche a janela para continuar")
    plt.show()
    
    plt.close()
    
    return True

def main():
    """Função principal"""
    parser = argparse.ArgumentParser(
        description="Visualização 3D para equipamento MECÂNICO"
    )
    parser.add_argument('--mpoint', type=str, required=True, help='ID do mpoint')
    
    args = parser.parse_args()
    
    if not criar_visualizacao_3d_mecanico(args.mpoint):
        print("\n[ERRO] Falha ao criar visualização")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("VISUALIZAÇÃO CONCLUÍDA COM SUCESSO!")
    print("="*80)

if __name__ == '__main__':
    main()

