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

def selecionar_top_intervalos_3_dias_mecanico(df_dados, num_intervalos=3, horas_desejadas=72):
    """
    Seleciona múltiplos intervalos não sobrepostos de 3 dias com mudanças de estado.
    """
    print(f"\nSelecionando até {num_intervalos} intervalos de {horas_desejadas}h (3 dias) com mudanças de estado...")
    
    if 'equipamento_status' not in df_dados.columns:
        print("  [AVISO] Coluna 'equipamento_status' não encontrada")
        tempo_inicio = df_dados['time'].min()
        tempo_fim = tempo_inicio + pd.Timedelta(hours=horas_desejadas)
        mask = (df_dados['time'] >= tempo_inicio) & (df_dados['time'] <= tempo_fim)
        return [df_dados[mask].copy()]
    
    # Encontrar transições
    df_dados['estado_anterior'] = df_dados['equipamento_status'].shift(1)
    transicoes = df_dados[df_dados['equipamento_status'] != df_dados['estado_anterior']].index.tolist()
    
    candidatos = []
    # Amostrar algumas posicoes de transição
    indices_teste = random.sample(transicoes, min(200, len(transicoes))) if transicoes else range(0, len(df_dados) - 100, 500)
    
    for idx in indices_teste:
        if idx >= len(df_dados) - 100:
            continue
            
        tempo_inicio = df_dados.loc[idx, 'time']
        tempo_fim = tempo_inicio + pd.Timedelta(hours=horas_desejadas)
        
        mask = (df_dados['time'] >= tempo_inicio) & (df_dados['time'] <= tempo_fim)
        df_intervalo = df_dados[mask]
        
        if len(df_intervalo) < 100:
            continue
            
        estados = df_intervalo['equipamento_status'].unique()
        if 'LIGADO' not in estados or 'DESLIGADO' not in estados:
            continue
            
        n_ligado = (df_intervalo['equipamento_status'] == 'LIGADO').sum()
        n_desligado = (df_intervalo['equipamento_status'] == 'DESLIGADO').sum()
        
        if n_ligado < 50 or n_desligado < 50:
            continue
            
        balanceamento = min(n_ligado, n_desligado) / max(n_ligado, n_desligado)
        trans_int = (df_intervalo['equipamento_status'] != df_intervalo['equipamento_status'].shift(1)).sum()
        
        score = balanceamento * 1000 + min(trans_int, 50) * 10
        candidatos.append({'inicio': tempo_inicio, 'fim': tempo_fim, 'score': score, 'df': df_intervalo.copy()})
        
    candidatos.sort(key=lambda x: x['score'], reverse=True)
    
    selecionados = []
    for cand in candidatos:
        if len(selecionados) >= num_intervalos:
            break
        sobreposto = False
        for sel in selecionados:
            if not (cand['fim'] <= sel['inicio'] or cand['inicio'] >= sel['fim']):
                sobreposto = True
                break
        if not sobreposto:
            selecionados.append(cand)
            
    if not selecionados:
        print("  [AVISO] Nenhum intervalo bom encontrado. Usando primeiros dias.")
        tempo_inicio = df_dados['time'].min()
        tempo_fim = tempo_inicio + pd.Timedelta(hours=horas_desejadas)
        mask = (df_dados['time'] >= tempo_inicio) & (df_dados['time'] <= tempo_fim)
        selecionados.append({'df': df_dados[mask].copy()})
        
    return [s['df'] for s in selecionados]

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
    
    # 1. Selecionar 3 intervalos de 3 dias com mudanças de estado
    intervalos_3dias = selecionar_top_intervalos_3_dias_mecanico(df, num_intervalos=3, horas_desejadas=72)
    
    resultados_plots = []
    
    for i, df_3dias in enumerate(intervalos_3dias):
        print(f"\n--- Processando gráfico {i+1} de {len(intervalos_3dias)} ---")
        
        # 2. Identificar colunas de temperatura e vibração
        colunas_temp = [col for col in df_3dias.columns if 'temp' in col.lower()]
        colunas_vibracao = [col for col in df_3dias.columns if 'vel_rms' in col.lower()]
        
        if not colunas_temp:
            df_3dias['object_temp'] = 0
            colunas_temp = ['object_temp']
        
        if not colunas_vibracao:
            df_3dias['vel_rms_media'] = 0
            colunas_vibracao = ['vel_rms_media']
        else:
            df_3dias['vel_rms_media'] = df_3dias[colunas_vibracao].mean(axis=1)
        
        # 3. Remover outliers
        colunas_para_limpar = [colunas_temp[0], 'vel_rms_media']
        df_limpo = remover_outliers_mecanico(df_3dias, colunas=colunas_para_limpar)
        
        # 4. Amostrar para visualização (1000 pontos)
        df_amostrado = amostrar_dados(df_limpo, n_amostras=1000)
        
        # 5. Preparar dados para plot
        timestamps = df_amostrado['time']
        tempo_inicio = timestamps.min()
        tempo_fim = timestamps.max()
        
        tempo_horas = (timestamps - tempo_inicio).dt.total_seconds() / 3600
        
        temperatura = df_amostrado[colunas_temp[0]].values
        vibracao = df_amostrado['vel_rms_media'].values
        status = df_amostrado['equipamento_status'].values if 'equipamento_status' in df_amostrado.columns else ['LIGADO'] * len(df_amostrado)
        
        z_inicio = tempo_horas.min()
        z_meio = (tempo_horas.min() + tempo_horas.max()) / 2
        z_fim = tempo_horas.max()
        
        label_inicio = tempo_inicio.strftime('%d/%m %H:%M')
        label_meio = (tempo_inicio + pd.Timedelta(hours=(z_fim - z_inicio) / 2)).strftime('%d/%m %H:%M')
        label_fim = tempo_fim.strftime('%d/%m %H:%M')
        
        # 6. Criar visualização
        cores = {'DESLIGADO': '#e74c3c', 'LIGADO': '#2ecc71'}
        
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
        
        ax1.set_zticks([z_inicio, z_meio, z_fim])
        ax1.set_zticklabels([label_inicio, label_meio, label_fim], fontsize=10)
        
        ax1.set_title(f'Gráfico {i+1} - Temp x Vib x Tempo ({label_inicio} a {label_fim})\n{mpoint}',
                      fontsize=14, fontweight='bold', pad=20)
        ax1.legend(loc='upper left', fontsize=10, framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        ax1.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        
        results_path = results_dir(mpoint, create=True)
        arquivo_plot = results_path / f'estados_temperatura_vibracao_tempo_3d_{i+1}_{mpoint}.png'
        plt.savefig(arquivo_plot, dpi=300, bbox_inches='tight')
        resultados_plots.append(arquivo_plot)
        print(f"Salvo: {arquivo_plot}")
    
    print("\n[INFO] Gráficos salvos com sucesso!")
    # Removido plt.show() para não travar o pipeline
    
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

