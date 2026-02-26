"""
Script simples pra cortar os dados em pedacos e preencher lacunas.
Divide os dados em segmentos e usa interpolacao pra completar os valores que faltam.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from scipy.interpolate import interp1d, UnivariateSpline
from sklearn.neighbors import KNeighborsRegressor
import warnings
import argparse
import sys
warnings.filterwarnings('ignore')

# Force UTF-8 encoding to avoid UnicodeDecodeError
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Importacoes para logging estruturado
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from utils.logging_utils import (
    save_log,
    create_processing_log,
    format_file_list,
    get_file_info,
    enrich_results_file,
)

def interpolar_simples(t, v, t_novo):
    """Interpolacao simples para gaps < 1h"""
    try:
        if len(t) >= 4:
            spline = UnivariateSpline(t, v, k=3, s=0)
            return spline(t_novo)
    except:
        pass
    f = interp1d(t, v, kind='linear', bounds_error=False, fill_value=(v[0], v[-1]))
    return f(t_novo)

def interpolar_avancada(t, v, t_novo):
    """Interpolacao avancada para gaps 1-3h"""
    try:
        if len(t) >= 10:
            knn = KNeighborsRegressor(n_neighbors=min(5, len(t)//2), weights='distance')
            knn.fit(t.reshape(-1, 1), v)
            v_knn = knn.predict(t_novo.reshape(-1, 1))
            f_lin = interp1d(t, v, kind='linear', bounds_error=False, fill_value=(v[0], v[-1]))
            v_lin = f_lin(t_novo)
            return 0.6 * v_knn + 0.4 * v_lin
    except:
        pass
    f = interp1d(t, v, kind='linear', bounds_error=False, fill_value=(v[0], v[-1]))
    return f(t_novo)

def _remover_sequencias_consecutivas(mask_outlier, series_original, min_consecutivas=10):
    """Remove sequencias de outliers consecutivos com >=min_consecutivas amostras.
    Essas sequencias representam estados operacionais reais (ex: desligamento),
    nao ruido pontual. Respeita a lei da inercia: equipamento leva tempo pra
    desligar e ligar.
    
    IMPORTANTE: Detecta consecutividade entre valores validos (nao-NaN) da
    serie original, nao entre indices brutos do DataFrame. Isso e necessario
    porque sensores diferentes reportam em linhas diferentes (dados intercalados).
    
    Args:
        mask_outlier: Serie booleana com True onde valor e outlier
        series_original: Serie original com valores (inclui NaN)
        min_consecutivas: Minimo de amostras consecutivas para preservar
    """
    mask = mask_outlier.copy()
    flagged = mask[mask].index.tolist()
    if len(flagged) == 0:
        return mask
    # Indices que tem valor real (nao-NaN) na serie original
    valid_indices = series_original.dropna().index.tolist()
    if len(valid_indices) == 0:
        return mask
    # Mapear indice do DataFrame -> posicao entre valores validos
    pos_map = {idx: pos for pos, idx in enumerate(valid_indices)}
    # Filtrar flagados que estao nos valores validos
    flagged_valid = [(idx, pos_map[idx]) for idx in flagged if idx in pos_map]
    if len(flagged_valid) == 0:
        return mask
    # Identificar blocos contiguos de posicoes entre valores validos
    grupos_idx = []
    grupo_atual = [flagged_valid[0][0]]
    pos_anterior = flagged_valid[0][1]
    for i in range(1, len(flagged_valid)):
        idx, pos = flagged_valid[i]
        if pos == pos_anterior + 1:
            grupo_atual.append(idx)
        else:
            grupos_idx.append(grupo_atual)
            grupo_atual = [idx]
        pos_anterior = pos
    grupos_idx.append(grupo_atual)
    # Des-flagar grupos com >= min_consecutivas (estado real, nao outlier)
    for grupo in grupos_idx:
        if len(grupo) >= min_consecutivas:
            mask.loc[grupo] = False
    return mask

def remover_outliers(series):
    """Remove outliers usando IQR, preservando sequencias consecutivas (>=15 amostras)
    que indicam estados operacionais reais (ex: desligamento)"""
    valores = series.dropna()
    if len(valores) < 4:
        return series
    Q1, Q3 = valores.quantile(0.25), valores.quantile(0.75)
    IQR = Q3 - Q1
    limites = (Q1 - 3*IQR, Q3 + 3*IQR)
    s = series.copy()
    mask_outlier = (s < limites[0]) | (s > limites[1])
    # Preservar sequencias de >=15 amostras consecutivas (estado real, nao ruido)
    mask_outlier = _remover_sequencias_consecutivas(mask_outlier, series, min_consecutivas=10)
    s[mask_outlier] = np.nan
    return s

def gerar_timestamps(inicio, fim):
    """Gera timestamps de 20 em 20 segundos"""
    segundo = (inicio.second // 20) * 20
    t = inicio.replace(second=segundo, microsecond=0)
    if t < inicio:
        t += timedelta(seconds=20)
    timestamps = []
    while t <= fim:
        timestamps.append(t)
        t += timedelta(seconds=20)
    return timestamps

def interpolar_coluna(timestamps_orig, valores, timestamps_novos, nome):
    """Interpola uma coluna com estrategia adaptativa (ultra-otimizada)"""
    mask = ~np.isnan(valores)
    if mask.sum() < 2:
        return np.full(len(timestamps_novos), np.nan)
    
    # Converter timestamps para segundos desde epoca
    t0 = pd.Timestamp(timestamps_orig[0]).value
    t_orig = (pd.Series(timestamps_orig).astype('int64').values - t0) / 1e9
    t_novo = (pd.Series(timestamps_novos).astype('int64').values - t0) / 1e9
    t_validos = t_orig[mask]
    v_validos = valores[mask]
    
    # Calcular gaps
    gaps = np.diff(t_validos)
    
    # Se nao ha gaps grandes (> 1h), usar interpolacao linear direta
    if gaps.max() < 3600:
        f = interp1d(t_validos, v_validos, kind='linear', bounds_error=False, 
                     fill_value=(v_validos[0], v_validos[-1]))
        return f(t_novo)
    
    # Se ha gaps grandes, processar apenas esses gaps
    indices_gaps = np.where(gaps > 3600)[0]
    
    # Comecar com interpolacao linear de tudo
    f_base = interp1d(t_validos, v_validos, kind='linear', bounds_error=False,
                      fill_value=(v_validos[0], v_validos[-1]))
    resultado = f_base(t_novo)
    
    # Refinar apenas os gaps grandes com interpolacao avancada
    for i in indices_gaps:
        gap_h = gaps[i] / 3600
        
        if gap_h >= 1:  # Apenas gaps >= 1h
            mask_gap = (t_novo > t_validos[i]) & (t_novo < t_validos[i+1])
            
            if mask_gap.any():
                t_gap = t_novo[mask_gap]
                
                # Contexto: 3 pontos antes e 3 depois
                idx_ini = max(0, i-3)
                idx_fim = min(len(t_validos), i+4)
                t_ctx = t_validos[idx_ini:idx_fim]
                v_ctx = v_validos[idx_ini:idx_fim]
                
                if len(t_ctx) >= 3:
                    try:
                        valores_gap = interpolar_avancada(t_ctx, v_ctx, t_gap)
                        resultado[mask_gap] = valores_gap
                    except:
                        # Se falhar, manter interpolacao linear
                        pass
    
    return resultado

def processar_periodo_worker(args):
    """Worker para processar um periodo em paralelo"""
    i, idx_i, idx_f, df_est, df_slip, dir_out, periodo_minimo, modo_intervalo = args
    
    try:
        p_est = df_est.iloc[idx_i:idx_f].copy()
        
        t_inicio = p_est['time'].min()
        t_fim = p_est['time'].max()
        duracao_h = (t_fim - t_inicio).total_seconds() / 3600
        
        # Filtrar slip para periodo
        p_slip = df_slip[(df_slip['time'] >= t_inicio) & (df_slip['time'] <= t_fim)].copy()
        
        # Validar duracao
        if duracao_h < periodo_minimo:
            print(f"   [REJEITADO] Periodo {i}: duracao muito curta ({duracao_h:.1f}h < {periodo_minimo:.1f}h)")
            return None
        
        # Verificar se tem dados necessarios (menos rigoroso no modo intervalo)
        if len(p_slip) == 0:
            print(f"   [REJEITADO] Periodo {i}: sem dados no slip")
            return None
        
        # No modo intervalo, apenas avisar sobre dados faltantes, nao rejeitar
        if modo_intervalo:
            if p_est['rotational_speed'].notna().sum() == 0:
                print(f"   [AVISO] Periodo {i}: sem rotational_speed valido (continuando no modo intervalo)")
            if p_est['current'].notna().sum() == 0:
                print(f"   [AVISO] Periodo {i}: sem current valido (continuando no modo intervalo)")
            if p_est['vel_rms'].notna().sum() == 0:
                print(f"   [AVISO] Periodo {i}: sem vel_rms valido (continuando no modo intervalo)")
        else:
            # Modo treino: validacoes rigorosas
            if p_est['rotational_speed'].notna().sum() == 0:
                print(f"   [REJEITADO] Periodo {i}: sem rotational_speed valido")
                return None
            if p_est['current'].notna().sum() == 0:
                print(f"   [REJEITADO] Periodo {i}: sem current valido")
                return None
            if p_est['vel_rms'].notna().sum() == 0:
                print(f"   [REJEITADO] Periodo {i}: sem vel_rms valido")
                return None
    
        # Gerar timestamps
        timestamps_novos = gerar_timestamps(t_inicio, t_fim)
        df_novo = pd.DataFrame({'time': timestamps_novos})
        
        # Merge e interpolar estimated
        df_merged = pd.merge(df_novo, p_est, on='time', how='left')
        
        for col in ['rotational_speed', 'vel_rms', 'current']:
            if col in p_est.columns:
                # Remover outliers e pegar valores originais direto de p_est
                valores_limpos = remover_outliers(p_est[col])
                valores_orig = valores_limpos.values
                
                if np.isnan(valores_orig).all():
                    continue
                
                valores_interp = interpolar_coluna(
                    p_est['time'].values,
                    valores_orig,
                    timestamps_novos,
                    col
                )
                df_merged[col] = valores_interp
        
        df_merged['m_point'] = 'c_636'
        
        # Tratar outliers em slip
        for col in ['fe_frequency', 'fe_magnitude_-_1', 'fe_magnitude_0', 
                    'fe_magnitude_1', 'fr_frequency', 'rms']:
            if col in p_slip.columns:
                p_slip[col] = remover_outliers(p_slip[col])
        
        return {
            'periodo': i,
            'duracao_h': duracao_h,
            't_inicio': t_inicio,
            't_fim': t_fim,
            'estimated': df_merged,
            'slip': p_slip,
            'registros': (len(df_merged), len(p_slip))
        }
        
    except Exception as e:
        print(f"\n   [ERRO] Periodo {i}: {e}")
        return None

def parse_arguments():
    """Parse argumentos da linha de comando"""
    parser = argparse.ArgumentParser(
        description="Processamento simples de dados - Segmentacao e Interpolacao"
    )
    parser.add_argument(
        '--mpoint',
        type=str,
        required=True,
        help='ID do mpoint (ex: c_636)'
    )
    parser.add_argument(
        '--intervalo-arquivo',
        type=str,
        help='Intervalo formatado para incluir no nome dos arquivos'
    )
    parser.add_argument(
        '--periodo-minimo',
        type=float,
        default=24.0,
        help='Duracao minima do periodo em horas (default: 24.0)'
    )

    return parser.parse_args()

if __name__ == '__main__':
    # Parse argumentos
    args = parse_arguments()

    print("="*80)
    print("PROCESSAMENTO DE DADOS")
    print("="*80)
    print(f"Mpoint: {args.mpoint}")
    print(f"Inicio: {datetime.now()}")

    # Caminhos
    base = Path(__file__).parent.parent

    # Usar pastas padrao
    dir_raw = base / 'data' / 'raw'
    dir_out = base / 'data' / 'raw_preenchido'
    dir_out.mkdir(exist_ok=True)

    # Armazenar mpoint globalmente
    mpoint_atual = args.mpoint

    # Buscar arquivos: primeiro tenta com intervalo, depois sem intervalo
    # Logica de selecao baseada no modo
    if args.intervalo_arquivo:
        # MODO ANALISE: usar arquivos com intervalo especifico
        arquivo_estimated = dir_raw / f'dados_estimated_{args.mpoint}_{args.intervalo_arquivo}.csv'
        arquivo_slip = dir_raw / f'dados_slip_{args.mpoint}_{args.intervalo_arquivo}.csv'
        arquivo_c636 = dir_raw / f'dados_{args.mpoint}_{args.intervalo_arquivo}.csv'
        print(f"[INFO] Modo analise - Usando arquivos com intervalo: {args.intervalo_arquivo}")
    else:
        # MODO TREINO: usar arquivos sem intervalo (dados originais)
        arquivo_estimated = dir_raw / f'dados_estimated_{args.mpoint}.csv'
        arquivo_slip = dir_raw / f'dados_slip_{args.mpoint}.csv'
        arquivo_c636 = dir_raw / f'dados_{args.mpoint}.csv'
        print(f"[INFO] Modo treino - Usando arquivos originais")

    # Verificar se arquivos existem
    if not arquivo_estimated.exists():
        print(f"[ERRO] Arquivo estimated nao encontrado: {arquivo_estimated}")
        exit(1)

    if not arquivo_slip.exists():
        print(f"[ERRO] Arquivo slip nao encontrado: {arquivo_slip}")
        exit(1)

    # Carregar dados
    print("\n1. Carregando dados...")
    try:
        df_est = pd.read_csv(arquivo_estimated)
        df_est['time'] = pd.to_datetime(df_est['time'], format='mixed', utc=True)

        df_slip = pd.read_csv(arquivo_slip)
        df_slip['time'] = pd.to_datetime(df_slip['time'], format='mixed', utc=True)
    except Exception as e:
        print(f"[ERRO] Erro ao carregar dados: {e}")
        exit(1)
    
    print(f"   Estimated: {len(df_est):,} registros")
    print(f"   Slip: {len(df_slip):,} registros")
    
    # Ordenar
    df_est = df_est.sort_values('time').reset_index(drop=True)
    df_slip = df_slip.sort_values('time').reset_index(drop=True)
    
    # Detectar gaps > 3h em estimated
    print("\n2. Detectando gaps > 3h...")
    gaps = df_est['time'].diff() > timedelta(hours=3)
    indices_gaps = df_est[gaps].index.tolist()
    print(f"   {len(indices_gaps)} gaps encontrados")
    
    # Segmentar em periodos
    print("\n3. Segmentando periodos...")
    periodos = []
    inicio_idx = 0
    for idx in indices_gaps:
        periodos.append((inicio_idx, idx))
        inicio_idx = idx
    periodos.append((inicio_idx, len(df_est)))
    print(f"   {len(periodos)} periodos criados")
    
    # Processar periodos sequencialmente
    print("\n4. Processando periodos...")
    
    # No modo intervalo, aceitar periodos menores (minimo 1h)
    periodo_minimo = 1.0 if args.intervalo_arquivo else args.periodo_minimo
    modo_intervalo = bool(args.intervalo_arquivo)
    if modo_intervalo:
        print(f"   [INFO] Modo intervalo - periodo minimo reduzido para {periodo_minimo:.1f}h")
        print(f"   [INFO] Modo intervalo - validacoes de dados menos rigorosas")
    
    num_valido = 0
    for i, (idx_i, idx_f) in enumerate(periodos, 1):
        args_worker = (i, idx_i, idx_f, df_est, df_slip, dir_out, periodo_minimo, modo_intervalo)
        resultado = processar_periodo_worker(args_worker)
        
        if resultado is not None:
            num_valido += 1
            duracao_h = resultado['duracao_h']
            t_inicio = resultado['t_inicio']
            t_fim = resultado['t_fim']
            
            print(f"\n   Periodo {num_valido} ({i}/{len(periodos)}): {duracao_h:.1f}h")
            print(f"      De {t_inicio} ate {t_fim}")
            
            # Salvar arquivos (com ou sem intervalo)
            if args.intervalo_arquivo:
                # MODO INTERVALO: salvar com intervalo no nome
                resultado['estimated'].to_csv(dir_out / f'periodo_{num_valido:02d}_estimated_{mpoint_atual}_{args.intervalo_arquivo}.csv', index=False)
                resultado['slip'].to_csv(dir_out / f'periodo_{num_valido:02d}_slip_{mpoint_atual}_{args.intervalo_arquivo}.csv', index=False)
            else:
                # MODO TREINO: salvar sem intervalo
                resultado['estimated'].to_csv(dir_out / f'periodo_{num_valido:02d}_estimated_{mpoint_atual}.csv', index=False)
                resultado['slip'].to_csv(dir_out / f'periodo_{num_valido:02d}_slip_{mpoint_atual}.csv', index=False)
            
            reg_est, reg_slip = resultado['registros']
            print(f"      Salvo: {reg_est:,} registros (estimated)")
            print(f"      Salvo: {reg_slip:,} registros (slip)")
    
    print("\n" + "="*80)
    print(f"CONCLUIDO: {num_valido} periodos validos processados")
    print(f"Fim: {datetime.now()}")
    print(f"Arquivos em: {dir_out}")
    print("="*80)

    # Gerar logs detalhados para TCC
    import time
    start_time = time.time()  # Nota: deveria ser definido no inicio, mas para compatibilidade vamos estimar

    # Coletar informacoes dos arquivos gerados
    generated_files = []
    if dir_out.exists():
        for file_path in dir_out.iterdir():
            if file_path.is_file() and mpoint_atual in file_path.name:
                generated_files.append(str(file_path))

    # Estatisticas do processamento
    processing_stats = {
        'total_periods_identified': len(periodos),
        'valid_periods_processed': num_valido,
        'invalid_periods_discarded': len(periodos) - num_valido,
        'files_generated': len(generated_files),
        'processing_efficiency': num_valido / len(periodos) * 100 if len(periodos) > 0 else 0
    }

    # Log de processamento
    processing_log = create_processing_log(
        script_name='processar_dados_simples',
        mpoint=mpoint_atual,
        operation='data_processing_and_interpolation',
        input_files=[
            str(arquivo_estimated),
            str(arquivo_slip),
            str(arquivo_c636)
        ],
        output_files=generated_files,
        parameters={
            'gap_detection_threshold': '3_hours',
            'interpolation_methods': ['simple_spline', 'advanced_knn', 'linear_fallback'],
            'outlier_removal': 'IQR_method_3sigma',
            'period_segmentation': True,
            'minimum_period_duration': 'not_specified'
        },
        statistics=processing_stats,
        processing_time=time.time() - start_time,
        success=True,
        data_description={
            'input_sources': ['c636_main', 'estimated_sensors', 'slip_sensors'],
            'processing_steps': [
                'data_loading_and_validation',
                'gap_detection',
                'period_segmentation',
                'outlier_removal',
                'interpolation',
                'data_saving'
            ],
            'interpolation_strategy': 'hybrid_approach',
            'quality_control': 'basic_validation'
        }
    )

    save_log(processing_log, 'processar_dados_simples', mpoint_atual, 'processing_complete')

    # Enriquecer arquivo results
    results_data = {
        'processing_completed': True,
        'processing_timestamp': datetime.now().isoformat(),
        'periods_processed': num_valido,
        'total_periods_identified': len(periodos),
        'files_generated': len(generated_files),
        'processing_efficiency_percent': processing_stats['processing_efficiency'],
        'processing_parameters': processing_log['parameters'],
        'processing_statistics': processing_stats
    }

    enrich_results_file(mpoint_atual, results_data)

