"""
Script para processar dados de equipamentos MECÂNICOS (sem estimated).
Equipamentos mecânicos só têm:
  - dados_c_XXX.csv (temperatura e vibração)
  - dados_slip_c_XXX.csv (análise de frequência)
  
NÃO TÊM:
  - dados_estimated_c_XXX.csv (sem campo magnético = sem RPM)
  - current (corrente)
  
Foco da análise:
  - Temperatura (object_temp): mudanças indicam operação
  - Vibração (vel_rms, mag_x/y/z): vibrações próximas de zero = desligado
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

# Importações para logging estruturado
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
    """Interpolação simples para gaps < 1h"""
    try:
        if len(t) >= 4:
            spline = UnivariateSpline(t, v, k=3, s=0)
            return spline(t_novo)
    except:
        pass
    f = interp1d(t, v, kind='linear', bounds_error=False, fill_value=(v[0], v[-1]))
    return f(t_novo)

def interpolar_avancada(t, v, t_novo):
    """Interpolação avançada para gaps 1-3h"""
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
    nao ruido pontual. Respeita a lei da inercia."""
    mask = mask_outlier.copy()
    flagged = mask[mask].index.tolist()
    if len(flagged) == 0:
        return mask
    valid_indices = series_original.dropna().index.tolist()
    if len(valid_indices) == 0:
        return mask
    pos_map = {idx: pos for pos, idx in enumerate(valid_indices)}
    flagged_valid = [(idx, pos_map[idx]) for idx in flagged if idx in pos_map]
    if len(flagged_valid) == 0:
        return mask
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
    for grupo in grupos_idx:
        if len(grupo) >= min_consecutivas:
            mask.loc[grupo] = False
    return mask

def remover_outliers(series):
    """Remove outliers usando IQR, preservando sequencias consecutivas (>=10 amostras)
    que indicam estados operacionais reais (ex: desligamento)"""
    valores = series.dropna()
    if len(valores) < 4:
        return series
    Q1, Q3 = valores.quantile(0.25), valores.quantile(0.75)
    IQR = Q3 - Q1
    limites = (Q1 - 3*IQR, Q3 + 3*IQR)
    s = series.copy()
    mask_outlier = (s < limites[0]) | (s > limites[1])
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
    """Interpola uma coluna com estratégia adaptativa (ultra-otimizada)"""
    mask = ~np.isnan(valores)
    if mask.sum() < 2:
        return np.full(len(timestamps_novos), np.nan)
    
    # Converter timestamps para segundos desde época
    t0 = pd.Timestamp(timestamps_orig[0]).value
    t_orig = (pd.Series(timestamps_orig).astype('int64').values - t0) / 1e9
    t_novo = (pd.Series(timestamps_novos).astype('int64').values - t0) / 1e9
    t_validos = t_orig[mask]
    v_validos = valores[mask]
    
    # Calcular gaps
    gaps = np.diff(t_validos)
    
    # Se não há gaps grandes (> 1h), usar interpolação linear direta
    if gaps.max() < 3600:
        f = interp1d(t_validos, v_validos, kind='linear', bounds_error=False, 
                     fill_value=(v_validos[0], v_validos[-1]))
        return f(t_novo)
    
    # Se há gaps grandes, processar apenas esses gaps
    indices_gaps = np.where(gaps > 3600)[0]
    
    # Começar com interpolação linear de tudo
    f_base = interp1d(t_validos, v_validos, kind='linear', bounds_error=False,
                      fill_value=(v_validos[0], v_validos[-1]))
    resultado = f_base(t_novo)
    
    # Refinar apenas os gaps grandes com interpolação avançada
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
                        # Se falhar, manter interpolação linear
                        pass
    
    return resultado

def processar_periodo_worker(args):
    """Worker para processar um período em paralelo (VERSÃO MECÂNICA)"""
    i, idx_i, idx_f, df_c, df_slip, dir_out, periodo_minimo, modo_intervalo = args
    
    try:
        p_c = df_c.iloc[idx_i:idx_f].copy()
        
        t_inicio = p_c['time'].min()
        t_fim = p_c['time'].max()
        duracao_h = (t_fim - t_inicio).total_seconds() / 3600
        
        # Filtrar slip para período
        p_slip = df_slip[(df_slip['time'] >= t_inicio) & (df_slip['time'] <= t_fim)].copy()
        
        # Validar duração
        if duracao_h < periodo_minimo:
            print(f"   [REJEITADO] Período {i}: duração muito curta ({duracao_h:.1f}h < {periodo_minimo:.1f}h)")
            return None
        
        # Verificar se tem dados necessários (menos rigoroso no modo intervalo)
        if len(p_slip) == 0:
            print(f"   [REJEITADO] Período {i}: sem dados no slip")
            return None
        
        # EQUIPAMENTO MECÂNICO: Verificar temperatura e vibração
        if modo_intervalo:
            if p_c['object_temp'].notna().sum() == 0:
                print(f"   [AVISO] Período {i}: sem object_temp válido (continuando no modo intervalo)")
            colunas_vibracao = [col for col in p_c.columns if 'vel_' in col or 'mag_' in col]
            if all(p_c[col].notna().sum() == 0 for col in colunas_vibracao):
                print(f"   [AVISO] Período {i}: sem dados de vibração válidos (continuando no modo intervalo)")
        else:
            # Modo treino: validações rigorosas
            if p_c['object_temp'].notna().sum() == 0:
                print(f"   [REJEITADO] Período {i}: sem object_temp válido")
                return None
            colunas_vibracao = [col for col in p_c.columns if 'vel_' in col or 'mag_' in col]
            if all(p_c[col].notna().sum() == 0 for col in colunas_vibracao):
                print(f"   [REJEITADO] Período {i}: sem dados de vibração válidos")
                return None
    
        # Gerar timestamps
        timestamps_novos = gerar_timestamps(t_inicio, t_fim)
        df_novo = pd.DataFrame({'time': timestamps_novos})
        
        # Merge e interpolar dados_c (temperatura + vibração + magnetômetro)
        df_merged = pd.merge(df_novo, p_c, on='time', how='left')
        
        # Colunas importantes para equipamento mecânico
        colunas_importantes = ['object_temp', 'vel_max_x', 'vel_max_y', 'vel_max_z', 
                              'vel_rms_x', 'vel_rms_y', 'vel_rms_z', 
                              'mag_x', 'mag_y', 'mag_z']
        
        for col in colunas_importantes:
            if col in p_c.columns:
                # Remover outliers e pegar valores originais direto de p_c
                valores_limpos = remover_outliers(p_c[col])
                valores_orig = valores_limpos.values
                
                if np.isnan(valores_orig).all():
                    continue
                
                valores_interp = interpolar_coluna(
                    p_c['time'].values,
                    valores_orig,
                    timestamps_novos,
                    col
                )
                df_merged[col] = valores_interp
        
        df_merged['m_point'] = p_c['m_point'].iloc[0] if 'm_point' in p_c.columns else 'unknown'
        
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
            'dados_c': df_merged,
            'slip': p_slip,
            'registros': (len(df_merged), len(p_slip))
        }
        
    except Exception as e:
        print(f"\n   [ERRO] Período {i}: {e}")
        return None

def parse_arguments():
    """Parse argumentos da linha de comando"""
    parser = argparse.ArgumentParser(
        description="Processamento simples de dados MECÂNICOS - Temperatura e Vibração"
    )
    parser.add_argument(
        '--mpoint',
        type=str,
        required=True,
        help='ID do mpoint (ex: c_640)'
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
        help='Duração mínima do período em horas (default: 24.0)'
    )

    return parser.parse_args()

if __name__ == '__main__':
    # Parse argumentos
    args = parse_arguments()

    print("="*80)
    print("PROCESSAMENTO DE DADOS - EQUIPAMENTO MECÂNICO")
    print("="*80)
    print(f"Mpoint: {args.mpoint}")
    print(f"Tipo: MECÂNICO (sem estimated, sem RPM, sem current)")
    print(f"Análise: Temperatura + Vibração")
    print(f"Inicio: {datetime.now()}")

    # Caminhos
    base = Path(__file__).parent.parent

    # Usar pastas padrão
    dir_raw = base / 'data' / 'raw'
    dir_out = base / 'data' / 'raw_preenchido'
    dir_out.mkdir(exist_ok=True)

    # Armazenar mpoint globalmente
    mpoint_atual = args.mpoint

    # Buscar arquivos: primeiro tenta com intervalo, depois sem intervalo
    if args.intervalo_arquivo:
        # MODO ANÁLISE: usar arquivos com intervalo específico
        arquivo_c = dir_raw / f'dados_{args.mpoint}_{args.intervalo_arquivo}.csv'
        arquivo_slip = dir_raw / f'dados_slip_{args.mpoint}_{args.intervalo_arquivo}.csv'
        print(f"[INFO] Modo análise - Usando arquivos com intervalo: {args.intervalo_arquivo}")
    else:
        # MODO TREINO: usar arquivos sem intervalo (dados originais)
        arquivo_c = dir_raw / f'dados_{args.mpoint}.csv'
        arquivo_slip = dir_raw / f'dados_slip_{args.mpoint}.csv'
        print(f"[INFO] Modo treino - Usando arquivos originais")

    # Verificar se arquivos existem
    if not arquivo_c.exists():
        print(f"[ERRO] Arquivo dados_c não encontrado: {arquivo_c}")
        exit(1)

    if not arquivo_slip.exists():
        print(f"[ERRO] Arquivo slip não encontrado: {arquivo_slip}")
        exit(1)

    # Carregar dados
    print("\n1. Carregando dados...")
    try:
        df_c = pd.read_csv(arquivo_c)
        df_c['time'] = pd.to_datetime(df_c['time'], format='mixed', utc=True)

        df_slip = pd.read_csv(arquivo_slip)
        df_slip['time'] = pd.to_datetime(df_slip['time'], format='mixed', utc=True)
    except Exception as e:
        print(f"[ERRO] Erro ao carregar dados: {e}")
        exit(1)
    
    print(f"   Dados_c (temperatura + vibração): {len(df_c):,} registros")
    print(f"   Slip: {len(df_slip):,} registros")
    
    # Ordenar
    df_c = df_c.sort_values('time').reset_index(drop=True)
    df_slip = df_slip.sort_values('time').reset_index(drop=True)
    
    # Detectar gaps > 3h em dados_c
    print("\n2. Detectando gaps > 3h...")
    gaps = df_c['time'].diff() > timedelta(hours=3)
    indices_gaps = df_c[gaps].index.tolist()
    print(f"   {len(indices_gaps)} gaps encontrados")
    
    # Segmentar em períodos
    print("\n3. Segmentando períodos...")
    periodos = []
    inicio_idx = 0
    for idx in indices_gaps:
        periodos.append((inicio_idx, idx))
        inicio_idx = idx
    periodos.append((inicio_idx, len(df_c)))
    print(f"   {len(periodos)} períodos criados")
    
    # Processar períodos sequencialmente
    print("\n4. Processando períodos...")
    
    # No modo intervalo, aceitar períodos menores (mínimo 1h)
    periodo_minimo = 1.0 if args.intervalo_arquivo else args.periodo_minimo
    modo_intervalo = bool(args.intervalo_arquivo)
    if modo_intervalo:
        print(f"   [INFO] Modo intervalo - período mínimo reduzido para {periodo_minimo:.1f}h")
        print(f"   [INFO] Modo intervalo - validações de dados menos rigorosas")
    
    num_valido = 0
    for i, (idx_i, idx_f) in enumerate(periodos, 1):
        args_worker = (i, idx_i, idx_f, df_c, df_slip, dir_out, periodo_minimo, modo_intervalo)
        resultado = processar_periodo_worker(args_worker)
        
        if resultado is not None:
            num_valido += 1
            duracao_h = resultado['duracao_h']
            t_inicio = resultado['t_inicio']
            t_fim = resultado['t_fim']
            
            print(f"\n   Período {num_valido} ({i}/{len(periodos)}): {duracao_h:.1f}h")
            print(f"      De {t_inicio} até {t_fim}")
            
            # Salvar arquivos (com ou sem intervalo)
            if args.intervalo_arquivo:
                # MODO INTERVALO: salvar com intervalo no nome
                resultado['dados_c'].to_csv(dir_out / f'periodo_{num_valido:02d}_c_{mpoint_atual}_{args.intervalo_arquivo}.csv', index=False)
                resultado['slip'].to_csv(dir_out / f'periodo_{num_valido:02d}_slip_{mpoint_atual}_{args.intervalo_arquivo}.csv', index=False)
            else:
                # MODO TREINO: salvar sem intervalo
                resultado['dados_c'].to_csv(dir_out / f'periodo_{num_valido:02d}_c_{mpoint_atual}.csv', index=False)
                resultado['slip'].to_csv(dir_out / f'periodo_{num_valido:02d}_slip_{mpoint_atual}.csv', index=False)
            
            reg_c, reg_slip = resultado['registros']
            print(f"      Salvo: {reg_c:,} registros (dados_c)")
            print(f"      Salvo: {reg_slip:,} registros (slip)")
    
    print("\n" + "="*80)
    print(f"CONCLUÍDO: {num_valido} períodos válidos processados")
    print(f"Fim: {datetime.now()}")
    print(f"Arquivos em: {dir_out}")
    print("="*80)

    # Gerar logs detalhados para TCC
    import time
    start_time = time.time()

    # Coletar informações dos arquivos gerados
    generated_files = []
    if dir_out.exists():
        for file_path in dir_out.iterdir():
            if file_path.is_file() and mpoint_atual in file_path.name:
                generated_files.append(str(file_path))

    # Estatísticas do processamento
    processing_stats = {
        'total_periods_identified': len(periodos),
        'valid_periods_processed': num_valido,
        'invalid_periods_discarded': len(periodos) - num_valido,
        'files_generated': len(generated_files),
        'processing_efficiency': num_valido / len(periodos) * 100 if len(periodos) > 0 else 0,
        'equipment_type': 'MECHANICAL'
    }

    # Log de processamento
    processing_log = create_processing_log(
        script_name='processar_dados_simples_mecanico',
        mpoint=mpoint_atual,
        operation='mechanical_equipment_data_processing',
        input_files=[
            str(arquivo_c),
            str(arquivo_slip)
        ],
        output_files=generated_files,
        parameters={
            'gap_detection_threshold': '3_hours',
            'interpolation_methods': ['simple_spline', 'advanced_knn', 'linear_fallback'],
            'outlier_removal': 'IQR_method_3sigma',
            'period_segmentation': True,
            'minimum_period_duration': f'{periodo_minimo}_hours',
            'equipment_type': 'MECHANICAL',
            'data_sources': ['temperature', 'vibration', 'magnetometer', 'slip_analysis'],
            'no_estimated_data': True,
            'no_current_rpm': True
        },
        statistics=processing_stats,
        processing_time=time.time() - start_time,
        success=True,
        data_description={
            'input_sources': ['dados_c_mechanical', 'slip_sensors'],
            'processing_steps': [
                'data_loading_and_validation',
                'gap_detection',
                'period_segmentation',
                'outlier_removal',
                'interpolation',
                'data_saving'
            ],
            'interpolation_strategy': 'hybrid_approach',
            'quality_control': 'basic_validation',
            'analysis_focus': 'temperature_and_vibration'
        }
    )

    save_log(processing_log, 'processar_dados_simples_mecanico', mpoint_atual, 'processing_complete')

    # Enriquecer arquivo results
    results_data = {
        'processing_completed': True,
        'processing_timestamp': datetime.now().isoformat(),
        'equipment_type': 'MECHANICAL',
        'periods_processed': num_valido,
        'total_periods_identified': len(periodos),
        'files_generated': len(generated_files),
        'processing_efficiency_percent': processing_stats['processing_efficiency'],
        'processing_parameters': processing_log['parameters'],
        'processing_statistics': processing_stats,
        'data_sources': ['temperature', 'vibration_only']
    }

    enrich_results_file(mpoint_atual, results_data)

