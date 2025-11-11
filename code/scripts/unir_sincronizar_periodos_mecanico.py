"""
Script para unir e sincronizar períodos de EQUIPAMENTOS MECÂNICOS.
Une dados_c (temperatura + vibração) com slip sincronizando pelos timestamps.
NÃO tem dados estimated (sem RPM, sem current).
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
import sys
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from utils.logging_utils import (
    save_log,
    create_processing_log,
    enrich_results_file,
)

def parse_arguments():
    """Parse argumentos da linha de comando"""
    parser = argparse.ArgumentParser(
        description="União e sincronização de períodos - EQUIPAMENTO MECÂNICO"
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
    return parser.parse_args()

def unir_sincronizar_periodo(arquivo_c, arquivo_slip, dir_out, periodo_num, mpoint, intervalo_arquivo=None):
    """Une e sincroniza dados_c com slip para equipamento mecânico"""
    try:
        # Carregar dados
        df_c = pd.read_csv(arquivo_c)
        df_c['time'] = pd.to_datetime(df_c['time'], format='mixed', utc=True)
        
        df_slip = pd.read_csv(arquivo_slip)
        df_slip['time'] = pd.to_datetime(df_slip['time'], format='mixed', utc=True)
        
        # Ordenar por tempo
        df_c = df_c.sort_values('time').reset_index(drop=True)
        df_slip = df_slip.sort_values('time').reset_index(drop=True)
        
        # Usar merge_asof para sincronizar (slip tem menor frequência)
        df_unificado = pd.merge_asof(
            df_c,
            df_slip,
            on='time',
            direction='nearest',
            tolerance=pd.Timedelta('2min'),  # Tolerância de 2 minutos
            suffixes=('', '_slip')
        )
        
        # Remover duplicatas de coluna 'm_point' se houver
        if 'm_point_slip' in df_unificado.columns:
            df_unificado = df_unificado.drop(columns=['m_point_slip'])
        
        # Remover duplicatas de outras colunas slip se houver
        colunas_duplicadas = [col for col in df_unificado.columns if col.endswith('_slip')]
        if colunas_duplicadas:
            df_unificado = df_unificado.drop(columns=colunas_duplicadas)
        
        # Salvar arquivo final
        if intervalo_arquivo:
            arquivo_saida = dir_out / f'periodo_{periodo_num:02d}_final_{mpoint}_{intervalo_arquivo}.csv'
        else:
            arquivo_saida = dir_out / f'periodo_{periodo_num:02d}_final_{mpoint}.csv'
        
        df_unificado.to_csv(arquivo_saida, index=False)
        
        return {
            'arquivo': arquivo_saida.name,
            'registros': len(df_unificado),
            'colunas': len(df_unificado.columns),
            't_inicio': df_unificado['time'].min(),
            't_fim': df_unificado['time'].max(),
            'duracao_h': (df_unificado['time'].max() - df_unificado['time'].min()).total_seconds() / 3600
        }
    
    except Exception as e:
        print(f"   [ERRO] Período {periodo_num}: {e}")
        return None

if __name__ == '__main__':
    args = parse_arguments()
    
    print("="*80)
    print("UNIÃO E SINCRONIZAÇÃO DE PERÍODOS - EQUIPAMENTO MECÂNICO")
    print("="*80)
    print(f"Mpoint: {args.mpoint}")
    print(f"Tipo: MECÂNICO (temperatura + vibração)")
    print(f"Inicio: {datetime.now()}")
    
    base = Path(__file__).parent.parent
    dir_raw_preenchido = base / 'data' / 'raw_preenchido'
    
    # Buscar arquivos processados
    if args.intervalo_arquivo:
        arquivos_c = sorted(dir_raw_preenchido.glob(f'periodo_*_c_{args.mpoint}_{args.intervalo_arquivo}.csv'))
        arquivos_slip = sorted(dir_raw_preenchido.glob(f'periodo_*_slip_{args.mpoint}_{args.intervalo_arquivo}.csv'))
    else:
        arquivos_c = sorted(dir_raw_preenchido.glob(f'periodo_*_c_{args.mpoint}.csv'))
        arquivos_slip = sorted(dir_raw_preenchido.glob(f'periodo_*_slip_{args.mpoint}.csv'))
    
    if len(arquivos_c) == 0:
        print(f"[ERRO] Nenhum arquivo dados_c encontrado em {dir_raw_preenchido}")
        print("Execute primeiro: python scripts/processar_dados_simples_mecanico.py --mpoint <mpoint>")
        exit(1)
    
    if len(arquivos_slip) == 0:
        print(f"[ERRO] Nenhum arquivo slip encontrado em {dir_raw_preenchido}")
        exit(1)
    
    print(f"\n   Arquivos dados_c encontrados: {len(arquivos_c)}")
    print(f"   Arquivos slip encontrados: {len(arquivos_slip)}")
    
    # Processar cada período
    print("\nProcessando períodos...")
    resultados = []
    
    for i, (arq_c, arq_slip) in enumerate(zip(arquivos_c, arquivos_slip), 1):
        print(f"\n   Período {i}/{len(arquivos_c)}")
        print(f"      dados_c: {arq_c.name}")
        print(f"      slip: {arq_slip.name}")
        
        resultado = unir_sincronizar_periodo(
            arq_c, arq_slip, dir_raw_preenchido, i, args.mpoint, args.intervalo_arquivo
        )
        
        if resultado:
            resultados.append(resultado)
            print(f"      -> {resultado['arquivo']}: {resultado['registros']:,} registros")
            print(f"         Colunas: {resultado['colunas']}")
            print(f"         Duracao: {resultado['duracao_h']:.1f}h")
    
    print("\n" + "="*80)
    print(f"CONCLUÍDO: {len(resultados)} períodos unificados")
    print(f"Total de registros: {sum(r['registros'] for r in resultados):,}")
    print(f"Arquivos em: {dir_raw_preenchido}")
    print("="*80)
    
    # Gerar logs
    import time
    start_time = time.time()
    
    generated_files = [str(dir_raw_preenchido / r['arquivo']) for r in resultados]
    
    processing_log = create_processing_log(
        script_name='unir_sincronizar_periodos_mecanico',
        mpoint=args.mpoint,
        operation='period_unification_and_synchronization_mechanical',
        input_files=[str(f) for f in arquivos_c + arquivos_slip],
        output_files=generated_files,
        parameters={
            'merge_method': 'merge_asof',
            'merge_tolerance': '2_minutes',
            'data_sources': ['dados_c_mechanical', 'slip_sensors'],
            'synchronization_key': 'timestamp',
            'equipment_type': 'MECHANICAL'
        },
        statistics={
            'total_periods_unified': len(resultados),
            'total_records': sum(r['registros'] for r in resultados),
            'average_columns': sum(r['colunas'] for r in resultados) / len(resultados) if resultados else 0,
            'total_duration_hours': sum(r['duracao_h'] for r in resultados)
        },
        processing_time=time.time() - start_time,
        success=True,
        data_description={
            'unified_sources': ['temperature', 'vibration', 'magnetometer', 'slip_analysis'],
            'synchronization_strategy': 'nearest_timestamp_with_2min_tolerance',
            'output_format': 'unified_csv_per_period',
            'equipment_type': 'MECHANICAL'
        }
    )
    
    save_log(processing_log, 'unir_sincronizar_periodos_mecanico', args.mpoint, 'unification_complete')
    
    results_data = {
        'unification_completed': True,
        'unification_timestamp': datetime.now().isoformat(),
        'equipment_type': 'MECHANICAL',
        'periods_unified': len(resultados),
        'total_records': sum(r['registros'] for r in resultados),
        'unification_parameters': processing_log['parameters']
    }
    
    enrich_results_file(args.mpoint, results_data)

