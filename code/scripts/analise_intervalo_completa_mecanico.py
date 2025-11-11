"""
Análise de intervalo para equipamentos MECÂNICOS.
Baixa dados do InfluxDB, processa e classifica usando modelo treinado.
Foco: Temperatura + Vibração (sem current, sem RPM).
"""
import sys
import subprocess
from pathlib import Path
from datetime import datetime
import argparse
import pandas as pd
import pickle
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from utils.artifact_paths import (
    kmeans_model_path,
    scaler_model_path,
    info_kmeans_path,
    results_dir,
)

def baixar_dados_influx_mecanico(mpoint, influx_ip, inicio, fim, intervalo_arquivo):
    """Baixa dados do InfluxDB para equipamento MECÂNICO"""
    print("\n" + "="*80)
    print("BAIXANDO DADOS DO INFLUXDB - EQUIPAMENTO MECÂNICO")
    print("="*80)
    
    # Baixar dados validated_default (temperatura + vibração)
    script_validated = BASE_DIR / 'scripts' / 'baixar_validated_default_intervalo.py'
    
    if not script_validated.exists():
        print(f"[ERRO] Script não encontrado: {script_validated}")
        return False
    
    cmd_validated = [
        sys.executable, str(script_validated),
        '--mpoint', mpoint,
        '--ip', influx_ip,
        '--inicio', inicio,
        '--fim', fim,
        '--intervalo-arquivo', intervalo_arquivo
    ]
    
    print(f"\n[1/2] Baixando dados validated_default (temperatura + vibração)...")
    resultado = subprocess.run(cmd_validated, cwd=str(BASE_DIR), text=True)
    
    if resultado.returncode != 0:
        print("[ERRO] Falha ao baixar dados validated_default")
        return False
    
    # Baixar dados slip
    script_slip = BASE_DIR / 'scripts' / 'baixar_validated_slip_intervalo.py'
    
    if not script_slip.exists():
        print(f"[ERRO] Script não encontrado: {script_slip}")
        return False
    
    cmd_slip = [
        sys.executable, str(script_slip),
        '--mpoint', mpoint,
        '--ip', influx_ip,
        '--inicio', inicio,
        '--fim', fim,
        '--intervalo-arquivo', intervalo_arquivo
    ]
    
    print(f"\n[2/2] Baixando dados slip...")
    resultado = subprocess.run(cmd_slip, cwd=str(BASE_DIR), text=True)
    
    if resultado.returncode != 0:
        print("[ERRO] Falha ao baixar dados slip")
        return False
    
    print("\n[OK] Dados baixados com sucesso!")
    return True

def processar_dados_mecanico(mpoint, intervalo_arquivo):
    """Processa dados para equipamento MECÂNICO"""
    print("\n" + "="*80)
    print("PROCESSANDO DADOS - EQUIPAMENTO MECÂNICO")
    print("="*80)
    
    scripts = [
        ('processar_dados_simples_mecanico.py', 'Processamento e interpolação'),
        ('unir_sincronizar_periodos_mecanico.py', 'União e sincronização'),
        ('normalizar_dados_kmeans_mecanico.py', 'Normalização')
    ]
    
    for script_name, descricao in scripts:
        print(f"\n[INFO] {descricao}: {script_name}")
        script_path = BASE_DIR / 'scripts' / script_name
        
        if not script_path.exists():
            print(f"[ERRO] Script não encontrado: {script_path}")
            return False
        
        cmd = [
            sys.executable, str(script_path),
            '--mpoint', mpoint,
            '--intervalo-arquivo', intervalo_arquivo
        ]

        resultado = subprocess.run(cmd, cwd=str(BASE_DIR), text=True)
        
        if resultado.returncode != 0:
            print(f"[ERRO] Falha em {script_name}")
            return False
    
    print("\n[OK] Processamento concluído!")
    return True

def classificar_estados_mecanico(mpoint, intervalo_arquivo):
    """Classifica estados usando modelo treinado - MECÂNICO"""
    print("\n" + "="*80)
    print("CLASSIFICANDO ESTADOS - EQUIPAMENTO MECÂNICO")
    print("="*80)
    
    # Carregar modelo treinado
    modelo_path = kmeans_model_path(mpoint)
    scaler_path = scaler_model_path(mpoint)
    info_path = info_kmeans_path(mpoint)
    
    if not all([modelo_path.exists(), scaler_path.exists(), info_path.exists()]):
        print("[ERRO] Modelo não encontrado. Execute o treino primeiro.")
        return False
    
    print("[INFO] Carregando modelo treinado...")
    
    with open(modelo_path, 'rb') as f:
        kmeans = pickle.load(f)
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    import json
    with open(info_path, 'r') as f:
        info_modelo = json.load(f)
    
    # Carregar dados normalizados
    from utils.artifact_paths import normalized_csv_path
    dados_path = normalized_csv_path(mpoint, intervalo_arquivo)
    
    if not dados_path.exists():
        print(f"[ERRO] Dados normalizados não encontrados: {dados_path}")
        return False
    
    print("[INFO] Carregando dados normalizados...")
    df = pd.read_csv(dados_path)
    
    # Separar timestamp
    timestamp = df['time']
    features = df.drop('time', axis=1)
    
    # Predizer clusters
    print("[INFO] Classificando dados...")
    clusters = kmeans.predict(features.values)
    
    # Aplicar classificação de estados (DESLIGADO vs LIGADO)
    thresholds = info_modelo.get('thresholds_desligado', {})
    clusters_desligado = thresholds.get('clusters_desligado', [0])
    
    df['cluster'] = clusters
    df['equipamento_status'] = 'LIGADO'
    
    for cluster_id in clusters_desligado:
        df.loc[df['cluster'] == cluster_id, 'equipamento_status'] = 'DESLIGADO'
    
    # Salvar resultados
    results_path = results_dir(mpoint, create=True)
    
    # Formatar intervalo para nome do arquivo
    inicio_fmt = intervalo_arquivo.split('_')[0] if '_' in intervalo_arquivo else 'unknown'
    fim_fmt = intervalo_arquivo.split('_')[-1] if '_' in intervalo_arquivo else 'unknown'
    
    arquivo_resultados = results_path / f'estados_classificados_{mpoint}_{inicio_fmt}_{fim_fmt}.csv'
    df.to_csv(arquivo_resultados, index=False)
    
    print(f"[OK] Resultados salvos: {arquivo_resultados}")
    
    # Estatísticas
    total = len(df)
    ligado = (df['equipamento_status'] == 'LIGADO').sum()
    desligado = (df['equipamento_status'] == 'DESLIGADO').sum()
    
    print("\n" + "="*80)
    print("ESTATÍSTICAS - EQUIPAMENTO MECÂNICO")
    print("="*80)
    print(f"Total de amostras: {total:,}")
    print(f"LIGADO: {ligado:,} ({ligado/total*100:.1f}%)")
    print(f"DESLIGADO: {desligado:,} ({desligado/total*100:.1f}%)")
    print("="*80)
    
    return True

def main():
    """Função principal"""
    parser = argparse.ArgumentParser(
        description="Análise de intervalo para equipamento MECÂNICO"
    )
    parser.add_argument('--mpoint', type=str, required=True, help='ID do mpoint')
    parser.add_argument('--influx-ip', type=str, required=True, help='IP do InfluxDB')
    parser.add_argument('--inicio', type=str, required=True, help='Data/hora início (YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--fim', type=str, required=True, help='Data/hora fim (YYYY-MM-DD HH:MM:SS)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("ANÁLISE DE INTERVALO - EQUIPAMENTO MECÂNICO")
    print("="*80)
    print(f"Mpoint: {args.mpoint}")
    print(f"Tipo: MECÂNICO (temperatura + vibração)")
    print(f"Período: {args.inicio} até {args.fim}")
    print(f"InfluxDB: {args.influx_ip}")
    print("="*80)
    
    # Formatar intervalo para nome de arquivo
    inicio_fmt = args.inicio.replace('-', '').replace(' ', '_').replace(':', ';')
    fim_fmt = args.fim.replace('-', '').replace(' ', '_').replace(':', ';')
    intervalo_arquivo = f"{inicio_fmt}_{fim_fmt}"
    
    # 1. Baixar dados do InfluxDB
    if not baixar_dados_influx_mecanico(args.mpoint, args.influx_ip, args.inicio, args.fim, intervalo_arquivo):
        print("\n[ERRO] Falha ao baixar dados")
        sys.exit(1)
    
    # 2. Processar dados
    if not processar_dados_mecanico(args.mpoint, intervalo_arquivo):
        print("\n[ERRO] Falha ao processar dados")
        sys.exit(1)
    
    # 3. Classificar estados
    if not classificar_estados_mecanico(args.mpoint, intervalo_arquivo):
        print("\n[ERRO] Falha ao classificar estados")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("ANÁLISE CONCLUÍDA COM SUCESSO!")
    print("="*80)
    print(f"Resultados em: results/{args.mpoint}/")

if __name__ == '__main__':
    main()

