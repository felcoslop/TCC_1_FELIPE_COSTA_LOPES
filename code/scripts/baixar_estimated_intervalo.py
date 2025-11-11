"""
Script pra baixar dados estimados do banco InfluxDB.
Primeiro baixa tudo de um periodo, depois filtra o que interessa.
"""

from influxdb import InfluxDBClient
import pandas as pd
from datetime import datetime, timezone, timedelta
import argparse
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Baixar dados estimated do InfluxDB")
    parser.add_argument('--mpoint', type=str, required=True, help='Mpoint especfico')
    parser.add_argument('--ip', type=str, required=True, help='IP do InfluxDB')
    parser.add_argument('--inicio', type=str, required=True, help='Data/hora inicial (UTC)')
    parser.add_argument('--fim', type=str, required=True, help='Data/hora final (UTC)')
    parser.add_argument('--intervalo-arquivo', type=str, help='Intervalo formatado para nome do arquivo')
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    dir_raw = base_dir / 'data' / 'raw'
    dir_raw.mkdir(exist_ok=True)

    mpoint = args.mpoint
    
    print(f" Processando dados estimated para mpoint {mpoint}...")
    
    # MODO INTERVALO: Sempre baixar do InfluxDB diretamente
    print(f" Baixando dados do InfluxDB...")
    print(f" InfluxDB: {args.ip}:8086")
    print(f" Período: {args.inicio} até {args.fim}")
    
    host = args.ip
    port = 8086
    database = "aihub"
    measurement = "estimated"
    retention_policy = "autogen"
    
    client = InfluxDBClient(host=host, port=port, database=database)
    
    # Query COM filtro de intervalo
    query = f"""
    SELECT "rotational_speed", "vel_rms", "current"
    FROM "{database}"."{retention_policy}"."{measurement}"
    WHERE time >= '{args.inicio}' AND time <= '{args.fim}'
      AND "m_point" = '{mpoint}'
    """
    
    print(" Executando query...")
    result = client.query(query)
    df = pd.DataFrame(result.get_points())
    
    if df.empty:
        print(" Nenhum dado encontrado no intervalo")
        return
    
    df['time'] = pd.to_datetime(df['time'], format='mixed')
    print(f" {len(df)} registros baixados do InfluxDB")
    
    # SALVAR IMEDIATAMENTE
    if args.intervalo_arquivo:
        # Usar formato personalizado com intervalo
        filename = dir_raw / f"dados_estimated_{mpoint}_{args.intervalo_arquivo}.csv"
    else:
        # Formato padrão
        inicio_dt = pd.to_datetime(args.inicio)
        fim_dt = pd.to_datetime(args.fim)
        inicio_str = inicio_dt.strftime('%d-%m-%Y')
        fim_str = fim_dt.strftime('%d-%m-%Y')
        filename = dir_raw / f"dados_estimated_{mpoint}_{inicio_str}_{fim_str}.csv"
    
    df.to_csv(filename, index=False)
    print(f" Dados salvos: {filename.name}")

if __name__ == "__main__":
    main()
