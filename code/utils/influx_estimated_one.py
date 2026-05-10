from influxdb import InfluxDBClient
import pandas as pd
from datetime import datetime, timedelta, timezone
import time
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Carregar configurações do .env
# O .env está na pasta code/
env_path = Path(__file__).parent.parent / 'config.env'
if env_path.exists():
    load_dotenv(env_path)
else:
    print(f"[ERROR] Arquivo {env_path} não encontrado!")
    sys.exit(1)

# -----------------------------
# Configurações do InfluxDB
# -----------------------------
host = os.getenv('INFLUXDB_IP')
port = int(os.getenv('INFLUXDB_PORT', 8086))
database = "aihub"
measurement = "estimated"
retention_policy = "autogen"

if not host:
    print("[ERROR] INFLUXDB_IP não configurado no arquivo config.env!")
    sys.exit(1)

# ARGUMENTOS: mpoint [start_date] [end_date]
# Ex: c_637 2024-09-02T00:00:00Z 2025-08-27T23:59:59Z
mpoint = sys.argv[1] if len(sys.argv) > 1 else "c_636"

if len(sys.argv) > 3:
    start_time_str = sys.argv[2]
    end_time_str = sys.argv[3]
    print(f"[ARG] Usando datas fornecidas: {start_time_str} ate {end_time_str}")
else:
    # Default: last 365 days
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=365)
    start_time_str = start_time.isoformat()
    end_time_str = end_time.isoformat()

print(f"INICIANDO exportacao de dados do InfluxDB (estimated) - MPOINT: {mpoint}")
print(f"[CONFIG] InfluxDB: {host}:{port}")

# Cria cliente
client = InfluxDBClient(host=host, port=port, database=database)

# -----------------------------
# Coleta dados do mpoint
# -----------------------------
# Salvar em code/data/raw para consistência
output_dir = Path(__file__).parent.parent / "data" / "raw"
filename = output_dir / f"dados_estimated_{mpoint}.csv"
os.makedirs(output_dir, exist_ok=True)

query = f"""
SELECT "rotational_speed",
       "vel_rms",
       "current"
FROM "{database}"."{retention_policy}"."{measurement}"
WHERE time >= '{start_time_str}' AND time <= '{end_time_str}'
  AND "m_point" = '{mpoint}'
"""

print(f"\n[PROCESSANDO] Coletando dados para m_point: {mpoint}...")
t1 = time.time()

try:
    result = client.query(query)
    df = pd.DataFrame(result.get_points())

    if df.empty:
        print(f"[WARNING] Sem dados para m_point: {mpoint}")
    else:
        df["m_point"] = mpoint
        df.to_csv(filename, index=False)
        print(f"[SUCCESS] {len(df)} registros salvos em {filename} (tempo: {time.time() - t1:.2f}s)")

except Exception as e:
    print(f"[ERROR] Erro ao coletar m_point {mpoint}: {e}")

print("\n[OK] FINALIZADO!")
