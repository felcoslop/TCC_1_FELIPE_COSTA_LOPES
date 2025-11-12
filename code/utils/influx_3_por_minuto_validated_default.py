from influxdb import InfluxDBClient
import pandas as pd
from datetime import datetime, timedelta, timezone
import time
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Carregar configurações do .env
env_path = Path(__file__).parent.parent / 'config.env'
if env_path.exists():
    load_dotenv(env_path)

# -----------------------------
# Configurações do InfluxDB
# -----------------------------
host = os.getenv('INFLUXDB_IP')
port = int(os.getenv('INFLUXDB_PORT', 8086))
database = "aihub"
measurement = "validated_default"
retention_policy = "autogen"

if not host:
    print("[ERROR] INFLUXDB_IP não configurado no arquivo config.env!")
    sys.exit(1)

print("INICIANDO exportacao de dados do InfluxDB...")
print(f"[CONFIG] InfluxDB: {host}:{port}")

# Cria cliente
client = InfluxDBClient(host=host, port=port, database=database)

# -----------------------------
# Carrega m_points - verifica se foi passado parâmetro específico
# -----------------------------
if len(sys.argv) > 1:
    mpoint_especifico = sys.argv[1]
    mpoints = [mpoint_especifico]
    print(f"[INFO] Modo ESPECIFICO: Processando apenas mpoint {mpoint_especifico}")
else:
    with open("lista_mpoints.txt", "r") as f:
        mpoints = [line.strip() for line in f if line.strip()]
    print(f"[INFO] Modo TODOS: Total de m_points carregados do txt: {len(mpoints)}")

# -----------------------------
# Define range de 1 ano (timezone-aware, UTC)
# -----------------------------
end_time = datetime.now(timezone.utc)
start_time = end_time - timedelta(days=365)
print(f"⏳ Intervalo de dados: {start_time.isoformat()} até {end_time.isoformat()}")

# -----------------------------
# Loop por cada m_point
# -----------------------------
total_com_dados = 0
total_sem_dados = 0
t0 = time.time()

for i, mp in enumerate(mpoints, start=1):
    filename = os.path.join("code", "data", "raw", f"dados_{mp}.csv")

    # Se já existir CSV desse m_point, pula
    if os.path.exists(filename):
        print(f"⏭️ {mp} já exportado, pulando...")
        continue

    print(f"\n➡️ [{i}/{len(mpoints)}] Coletando dados para m_point: {mp}")
    try:
        t1 = time.time()
        query = f"""
        SELECT MEAN("mag_x") AS "mag_x",
               MEAN("mag_y") AS "mag_y",
               MEAN("mag_z") AS "mag_z",
               MEAN("object_temp") AS "object_temp",
               MEAN("vel_max_x") AS "vel_max_x",
               MEAN("vel_max_y") AS "vel_max_y",
               MEAN("vel_rms_x") AS "vel_rms_x",
               MEAN("vel_max_z") AS "vel_max_z",
               MEAN("vel_rms_y") AS "vel_rms_y",
               MEAN("vel_rms_z") AS "vel_rms_z"
        FROM "{database}"."{retention_policy}"."{measurement}"
        WHERE time >= '{start_time.isoformat()}' AND time <= '{end_time.isoformat()}'
          AND "m_point" = '{mp}'
        GROUP BY time(20s) fill(none)
        """

        result = client.query(query)
        df = pd.DataFrame(result.get_points())

        if df.empty:
            total_sem_dados += 1
            print(f"⚠️ Sem dados para m_point: {mp}")
            continue

        df["m_point"] = mp

        # Salva CSV individual
        df.to_csv(filename, index=False)
        print(f"✅ {len(df)} registros salvos em {filename} (tempo: {time.time() - t1:.2f}s)")

        total_com_dados += 1

    except Exception as e:
        print(f"❌ Erro no m_point {mp}: {e}")
        continue

# -----------------------------
# Resumo final
# -----------------------------
print("\n📊 Resumo final:")
print(f"   ✅ m_points com dados: {total_com_dados}")
print(f"   ⚠️ m_points sem dados: {total_sem_dados}")
print(f"   🕒 Tempo total: {time.time() - t0:.2f}s")
