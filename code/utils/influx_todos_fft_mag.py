from influxdb import InfluxDBClient
import pandas as pd
from datetime import datetime, timedelta, timezone
import time
import os
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError
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
measurement = "validated_fft_mag"
retention_policy = "autogen"

if not host:
    print("[ERROR] INFLUXDB_IP não configurado no arquivo config.env!")
    sys.exit(1)

print("INICIANDO exportacao de dados do InfluxDB (validated_fft_mag)...")
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
# Função que coleta dados de 1 m_point (todas as amostras)
# -----------------------------
def coletar_mpoint(mp, filename):
    query = f"""
    SELECT "f", "x", "y", "z", "N", "fs"
    FROM "{database}"."{retention_policy}"."{measurement}"
    WHERE time >= '{start_time.isoformat()}' AND time <= '{end_time.isoformat()}'
      AND "m_point" = '{mp}'
    """

    result = client.query(query)
    df = pd.DataFrame(result.get_points())

    if df.empty:
        return None

    df["m_point"] = mp
    df.to_csv(filename, index=False)
    return len(df)

# -----------------------------
# Loop por cada m_point
# -----------------------------
total_com_dados = 0
total_sem_dados = 0
timeouts = []
t0 = time.time()

executor = ThreadPoolExecutor(max_workers=1)

for i, mp in enumerate(mpoints, start=1):
    filename = os.path.join("code", "data", "raw", f"dados_fft_mag_{mp}.csv")

    if os.path.exists(filename):
        print(f"⏭️ {mp} já exportado, pulando...")
        continue

    print(f"\n➡️ [{i}/{len(mpoints)}] Coletando dados para m_point: {mp}")
    t1 = time.time()

    future = executor.submit(coletar_mpoint, mp, filename)

    try:
        registros = future.result(timeout=400)  # timeout de 400s
        if registros is None:
            total_sem_dados += 1
            print(f"⚠️ Sem dados para m_point: {mp}")
        else:
            total_com_dados += 1
            print(f"✅ {registros} registros salvos em {filename} (tempo: {time.time() - t1:.2f}s)")

    except TimeoutError:
        print(f"⏰ Timeout excedido para m_point {mp}, pulando...")
        timeouts.append(mp)
        future.cancel()
    except Exception as e:
        print(f"❌ Erro no m_point {mp}: {e}")
        continue

# -----------------------------
# Resumo final
# -----------------------------
print("\n📊 Resumo final:")
print(f"   ✅ m_points com dados: {total_com_dados}")
print(f"   ⚠️ m_points sem dados: {total_sem_dados}")
print(f"   ⏰ m_points com timeout: {len(timeouts)}")
print(f"   🕒 Tempo total: {time.time() - t0:.2f}s")

# -----------------------------
# Salva lista de timeouts
# -----------------------------
if timeouts:
    with open("m_points_timeout_fft_mag.txt", "w") as f:
        for mp in timeouts:
            f.write(mp + "\n")
    print("📂 Lista de m_points que excederam tempo salva em m_points_timeout_fft_mag.txt")
