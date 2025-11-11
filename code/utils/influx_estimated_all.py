from influxdb import InfluxDBClient
import pandas as pd
from datetime import datetime, timedelta, timezone
import time
import os
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# -----------------------------
# Configurações do InfluxDB
# -----------------------------
host = "10.8.0.121"  # novo host solicitado
port = 8086
database = "aihub"
measurement = "estimated"  # solicitado
retention_policy = "autogen"

print("INICIANDO exportacao de dados do InfluxDB (estimated) - TODOS OS MPOINTS...")

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
    try:
        with open("code/utils/lista_mpoints.txt", "r") as f:
            mpoints = [linha.strip() for linha in f if linha.strip()]
        print(f"[INFO] Modo TODOS: Total de m_points carregados da lista: {len(mpoints)}")
    except FileNotFoundError:
        print("[ERROR] Arquivo lista_mpoints.txt nao encontrado!")
        mpoints = []
    except Exception as e:
        print(f"[ERROR] Erro ao carregar lista de m_points: {e}")
        mpoints = []

# -----------------------------
# Define range de 1 ano (timezone-aware, UTC)
# -----------------------------
end_time = datetime.now(timezone.utc)
start_time = end_time - timedelta(days=365)
print(f"[INTERVALO] Intervalo de dados: {start_time.isoformat()} ate {end_time.isoformat()}")

# -----------------------------
# Função que coleta dados de 1 m_point (todas as amostras cruas)
# -----------------------------
def coletar_mpoint(mp, filename):
    query = f"""
    SELECT "rotational_speed",
           "vel_rms",
           "current"
    FROM "{database}"."{retention_policy}"."{measurement}"
    WHERE time >= '{start_time.isoformat()}' AND time <= '{end_time.isoformat()}'
      AND "m_point" = '{mp}'
    """

    result = client.query(query)
    df = pd.DataFrame(result.get_points())

    if df.empty:
        return None

    df["m_point"] = mp

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)
    return len(df)

# -----------------------------
# Loop por TODOS os m_points dinamicamente carregados
# -----------------------------
total_com_dados = 0
total_sem_dados = 0
timeouts = []
t0 = time.time()

executor = ThreadPoolExecutor(max_workers=1)

for i, mp in enumerate(mpoints, start=1):
    filename = os.path.join("code", "data", "raw", f"dados_estimated_{mp}.csv")

    if os.path.exists(filename):
        print(f"[SKIP] {mp} já exportado, pulando...")
        continue

    print(f"\n[PROCESSANDO] [{i}/{len(mpoints)}] Coletando dados para m_point: {mp}")
    t1 = time.time()

    future = executor.submit(coletar_mpoint, mp, filename)

    try:
        registros = future.result(timeout=400)
        if registros is None:
            total_sem_dados += 1
            print(f"[WARNING] Sem dados para m_point: {mp}")
        else:
            total_com_dados += 1
            print(f"[SUCCESS] {registros} registros salvos em {filename} (tempo: {time.time() - t1:.2f}s)")

    except TimeoutError:
        print(f"[TIMEOUT] Timeout excedido para m_point {mp}, pulando...")
        timeouts.append(mp)
        future.cancel()
    except Exception as e:
        print(f"[ERROR] Erro no m_point {mp}: {e}")
        continue

# -----------------------------
# Resumo final
# -----------------------------
print("\n[RESUMO] Resumo final:")
print(f"   [SUCCESS] m_points com dados: {total_com_dados}")
print(f"   [WARNING] m_points sem dados: {total_sem_dados}")
print(f"   [TIMEOUT] m_points com timeout: {len(timeouts)}")
print(f"   [TIME] Tempo total: {time.time() - t0:.2f}s")

# -----------------------------
# Salva lista de timeouts
# -----------------------------
if timeouts:
    with open("m_points_timeout_estimated.txt", "w") as f:
        for mp in timeouts:
            f.write(mp + "\n")
    print("[SAVE] Lista de m_points que excederam tempo salva em m_points_timeout_estimated.txt")



