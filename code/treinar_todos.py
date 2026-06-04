"""
Treina (do zero, a partir dos dados brutos em data/raw/) TODOS os equipamentos
disponiveis, gerando as tabelas unificadas e classificadas em data/processed/.

Detecta automaticamente o tipo:
  - ELETRICO  -> existe data/raw/dados_estimated_<mpoint>.csv
  - MECANICO  -> nao existe arquivo estimated

Uso:
    cd code
    python treinar_todos.py            # treina todos
    python treinar_todos.py c_636 c_1518   # treina apenas os informados

Roda 100% offline a partir dos CSVs brutos (nao acessa o InfluxDB).
Nao abre janelas nem gera as visualizacoes 3D (foco em produzir os dados).
"""
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from pipeline_deteccao_estados import PipelineDeteccaoEstados
from pipeline_deteccao_estados_mecanico import PipelineDeteccaoEstadosMecanico

DIR_RAW = BASE_DIR / "data" / "raw"


def descobrir_mpoints():
    """Lista mpoints a partir dos arquivos dados_c_*.csv em data/raw/."""
    mpoints = []
    for arq in sorted(DIR_RAW.glob("dados_c_*.csv")):
        nome = arq.stem  # dados_c_636
        mp = nome.replace("dados_", "")  # c_636
        mpoints.append(mp)
    return mpoints


def eh_eletrico(mpoint):
    """Eletrico se possui arquivo estimated; caso contrario, mecanico."""
    return (DIR_RAW / f"dados_estimated_{mpoint}.csv").exists()


def treinar_eletrico(mpoint):
    p = PipelineDeteccaoEstados(mpoint=mpoint)
    if not p.executar_processamento(mpoint):
        return False
    if not p.executar_treino_kmeans(mpoint):
        return False
    if not p.gerar_parametros(mpoint):
        return False
    return True


def treinar_mecanico(mpoint):
    p = PipelineDeteccaoEstadosMecanico(mpoint=mpoint)
    if not p.executar_processamento_mecanico(mpoint):
        return False
    if not p.executar_treino_kmeans_mecanico(mpoint):
        return False
    if not p.gerar_parametros_mecanico(mpoint):
        return False
    return True


def main():
    alvo = sys.argv[1:] if len(sys.argv) > 1 else descobrir_mpoints()
    if not alvo:
        print("[ERRO] Nenhum mpoint encontrado em data/raw/ (dados_c_*.csv).")
        return 1

    print("=" * 80)
    print(f"TREINANDO {len(alvo)} EQUIPAMENTO(S): {', '.join(alvo)}")
    print("=" * 80)

    resultados = {}
    t0 = time.time()
    for mp in alvo:
        tipo = "ELETRICO" if eh_eletrico(mp) else "MECANICO"
        print("\n" + "#" * 80)
        print(f"# {mp}  ({tipo})")
        print("#" * 80)
        ti = time.time()
        try:
            ok = treinar_eletrico(mp) if tipo == "ELETRICO" else treinar_mecanico(mp)
        except Exception as e:
            print(f"[ERRO] Excecao ao treinar {mp}: {e}")
            ok = False
        resultados[mp] = (tipo, ok, (time.time() - ti) / 60.0)

    print("\n" + "=" * 80)
    print("RESUMO DO TREINO")
    print("=" * 80)
    for mp, (tipo, ok, mins) in resultados.items():
        status = "[OK]" if ok else "[FALHOU]"
        print(f"   {status:9} {mp:8} ({tipo})  -  {mins:.1f} min")
    print(f"\nTempo total: {(time.time() - t0) / 60.0:.1f} min")

    return 0 if all(ok for _, ok, _ in resultados.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
