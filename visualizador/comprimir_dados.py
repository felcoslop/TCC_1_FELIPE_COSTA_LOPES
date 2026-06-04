"""
Comprime os CSV de dados/ para .csv.gz (para o deploy: ~196 MB -> ~40 MB).
O nginx serve esses .gz automaticamente quando o app pede o .csv.

Uso:
    cd visualizador
    python comprimir_dados.py
"""
import gzip
import glob
import os
import shutil

DADOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dados")


def main():
    csvs = sorted(glob.glob(os.path.join(DADOS, "*.csv")))
    if not csvs:
        print("Nenhum .csv encontrado em dados/. Rode antes: python ../code/exportar_visualizador_web.py")
        return
    tot_in = tot_out = 0
    for f in csvs:
        out = f + ".gz"
        with open(f, "rb") as fi, gzip.open(out, "wb", compresslevel=9) as fo:
            shutil.copyfileobj(fi, fo, length=1024 * 1024)
        si, so = os.path.getsize(f), os.path.getsize(out)
        tot_in += si
        tot_out += so
        print(f"  {os.path.basename(f):16} {si/1048576:6.1f} MB -> {so/1048576:5.1f} MB")
    print("-" * 48)
    print(f"  {'TOTAL':16} {tot_in/1048576:6.1f} MB -> {tot_out/1048576:5.1f} MB")
    print("\nPronto. Suba os arquivos .csv.gz e o manifest.json (o build usa esses).")


if __name__ == "__main__":
    main()
