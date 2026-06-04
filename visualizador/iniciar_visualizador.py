"""
Inicia um servidor web local (offline, sem internet) servindo esta pasta e
abre o navegador no visualizador. Encerre com Ctrl+C.
"""
import http.server
import socketserver
import threading
import webbrowser
import time
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))

PORTA_INICIAL = 8000
PORTA_FINAL = 8010


class Handler(http.server.SimpleHTTPRequestHandler):
    # Garante MIME corretos (alguns Windows nao mapeiam .csv/.js)
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".js": "application/javascript",
        ".csv": "text/csv",
        ".json": "application/json",
    }

    def end_headers(self):
        # Forca o navegador a SEMPRE buscar a versao mais recente (sem cache),
        # evitando rodar um app.js antigo guardado em cache.
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def log_message(self, *args):
        pass  # silencia o log de cada requisicao


def achar_porta():
    for porta in range(PORTA_INICIAL, PORTA_FINAL + 1):
        try:
            httpd = socketserver.TCPServer(("", porta), Handler)
            return httpd, porta
        except OSError:
            continue
    return None, None


def main():
    httpd, porta = achar_porta()
    if httpd is None:
        print(f"[ERRO] Nenhuma porta livre entre {PORTA_INICIAL} e {PORTA_FINAL}.")
        return 1

    url = f"http://localhost:{porta}/"
    print("=" * 60, flush=True)
    print("  VISUALIZADOR DE ESTADOS OPERACIONAIS", flush=True)
    print("=" * 60, flush=True)
    print(f"  >>> ABRA NO NAVEGADOR:  {url}", flush=True)
    print(f"  >>> PORTA: {porta}", flush=True)
    print("  (Mantenha esta janela aberta. Ctrl+C para encerrar.)", flush=True)
    print("=" * 60, flush=True)

    def abrir():
        time.sleep(1.0)
        webbrowser.open(url)

    threading.Thread(target=abrir, daemon=True).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nEncerrando servidor...")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
