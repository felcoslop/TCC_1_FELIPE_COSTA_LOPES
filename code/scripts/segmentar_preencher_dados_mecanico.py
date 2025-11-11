"""
Script de segmentação inicial para equipamentos MECÂNICOS.
Separa dados_c e dados_slip em períodos baseados em gaps de tempo.
Este script é um wrapper que chama processar_dados_simples_mecanico.py.
"""
import sys
import subprocess
from pathlib import Path
import argparse

BASE_DIR = Path(__file__).resolve().parent.parent

def main():
    """Wrapper que chama o script de processamento mecânico"""
    parser = argparse.ArgumentParser(
        description="Segmentação de dados para equipamento MECÂNICO"
    )
    parser.add_argument('--mpoint', type=str, required=True, help='ID do mpoint (ex: c_1518)')
    parser.add_argument('--intervalo-arquivo', type=str, help='Intervalo formatado')
    
    args = parser.parse_args()
    
    print("="*80)
    print("SEGMENTAÇÃO E PREENCHIMENTO - EQUIPAMENTO MECÂNICO")
    print("="*80)
    print(f"Mpoint: {args.mpoint}")
    print("Redirecionando para processar_dados_simples_mecanico.py...")
    print("="*80)
    
    # Chamar script de processamento
    script_path = BASE_DIR / 'scripts' / 'processar_dados_simples_mecanico.py'
    
    cmd = [sys.executable, str(script_path), '--mpoint', args.mpoint]
    if args.intervalo_arquivo:
        cmd.extend(['--intervalo-arquivo', args.intervalo_arquivo])

    resultado = subprocess.run(cmd, cwd=str(BASE_DIR), text=True)
    
    sys.exit(resultado.returncode)

if __name__ == '__main__':
    main()

