#!/usr/bin/env python3
"""
Script para comprimir todos os arquivos PNG na pasta latex.
Comprime os arquivos PNG usando otimização e sobrescreve os originais.
"""

import os
import glob
from PIL import Image
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def comprimir_png(caminho_arquivo):
    """
    Comprime um arquivo PNG usando otimização do Pillow.

    Args:
        caminho_arquivo (str): Caminho completo para o arquivo PNG
    """
    try:
        # Abrir a imagem
        with Image.open(caminho_arquivo) as img:
            # Obter o tamanho original
            tamanho_original = os.path.getsize(caminho_arquivo)

            # Salvar com otimização máxima
            # optimize=True reduz significativamente o tamanho dos PNGs
            # Para PNG, não usamos quality como em JPEG, mas optimize=True
            img.save(caminho_arquivo, 'PNG', optimize=True)

            # Obter o tamanho comprimido
            tamanho_comprimido = os.path.getsize(caminho_arquivo)

            # Calcular redução
            reducao = tamanho_original - tamanho_comprimido
            percentual = (reducao / tamanho_original) * 100 if tamanho_original > 0 else 0

            logging.info(".2f")

    except Exception as e:
        logging.error(f"Erro ao comprimir {caminho_arquivo}: {str(e)}")

def main():
    """
    Função principal que encontra e comprime todos os PNGs na pasta latex.
    """
    # Caminho para a pasta latex (um nível acima de code/)
    pasta_latex = os.path.join(os.path.dirname(__file__), '..', 'latex')

    # Verificar se a pasta existe
    if not os.path.exists(pasta_latex):
        logging.error(f"Pasta latex não encontrada: {pasta_latex}")
        return

    # Encontrar todos os arquivos PNG recursivamente
    padrao_png = os.path.join(pasta_latex, '**', '*.png')
    arquivos_png = glob.glob(padrao_png, recursive=True)

    if not arquivos_png:
        logging.warning("Nenhum arquivo PNG encontrado na pasta latex.")
        return

    logging.info(f"Encontrados {len(arquivos_png)} arquivos PNG para compressão.")

    # Contadores para estatísticas
    total_original = 0
    total_comprimido = 0

    # Processar cada arquivo PNG
    for arquivo in arquivos_png:
        tamanho_original = os.path.getsize(arquivo)
        total_original += tamanho_original

        comprimir_png(arquivo)

        tamanho_comprimido = os.path.getsize(arquivo)
        total_comprimido += tamanho_comprimido

    # Estatísticas finais
    reducao_total = total_original - total_comprimido
    percentual_total = (reducao_total / total_original) * 100 if total_original > 0 else 0

    logging.info("""
=== RESUMO DA COMPRESSÃO ===
Tamanho total original: %.2f MB
Tamanho total comprimido: %.2f MB
Redução total: %.2f MB (%.2f%%)""" % (
        total_original / (1024 * 1024),
        total_comprimido / (1024 * 1024),
        reducao_total / (1024 * 1024),
        percentual_total
    ))

if __name__ == "__main__":
    main()
