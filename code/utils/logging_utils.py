"""
Funcoes auxiliares pra criar logs organizados pro TCC.
Esses logs vao ser usados depois pra gerar os textos e tabelas do trabalho.
Ajuda a manter tudo documentado e facil de encontrar depois.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional


def ensure_logs_dir() -> Path:
    """Cria o diretorio de logs se nao existir"""
    logs_dir = Path(__file__).parent.parent / 'logs'
    logs_dir.mkdir(exist_ok=True)
    return logs_dir


def generate_log_filename(script_name: str, mpoint: str, operation: str = None) -> str:
    """
    Cria nome padronizado pros arquivos de log

    Args:
        script_name: Nome do script que rodou (ex: 'normalizar_dados_kmeans')
        mpoint: Codigo do equipamento (ex: 'c_636')
        operation: Tipo de operacao se quiser especificar (opcional)

    Returns:
        Nome do arquivo de log
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if operation:
        return f"{script_name}_{operation}_{mpoint}_{timestamp}.txt"
    else:
        return f"{script_name}_{mpoint}_{timestamp}.txt"


def save_log(log_data: Dict[str, Any], script_name: str, mpoint: str, operation: str = None) -> Path:
    """
    Salva dados de log em arquivo .txt estruturado

    Args:
        log_data: Dicionário com dados do log
        script_name: Nome do script
        mpoint: ID do mpoint
        operation: Operação específica (opcional)

    Returns:
        Caminho do arquivo salvo
    """
    logs_dir = ensure_logs_dir()
    filename = generate_log_filename(script_name, mpoint, operation)
    filepath = logs_dir / filename

    # Adicionar metadados
    log_data['_metadata'] = {
        'timestamp': datetime.now().isoformat(),
        'script': script_name,
        'mpoint': mpoint,
        'operation': operation,
        'version': '1.0'
    }

    # Salvar como JSON formatado para fácil parsing pela IA
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

    print(f"  [LOG] Arquivo de log salvo: {filepath}")
    return filepath


def create_processing_log(
    script_name: str,
    mpoint: str,
    operation: str,
    input_files: List[str] = None,
    output_files: List[str] = None,
    parameters: Dict[str, Any] = None,
    statistics: Dict[str, Any] = None,
    processing_time: float = None,
    success: bool = True,
    error_message: str = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Cria log padronizado para operações de processamento

    Args:
        script_name: Nome do script
        mpoint: ID do mpoint
        operation: Operação realizada
        input_files: Lista de arquivos de entrada
        output_files: Lista de arquivos de saída
        parameters: Parâmetros utilizados
        statistics: Estatísticas do processamento
        processing_time: Tempo de processamento em segundos
        success: Se a operação foi bem-sucedida
        error_message: Mensagem de erro se houver
        **kwargs: Outros dados específicos

    Returns:
        Dicionário com dados do log
    """
    log_data = {
        'tipo': 'processing_log',
        'script_name': script_name,
        'mpoint': mpoint,
        'operation': operation,
        'input_files': input_files or [],
        'output_files': output_files or [],
        'parameters': parameters or {},
        'statistics': statistics or {},
        'processing_time_seconds': processing_time,
        'success': success,
        'error_message': error_message,
        **kwargs
    }

    return log_data


def create_visualization_log(
    script_name: str,
    mpoint: str,
    chart_type: str,
    data_description: Dict[str, Any],
    chart_files: List[str],
    period_info: Dict[str, Any] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Cria log padronizado para visualizações/gráficos

    Args:
        script_name: Nome do script
        mpoint: ID do mpoint
        chart_type: Tipo de gráfico (ex: '3d_scatter', 'histogram', etc.)
        data_description: Descrição dos dados utilizados
        chart_files: Lista de arquivos de gráficos gerados
        period_info: Informações sobre o período dos dados
        **kwargs: Outros dados específicos

    Returns:
        Dicionário com dados do log
    """
    log_data = {
        'tipo': 'visualization_log',
        'script_name': script_name,
        'mpoint': mpoint,
        'chart_type': chart_type,
        'data_description': data_description,
        'chart_files': chart_files,
        'period_info': period_info or {},
        **kwargs
    }

    return log_data


def create_training_log(
    script_name: str,
    mpoint: str,
    model_info: Dict[str, Any],
    training_data_info: Dict[str, Any],
    performance_metrics: Dict[str, Any],
    model_files: List[str],
    **kwargs
) -> Dict[str, Any]:
    """
    Cria log padronizado para treinamento de modelos

    Args:
        script_name: Nome do script
        mpoint: ID do mpoint
        model_info: Informações sobre o modelo treinado
        training_data_info: Informações sobre os dados de treino
        performance_metrics: Métricas de performance
        model_files: Arquivos do modelo salvos
        **kwargs: Outros dados específicos

    Returns:
        Dicionário com dados do log
    """
    log_data = {
        'tipo': 'training_log',
        'script_name': script_name,
        'mpoint': mpoint,
        'model_info': model_info,
        'training_data_info': training_data_info,
        'performance_metrics': performance_metrics,
        'model_files': model_files,
        **kwargs
    }

    return log_data


def create_analysis_log(
    script_name: str,
    mpoint: str,
    analysis_type: str,
    input_period: Dict[str, Any],
    results_summary: Dict[str, Any],
    generated_files: List[str],
    **kwargs
) -> Dict[str, Any]:
    """
    Cria log padronizado para análises de intervalo

    Args:
        script_name: Nome do script
        mpoint: ID do mpoint
        analysis_type: Tipo de análise (ex: 'interval_analysis', 'real_time')
        input_period: Período analisado
        results_summary: Resumo dos resultados
        generated_files: Arquivos gerados na análise
        **kwargs: Outros dados específicos

    Returns:
        Dicionário com dados do log
    """
    log_data = {
        'tipo': 'analysis_log',
        'script_name': script_name,
        'mpoint': mpoint,
        'analysis_type': analysis_type,
        'input_period': input_period,
        'results_summary': results_summary,
        'generated_files': generated_files,
        **kwargs
    }

    return log_data


def format_file_list(files: List[Path]) -> List[str]:
    """Formata lista de arquivos para log"""
    return [str(f) for f in files]


def get_file_info(filepath: Path) -> Dict[str, Any]:
    """Obtém informações sobre um arquivo"""
    if not filepath.exists():
        return {'exists': False}

    stat = filepath.stat()
    return {
        'exists': True,
        'size_bytes': stat.st_size,
        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
        'path': str(filepath)
    }


def enrich_results_file(mpoint: str, new_data: Dict[str, Any]) -> None:
    """
    Enriquece o arquivo results existente com novas informações

    Args:
        mpoint: ID do mpoint
        new_data: Novos dados para adicionar ao results
    """
    results_dir = Path(__file__).parent.parent / 'results'
    results_file = results_dir / f'results_{mpoint}.txt'

    # Carregar dados existentes se o arquivo existir
    existing_data = {}
    if results_file.exists():
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        except:
            existing_data = {}

    # Mesclar dados
    existing_data.update(new_data)
    existing_data['_last_updated'] = datetime.now().isoformat()

    # Salvar arquivo enriquecido
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, indent=2, ensure_ascii=False)

    print(f"  [RESULTS] Arquivo results enriquecido: {results_file}")
