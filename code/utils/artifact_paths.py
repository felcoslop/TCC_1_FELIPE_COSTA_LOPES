"""Utility helpers to centralize artifact paths per mpoint.

This module standardizes where intermediate and final artifacts are stored
for each monitored `mpoint`.  All generated files now include the `mpoint`
identifier in their names as required and no longer rely on the legacy
`parametros/` directory structure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict


BASE_DIR = Path(__file__).resolve().parent.parent


def _validate_mpoint(mpoint: str) -> str:
    value = (mpoint or "").strip()
    if not value:
        raise ValueError("mpoint must be provided")
    return value


def ensure_base_structure() -> None:
    """Ensure shared directories exist."""

    for path in (
        BASE_DIR / "data" / "raw",
        BASE_DIR / "data" / "raw_preenchido",
        BASE_DIR / "data" / "processed",
        BASE_DIR / "data" / "normalized",
        BASE_DIR / "models",
        BASE_DIR / "results",
    ):
        path.mkdir(parents=True, exist_ok=True)


def get_mpoint_dirs(mpoint: str, create: bool = False) -> Dict[str, Path]:
    """Return per-mpoint directories.

    Args:
        mpoint: Identifier such as ``"c_636"``.
        create: When ``True`` the directories are created on the filesystem.
    """

    value = _validate_mpoint(mpoint)
    dirs = {
        "models": BASE_DIR / "models" / value,
        "results": BASE_DIR / "results" / value,
    }

    if create:
        for path in dirs.values():
            path.mkdir(parents=True, exist_ok=True)

    return dirs


def normalized_csv_path(mpoint: str, intervalo_arquivo: str = None) -> Path:
    value = _validate_mpoint(mpoint)
    ensure_base_structure()
    intervalo_tag = f"_{intervalo_arquivo}" if intervalo_arquivo else ""
    return BASE_DIR / "data" / "normalized" / f"dados_kmeans_{value}{intervalo_tag}.csv"


def normalized_numpy_path(mpoint: str, intervalo_arquivo: str = None) -> Path:
    value = _validate_mpoint(mpoint)
    ensure_base_structure()
    intervalo_tag = f"_{intervalo_arquivo}" if intervalo_arquivo else ""
    return BASE_DIR / "data" / "normalized" / f"dados_normalizados_completos_{value}{intervalo_tag}.npy"


def processed_unificado_path(mpoint: str) -> Path:
    value = _validate_mpoint(mpoint)
    ensure_base_structure()
    return BASE_DIR / "data" / "processed" / f"dados_unificados_final_{value}.csv"


def processed_classificado_path(mpoint: str) -> Path:
    value = _validate_mpoint(mpoint)
    ensure_base_structure()
    return BASE_DIR / "data" / "processed" / f"dados_classificados_kmeans_moderado_{value}.csv"


def processed_rotulado_path(mpoint: str) -> Path:
    value = _validate_mpoint(mpoint)
    ensure_base_structure()
    return BASE_DIR / "data" / "processed" / f"dados_kmeans_rotulados_conservador_{value}.csv"


def kmeans_model_path(mpoint: str, create: bool = False) -> Path:
    dirs = get_mpoint_dirs(mpoint, create=create)
    value = _validate_mpoint(mpoint)
    return dirs["models"] / f"kmeans_model_moderado_{value}.pkl"


def scaler_model_path(mpoint: str, create: bool = False) -> Path:
    dirs = get_mpoint_dirs(mpoint, create=create)
    value = _validate_mpoint(mpoint)
    return dirs["models"] / f"scaler_model_moderado_{value}.pkl"


def info_kmeans_path(mpoint: str, create: bool = False) -> Path:
    dirs = get_mpoint_dirs(mpoint, create=create)
    value = _validate_mpoint(mpoint)
    return dirs["models"] / f"info_kmeans_model_moderado_{value}.json"


def info_normalizacao_path(mpoint: str, create: bool = False) -> Path:
    dirs = get_mpoint_dirs(mpoint, create=create)
    value = _validate_mpoint(mpoint)
    return dirs["models"] / f"info_normalizacao_{value}.json"


def preprocess_pipeline_path(mpoint: str, create: bool = False) -> Path:
    dirs = get_mpoint_dirs(mpoint, create=create)
    value = _validate_mpoint(mpoint)
    return dirs["models"] / f"preprocess_pipeline_{value}.pkl"


def scaler_maxmin_path(mpoint: str, create: bool = False) -> Path:
    dirs = get_mpoint_dirs(mpoint, create=create)
    value = _validate_mpoint(mpoint)
    return dirs["models"] / f"scaler_maxmin_{value}.pkl"


def config_path(mpoint: str, create: bool = False) -> Path:
    dirs = get_mpoint_dirs(mpoint, create=create)
    value = _validate_mpoint(mpoint)
    return dirs["models"] / f"config_{value}.json"


def results_dir(mpoint: str, create: bool = False) -> Path:
    dirs = get_mpoint_dirs(mpoint, create=create)
    return dirs["results"]


def resultado_intervalo_csv(mpoint: str) -> Path:
    value = _validate_mpoint(mpoint)
    return results_dir(mpoint, create=True) / f"predicao_intervalo_{value}.csv"


def grafico_intervalo_path(mpoint: str, sufixo: str) -> Path:
    value = _validate_mpoint(mpoint)
    return results_dir(mpoint, create=True) / f"grafico_estados_{value}_{sufixo}.png"


def relatorio_intervalo_path(mpoint: str, sufixo: str) -> Dict[str, Path]:
    value = _validate_mpoint(mpoint)
    base_name = f"analise_completa_{value}_{sufixo}"
    dir_resultados = results_dir(mpoint, create=True)
    return {
        "normalizado": dir_resultados / f"{base_name}_normalizado.csv",
        "resultados": dir_resultados / f"{base_name}_resultados.csv",
        "relatorio": dir_resultados / f"{base_name}_relatorio.txt",
    }


