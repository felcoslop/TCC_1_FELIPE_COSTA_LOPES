"""
Script de normalização para equipamentos MECÂNICOS.
Foco: Temperatura e Vibração (sem current, sem RPM).

Características dos dados mecânicos:
- object_temp: Temperatura do equipamento
- vel_rms_x/y/z: Vibração RMS em 3 eixos  
- vel_max_x/y/z: Picos de vibração
- mag_x/y/z: Magnetômetro (para detectar vibrações residuais)
- Dados de slip: Análise de frequência

Análise de estados:
- DESLIGADO: Temperatura ambiente + vibrações próximas de zero/residuais
- LIGADO: Aumento de temperatura + vibrações significativas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer
from sklearn.decomposition import PCA
import argparse
import json
import pickle
import warnings
from datetime import datetime
from pathlib import Path

import sys

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from utils.artifact_paths import (
    ensure_base_structure,
    info_normalizacao_path,
    info_kmeans_path,
    normalized_csv_path,
    normalized_numpy_path,
    preprocess_pipeline_path,
    processed_unificado_path,
    scaler_maxmin_path,
)
from utils.logging_utils import (
    save_log,
    create_processing_log,
    create_visualization_log,
    format_file_list,
    get_file_info,
    enrich_results_file,
)

warnings.filterwarnings('ignore')

DIR_NORMALIZED = BASE_DIR / 'data' / 'normalized'
DIR_MODELS = BASE_DIR / 'models'
DIR_PROCESSED = BASE_DIR / 'data' / 'processed'
DIR_RESULTS = BASE_DIR / 'results'
DIR_PLOTS = BASE_DIR / 'plots'

def criar_diretorios():
    """Cria diretórios necessários"""
    ensure_base_structure()
    for diretorio in [DIR_RESULTS, DIR_PLOTS]:
        diretorio.mkdir(parents=True, exist_ok=True)
    print("Diretórios criados/verificados com sucesso!")

def validar_timestamp(df):
    """Valida a coluna de timestamp para monotonicidade, duplicatas e timezone"""
    print("\nValidando coluna de timestamp...")
    if 'time' not in df.columns:
        print("  - Coluna 'time' não encontrada!")
        return
    
    # Verificar se já é datetime, se não, converter
    if not pd.api.types.is_datetime64_any_dtype(df['time']):
        try:
            df['time'] = pd.to_datetime(df['time'])
        except Exception:
            print("  - Falha ao converter 'time' para datetime.")
            return
    
    is_monotonic = df['time'].is_monotonic_increasing
    num_dups = df['time'].duplicated().sum()
    tzinfo = df['time'].dt.tz is not None
    print(f"  - Monotônica crescente: {'Sim' if is_monotonic else 'Não'}")
    print(f"  - Duplicatas de timestamp: {num_dups:,}")
    print(f"  - Possui timezone: {'Sim' if tzinfo else 'Não'}")
    if not is_monotonic:
        print("  - Aviso: timestamps não estão estritamente ordenados. Mantendo ordem original.")

def carregar_dados(mpoint=None, intervalo_arquivo=None):
    """Carrega e concatena todos os períodos finalizados (VERSÃO MECÂNICA)"""
    print("Carregando períodos finalizados (equipamento mecânico)...")

    dir_uniao = BASE_DIR / 'data' / 'raw_preenchido'
    
    # Buscar com ou sem intervalo
    if intervalo_arquivo:
        print(f"  - Modo intervalo: buscando arquivos com {intervalo_arquivo}")
        arquivos_final = sorted(dir_uniao.glob(f'periodo_*_final_{mpoint}_{intervalo_arquivo}.csv'))
    else:
        print(f"  - Modo treino: buscando arquivos originais")
        arquivos_final = sorted(dir_uniao.glob(f'periodo_*_final_{mpoint}.csv'))

    if len(arquivos_final) == 0:
        print(f"  - Nenhum arquivo final encontrado em {dir_uniao}")
        print("Execute primeiro:")
        print(f"  1. python scripts/processar_dados_simples_mecanico.py --mpoint {mpoint}")
        print(f"  2. python scripts/unir_sincronizar_periodos_mecanico.py --mpoint {mpoint}")
        return None, []

    dataframes = []
    arquivos_origem = []

    for arq in arquivos_final:
        try:
            df_periodo = pd.read_csv(arq)
        except Exception as e:
            print(f"  - [AVISO] Falha ao ler {arq.name}: {e}")
            continue

        if 'time' not in df_periodo.columns:
            print(f"  - [AVISO] Arquivo {arq.name} sem coluna 'time' - ignorando")
            continue

        # Converter timestamps
        df_periodo['time'] = pd.to_datetime(df_periodo['time'], format='mixed', utc=True)

        # Garantir preenchimento de m_point se existir
        if 'm_point' in df_periodo.columns:
            df_periodo['m_point'] = df_periodo['m_point'].ffill().bfill()

        df_periodo['arquivo_origem'] = arq.name
        dataframes.append(df_periodo)
        arquivos_origem.append(arq.name)

    if len(dataframes) == 0:
        print("  - Nenhum arquivo válido encontrado após leitura")
        return None, []

    df = pd.concat(dataframes, ignore_index=True)
    df = df.sort_values('time').drop_duplicates(subset='time', keep='first').reset_index(drop=True)

    print(f"  - Arquivos combinados: {len(arquivos_origem)}")
    print(f"  - Total de linhas: {len(df):,}")
    print(f"  - Intervalo temporal: {df['time'].min()} até {df['time'].max()}")

    validar_timestamp(df)

    # Salvar versão combinada para compatibilidade com outros scripts
    DIR_PROCESSED.mkdir(parents=True, exist_ok=True)
    arquivo_unificado = processed_unificado_path(mpoint)
    df.to_csv(arquivo_unificado, index=False)
    print(f"  - Arquivo combinado salvo em: {arquivo_unificado}")

    return df, arquivos_origem

def remover_colunas_auxiliares(df):
    """Remove colunas categóricas e auxiliares (m_point, interpolado, identificadores)"""
    print("\nRemovendo colunas auxiliares...")

    colunas_remover = []
    for col in df.columns:
        if col == 'time':
            continue
        col_lower = col.lower()
        if ('m_point' in col_lower or 'interpolado' in col_lower or
                'periodo_id' in col_lower or 'arquivo_origem' in col_lower):
            colunas_remover.append(col)
        elif df[col].dtype == bool:
            colunas_remover.append(col)
        elif df[col].dtype == object:
            colunas_remover.append(col)

    colunas_remover = sorted(set(colunas_remover))
    if colunas_remover:
        print(f"  - Colunas removidas: {colunas_remover}")
    else:
        print("  - Nenhuma coluna auxiliar encontrada")

    df_limpo = df.drop(columns=colunas_remover, errors='ignore')

    print(f"  - Colunas após limpeza: {len(df_limpo.columns)}")
    print(f"  - Linhas mantidas: {len(df_limpo):,}")

    return df_limpo

def analisar_dados_mecanicos(df):
    """Analisa características dos dados mecânicos"""
    print("\nAnalisando características dos dados MECÂNICOS...")
    
    # Informações gerais
    print(f"  - Total de linhas: {len(df):,}")
    print(f"  - Total de colunas: {len(df.columns)}")
    
    # Identificar colunas numéricas (excluindo time)
    colunas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'time' in colunas_numericas:
        colunas_numericas.remove('time')
    
    print(f"  - Colunas numéricas: {len(colunas_numericas)}")
    
    # Identificar colunas importantes para equipamentos mecânicos
    colunas_temp = [col for col in colunas_numericas if 'temp' in col.lower()]
    colunas_vibracao = [col for col in colunas_numericas if 'vel_' in col.lower()]
    colunas_mag = [col for col in colunas_numericas if 'mag_' in col.lower()]
    colunas_slip = [col for col in colunas_numericas if any(x in col.lower() for x in ['fe_', 'fr_', 'rms'])]
    
    print(f"\n  Colunas por tipo:")
    print(f"  - Temperatura: {len(colunas_temp)} ({', '.join(colunas_temp[:3])}...)")
    print(f"  - Vibração: {len(colunas_vibracao)} ({', '.join(colunas_vibracao[:3])}...)")
    print(f"  - Magnetômetro: {len(colunas_mag)} ({', '.join(colunas_mag[:3])}...)")
    print(f"  - Slip: {len(colunas_slip)}")
    
    # Análise de valores nulos
    print("\nAnálise de valores nulos:")
    colunas_com_nulos = df[colunas_numericas].isnull().sum()
    colunas_com_nulos = colunas_com_nulos[colunas_com_nulos > 0].sort_values(ascending=False)
    
    if len(colunas_com_nulos) > 0:
        print(f"  - Colunas com valores nulos: {len(colunas_com_nulos)}")
        for col, n_nulos in colunas_com_nulos.head(10).items():
            pct = (n_nulos / len(df)) * 100
            print(f"    - {col}: {n_nulos:,} nulos ({pct:.1f}%)")
    else:
        print("  - Nenhuma coluna com valores nulos!")
    
    # Análise de estatísticas básicas
    print("\nEstatísticas básicas - Temperatura:")
    if colunas_temp:
        for col in colunas_temp:
            print(f"  - {col}: mean={df[col].mean():.2f}, std={df[col].std():.2f}, min={df[col].min():.2f}, max={df[col].max():.2f}")
    
    print("\nEstatísticas básicas - Vibração:")
    if colunas_vibracao:
        for col in colunas_vibracao[:5]:  # Primeiras 5
            print(f"  - {col}: mean={df[col].mean():.4f}, std={df[col].std():.4f}, min={df[col].min():.4f}, max={df[col].max():.4f}")
    
    return colunas_numericas

def preparar_dados_para_normalizacao(df, colunas_numericas):
    """Prepara dados para normalização, mantendo timestamp intacto"""
    print("\nPreparando dados para normalização...")
    
    # Remover colunas com muitos valores nulos (>50%) - mais conservador
    colunas_validas = []
    colunas_removidas = []
    
    for col in colunas_numericas:
        if col != 'time':
            pct_nulos = (df[col].isnull().sum() / len(df)) * 100
            if pct_nulos < 50:  # Manter colunas com menos de 50% nulos
                colunas_validas.append(col)
            else:
                colunas_removidas.append((col, pct_nulos))
    
    print(f"  - Colunas selecionadas: {len(colunas_validas)}")
    print(f"  - Colunas removidas: {len(colunas_removidas)}")
    
    if colunas_removidas:
        print("  - Colunas removidas (com muitos nulos):")
        for col, pct in colunas_removidas[:10]:  # Mostrar apenas as primeiras 10
            print(f"    - {col}: {pct:.1f}% nulos")
    
    # Criar dataset para normalização (incluindo timestamp)
    df_norm = df[['time'] + colunas_validas].copy()
    
    n_linhas_orig = len(df_norm)
    n_nulos_total = int(df_norm[colunas_validas].isnull().sum().sum())
    print(f"  - Valores nulos nas features (antes do pipeline): {n_nulos_total:,}")
    print(f"  - Linhas consideradas: {n_linhas_orig:,}")
    print(f"  - Timestamp mantido intacto")
    
    return df_norm, colunas_validas

def construir_pipeline_preprocessamento(
    scaler_tipo: str = 'minmax',
    feature_range=(0, 1),
    power: str = 'none',
    quantile: str = 'none',
    variance_threshold: float = 0.0,
    pca_components: int = 0,
    pca_variance: float = 0.0
):
    """Constrói pipeline: Imputer → (Power|Quantile opcional) → Scaler → VarianceThreshold → PCA opcional"""
    steps = []
    # 1) Imputer
    steps.append(('imputer', SimpleImputer(strategy='median')))
    # 2) Power/Quantile transform (mutuamente exclusivos)
    if quantile in ('normal', 'uniform'):
        steps.append(('quantile', QuantileTransformer(output_distribution=quantile, subsample=100000, random_state=42, copy=True)))
    elif power == 'yeo-johnson':
        steps.append(('power', PowerTransformer(method='yeo-johnson', standardize=False)))
    # 3) Scaler
    if scaler_tipo == 'minmax':
        steps.append(('scaler', MinMaxScaler(feature_range=feature_range)))
    elif scaler_tipo == 'standard':
        steps.append(('scaler', StandardScaler()))
    elif scaler_tipo == 'robust':
        steps.append(('scaler', RobustScaler()))
    else:
        steps.append(('scaler', MinMaxScaler(feature_range=feature_range)))
    # 4) VarianceThreshold
    steps.append(('var', VarianceThreshold(threshold=variance_threshold)))
    # 5) PCA (opcional)
    if pca_components > 0:
        steps.append(('pca', PCA(n_components=pca_components, random_state=42)));
    elif pca_variance > 0.0:
        steps.append(('pca', PCA(n_components=pca_variance, svd_solver='full', random_state=42)))
    pipeline = Pipeline(steps=steps)
    return pipeline

def clip_outliers(df_features, percentile=99.5):
    """Aplica clipping de outliers em TODAS as colunas usando percentis"""
    print(f"\nAplicando clipping de outliers em TODAS as colunas (percentil {percentile})...")
    
    df_clipped = df_features.copy()
    outliers_info = []
    colunas_processadas = 0
    
    for col in df_features.columns:
        # Calcular limites usando percentis (ignorando NaNs)
        lower = df_features[col].quantile((100 - percentile) / 100)
        upper = df_features[col].quantile(percentile / 100)
        
        # Contar outliers antes do clipping
        outliers_lower = (df_features[col] < lower).sum()
        outliers_upper = (df_features[col] > upper).sum()
        total_outliers = outliers_lower + outliers_upper
        
        # SEMPRE aplicar clipping em TODAS as colunas
        df_clipped[col] = df_features[col].clip(lower=lower, upper=upper)
        colunas_processadas += 1
        
        # Registrar informações para relatório
        outliers_info.append({
            'coluna': col,
            'outliers': total_outliers,
            'pct': (total_outliers / len(df_features)) * 100,
            'lower': lower,
            'upper': upper
        })
    
    print(f"  - Total de colunas processadas: {colunas_processadas}/{len(df_features.columns)}")
    print(f"  - Colunas com outliers detectados e removidos: {sum(1 for info in outliers_info if info['outliers'] > 0)}")
    
    return df_clipped

def normalizar_dados_maxmin(df_norm, colunas_validas, args=None):
    """Normaliza dados usando Pipeline sklearn, mantendo timestamp intacto"""
    
    # Separar timestamp das features numéricas
    timestamp = df_norm['time'].copy()
    df_features = df_norm[colunas_validas].copy()
    
    # MODO INTERVALO: Carregar pipeline existente e apenas TRANSFORMAR
    if args and args.intervalo_arquivo:
        print("\n[MODO INTERVALO] Carregando pipeline de normalização do treino...")
        
        try:
            pipeline_path = preprocess_pipeline_path(args.mpoint)
            if not pipeline_path.exists():
                raise FileNotFoundError(f"Pipeline não encontrado: {pipeline_path}")
            
            # Carregar pipeline treinado
            pipeline = pickle.load(open(pipeline_path, 'rb'))
            print(f"  - Pipeline carregado: {pipeline_path}")
            
            # Carregar info de normalização para saber quais colunas usar
            info_norm_path = info_normalizacao_path(args.mpoint)
            with open(info_norm_path, 'r') as f:
                info_norm = json.load(f)
                colunas_usadas_treino = info_norm['colunas_utilizadas_finais']
            
            # Garantir mesmas colunas que o treino
            colunas_faltando = set(colunas_usadas_treino) - set(df_features.columns)
            if colunas_faltando:
                print(f"  [AVISO] Colunas faltando (preenchidas com 0): {colunas_faltando}")
                for col in colunas_faltando:
                    df_features[col] = 0
            
            # Reordenar para corresponder ao treino
            df_features = df_features[colunas_usadas_treino]
            
            # APENAS TRANSFORMAR (não fit!)
            dados_normalizados = pipeline.transform(df_features)
            colunas_finais = colunas_usadas_treino
            
            print(f"  - Dados transformados usando pipeline do treino")
            print(f"  - Shape: {dados_normalizados.shape}")
            
        except Exception as e:
            print(f"  [ERRO] Falha ao carregar pipeline do treino: {e}")
            raise
    
    # MODO TREINO: Criar e treinar novo pipeline
    else:
        print("\n[MODO TREINO] Criando e treinando novo pipeline...")
        
        # NOVO: Aplicar clipping de outliers ANTES da normalização
        df_features = clip_outliers(df_features, percentile=99.5)
        
        # Construir e ajustar pipeline
        pipeline = construir_pipeline_preprocessamento(
            scaler_tipo=(args.scaler if args else 'minmax'),
            feature_range=(0, 1),
            power=(args.power if args else 'none'),
            quantile=(args.quantile if args else 'none'),
            variance_threshold=(args.variance_threshold if args else 0.0),
            pca_components=(args.pca_components if args else 0),
            pca_variance=(args.pca_variance if args else 0.0),
        )
        dados_normalizados = pipeline.fit_transform(df_features)
        
        # Identificar e manter nomes das colunas após remoção de variância zero
        var_selector = pipeline.named_steps['var']
        support_mask = var_selector.get_support()
        colunas_pos_var = [c for c, keep in zip(colunas_validas, support_mask) if keep]

        # Se PCA estiver ativo, substitui nomes por pca_*
        colunas_finais = colunas_pos_var
        if 'pca' in pipeline.named_steps:
            n_out = dados_normalizados.shape[1]
            colunas_finais = [f'pca_{i+1}' for i in range(n_out)]
        
        print("  - Dados normalizados com sucesso!")
        print(f"  - Shape features (após seleção): {dados_normalizados.shape}")
        print(f"  - Range: [{dados_normalizados.min():.6f}, {dados_normalizados.max():.6f}]")
        print(f"  - Média: {dados_normalizados.mean():.6f}")
        print(f"  - Desvio padrão: {dados_normalizados.std():.6f}")
        print(f"  - Features removidas (variância zero): {len(colunas_validas) - len(colunas_pos_var)}")
        print(f"  - Timestamp mantido separadamente: {len(timestamp)} registros")
    
    # Para compatibilidade, também expomos o scaler interno já ajustado
    scaler_ajustado = pipeline.named_steps['scaler']
    
    return dados_normalizados, scaler_ajustado, timestamp, pipeline, colunas_finais

def preparar_dados_kmeans(dados_normalizados, colunas_validas, timestamp, mpoint=None, intervalo_arquivo=None):
    """Prepara dados normalizados para K-means, incluindo timestamp"""
    print("\nPreparando dados para K-means...")
    
    # Criar DataFrame com features normalizadas
    df_features = pd.DataFrame(dados_normalizados, columns=colunas_validas)
    
    # NO MODO INTERVALO: Reordenar colunas para corresponder ao modelo treinado
    if intervalo_arquivo and mpoint:
        try:
            info_kmeans_file = info_kmeans_path(mpoint)
            if info_kmeans_file.exists():
                with open(info_kmeans_file, 'r') as f:
                    info_kmeans = json.load(f)
                    ordem_modelo = info_kmeans.get('colunas_utilizadas', [])
                    
                    if ordem_modelo:
                        # Reordenar para corresponder ao modelo
                        colunas_disponiveis = [col for col in ordem_modelo if col in df_features.columns]
                        df_features = df_features[colunas_disponiveis]
                        print(f"  - Colunas reordenadas para corresponder ao modelo treinado")
                        print(f"  - Ordem: {colunas_disponiveis[:5]}... (primeiras 5)")
        except Exception as e:
            print(f"  [AVISO] Não foi possível reordenar colunas: {e}")
    
    # Adicionar timestamp
    df_kmeans = pd.concat([timestamp.reset_index(drop=True), df_features], axis=1)
    
    print(f"  - Shape para K-means: {df_kmeans.shape}")
    print(f"  - Colunas: {len(df_kmeans.columns)}")
    print(f"  - Timestamp incluído: Sim")
    
    return df_kmeans

def criar_visualizacoes(dados_normalizados, colunas_validas, df_original, args):
    """Cria visualizações dos dados normalizados"""
    print("\nCriando visualizações...")
    
    # Selecionar algumas colunas para visualização (primeiras 20)
    colunas_viz = colunas_validas[:20]
    
    # Criar figura
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Distribuição antes da normalização
    df_original_viz = df_original[colunas_viz]
    axes[0,0].boxplot([df_original_viz[col].dropna() for col in colunas_viz[:10]], 
                      labels=colunas_viz[:10])
    axes[0,0].set_title('Distribuição Original (primeiras 10 colunas)')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # 2. Distribuição após normalização
    dados_viz = dados_normalizados[:, :10]
    axes[0,1].boxplot([dados_viz[:, i] for i in range(10)], 
                      labels=colunas_viz[:10])
    axes[0,1].set_title('Distribuição Normalizada (primeiras 10 colunas)')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # 3. Histograma de uma coluna específica (antes)
    col_exemplo = colunas_viz[0]
    axes[1,0].hist(df_original[col_exemplo].dropna(), bins=50, alpha=0.7, color='blue')
    axes[1,0].set_title(f'Histograma Original - {col_exemplo}')
    axes[1,0].set_xlabel('Valor')
    axes[1,0].set_ylabel('Frequência')
    
    # 4. Histograma de uma coluna específica (depois)
    col_idx = colunas_validas.index(col_exemplo)
    axes[1,1].hist(dados_normalizados[:, col_idx], bins=50, alpha=0.7, color='red')
    axes[1,1].set_title(f'Histograma Normalizado - {col_exemplo}')
    axes[1,1].set_xlabel('Valor Normalizado')
    axes[1,1].set_ylabel('Frequência')
    
    plt.tight_layout()
    caminho_plot = DIR_PLOTS / f'dados_normalizados_analise_{args.mpoint}_mecanico.png'
    plt.savefig(caminho_plot, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  - Visualizações salvas em: {caminho_plot}")
    return caminho_plot

def salvar_dados_e_modelos(dados_normalizados, scaler, colunas_validas, X_kmeans, pipeline, colunas_selecionadas, args=None, arquivos_origem=None, mpoint=None, intervalo_arquivo=None):
    """Salva dados normalizados e modelos para K-means"""
    print("\nSalvando dados normalizados e modelos...")

    if not mpoint:
        raise ValueError("mpoint deve ser informado para salvar artefatos")

    scaler_path = scaler_maxmin_path(mpoint, create=True)
    pipeline_path = preprocess_pipeline_path(mpoint, create=True)
    dados_numpy_path = normalized_numpy_path(mpoint, intervalo_arquivo)
    dados_csv_path = normalized_csv_path(mpoint, intervalo_arquivo)
    info_path = info_normalizacao_path(mpoint, create=True)

    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    dados_numpy_path.parent.mkdir(parents=True, exist_ok=True)

    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"  - Scaler salvo: {scaler_path}")

    with open(pipeline_path, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"  - Pipeline salvo: {pipeline_path}")

    np.save(dados_numpy_path, dados_normalizados)
    print(f"  - Dados normalizados completos: {dados_numpy_path}")

    X_kmeans.to_csv(dados_csv_path, index=False)
    print(f"  - Dados para K-means: {dados_csv_path}")
    
    # Salvar informações do processamento
    info_processamento = {
        'timestamp': datetime.now().isoformat(),
        'equipment_type': 'MECHANICAL',
        'arquivos_origem': arquivos_origem or [],
        'arquivo_origem': f'periodo_*_final_{mpoint}.csv',
        'colunas_utilizadas_iniciais': colunas_validas,
        'colunas_utilizadas_finais': colunas_selecionadas,
        'numero_colunas': len(colunas_validas),
        'numero_colunas_pos_variance_threshold': len(colunas_selecionadas),
        'numero_amostras': len(dados_normalizados),
        'shape_dados_normalizados': list(dados_normalizados.shape),
        'shape_kmeans': list(X_kmeans.shape),
        'range_normalizacao': [float(dados_normalizados.min()), float(dados_normalizados.max())],
        'media_normalizada': float(dados_normalizados.mean()),
        'desvio_padrao_normalizado': float(dados_normalizados.std()),
        'preprocessamento': {
            'imputer': 'SimpleImputer(strategy=median)',
            'power': (args.power if args else 'none'),
            'quantile': (args.quantile if args else 'none'),
            'scaler': (args.scaler if args else 'minmax'),
            'feature_range': [0, 1],
            'variance_threshold': (args.variance_threshold if args else 0.0),
            'pca_components': (args.pca_components if args else 0),
            'pca_variance': (args.pca_variance if args else 0.0)
        },
        'colunas_auxiliares_removidas': True,
        'observacao': f'Equipamento MECÂNICO - Dados: temperatura + vibração (sem current, sem RPM)'
    }
    
    with open(info_path, 'w') as f:
        json.dump(info_processamento, f, indent=2)

    print(f"  - Informações do processamento: {info_path}")

def main():
    """Função principal"""
    print("=== NORMALIZAÇÃO DE DADOS PARA K-MEANS - EQUIPAMENTO MECÂNICO ===")
    print("=" * 70)
    print("Análise: Temperatura + Vibração (sem current, sem RPM)")
    print("=" * 70)
    
    try:
        # Parse de argumentos
        parser = argparse.ArgumentParser(description='Normalização de dados para K-means - EQUIPAMENTO MECÂNICO')
        parser.add_argument('--mpoint', type=str, required=True, help='ID do mpoint (ex: c_640)')
        parser.add_argument('--scaler', choices=['minmax', 'standard', 'robust'], default='minmax', help='Tipo de scaler a utilizar')
        parser.add_argument('--power', choices=['none', 'yeo-johnson'], default='none', help='Transformação de potência (antes do scaler)')
        parser.add_argument('--quantile', choices=['none', 'normal', 'uniform'], default='none', help='QuantileTransformer (incompatível com power)')
        parser.add_argument('--variance-threshold', type=float, default=0.0, help='Remover features com variância <= threshold')
        parser.add_argument('--pca-components', type=int, default=0, help='Número de componentes PCA (0 = desabilitado)')
        parser.add_argument('--pca-variance', type=float, default=0.0, help='Variância explicada alvo para PCA (0 = desabilitado)')
        parser.add_argument('--intervalo-arquivo', type=str, help='Intervalo formatado para incluir no nome dos arquivos')
        args = parser.parse_args()

        # 1. Criar diretórios
        criar_diretorios()
        
        # 2. Carregar dados unificados finais
        df, arquivos_origem = carregar_dados(mpoint=args.mpoint, intervalo_arquivo=args.intervalo_arquivo)
        if df is None:
            print("[ERRO] Falha ao carregar dados unificados finais")
            return
        
        # 3. Remover colunas auxiliares
        df_sem_aux = remover_colunas_auxiliares(df)
        
        # 4. Analisar dados mecânicos
        colunas_numericas = analisar_dados_mecanicos(df_sem_aux)
        
        # 5. Preparar dados para normalização
        df_norm, colunas_validas = preparar_dados_para_normalizacao(df_sem_aux, colunas_numericas)
        
        # 6. Normalizar dados
        dados_normalizados, scaler, timestamp, pipeline, colunas_selecionadas = normalizar_dados_maxmin(df_norm, colunas_validas, args=args)
        
        # 7. Preparar dados para K-means
        X_kmeans = preparar_dados_kmeans(dados_normalizados, colunas_selecionadas, timestamp, mpoint=args.mpoint, intervalo_arquivo=args.intervalo_arquivo)
        
        # 8. Criar visualizações
        caminho_plot = criar_visualizacoes(dados_normalizados, colunas_selecionadas, df_sem_aux, args)
        
        # 9. Salvar dados e modelos
        salvar_dados_e_modelos(
            dados_normalizados,
            scaler,
            colunas_validas,
            X_kmeans,
            pipeline,
            colunas_selecionadas,
            args=args,
            arquivos_origem=arquivos_origem,
            mpoint=args.mpoint,
            intervalo_arquivo=args.intervalo_arquivo
        )
        
        print("\n=== PROCESSO CONCLUÍDO COM SUCESSO ===")
        print("\nDados preparados para K-means:")
        print(f"  - Tipo: EQUIPAMENTO MECÂNICO (temperatura + vibração)")
        print(f"  - Arquivo principal: {normalized_csv_path(args.mpoint, args.intervalo_arquivo)}")
        print(f"  - Dados normalizados: {normalized_numpy_path(args.mpoint, args.intervalo_arquivo)}")

        # Gerar logs
        import time
        start_time = time.time()

        processing_log = create_processing_log(
            script_name='normalizar_dados_kmeans_mecanico',
            mpoint=args.mpoint,
            operation='mechanical_equipment_data_normalization',
            input_files=arquivos_origem,
            output_files=[
                str(normalized_csv_path(args.mpoint, args.intervalo_arquivo)),
                str(normalized_numpy_path(args.mpoint, args.intervalo_arquivo)),
                str(scaler_maxmin_path(args.mpoint)),
                str(info_normalizacao_path(args.mpoint)),
                str(preprocess_pipeline_path(args.mpoint)),
                str(caminho_plot)
            ],
            parameters={
                'equipment_type': 'MECHANICAL',
                'data_focus': 'temperature_and_vibration',
                'no_current_rpm': True,
                'scaler_type': args.scaler,
                'feature_range': [0, 1],
                'outlier_clipping_percentile': 99.5
            },
            statistics={
                'total_samples': len(dados_normalizados),
                'original_features': len(colunas_validas),
                'final_features': len(colunas_selecionadas),
                'normalization_range': [float(dados_normalizados.min()), float(dados_normalizados.max())],
            },
            processing_time=time.time() - start_time,
            success=True,
            data_description={
                'equipment_type': 'MECHANICAL',
                'source_files': arquivos_origem,
                'features_used': colunas_selecionadas,
                'normalization_method': 'MinMaxScaler with outlier clipping'
            }
        )

        save_log(processing_log, 'normalizar_dados_kmeans_mecanico', args.mpoint, 'normalization_complete')

        results_data = {
            'normalization_completed': True,
            'normalization_timestamp': datetime.now().isoformat(),
            'equipment_type': 'MECHANICAL',
            'normalized_samples': len(dados_normalizados),
            'normalized_features': len(colunas_selecionadas),
        }

        enrich_results_file(args.mpoint, results_data)

    except Exception as e:
        print(f"\nErro durante o processamento: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

