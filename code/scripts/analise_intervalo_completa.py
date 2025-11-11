"""
Script pra fazer analise completa de um periodo especifico.
Baixa dados do banco (InfluxDB), processa tudo e classifica se ta ligado/desligado.
Usa os scripts que ja existem pra fazer o trabalho pesado.
"""

import pandas as pd
import numpy as np
import joblib
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys
import subprocess

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from utils.artifact_paths import (
    config_path,
    info_kmeans_path,
    kmeans_model_path,
    results_dir,
    scaler_model_path,
)
from utils.logging_utils import (
    save_log,
    create_analysis_log,
    create_visualization_log,
    format_file_list,
    get_file_info,
    enrich_results_file,
)

warnings.filterwarnings('ignore')

class AnalisadorIntervaloCompleto:
    def __init__(self, mpoint, influx_ip, data_inicio, data_fim):
        """
        Inicializa analisador completo para intervalo

        Args:
            mpoint (str): ID do mpoint
            influx_ip (str): IP do InfluxDB
            data_inicio (str): Data/hora inicial (GMT-3)
            data_fim (str): Data/hora final (GMT-3)
        """
        self.mpoint = mpoint
        self.influx_ip = influx_ip

        # Verificar intervalo mínimo de 24 horas
        from datetime import datetime
        dt_inicio = datetime.strptime(data_inicio, '%Y-%m-%d %H:%M:%S')
        dt_fim = datetime.strptime(data_fim, '%Y-%m-%d %H:%M:%S')
        duracao_horas = (dt_fim - dt_inicio).total_seconds() / 3600
        
        if duracao_horas < 24:
            raise ValueError(f"Intervalo muito curto ({duracao_horas:.1f}h). Mínimo necessário: 24 horas (1 dia)")

        # Formatar intervalo para incluir nos nomes dos arquivos
        # Exemplo: 06-11-25_12;00;00_06-11-25_15;00;00
        data_inicio_fmt = data_inicio.replace("-", "").replace(" ", "_").replace(":", ";")
        data_fim_fmt = data_fim.replace("-", "").replace(" ", "_").replace(":", ";")
        self.intervalo_arquivo = f"{data_inicio_fmt}_{data_fim_fmt}"

        # Converter datas de GMT-3 para UTC para uso no InfluxDB
        self.data_inicio_utc = self.converter_para_utc(data_inicio)
        self.data_fim_utc = self.converter_para_utc(data_fim)

        # Manter datas originais para exibio
        self.data_inicio = data_inicio
        self.data_fim = data_fim

        # Caminhos base (usar pastas padrão)
        self.base_dir = Path(__file__).parent.parent
        self.dir_data_raw = self.base_dir / 'data' / 'raw'
        self.dir_data_raw_preenchido = self.base_dir / 'data' / 'raw_preenchido'
        self.dir_data_processed = self.base_dir / 'data' / 'processed'

        self.dir_results = results_dir(mpoint, create=True)

        # Modelos e parmetros
        self.kmeans = None
        self.scaler = None
        self.feature_columns = None
        self.centroides_info = None
        self.clusters_desligado = []
        self.clusters_ligado = []

        # Verificar se parmetros existem
        if not self.verificar_parametros():
            raise FileNotFoundError(f"Parmetros no encontrados para mpoint {mpoint}")

        # Carregar modelos
        self.carregar_modelos()

    def converter_para_utc(self, data_str):
        """
        Converte data/hora de GMT-3 para UTC
        """
        from datetime import datetime, timezone, timedelta

        try:
            dt_local = datetime.strptime(data_str, '%Y-%m-%d %H:%M:%S')
            tz_local = timezone(timedelta(hours=-3))
            dt_com_tz = dt_local.replace(tzinfo=tz_local)
            dt_utc = dt_com_tz.astimezone(timezone.utc)
            return dt_utc.strftime('%Y-%m-%dT%H:%M:%SZ')
        except Exception as e:
            print(f"[ERRO] Erro ao converter data '{data_str}': {e}")
            raise

    def verificar_parametros(self):
        """Verifica se os parmetros treinados existem"""
        required_paths = [
            kmeans_model_path(self.mpoint),
            scaler_model_path(self.mpoint),
            info_kmeans_path(self.mpoint),
            config_path(self.mpoint),
        ]

        for path in required_paths:
            if not path.exists():
                return False

        return True

    def carregar_modelos(self):
        """Carrega scaler e informações dos modelos treinados (não carrega K-means)"""
        print("[INFO] Carregando scaler e informações dos modelos...")

        try:
            scaler_path = scaler_model_path(self.mpoint)
            info_path = info_kmeans_path(self.mpoint)

            if not scaler_path.exists():
                raise FileNotFoundError(f"Arquivo scaler não encontrado: {scaler_path}")
            if not info_path.exists():
                raise FileNotFoundError(f"Arquivo de informações não encontrado: {info_path}")

            # Não carrega mais o modelo K-means
            self.kmeans = None
            self.scaler = joblib.load(scaler_path)

            with open(info_path, 'r') as f:
                info = json.load(f)
                self.feature_columns = info.get('colunas_utilizadas', []) or info.get('colunas_utilizadas_finais', [])

            print(f"  - Features: {len(self.feature_columns)}")
            print("  - Scaler e informações carregados com sucesso!")

        except Exception as e:
            print(f"[ERRO] Erro ao carregar scaler/informações: {str(e)}")
            raise

        # Carregar thresholds dinâmicos do estado desligado (mais importante agora)
        self.carregar_thresholds_desligado()

    def carregar_thresholds_desligado(self):
        """Carrega os thresholds dinâmicos REAIS (não normalizados) do estado desligado"""
        try:
            info_path = self.base_dir / 'models' / self.mpoint / f'info_kmeans_model_moderado_{self.mpoint}.json'

            if info_path.exists():
                with open(info_path, 'r') as f:
                    info_kmeans = json.load(f)

                # Carregar thresholds se existirem
                if 'thresholds_desligado' in info_kmeans:
                    # Thresholds JÁ VÊM COMO VALORES REAIS (não normalizados)
                    self.thresholds_desligado = info_kmeans['thresholds_desligado']
                    print("  [OK] Thresholds REAIS carregados (valores não normalizados):")
                    print(f"    - {len(self.thresholds_desligado)} parametros disponiveis")
                    # Mostrar thresholds importantes com unidades
                    unidades = {'vel_rms_max': 'mm/s', 'current_max': 'A', 'rpm_max': 'RPM'}
                    for key in ['vel_rms_max', 'current_max', 'rpm_max']:
                        if key in self.thresholds_desligado:
                            unidade = unidades.get(key, '')
                            print(f"    - {key}: {self.thresholds_desligado[key]:.3f} {unidade}")
                else:
                    self.thresholds_desligado = {}
                    print("  [AVISO] Thresholds dinamicos nao encontrados no modelo")
            else:
                self.thresholds_desligado = {}
                print("  [AVISO] Arquivo de info do modelo nao encontrado")

        except Exception as e:
            print(f"  [AVISO] Erro ao carregar thresholds dinamicos: {e}")
            self.thresholds_desligado = {}

    def _normalizar_thresholds(self, thresholds_originais):
        """Normaliza os thresholds originais usando o scaler carregado"""
        try:
            import pandas as pd

            # Mapeamento de nomes dos thresholds para nomes das features
            mapeamento_thresholds = {
                'vel_rms_max': 'vel_rms',
                'current_max': 'current',
                'rpm_max': 'rotational_speed'
            }

            # Criar um DataFrame com uma linha dummy para todas as features
            dummy_data = {}
            for col in self.feature_columns:
                dummy_data[col] = [0.0]  # Valores dummy, só para estrutura

            # Adicionar os valores reais dos thresholds que queremos normalizar
            for threshold_key, feature_key in mapeamento_thresholds.items():
                if threshold_key in thresholds_originais and feature_key in self.feature_columns:
                    dummy_data[feature_key] = [thresholds_originais[threshold_key]]
                    print(f"    Mapeando {threshold_key} ({thresholds_originais[threshold_key]:.3f}) -> {feature_key}")

            df_dummy = pd.DataFrame(dummy_data)

            # Aplicar o scaler (transform normaliza baseado nos parâmetros aprendidos)
            dados_normalizados = self.scaler.transform(df_dummy)

            # Criar dicionário com valores normalizados
            thresholds_normalizados = {}
            for i, col in enumerate(self.feature_columns):
                if col in dummy_data and dummy_data[col][0] != 0.0:  # Se foi modificado
                    thresholds_normalizados[col] = dados_normalizados[0, i]
                    print(f"    {col}: normalizado para {thresholds_normalizados[col]:.3f}")

            # Adicionar thresholds que não são features (como contagens, percentuais, etc.)
            for key, value in thresholds_originais.items():
                if key not in mapeamento_thresholds:  # Não é um threshold que foi normalizado
                    thresholds_normalizados[key] = value

            return thresholds_normalizados

        except Exception as e:
            print(f"  [ERRO] Falha ao normalizar thresholds: {e}")
            import traceback
            traceback.print_exc()
            # Retornar thresholds originais como fallback
            return thresholds_originais

    def analisar_centroides(self):
        """Define clusters simplificado: 0=DESLIGADO, 1=LIGADO (usando apenas thresholds)"""
        print("  [INFO] Usando classificação simplificada baseada em thresholds:")
        print("    DESLIGADO: cluster 0 (valores dentro dos thresholds)")
        print("    LIGADO:    cluster 1 (valores acima dos thresholds)")

        self.clusters_desligado = [0]
        self.clusters_ligado = [1]
    def validar_rotational_speed(self):
        """Valida se há pelo menos 70% de rotational_speed e current preenchidos (baseado em TEMPO)
        
        IMPORTANTE: Valores 0.0 são VÁLIDOS (representam equipamento desligado).
                   Apenas valores vazios (None/NaN) por longos períodos são inválidos.
        """
        print("\n[VALIDAÇÃO] Verificando dados de rotational_speed e current...")
        print("  NOTA: Valores 0.0 são considerados VÁLIDOS (equipamento desligado)")
        
        # Carregar arquivo estimated baixado
        arquivo_estimated = self.dir_data_raw / f'dados_estimated_{self.mpoint}_{self.intervalo_arquivo}.csv'
        
        if not arquivo_estimated.exists():
            raise FileNotFoundError(f"Arquivo estimated não encontrado: {arquivo_estimated}")
        
        df = pd.read_csv(arquivo_estimated)
        
        # Converter time para datetime (format='mixed' para lidar com formatos inconsistentes)
        df['time'] = pd.to_datetime(df['time'], format='mixed', utc=True)
        df = df.sort_values('time').reset_index(drop=True)
        
        # Calcular intervalo total em horas
        tempo_total_horas = (df['time'].max() - df['time'].min()).total_seconds() / 3600
        total_registros = len(df)
        
        erros = []
        
        # ========== VALIDAR ROTATIONAL_SPEED ==========
        if 'rotational_speed' not in df.columns:
            erros.append("Coluna 'rotational_speed' não encontrada no arquivo estimated")
        else:
            # Filtrar valores PREENCHIDOS (não-nulos) - INCLUINDO zeros
            df_rpm_preenchido = df[df['rotational_speed'].notna()].copy()
            
            if len(df_rpm_preenchido) == 0:
                erros.append("Nenhum dado de rotational_speed encontrado (todos valores são nulos/vazios)")
            else:
                # Calcular cobertura temporal dos dados PREENCHIDOS
                tempo_rpm_coberto = (df_rpm_preenchido['time'].max() - df_rpm_preenchido['time'].min()).total_seconds() / 3600
                pct_rpm_tempo = (tempo_rpm_coberto / tempo_total_horas) * 100 if tempo_total_horas > 0 else 0
                pct_rpm_registros = (len(df_rpm_preenchido) / total_registros) * 100 if total_registros > 0 else 0
                
                # Estatísticas dos valores (incluindo zeros)
                rpm_mean = df_rpm_preenchido['rotational_speed'].mean()
                rpm_max = df_rpm_preenchido['rotational_speed'].max()
                rpm_min = df_rpm_preenchido['rotational_speed'].min()
                
                # Contar quantos valores são zero vs não-zero
                rpm_zeros = (df_rpm_preenchido['rotational_speed'] == 0).sum()
                rpm_nao_zeros = (df_rpm_preenchido['rotational_speed'] != 0).sum()
                pct_zeros = (rpm_zeros / len(df_rpm_preenchido)) * 100 if len(df_rpm_preenchido) > 0 else 0
                
                print(f"\n  [rotational_speed]")
                print(f"    Período coberto: {tempo_rpm_coberto:.1f}h de {tempo_total_horas:.1f}h")
                print(f"    Percentual de TEMPO: {pct_rpm_tempo:.1f}%")
                print(f"    Registros preenchidos: {len(df_rpm_preenchido):,} de {total_registros:,} ({pct_rpm_registros:.1f}%)")
                print(f"    Valores - Min: {rpm_min:.1f}, Média: {rpm_mean:.1f}, Máximo: {rpm_max:.1f} RPM")
                print(f"    Distribuição - Zeros: {rpm_zeros:,} ({pct_zeros:.1f}%), Não-zeros: {rpm_nao_zeros:,} ({100-pct_zeros:.1f}%)")
                
                # Validação: pelo menos 70% de cobertura temporal com dados preenchidos
                if pct_rpm_tempo < 70.0:
                    erros.append(f"rotational_speed: cobertura temporal insuficiente ({pct_rpm_tempo:.1f}% < 70%)")
                else:
                    print(f"    [OK] Cobertura temporal adequada")
        
        # ========== VALIDAR CURRENT ==========
        if 'current' not in df.columns:
            erros.append("Coluna 'current' não encontrada no arquivo estimated")
        else:
            # Filtrar valores PREENCHIDOS (não-nulos) - INCLUINDO zeros
            df_current_preenchido = df[df['current'].notna()].copy()
            
            if len(df_current_preenchido) == 0:
                erros.append("Nenhum dado de current encontrado (todos valores são nulos/vazios)")
            else:
                # Calcular cobertura temporal dos dados PREENCHIDOS
                tempo_current_coberto = (df_current_preenchido['time'].max() - df_current_preenchido['time'].min()).total_seconds() / 3600
                pct_current_tempo = (tempo_current_coberto / tempo_total_horas) * 100 if tempo_total_horas > 0 else 0
                pct_current_registros = (len(df_current_preenchido) / total_registros) * 100 if total_registros > 0 else 0
                
                # Estatísticas dos valores (incluindo zeros)
                current_mean = df_current_preenchido['current'].mean()
                current_max = df_current_preenchido['current'].max()
                current_min = df_current_preenchido['current'].min()
                
                # Contar quantos valores são zero vs não-zero
                current_zeros = (df_current_preenchido['current'] == 0).sum()
                current_nao_zeros = (df_current_preenchido['current'] != 0).sum()
                pct_zeros = (current_zeros / len(df_current_preenchido)) * 100 if len(df_current_preenchido) > 0 else 0
                
                print(f"\n  [current]")
                print(f"    Período coberto: {tempo_current_coberto:.1f}h de {tempo_total_horas:.1f}h")
                print(f"    Percentual de TEMPO: {pct_current_tempo:.1f}%")
                print(f"    Registros preenchidos: {len(df_current_preenchido):,} de {total_registros:,} ({pct_current_registros:.1f}%)")
                print(f"    Valores - Min: {current_min:.1f}, Média: {current_mean:.1f}, Máximo: {current_max:.1f} A")
                print(f"    Distribuição - Zeros: {current_zeros:,} ({pct_zeros:.1f}%), Não-zeros: {current_nao_zeros:,} ({100-pct_zeros:.1f}%)")
                
                # Validação: pelo menos 70% de cobertura temporal com dados preenchidos
                if pct_current_tempo < 70.0:
                    erros.append(f"current: cobertura temporal insuficiente ({pct_current_tempo:.1f}% < 70%)")
                else:
                    print(f"    [OK] Cobertura temporal adequada")
        
        # Se houver erros, lançar exceção
        if erros:
            raise ValueError(
                f"VALIDAÇÃO FALHOU - Dados insuficientes:\n" + 
                "\n".join(f"  - {e}" for e in erros) + 
                f"\n\n⚠️  MOTIVO: O intervalo selecionado não possui cobertura temporal suficiente (>70%)."
                f"\n⚠️  NOTA: Valores 0.0 são válidos - o problema são valores vazios (None/NaN) por longos períodos."
                f"\n⚠️  IMPACTO: Sem cobertura adequada, o modelo não pode classificar os estados corretamente."
                f"\n\n💡 SOLUÇÃO: Selecione um intervalo com dados mais completos no InfluxDB."
            )
        
        print("\n  [OK] Validação OK - cobertura temporal suficiente de rotational_speed e current")
        return True

    def baixar_dados_influx(self):
        """Baixa dados do InfluxDB usando scripts existentes modificados"""
        print("[DOWNLOAD] Baixando dados do InfluxDB...")

        # Scripts de download modificados
        scripts_download = [
            ('baixar_estimated_intervalo.py', 'estimated'),
            ('baixar_validated_default_intervalo.py', 'validated_default'),
            ('baixar_validated_slip_intervalo.py', 'validated_slip')
        ]

        for script_nome, tabela in scripts_download:
            print(f"  - Baixando dados de {tabela}...")

            script_path = self.base_dir / 'scripts' / script_nome
            if not script_path.exists():
                print(f"     Script {script_nome} não encontrado")
                continue

            try:
                cmd = [
                    sys.executable, str(script_path),
                    '--mpoint', self.mpoint,
                    '--ip', self.influx_ip,
                    '--inicio', self.data_inicio_utc,
                    '--fim', self.data_fim_utc,
                    '--intervalo-arquivo', self.intervalo_arquivo
                ]

                print(f"    - Executando: {' '.join(cmd)}")
                result = subprocess.run(cmd, cwd=self.base_dir, capture_output=True, text=True, encoding='utf-8')

                if result.returncode == 0:
                    print("     Download concluído com sucesso")
                    
                    # VALIDAR rotational_speed logo após baixar estimated
                    if tabela == 'estimated':
                        self.validar_rotational_speed()
                else:
                    print("     Erro no download:")
                    print(result.stderr)

                    # TRATAR ERROS DE CONEXÃO
                    erro_conexao = self._verificar_erro_conexao(result.stderr)
                    if erro_conexao:
                        self._tratar_erro_conexao()

            except Exception as e:
                print(f"     Erro ao executar download: {e}")

                # TRATAR ERROS DE CONEXÃO
                erro_conexao = self._verificar_erro_conexao(str(e))
                if erro_conexao:
                    self._tratar_erro_conexao()
                else:
                    raise

    def processar_dados(self):
        """Processa dados baixados usando scripts existentes"""
        print(" Processando dados baixados...")

        scripts_processamento = [
            ('segmentar_preencher_dados.py', 'Segmentação inicial'),
            ('processar_dados_simples.py', 'Processamento e interpolação'),
            ('unir_sincronizar_periodos.py', 'União e sincronização'),
            ('normalizar_dados_kmeans.py', 'Normalização para K-means')
        ]

        for script_nome, descricao in scripts_processamento:
            print(f"  - {descricao}...")

            script_path = self.base_dir / 'scripts' / script_nome
            if not script_path.exists():
                print(f"     Script {script_nome} não encontrado")
                continue

            try:
                cmd = [sys.executable, str(script_path), '--mpoint', self.mpoint]

                # Passar o intervalo para incluir no nome dos arquivos
                cmd.extend(['--intervalo-arquivo', self.intervalo_arquivo])

                result = subprocess.run(cmd, cwd=self.base_dir, capture_output=True, text=True, encoding='utf-8')

                if result.returncode == 0:
                    print("     Processamento concluído")
                else:
                    print("     Erro no processamento:")
                    print(result.stderr)
                    # Se a normalização falhar, isso é crítico
                    if script_nome == 'normalizar_dados_kmeans.py':
                        raise RuntimeError(f"Falha na normalização: {result.stderr}")

            except Exception as e:
                print(f"     Erro ao executar processamento: {e}")
                raise

    def carregar_dados_processados(self):
        """Carrega dados processados finais desta execução específica"""
        print(" Carregando dados processados...")

        # Usar a pasta específica desta execução
        dir_preenchido = self.dir_data_raw_preenchido

        # MODO INTERVALO: Procurar arquivos finais com intervalo no nome
        arquivos_finais = list(dir_preenchido.glob(f'periodo_*_final_{self.mpoint}_{self.intervalo_arquivo}.csv'))

        if not arquivos_finais:
            raise FileNotFoundError(f"Nenhum arquivo periodo_*_final_{self.mpoint}_{self.intervalo_arquivo}.csv encontrado nesta execução")

        print(f"  - Encontrados {len(arquivos_finais)} arquivo(s) desta execução")
        arquivos_a_carregar = arquivos_finais

        # Carregar e concatenar arquivos
        dfs = []
        for arquivo in sorted(arquivos_a_carregar):
            print(f"  - Carregando: {arquivo.name}")
            df = pd.read_csv(arquivo)
            df['time'] = pd.to_datetime(df['time'], format='mixed')
            dfs.append(df)

        # Concatenar todos os DataFrames
        if len(dfs) == 1:
            df_final = dfs[0]
        else:
            df_final = pd.concat(dfs, ignore_index=True)
            # Ordenar por tempo e remover duplicatas se houver
            df_final = df_final.sort_values('time').drop_duplicates(subset='time').reset_index(drop=True)

        print(f"   {len(df_final)} registros carregados (de {len(arquivos_finais)} arquivo(s))")
        
        # FILTRAR APENAS DADOS DO INTERVALO SOLICITADO
        # Usar as datas já convertidas para UTC (data_inicio_utc e data_fim_utc são strings ISO)
        data_inicio_dt = pd.to_datetime(self.data_inicio_utc, format='%Y-%m-%dT%H:%M:%SZ', utc=True)
        data_fim_dt = pd.to_datetime(self.data_fim_utc, format='%Y-%m-%dT%H:%M:%SZ', utc=True)
        
        # Garantir que a coluna 'time' está em datetime com UTC
        df_final['time'] = pd.to_datetime(df_final['time'], format='mixed', utc=True)
        
        # Filtrar pelo intervalo
        mask = (df_final['time'] >= data_inicio_dt) & (df_final['time'] <= data_fim_dt)
        df_filtrado = df_final[mask].copy()
        
        print(f"   {len(df_filtrado)} registros no intervalo solicitado ({self.data_inicio} até {self.data_fim})")
        
        if len(df_filtrado) == 0:
            raise ValueError(f"Nenhum dado encontrado no intervalo {self.data_inicio} até {self.data_fim}")
        
        return df_filtrado

    def gerar_tabela_normalizada(self, df):
        """Gera tabela com 19 features normalizadas"""
        print(" Gerando tabela normalizada...")

        colunas_disponiveis = [col for col in self.feature_columns if col in df.columns]

        if len(colunas_disponiveis) != len(self.feature_columns):
            faltando = set(self.feature_columns) - set(colunas_disponiveis)
            print(f" Colunas faltando: {faltando}")

        X = df[colunas_disponiveis].values
        X_norm = self.scaler.transform(X)

        df_normalizado = pd.DataFrame(X_norm, columns=colunas_disponiveis, index=df.index)
        df_normalizado.insert(0, 'time', df['time'])

        print(f" Tabela normalizada: {len(df_normalizado)} registros  {len(colunas_disponiveis)} features")
        return df_normalizado, colunas_disponiveis

    def verificar_dados_desligamento_suficientes(self, df_normalizado, colunas_features):
        """
        Verifica se há dados suficientes de estado desligado no intervalo atual
        baseados nos thresholds dinâmicos treinados
        """
        if not hasattr(self, 'thresholds_desligado') or not self.thresholds_desligado:
            print("  Thresholds dinamicos nao disponiveis, assumindo dados insuficientes")
            return False

        # Identificar colunas relevantes
        vel_rms_cols = [col for col in colunas_features if 'vel_rms' in col.lower()]
        current_cols = [col for col in colunas_features if 'current' in col.lower()]
        rpm_cols = [col for col in colunas_features if 'rpm' in col.lower() or 'rotational_speed' in col.lower()]

        # Contar amostras que se enquadram nos critérios de desligado
        criterios_desligado = []

        # Critério 1: vel_rms abaixo do threshold máximo observado no treino
        if vel_rms_cols and 'vel_rms_max' in self.thresholds_desligado:
            vel_rms_baixo = df_normalizado[vel_rms_cols].max(axis=1) <= self.thresholds_desligado['vel_rms_max']
            criterios_desligado.append(vel_rms_baixo)

        # Critério 2: current abaixo do threshold máximo observado no treino
        if current_cols and 'current_max' in self.thresholds_desligado:
            current_baixo = df_normalizado[current_cols].max(axis=1) <= self.thresholds_desligado['current_max']
            criterios_desligado.append(current_baixo)

        # Critério 3: RPM próximo ao mínimo (próximo de 0 ou abaixo do threshold)
        if rpm_cols and 'rpm_max' in self.thresholds_desligado:
            rpm_baixo = df_normalizado[rpm_cols].max(axis=1) <= self.thresholds_desligado['rpm_max']
            criterios_desligado.append(rpm_baixo)

        # Se temos pelo menos 2 critérios, usar combinação
        if len(criterios_desligado) >= 2:
            # Pelo menos 2 critérios devem ser atendidos
            criterios_combinados = np.column_stack(criterios_desligado)
            amostras_desligado = np.sum(np.sum(criterios_combinados, axis=1) >= 2)
        elif len(criterios_desligado) == 1:
            amostras_desligado = np.sum(criterios_desligado[0])
        else:
            amostras_desligado = 0

        pct_desligado = amostras_desligado / len(df_normalizado) * 100

        # Threshold mínimo: pelo menos 5% dos dados devem indicar estado desligado
        # ou pelo menos 1000 amostras (o que for menor)
        minimo_pct = 5.0
        minimo_absoluto = min(1000, len(df_normalizado) * 0.05)

        dados_suficientes = pct_desligado >= minimo_pct and amostras_desligado >= minimo_absoluto

        print("  Analise de dados desligado no intervalo:")
        print(f"    - Amostras identificadas como DESLIGADO: {amostras_desligado:,} ({pct_desligado:.1f}%)")
        print(f"    - Threshold minimo: {minimo_pct:.1f}% ou {minimo_absoluto:.0f} amostras")
        print(f"    - Dados suficientes: {'SIM' if dados_suficientes else 'NAO'}")

        return dados_suficientes

    def classificar_estados(self, df_normalizado, colunas_features):
        """Classifica estados usando thresholds REAIS do estado desligado"""
        print(" Classificando estados usando thresholds REAIS (nao normalizados)...")

        # Carregar dados originais para verificar valores cruciais
        arquivo_final = self.base_dir / 'data' / 'raw_preenchido' / f'periodo_01_final_{self.mpoint}_{self.intervalo_arquivo}.csv'
        df_original = None

        if arquivo_final.exists():
            df_original = pd.read_csv(arquivo_final)
            df_original['time'] = pd.to_datetime(df_original['time'], format='mixed', utc=True)

            # Merge para ter valores originais junto com normalizados
            df_completo = pd.merge(
                df_normalizado,
                df_original[['time', 'rotational_speed', 'current', 'vel_rms']],
                on='time',
                how='left',
                suffixes=('_norm', '')
            )
        else:
            print("  [ERRO] Arquivo original não encontrado - não é possível classificar sem dados reais")
            return None

        estados = []
        clusters = []

        # Thresholds REAIS do estado desligado (JÁ VÊM COM MARGEM DE SEGURANÇA)
        if hasattr(self, 'thresholds_desligado') and self.thresholds_desligado:
            # Thresholds já incluem margem de segurança (vel_rms * 1.2, current * 1.3, rpm * 1.1)
            vel_rms_max = self.thresholds_desligado.get('vel_rms_max', 2.0)  # mm/s
            current_max = self.thresholds_desligado.get('current_max', 60.0)  # A
            rpm_max = self.thresholds_desligado.get('rpm_max', 100.0)  # RPM

            print(f"  Usando thresholds REAIS do estado desligado (ja com margem):")
            print(f"    - vel_rms_max: {vel_rms_max:.3f} mm/s")
            print(f"    - current_max: {current_max:.3f} A")
            print(f"    - rpm_max: {rpm_max:.3f} RPM")

            for idx, row in df_completo.iterrows():
                # Valores REAIS (originais, não normalizados)
                current_real = row.get('current', 0)
                rpm_real = row.get('rotational_speed', 0)
                vel_rms_real = row.get('vel_rms', 0)

                # LÓGICA DE CLASSIFICAÇÃO BASEADA EM THRESHOLDS REAIS
                # PRIORIDADE: current e rpm (mais confiáveis que vibração)
                
                is_desligado = False
                
                # Critério OBRIGATÓRIO 1: Current E RPM abaixo dos thresholds
                if current_real <= current_max and rpm_real <= rpm_max:
                    # Current e RPM baixos - possível DESLIGADO
                    # Verificar vibração para confirmar
                    if vel_rms_real <= vel_rms_max:
                        # Tudo abaixo dos thresholds = DESLIGADO
                        is_desligado = True
                    else:
                        # Vibração acima do threshold mas current/rpm baixos
                        # Pode ser vibração residual OU operação anormal
                        # Usar threshold de vibração mais alto (3.0 mm/s) para decisão final
                        if vel_rms_real <= 3.0:
                            # Vibração residual aceitável
                            is_desligado = True
                        else:
                            # Vibração muito alta - considerar LIGADO
                            is_desligado = False
                else:
                    # Current OU RPM acima dos thresholds = LIGADO
                    is_desligado = False

                if is_desligado:
                    estados.append('DESLIGADO')
                    clusters.append(0)  # Cluster 0 sempre representa desligado
                else:
                    estados.append('LIGADO')
                    clusters.append(1)  # Cluster 1 representa ligado
        else:
            print("  [ERRO] Thresholds do estado desligado não disponíveis!")
            # Fallback: tudo LIGADO
            estados = ['LIGADO'] * len(df_completo)
            clusters = [1] * len(df_completo)

        df_resultados = df_normalizado.copy()
        df_resultados['cluster'] = clusters
        df_resultados['estado'] = estados
        df_resultados['equipamento_status'] = estados  # Compatibilidade com outros scripts

        distribuicao = df_resultados['estado'].value_counts().to_dict()
        print(" Distribuicao (usando thresholds do estado desligado):")
        for estado, count in distribuicao.items():
            pct = count / len(df_resultados) * 100
            print(f"    {estado}: {count} ({pct:.1f}%)")

        return df_resultados
    
    def corrigir_estados_por_threshold(self, df_resultados, rpm_threshold=1000, current_threshold=600):
        """
        Corrige estados baseado em thresholds físicos.
        Se RPM > rpm_threshold OU current > current_threshold, força estado = LIGADO
        
        Isso resolve o problema de K-means classificar dados de equipamento ligado
        como DESLIGADO quando o intervalo não contém dados reais de desligamento.
        """
        print(f"\n Aplicando correção por thresholds físicos (RPM > {rpm_threshold} ou Current > {current_threshold})...")
        
        # Carregar dados originais (desnormalizados) para verificar thresholds
        arquivo_final = self.base_dir / 'data' / 'raw_preenchido' / f'periodo_01_final_{self.mpoint}_{self.intervalo_arquivo}.csv'
        
        if not arquivo_final.exists():
            print(f"  [AVISO] Arquivo {arquivo_final.name} não encontrado, pulando correção por threshold")
            return df_resultados
        
        df_original = pd.read_csv(arquivo_final)
        df_original['time'] = pd.to_datetime(df_original['time'], format='mixed', utc=True)
        
        # Merge para pegar valores originais
        df_merged = pd.merge(
            df_resultados,
            df_original[['time', 'rotational_speed', 'current']],
            on='time',
            how='left',
            suffixes=('', '_original')
        )
        
        # Identificar registros que devem ser LIGADO
        rpm_col = 'rotational_speed_original' if 'rotational_speed_original' in df_merged.columns else 'rotational_speed'
        current_col = 'current_original' if 'current_original' in df_merged.columns else 'current'
        
        mask_deveria_ser_ligado = (
            (df_merged[rpm_col] > rpm_threshold) | 
            (df_merged[current_col] > current_threshold)
        )
        
        # Contar quantos estão classificados como DESLIGADO mas deveriam ser LIGADO
        mask_errado = mask_deveria_ser_ligado & (df_merged['estado'] == 'DESLIGADO')
        n_corrigidos = mask_errado.sum()
        
        if n_corrigidos > 0:
            print(f"  - Corrigindo {n_corrigidos} registros classificados incorretamente como DESLIGADO")
            df_merged.loc[mask_errado, 'estado'] = 'LIGADO'
        else:
            print(f"  - Nenhuma correção necessária")
        
        # Remover colunas auxiliares
        cols_to_drop = [col for col in df_merged.columns if col.endswith('_original')]
        df_merged = df_merged.drop(columns=cols_to_drop)
        
        # Mostrar nova distribuição
        if n_corrigidos > 0:
            distribuicao_nova = df_merged['estado'].value_counts().to_dict()
            print(f"\n Distribuição (após correção por threshold):")
            for estado, count in distribuicao_nova.items():
                pct = count / len(df_merged) * 100
                print(f"    {estado}: {count} ({pct:.1f}%)")
        
        return df_merged
    
    def filtrar_outliers_estado(self, df_resultados, duracao_minima_minutos=10):
        """
        Remove outliers de estado: sequências < 10 minutos são substituídas pelo estado dominante ao redor
        
        Args:
            df_resultados: DataFrame com coluna 'estado' e 'time'
            duracao_minima_minutos: Duração mínima para considerar uma mudança de estado válida
        """
        print(f"\n Filtrando outliers de estado (sequências < {duracao_minima_minutos} min)...")
        
        df = df_resultados.copy()
        df = df.sort_values('time').reset_index(drop=True)
        
        # Identificar mudanças de estado
        df['estado_anterior'] = df['estado'].shift(1)
        df['mudou_estado'] = df['estado'] != df['estado_anterior']
        
        # Criar ID de sequência para cada período contínuo do mesmo estado
        df['sequencia_id'] = df['mudou_estado'].cumsum()
        
        # Calcular duração de cada sequência
        sequencias = df.groupby('sequencia_id').agg({
            'time': ['first', 'last', 'count'],
            'estado': 'first'
        })
        
        sequencias.columns = ['tempo_inicio', 'tempo_fim', 'n_registros', 'estado']
        sequencias['duracao_minutos'] = (sequencias['tempo_fim'] - sequencias['tempo_inicio']).dt.total_seconds() / 60
        
        # Identificar sequências outliers (< 10 minutos)
        outliers_ids = sequencias[sequencias['duracao_minutos'] < duracao_minima_minutos].index.tolist()
        
        print(f"  - Total de sequências: {len(sequencias)}")
        print(f"  - Sequências outliers (< {duracao_minima_minutos} min): {len(outliers_ids)}")
        
        if len(outliers_ids) == 0:
            print("  - Nenhum outlier encontrado")
            return df_resultados
        
        # Substituir outliers pelo estado dominante ao redor
        estados_corrigidos = 0
        for seq_id in outliers_ids:
            mask_seq = df['sequencia_id'] == seq_id
            indices_seq = df[mask_seq].index
            
            if len(indices_seq) == 0:
                continue
            
            # Pegar índices anterior e posterior
            idx_inicio = indices_seq[0]
            idx_fim = indices_seq[-1]
            
            # Estado anterior e posterior
            estado_antes = df.loc[idx_inicio - 1, 'estado'] if idx_inicio > 0 else None
            estado_depois = df.loc[idx_fim + 1, 'estado'] if idx_fim < len(df) - 1 else None
            
            # Decidir qual estado usar
            if estado_antes == estado_depois and estado_antes is not None:
                # Estados ao redor são iguais - usar esse
                novo_estado = estado_antes
            elif estado_antes is not None:
                # Usar estado anterior
                novo_estado = estado_antes
            elif estado_depois is not None:
                # Usar estado posterior
                novo_estado = estado_depois
            else:
                # Primeiro/último da série, manter como está
                continue
            
            # Substituir
            df.loc[mask_seq, 'estado'] = novo_estado
            estados_corrigidos += len(indices_seq)
        
        print(f"  - Registros corrigidos: {estados_corrigidos}")
        
        # Mostrar nova distribuição
        distribuicao_nova = df['estado'].value_counts().to_dict()
        print(f"\n Distribuição (após filtrar outliers):")
        for estado, count in distribuicao_nova.items():
            pct = count / len(df) * 100
            print(f"    {estado}: {count} ({pct:.1f}%)")
        
        # Remover colunas auxiliares
        df = df.drop(columns=['estado_anterior', 'mudou_estado', 'sequencia_id'])
        
        return df

    def calcular_tempo_ligado_desligado(self, df_resultados):
        """Calcula tempo ligado/desligado baseado nas diferenças temporais entre pontos"""
        print(" Calculando tempo ligado/desligado...")

        # FILTRAR APENAS DADOS DO PERÍODO SOLICITADO
        data_inicio_dt = pd.to_datetime(self.data_inicio_utc, format='%Y-%m-%dT%H:%M:%SZ', utc=True)
        data_fim_dt = pd.to_datetime(self.data_fim_utc, format='%Y-%m-%dT%H:%M:%SZ', utc=True)

        # Filtrar dados apenas do período solicitado
        mask_periodo = (df_resultados['time'] >= data_inicio_dt) & (df_resultados['time'] <= data_fim_dt)
        df_periodo = df_resultados[mask_periodo].copy()

        print(f"  - Dados originais: {len(df_resultados)} registros")
        print(f"  - Dados no período solicitado: {len(df_periodo)} registros")
        print(f"  - Período solicitado: {self.data_inicio} até {self.data_fim}")

        # Se não há dados no período solicitado, retornar zeros
        if len(df_periodo) == 0:
            periodo_solicitado = data_fim_dt - data_inicio_dt
            print(f"  - AVISO: Nenhum dado encontrado no período solicitado!")
            return {
                'tempo_desligado': pd.Timedelta(0),
                'tempo_ligado': periodo_solicitado,  # Assumir tudo ligado se não há dados
                'tempo_total': periodo_solicitado,
                'pct_desligado': 0.0,
                'pct_ligado': 100.0
            }

        # Debug: mostrar período dos dados filtrados
        primeiro_timestamp = df_periodo['time'].min()
        ultimo_timestamp = df_periodo['time'].max()
        periodo_coberto = ultimo_timestamp - primeiro_timestamp

        print(f"  - Primeiro timestamp (filtrado): {primeiro_timestamp}")
        print(f"  - Último timestamp (filtrado): {ultimo_timestamp}")
        print(f"  - Período coberto pelos dados: {periodo_coberto}")

        # Calcular estatísticas dos intervalos
        if len(df_periodo) > 1:
            df_sorted_temp = df_periodo.sort_values('time').copy()
            diffs = df_sorted_temp['time'].diff().dropna()
            intervalo_medio = diffs.mean()
            intervalo_max = diffs.max()
            intervalo_min = diffs.min()
            print(f"  - Intervalo médio entre pontos: {intervalo_medio}")
            print(f"  - Intervalo máximo: {intervalo_max}")
            print(f"  - Intervalo mínimo: {intervalo_min}")

        # Ordenar por tempo para garantir sequência correta
        df_sorted = df_periodo.sort_values('time').copy()
        df_sorted['time'] = pd.to_datetime(df_sorted['time'])

        # Método correto: cada ponto representa o estado até o próximo ponto
        tempo_desligado = pd.Timedelta(0)
        tempo_ligado = pd.Timedelta(0)

        for i in range(len(df_sorted)):
            estado_atual = df_sorted['estado'].iloc[i]
            timestamp_atual = df_sorted['time'].iloc[i]

            # Calcular até quando este estado dura
            if i < len(df_sorted) - 1:  # Não é o último ponto
                timestamp_proximo = df_sorted['time'].iloc[i + 1]
                duracao = timestamp_proximo - timestamp_atual
            else:  # Último ponto - deve representar até o final do período solicitado
                duracao = data_fim_dt - timestamp_atual

                # Limitar duração máxima para evitar períodos muito longos
                # Máximo 1 hora para o último ponto se não houver dados recentes
                max_duracao = pd.Timedelta(hours=1)
                if duracao > max_duracao:
                    print(f"  - AVISO: Último ponto tem duração {duracao}, limitando para {max_duracao}")
                    duracao = max_duracao

            if estado_atual == 'DESLIGADO':
                tempo_desligado += duracao
            else:  # LIGADO
                tempo_ligado += duracao

        tempo_total_calculado = tempo_desligado + tempo_ligado

        # IMPORTANTE: O tempo total deve ser o PERÍODO SOLICITADO, não apenas o tempo calculado
        # Isso garante que percentuais reflitam o período completo solicitado
        periodo_solicitado_total = data_fim_dt - data_inicio_dt

        # Calcular percentuais baseado no período solicitado
        if periodo_solicitado_total > pd.Timedelta(0):
            pct_desligado = tempo_desligado / periodo_solicitado_total * 100
            pct_ligado = tempo_ligado / periodo_solicitado_total * 100
        else:
            pct_desligado = 0
            pct_ligado = 100

        print(f"  - Tempo DESLIGADO: {tempo_desligado}")
        print(f"  - Tempo LIGADO: {tempo_ligado}")
        print(f"  - Período solicitado total: {periodo_solicitado_total}")
        print(f"  - Total de pontos analisados (filtrados): {len(df_periodo)}")

        return {
            'tempo_desligado': tempo_desligado,
            'tempo_ligado': tempo_ligado,
            'tempo_total': periodo_solicitado_total,  # Retornar o período solicitado
            'pct_desligado': pct_desligado,
            'pct_ligado': pct_ligado
        }


    def _verificar_erro_conexao(self, erro_msg):
        """
        Verifica se o erro é relacionado a problemas de conexão com o InfluxDB
        """
        indicadores_conexao = [
            'Connection to',
            'timed out',
            'Max retries exceeded',
            'ConnectionError',
            'TimeoutError',
            'ConnectTimeoutError',
            'HTTPConnectionPool',
            '[WinError 10060]',
            'falhou porque o componente conectado não respondeu'
        ]

        erro_lower = str(erro_msg).lower()
        for indicador in indicadores_conexao:
            if indicador.lower() in erro_lower:
                return True
        return False

    def _tratar_erro_conexao(self):
        """
        Trata erros de conexão com o InfluxDB, dando opções ao usuário
        """
        print("\n" + "="*70)
        print("[ERRO] CONEXAO COM O BANCO DE DADOS")
        print("="*70)
        print("Não foi possível conectar ao InfluxDB. Possíveis causas:")
        print("  • Servidor indisponível")
        print("  • Problemas de rede")
        print("  • Porta/firewall bloqueada")
        print("  • Timeout de conexão")
        print()

        while True:
            print("Opções:")
            print("1. Tentar novamente com NOVO INTERVALO de datas")
            print("2. Sair do programa")
            print()

            try:
                opcao = input("Escolha uma opção (1 ou 2): ").strip()

                if opcao == '1':
                    print("\n" + "-"*50)
                    print("INSERINDO NOVO INTERVALO DE DATAS")
                    print("-"*50)

                    # Pedir novas datas
                    while True:
                        try:
                            data_inicio = input("Digite a data/hora inicial (YYYY-MM-DD HH:MM:SS em GMT-3): ").strip()
                            if not data_inicio:
                                print("Data inicial é obrigatória!")
                                continue

                            data_fim = input("Digite a data/hora final (YYYY-MM-DD HH:MM:SS em GMT-3): ").strip()
                            if not data_fim:
                                print("Data final é obrigatória!")
                                continue

                            # Atualizar os atributos da classe
                            self.data_inicio = data_inicio
                            self.data_fim = data_fim
                            self.data_inicio_utc = self._converter_para_utc(data_inicio)
                            self.data_fim_utc = self._converter_para_utc(data_fim)

                            print(f"\nNovo intervalo definido:")
                            print(f"  Início: {data_inicio} (GMT-3)")
                            print(f"  Fim: {data_fim} (GMT-3)")
                            print(f"  UTC: {self.data_inicio_utc} até {self.data_fim_utc}")

                            # Reiniciar o processo
                            print("\n[INFO] Reiniciando análise com novo intervalo...")
                            return  # Sai da função e continua o fluxo normal

                        except KeyboardInterrupt:
                            print("\nOperação cancelada pelo usuário.")
                            sys.exit(0)

                elif opcao == '2':
                    print("\n[INFO] Encerrando programa...")
                    sys.exit(0)

                else:
                    print("[ERRO] Opção inválida! Digite 1 ou 2.")

            except KeyboardInterrupt:
                print("\n\n[INFO] Programa interrompido pelo usuário.")
                sys.exit(0)

    def _converter_para_utc(self, data_gmt3):
        """
        Converte data GMT-3 para UTC (simples para este contexto)
        """
        # Para simplificar, adiciona 3 horas para converter GMT-3 para UTC
        from datetime import datetime, timedelta
        dt = datetime.strptime(data_gmt3, '%Y-%m-%d %H:%M:%S')
        dt_utc = dt + timedelta(hours=3)
        return dt_utc.strftime('%Y-%m-%dT%H:%M:%SZ')

    def gerar_grafico_estados(self, df_resultados, estatisticas_tempo):
        """Gera gráfico de estados COM DATAS CORRETAS DO INTERVALO"""
        print(" Gerando gráfico...")

        # IMPORTANTE: df_resultados já deve estar filtrado pelo intervalo
        # Verificar se os dados estão no intervalo correto
        data_min = df_resultados['time'].min()
        data_max = df_resultados['time'].max()
        
        print(f" Dados do gráfico: {len(df_resultados)} registros")
        print(f" Período: {data_min} até {data_max}")

        fig, ax = plt.subplots(figsize=(15, 8))

        times = df_resultados['time']
        estados = df_resultados['estado']
        estado_numeric = estados.map({'DESLIGADO': 0, 'LIGADO': 1})

        # Criar áreas coloridas para cada estado
        ax.fill_between(times, 0, estado_numeric, where=(estado_numeric == 0), 
                        color='red', alpha=0.5, label='DESLIGADO', step='post')
        ax.fill_between(times, estado_numeric, 1, where=(estado_numeric == 1), 
                        color='green', alpha=0.5, label='LIGADO', step='post')

        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['DESLIGADO', 'LIGADO'])

        # Formatação do eixo X
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, len(times) // 20)))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        # Título com as datas REAIS dos dados (já filtrados)
        data_inicio_grafico = data_min.strftime('%Y-%m-%d %H:%M')
        data_fim_grafico = data_max.strftime('%Y-%m-%d %H:%M')
        
        ax.set_title(f'Análise de Estados - Mpoint {self.mpoint}\n{data_inicio_grafico} até {data_fim_grafico}', 
                     fontsize=14, pad=20)
        ax.legend(loc='upper right')

        # Estatísticas do gráfico (baseadas nos dados filtrados)
        pct_desligado = (estados == 'DESLIGADO').sum() / len(estados) * 100
        pct_ligado = (estados == 'LIGADO').sum() / len(estados) * 100
        
        texto_stats = f"DESLIGADO: {pct_desligado:.1f}%\nLIGADO: {pct_ligado:.1f}%"
        ax.text(0.02, 0.98, texto_stats, transform=ax.transAxes, verticalalignment='top', 
                fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nome_arquivo = f"grafico_estados_{self.mpoint}_{timestamp}.png"
        caminho_arquivo = self.dir_results / nome_arquivo
        self.dir_results.mkdir(exist_ok=True)
        plt.savefig(caminho_arquivo, dpi=300, bbox_inches='tight')
        plt.close()

        print(f" Gráfico salvo: {caminho_arquivo}")
        return caminho_arquivo

    def salvar_resultados(self, df_normalizado, df_resultados, estatisticas_tempo, caminho_grafico):
        """Salva resultados"""
        print(" Salvando resultados...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_nome = f"analise_completa_{self.mpoint}_{timestamp}"

        self.dir_results.mkdir(exist_ok=True)

        arquivo_normalizado = self.dir_results / f"{base_nome}_normalizado.csv"
        df_normalizado.to_csv(arquivo_normalizado, index=False)

        arquivo_resultados = self.dir_results / f"{base_nome}_resultados.csv"
        df_resultados.to_csv(arquivo_resultados, index=False)

        arquivo_relatorio = self.dir_results / f"{base_nome}_relatorio.txt"
        with open(arquivo_relatorio, 'w', encoding='utf-8') as f:
            f.write(f"RELATRIO DE ANLISE - MPOINT {self.mpoint}\n")
            f.write("="*50 + "\n\n")
            f.write(f"Perodo: {self.data_inicio} at {self.data_fim}\n\n")
            f.write("ESTATSTICAS:\n")
            f.write(f"- Tempo DESLIGADO: {estatisticas_tempo['tempo_desligado']} ({estatisticas_tempo['pct_desligado']:.1f}%)\n")
            f.write(f"- Tempo LIGADO: {estatisticas_tempo['tempo_ligado']} ({estatisticas_tempo['pct_ligado']:.1f}%)\n")
            f.write("\nARQUIVOS:\n")
            f.write(f"- Normalizado: {arquivo_normalizado.name}\n")
            f.write(f"- Resultados: {arquivo_resultados.name}\n")
            f.write(f"- Grfico: {caminho_grafico.name}\n")

        # Salvar também arquivo .txt com nome específico que o GUI procura
        arquivo_txt_gui = self.dir_results / f"estados_intervalo_{self.mpoint}.txt"
        with open(arquivo_txt_gui, 'w', encoding='utf-8') as f:
            f.write(f"ANÁLISE DE INTERVALO - MPOINT {self.mpoint}\n")
            f.write("="*60 + "\n\n")
            f.write(f"Período analisado: {self.data_inicio} até {self.data_fim}\n")
            f.write(f"InfluxDB: {self.influx_ip}\n\n")
            f.write("ESTATÍSTICAS DE TEMPO:\n")
            f.write(f"- Tempo DESLIGADO: {estatisticas_tempo['tempo_desligado']} ({estatisticas_tempo['pct_desligado']:.1f}%)\n")
            f.write(f"- Tempo LIGADO: {estatisticas_tempo['tempo_ligado']} ({estatisticas_tempo['pct_ligado']:.1f}%)\n")
            f.write(f"- Total de registros: {len(df_resultados)}\n\n")
            f.write("TRANSIÇÕES DE ESTADO DETECTADAS:\n")

            # Calcular transições
            estados = df_resultados['equipamento_status'].values
            transicoes_ligado = 0
            transicoes_desligado = 0

            for i in range(1, len(estados)):
                if estados[i-1] == 'DESLIGADO' and estados[i] == 'LIGADO':
                    transicoes_ligado += 1
                elif estados[i-1] == 'LIGADO' and estados[i] == 'DESLIGADO':
                    transicoes_desligado += 1

            f.write(f"- Ligações (DESLIGADO → LIGADO): {transicoes_ligado}\n")
            f.write(f"- Desligações (LIGADO → DESLIGADO): {transicoes_desligado}\n\n")
            f.write("ARQUIVOS GERADOS:\n")
            f.write(f"- {arquivo_normalizado.name}\n")
            f.write(f"- {arquivo_resultados.name}\n")
            f.write(f"- {caminho_grafico.name}\n")

        # Executar visualização 3D em janela do Windows
        self.executar_visualizacao_3d_intervalo(df_resultados)

        print(f" Resultados salvos em: {self.dir_results}")

        return {
            'normalizado': arquivo_normalizado,
            'resultados': arquivo_resultados,
            'relatorio': arquivo_relatorio,
            'grafico': caminho_grafico
        }

    def executar_visualizacao_3d_intervalo(self, df_resultados):
        """Executa visualização 3D em janela do Windows usando o script visualizar_clusters_3d_simples.py"""
        print(" Executando visualização 3D em janela do Windows...")

        try:
            script_path = self.base_dir / 'scripts' / 'visualizar_clusters_3d_simples.py'

            if not script_path.exists():
                print("   [ERRO] Script de visualização 3D não encontrado")
                return False

            cmd = [
                sys.executable,
                str(script_path),
                '--mpoint', self.mpoint,
                '--data-inicio', self.data_inicio,
                '--data-fim', self.data_fim,
                '--intervalo-arquivo', self.intervalo_arquivo
            ]

            print(f"   [INFO] Executando: {' '.join(cmd)}")
            print("   " + "="*70)

            import subprocess
            resultado = subprocess.run(
                cmd,
                timeout=300  # 5 minutos timeout
            )

            print("   " + "="*70)

            if resultado.returncode == 0:
                print("   [OK] Visualização 3D concluída")
                return True
            else:
                print(f"   [ERRO] Falha na visualização 3D (código: {resultado.returncode})")
                return False

        except subprocess.TimeoutExpired:
            print("   [ERRO] Timeout na visualização 3D (>5 minutos)")
            return False
        except Exception as e:
            print(f"   [ERRO] Erro ao executar visualização 3D: {e}")
            return False

    def executar_analise_completa(self):
        """Executa visualização 3D em janela do Windows usando o script visualizar_clusters_3d_simples.py"""
        print(" Executando visualização 3D em janela do Windows...")

        try:
            script_path = self.base_dir / 'scripts' / 'visualizar_clusters_3d_simples.py'

            if not script_path.exists():
                print("   [ERRO] Script de visualização 3D não encontrado")
                return False

            cmd = [
                sys.executable,
                str(script_path),
                '--mpoint', self.mpoint,
                '--data-inicio', self.data_inicio,
                '--data-fim', self.data_fim,
                '--intervalo-arquivo', self.intervalo_arquivo
            ]

            print(f"   [INFO] Executando: {' '.join(cmd)}")
            print("   " + "="*70)

            import subprocess
            resultado = subprocess.run(
                cmd,
                timeout=300  # 5 minutos timeout
            )

            print("   " + "="*70)

            if resultado.returncode == 0:
                print("   [OK] Visualização 3D concluída")
                return True
            else:
                print(f"   [ERRO] Falha na visualização 3D (código: {resultado.returncode})")
                return False

        except subprocess.TimeoutExpired:
            print("   [ERRO] Timeout na visualização 3D (>5 minutos)")
            return False
        except Exception as e:
            print(f"   [ERRO] Erro ao executar visualização 3D: {e}")
            return False

    def executar_analise_completa(self):
        """Executa análise completa por intervalo"""
        print("="*80)
        print(f"ANÁLISE COMPLETA POR INTERVALO - MPOINT {self.mpoint}")
        print("="*80)

        try:
            # 1. Baixar dados
            self.baixar_dados_influx()

            # 2. Processar dados
            self.processar_dados()

            # 3. Carregar dados já normalizados (gerados pelo normalizar_dados_kmeans.py)
            from utils.artifact_paths import normalized_csv_path
            arquivo_normalizado = normalized_csv_path(self.mpoint, self.intervalo_arquivo)
            
            if not arquivo_normalizado.exists():
                raise FileNotFoundError(f"Arquivo normalizado não encontrado: {arquivo_normalizado}\n"
                                       f"Execute: python scripts/normalizar_dados_kmeans.py --mpoint {self.mpoint} --intervalo-arquivo \"{self.intervalo_arquivo}\"")
            
            print(f" Carregando dados normalizados de: {arquivo_normalizado.name}")
            df_normalizado = pd.read_csv(arquivo_normalizado)
            df_normalizado['time'] = pd.to_datetime(df_normalizado['time'], format='mixed', utc=True)
            
            # Os dados já foram gerados apenas para o intervalo solicitado
            # Não é necessário filtrar novamente
            print(f"   {len(df_normalizado)} registros carregados (já filtrados pelo intervalo)")

            # 4. Garantir que as features estejam na MESMA ORDEM do treinamento
            # As features no modelo estão em uma ordem específica
            colunas_features_ordenadas = [col for col in self.feature_columns if col in df_normalizado.columns]
            
            # Verificar se temos todas as features necessárias
            features_faltando = set(self.feature_columns) - set(df_normalizado.columns)
            if features_faltando:
                print(f" [AVISO] Features faltando (serão preenchidas com 0): {features_faltando}")
                for col in features_faltando:
                    df_normalizado[col] = 0
                colunas_features_ordenadas = self.feature_columns
            
            # Reordenar dataframe para match com o modelo
            df_normalizado_ordenado = df_normalizado[['time'] + colunas_features_ordenadas].copy()
            
            print(f" Tabela normalizada: {len(df_normalizado_ordenado)} registros  {len(colunas_features_ordenadas)} features")
            print(f" Ordem das features: {colunas_features_ordenadas[:5]}... (primeiras 5)")

            # 5. Classificar estados (com tratamento especial para intervalos sem dados de desligamento)
            df_resultados = self.classificar_estados(df_normalizado_ordenado, colunas_features_ordenadas)
            
            # 6. Classificação rigorosa já aplicada - prosseguir com correções

            # Aplicar correção baseada em thresholds físicos apenas se houver dados suficientes de desligamento
            if hasattr(self, 'thresholds_desligado') and self.thresholds_desligado:
                # Verificar novamente se há dados suficientes para decidir se aplica correção
                dados_suficientes = self.verificar_dados_desligamento_suficientes(df_normalizado_ordenado, colunas_features_ordenadas)
                if dados_suficientes:
                    # Só aplicar correção se há dados suficientes de desligamento
                    df_resultados = self.corrigir_estados_por_threshold(df_resultados, rpm_threshold=1000, current_threshold=600)
                    print("  Aplicando correcao por thresholds fisicos (dados suficientes encontrados)")
                else:
                    print("  Pulando correcao por thresholds fisicos (dados insuficientes de desligamento)")
            else:
                print("  Thresholds dinamicos nao disponiveis - aplicando correcao por thresholds fisicos padrao")
            df_resultados = self.corrigir_estados_por_threshold(df_resultados, rpm_threshold=1000, current_threshold=600)
            
            # 7. Filtrar outliers de estado (sequências < 10 minutos)
            df_resultados = self.filtrar_outliers_estado(df_resultados, duracao_minima_minutos=10)

            # 8. Calcular estatísticas
            estatisticas_tempo = self.calcular_tempo_ligado_desligado(df_resultados)

            # 9. Gerar gráfico
            caminho_grafico = self.gerar_grafico_estados(df_resultados, estatisticas_tempo)

            # 10. Salvar resultados
            arquivos = self.salvar_resultados(df_normalizado_ordenado, df_resultados, estatisticas_tempo, caminho_grafico)

            print("\n [OK] ANALISE CONCLUIDA COM SUCESSO!")
            print(f" [INFO] Resultados em: {self.dir_results}")

            # Logs detalhados
            import time
            start_time = time.time()

            # Coletar informações dos resultados gerados
            generated_files = []
            chart_files = []

            # Listar arquivos gerados na pasta results
            if self.dir_results.exists():
                for file_path in self.dir_results.iterdir():
                    if file_path.is_file():
                        generated_files.append(str(file_path))
                        if file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.svg']:
                            chart_files.append(str(file_path))

            # Estatísticas da análise
            analysis_summary = {
                'mpoint': self.mpoint,
                'analysis_type': 'interval_complete_analysis',
                'influxdb_connection': {
                    'ip': self.influx_ip,
                    'url': f"http://{self.influx_ip}:8086"
                },
                'processing_steps': [
                    'data_download_from_influx',
                    'data_segmentation',
                    'outlier_detection_IQR',
                    'interpolation_KNN_temporal',
                    'data_unification',
                    'feature_normalization',
                    'K-means_clustering',
                    'state_classification',
                    'time_statistics_calculation',
                    'results_visualization'
                ],
                'time_period': {
                    'start_utc': self.data_inicio_utc,
                    'end_utc': self.data_fim_utc,
                    'start_local': self.data_inicio,
                    'end_local': self.data_fim,
                    'timezone': 'GMT-3'
                },
                'files_generated': len(generated_files),
                'charts_generated': len(chart_files),
                'analysis_type': 'interval_complete_analysis'
            }

            # Log de análise
            analysis_log = create_analysis_log(
                script_name='analise_intervalo_completa',
                mpoint=self.mpoint,
                analysis_type='interval_analysis',
                input_period={
                    'start_datetime': self.data_inicio,
                    'end_datetime': self.data_fim,
                    'timezone': 'GMT-3',
                    'influxdb_url': f"http://{self.influx_ip}:8086"
                },
                results_summary=analysis_summary,
                generated_files=generated_files,
                processing_time=time.time() - start_time,
                analysis_parameters={
                    'data_source': 'InfluxDB_validated_default',
                    'processing_steps': [
                        'data_download_from_influx',
                        'data_preprocessing',
                        'feature_engineering',
                        'model_prediction',
                        'state_classification',
                        'visualization_generation'
                    ],
                    'model_used': 'K-means_trained_model',
                    'output_format': 'charts_and_reports'
                },
                data_characteristics={
                    'mpoint_analyzed': self.mpoint,
                    'time_range_hours': None,  # Poderia calcular se tivesse os dados
                    'data_quality': 'validated_default_from_influxdb'
                }
            )

            save_log(analysis_log, 'analise_intervalo_completa', self.mpoint, 'interval_analysis_complete')

            # Log de visualização se houver gráficos
            if chart_files:
                viz_log = create_visualization_log(
                    script_name='analise_intervalo_completa',
                    mpoint=self.mpoint,
                    chart_type='interval_analysis_charts',
                    data_description={
                        'analysis_type': 'time_interval_analysis',
                        'data_source': 'InfluxDB_validated_default',
                        'mpoint': self.mpoint,
                        'charts_count': len(chart_files)
                    },
                    chart_files=chart_files,
                    period_info={
                        'start_datetime': self.data_inicio,
                        'end_datetime': self.data_fim,
                        'timezone': 'GMT-3',
                        'analysis_duration_hours': None
                    }
                )

                save_log(viz_log, 'analise_intervalo_completa', self.mpoint, 'charts_generation')

            # Enriquecer arquivo results
            results_data = {
                'interval_analysis_completed': True,
                'interval_analysis_timestamp': datetime.now().isoformat(),
                'influxdb_ip': self.influx_ip,
                'analysis_period': {
                    'start': self.data_inicio,
                    'end': self.data_fim,
                    'timezone': 'GMT-3'
                },
                'files_generated': generated_files,
                'charts_generated': chart_files,
                'analysis_parameters': analysis_log['analysis_parameters']
            }

            enrich_results_file(self.mpoint, results_data)

        except Exception as e:
            print(f"\n ERRO: {str(e)}")
            import traceback
            traceback.print_exc()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Anlise completa por intervalo")
    parser.add_argument('--mpoint', type=str, required=True, help='ID do mpoint')
    parser.add_argument('--influx-ip', type=str, required=True, help='IP do InfluxDB')
    parser.add_argument('--inicio', type=str, required=True, help='Data/hora inicial (GMT-3)')
    parser.add_argument('--fim', type=str, required=True, help='Data/hora final (GMT-3)')
    return parser.parse_args()


def main():
    try:
        args = parse_arguments()

        analisador = AnalisadorIntervaloCompleto(
            mpoint=args.mpoint,
            influx_ip=args.influx_ip,
            data_inicio=args.inicio,
            data_fim=args.fim
        )

        analisador.executar_analise_completa()

    except Exception as e:
        print(f"\n Erro: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()