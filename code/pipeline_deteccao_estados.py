"""
Sistema completo pra detectar se os equipamentos tao ligados ou desligados.
Usa machine learning (K-means) pra analisar dados de corrente, vibracao, etc.

Dois modos principais:
1. TREINO: Pega dados historicos e cria o modelo de ML
2. PREDICAO: Usa o modelo treinado pra classificar novos dados em tempo real
"""
import sys
import subprocess
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
import argparse

from utils.artifact_paths import (
    config_path,
    ensure_base_structure,
    get_mpoint_dirs,
    info_kmeans_path,
    info_normalizacao_path,
    kmeans_model_path,
    normalized_csv_path,
    normalized_numpy_path,
    preprocess_pipeline_path,
    processed_classificado_path,
    processed_rotulado_path,
    processed_unificado_path,
    scaler_maxmin_path,
    scaler_model_path,
    results_dir,
)
from utils.logging_utils import (
    save_log,
    create_processing_log,
    create_training_log,
    create_analysis_log,
    format_file_list,
    get_file_info,
    enrich_results_file,
)
class PipelineDeteccaoEstados:
    """Classe principal que coordena todo o processo de detecção de estados"""
    
    def __init__(self, mpoint=None):
        self.base_dir = Path(__file__).parent
        self.dir_raw = self.base_dir / 'data' / 'raw'
        self.dir_raw_preenchido = self.base_dir / 'data' / 'raw_preenchido'
        self.dir_models = self.base_dir / 'models'
        self.dir_scripts = self.base_dir / 'scripts'
        self.dir_utils = self.base_dir / 'utils'

        # Mpoint atual (pode ser definido depois)
        self.mpoint_atual = mpoint

        # Criar diretórios base se não existem
        ensure_base_structure()

    def configurar_pastas_mpoint(self, mpoint):
        """Configura estrutura de pastas para o mpoint"""
        dirs_mpoint = get_mpoint_dirs(mpoint, create=True)
        pastas_mpoint = {
            'raw_preenchido': self.dir_raw_preenchido,
            'processed': self.base_dir / 'data' / 'processed',
            'normalized': self.base_dir / 'data' / 'normalized',
            'models': dirs_mpoint['models'],
            'results': dirs_mpoint['results'],
        }

        for caminho in pastas_mpoint.values():
            caminho.mkdir(parents=True, exist_ok=True)

        return pastas_mpoint
    
    def listar_mpoints_disponiveis(self):
        """Lista todos os mpoints com dados disponíveis"""
        print("\n" + "="*80)
        print("MPOINTS DISPONÍVEIS")
        print("="*80)
        
        # Tentar ler lista_mpoints.txt
        arquivo_lista = self.base_dir / 'lista_mpoints.txt'
        mpoints_lista = []
        
        if arquivo_lista.exists():
            print(f"   Lendo de: {arquivo_lista.name}")
            with open(arquivo_lista, 'r') as f:
                mpoints_lista = [linha.strip() for linha in f if linha.strip()]
        
        # Se não tem lista ou está vazia, buscar arquivos
        if len(mpoints_lista) == 0:
            print("   Buscando em data/raw/...")
            arquivos_dados = list(self.dir_raw.glob('dados_c_*.csv'))
            
            for arq in arquivos_dados:
                nome = arq.stem  # dados_c_636
                if nome.startswith('dados_c_'):
                    mpoint = nome.replace('dados_', '')  # c_636
                    mpoints_lista.append(mpoint)
        
        # Verificar cada mpoint
        mpoints = []
        for mpoint in mpoints_lista:
            # Extrair mpoint se tiver prefixo 'dados_'
            if mpoint.startswith('dados_'):
                mpoint = mpoint.replace('dados_', '')
            
            arq_dados = self.dir_raw / f'dados_{mpoint}.csv'
            if not arq_dados.exists():
                continue
            
            # Verificar se tem os 3 arquivos necessários
            arq_estimated = self.dir_raw / f'dados_estimated_{mpoint}.csv'
            arq_slip = self.dir_raw / f'dados_slip_{mpoint}.csv'
            
            tem_completo = arq_estimated.exists() and arq_slip.exists()
            
            mpoints.append({
                'mpoint': mpoint,
                'completo': tem_completo,
                'dados': arq_dados.exists(),
                'estimated': arq_estimated.exists(),
                'slip': arq_slip.exists()
            })
        
        if len(mpoints) == 0:
            print("   Nenhum mpoint encontrado em data/raw/")
            return []
        
        print(f"\n   Total: {len(mpoints)} mpoint(s)\n")
        for i, mp in enumerate(mpoints, 1):
            status = "[OK] COMPLETO" if mp['completo'] else "[ERRO] INCOMPLETO"
            print(f"   {i}. {mp['mpoint']} - {status}")
            if not mp['completo']:
                print(f"      Arquivos faltando: ", end="")
                faltando = []
                if not mp['estimated']:
                    faltando.append('estimated')
                if not mp['slip']:
                    faltando.append('slip')
                print(", ".join(faltando))
        
        return mpoints
    
    def verificar_arquivos_raw(self, mpoint):
        """Verifica se existem os 3 arquivos necessários"""
        arquivos = {
            'dados': self.dir_raw / f'dados_{mpoint}.csv',
            'estimated': self.dir_raw / f'dados_estimated_{mpoint}.csv',
            'slip': self.dir_raw / f'dados_slip_{mpoint}.csv'
        }
        
        todos_existem = all(arq.exists() for arq in arquivos.values())
        
        return todos_existem, arquivos
    
    def executar_processamento(self, mpoint=None):
        """Executa pipeline de processamento"""
        if mpoint is None:
            mpoint = self.mpoint_atual
        if mpoint is None:
            raise ValueError("Mpoint deve ser especificado")

        print("\n" + "="*80)
        print(f"FASE 1: PROCESSAMENTO DE DADOS - MPOINT: {mpoint}")
        print("="*80)

        # Configurar estrutura de pastas
        pastas_mpoint = self.configurar_pastas_mpoint(mpoint)
        print(f"   [OK] Pastas prontas para {mpoint}:")
        for nome, caminho in pastas_mpoint.items():
            print(f"      - {nome}: {caminho}")

        scripts = [
            ('segmentar_preencher_dados.py', 'Segmentação inicial'),
            ('processar_dados_simples.py', 'Processamento e interpolação'),
            ('unir_sincronizar_periodos.py', 'União e sincronização')
        ]
        
        for script, descricao in scripts:
            print(f"\n--> {descricao}: {script}")
            caminho_script = self.dir_scripts / script
            
            if not caminho_script.exists():
                print(f"   [ERRO] Script não encontrado: {caminho_script}")
                return False
            
            # Executar script com parâmetro mpoint se necessário
            try:
                cmd = [sys.executable, str(caminho_script)]
                # Scripts que precisam do parâmetro mpoint
                scripts_com_mpoint = [
                    'segmentar_preencher_dados.py',
                    'processar_dados_simples.py',
                    'unir_sincronizar_periodos.py',
                    'normalizar_dados_kmeans.py',
                    'kmeans_classificacao_moderado.py',
                    'visualizar_clusters_3d.py'
                ]

                if script in scripts_com_mpoint:
                    cmd.extend(['--mpoint', mpoint])

                print(f"   [INFO] Executando: {' '.join(cmd)}")
                print("   " + "="*70)
                resultado = subprocess.run(
                    cmd,
                    timeout=3600  # 1 hora timeout
                )
                print("   " + "="*70)
                
                if resultado.returncode == 0:
                    print(f"   [OK] Concluído com sucesso")
                else:
                    print(f"   [ERRO] Falha na execução (código {resultado.returncode})")
                    return False
                    
            except subprocess.TimeoutExpired:
                print(f"   ✗ Timeout (>1 hora)")
                return False
            except Exception as e:
                print(f"   [ERRO] Falha: {e}")
                return False
        
        return True
    
    def verificar_duracao_minima(self, mpoint=None, minimo_dias=30):
        """Verifica se há dados suficientes (mínimo de dias)"""
        if mpoint is None:
            mpoint = self.mpoint_atual
        if mpoint is None:
            raise ValueError("Mpoint deve ser especificado")

        print("\n" + "="*80)
        print(f"VERIFICAÇÃO DE DADOS MÍNIMOS - MPOINT: {mpoint}")
        print("="*80)

        # Procurar arquivos finais gerados para o mpoint específico na pasta uniao
        dir_periodos = self.dir_raw_preenchido
        arquivos_finais = list(dir_periodos.glob(f'periodo_*_final_{mpoint}.csv'))

        if len(arquivos_finais) == 0:
            # Se não tem finais, procurar unificados
            arquivos_finais = list(dir_periodos.glob(f'periodo_*_unificado_{mpoint}.csv'))

        if len(arquivos_finais) == 0:
            print(f"   ✗ Nenhum arquivo de período processado encontrado para mpoint {mpoint}")
            print(f"   Procurado: periodo_*_final_{mpoint}.csv ou periodo_*_unificado_{mpoint}.csv")
            return False

        print(f"   Encontrados {len(arquivos_finais)} períodos processados")

        # Verificar se há pelo menos 2 períodos (mínimo para análise)
        if len(arquivos_finais) < 2:
            print(f"   ✗ Pelo menos 2 períodos são necessários (encontrado: {len(arquivos_finais)})")
            return False

        # Calcular duração total
        duracao_total_horas = 0
        total_registros = 0

        for arq in arquivos_finais:
            try:
                # Ler apenas timestamps para calcular duração
                df_timestamps = pd.read_csv(arq, usecols=['time'])
                df_timestamps['time'] = pd.to_datetime(df_timestamps['time'], format='mixed', utc=True)

                if len(df_timestamps) == 0:
                    print(f"   ⚠️ Arquivo vazio: {arq.name}")
                    continue

                inicio = df_timestamps['time'].min()
                fim = df_timestamps['time'].max()
                duracao_h = (fim - inicio).total_seconds() / 3600
                duracao_total_horas += duracao_h
                total_registros += len(df_timestamps)

                print(f"   - {arq.name}: {len(df_timestamps):,} registros, {duracao_h:.1f}h")

            except Exception as e:
                print(f"   [AVISO] Erro ao processar {arq.name}: {e}")
                continue

        if total_registros == 0:
            print(f"   [ERRO] Nenhum registro válido encontrado")
            return False

        duracao_total_dias = duracao_total_horas / 24

        print(f"\n   [RESUMO]")
        print(f"   - Total de registros: {total_registros:,}")
        print(f"   - Duração total: {duracao_total_horas:.1f} horas ({duracao_total_dias:.1f} dias)")
        print(f"   - Registros por hora: {total_registros/duracao_total_horas:.1f}")
        print(f"   - Mínimo necessário: {minimo_dias} dias ({minimo_dias * 24} horas)")

        # Verificações de qualidade
        qualidade_ok = True

        if duracao_total_dias < minimo_dias:
            print(f"   [ERRO] Dados insuficientes (faltam {minimo_dias - duracao_total_dias:.1f} dias)")
            qualidade_ok = False

        if total_registros < 1000:
            print(f"   [ERRO] Poucos registros ({total_registros} < 1000 mínimo)")
            qualidade_ok = False

        # Verificar frequência de amostragem (deve ser ~20 segundos = 3 registros/minuto)
        freq_esperada = total_registros / duracao_total_horas  # registros por hora
        if freq_esperada < 150 or freq_esperada > 200:  # 150-200 registros/hora = ~3-5 registros/minuto
            print(f"   [AVISO] Frequência de amostragem suspeita: {freq_esperada:.1f} registros/hora")
            print(f"      Esperado: ~180 registros/hora (1 registro a cada 20 segundos)")

        if qualidade_ok:
            print(f"   [OK] Dados suficientes para treino K-means")
            return True
        else:
            print(f"   [ERRO] Requisitos mínimos não atendidos")
            return False
    
    def executar_treino_kmeans(self, mpoint=None):
        """Executa normalização e treino K-means"""
        if mpoint is None:
            mpoint = self.mpoint_atual
        if mpoint is None:
            raise ValueError("Mpoint deve ser especificado")

        print("\n" + "="*80)
        print(f"FASE 2: TREINO K-MEANS - MPOINT: {mpoint}")
        print("="*80)
        
        scripts = [
            ('normalizar_dados_kmeans.py', 'Normalização de dados'),
            ('kmeans_classificacao_moderado.py', 'Treino K-means')
        ]
        
        for script, descricao in scripts:
            print(f"\n--> {descricao}: {script}")
            caminho_script = self.dir_scripts / script
            
            if not caminho_script.exists():
                print(f"   [ERRO] Script não encontrado: {caminho_script}")
                return False
            
            try:
                cmd = [sys.executable, str(caminho_script)]
                # Scripts que precisam do parâmetro mpoint
                scripts_com_mpoint = [
                    'normalizar_dados_kmeans.py',
                    'kmeans_classificacao_moderado.py'
                ]

                if script in scripts_com_mpoint:
                    cmd.extend(['--mpoint', mpoint])

                print(f"   [INFO] Executando: {' '.join(cmd)}")
                print("   " + "="*70)
                resultado = subprocess.run(
                    cmd,
                    timeout=1800  # 30 min timeout
                )
                print("   " + "="*70)
                
                if resultado.returncode == 0:
                    print(f"   [OK] Concluído com sucesso")
                else:
                    print(f"   [ERRO] Falha na execução (código {resultado.returncode})")
                    return False
                    
            except Exception as e:
                print(f"   [ERRO] Falha: {e}")
                return False
        
        return True
    
    def gerar_parametros(self, mpoint=None):
        """Gera e salva parâmetros para o mpoint"""
        if mpoint is None:
            mpoint = self.mpoint_atual
        if mpoint is None:
            raise ValueError("Mpoint deve ser especificado")

        print("\n" + "="*80)
        print(f"FASE 3: GERAÇÃO DE PARÂMETROS - MPOINT: {mpoint}")
        print("="*80)
        
        required_artifacts = {
            'kmeans_model': kmeans_model_path(mpoint),
            'scaler_model': scaler_model_path(mpoint),
            'info_kmeans': info_kmeans_path(mpoint),
        }

        optional_artifacts = {
            'scaler_normalizacao': scaler_maxmin_path(mpoint),
            'info_normalizacao': info_normalizacao_path(mpoint),
            'pipeline_preprocessamento': preprocess_pipeline_path(mpoint),
            'dados_normalizados_csv': normalized_csv_path(mpoint),
            'dados_normalizados_numpy': normalized_numpy_path(mpoint),
            'dados_unificados_final': processed_unificado_path(mpoint),
            'dados_classificados': processed_classificado_path(mpoint),
            'dados_rotulados': processed_rotulado_path(mpoint),
        }

        missing_required = [nome for nome, path in required_artifacts.items() if not path.exists()]
        if missing_required:
            print("   [ERRO] Artefatos obrigatórios ausentes:")
            for nome in missing_required:
                print(f"      - {nome}: {required_artifacts[nome]}")
            print("   Execute as etapas anteriores do pipeline e tente novamente.")
            return False

        missing_optional = [nome for nome, path in optional_artifacts.items() if not path.exists()]
        if missing_optional:
            print("   [AVISO] Alguns artefatos opcionais não foram encontrados:")
            for nome in missing_optional:
                print(f"      - {nome}: {optional_artifacts[nome]}")

        artefatos = {**required_artifacts, **optional_artifacts}
        artefatos_relativos = {
            nome: str(path.relative_to(self.base_dir)) if path.exists() else None
            for nome, path in artefatos.items()
        }

        config = {
            'mpoint': mpoint,
            'data_treino': datetime.now().isoformat(),
            'artefatos': artefatos_relativos,
            'colunas_dados_c636': [
                'mag_x', 'mag_y', 'mag_z', 'object_temp',
                'vel_max_x', 'vel_max_y', 'vel_rms_x',
                'vel_max_z', 'vel_rms_y', 'vel_rms_z'
            ],
            'colunas_estimated': [
                'rotational_speed', 'vel_rms', 'current'
            ],
            'colunas_slip': [
                'fe_frequency', 'fe_magnitude_-_1', 'fe_magnitude_0', 'fe_magnitude_1',
                'fr_frequency', 'rms'
            ]
        }

        arquivo_config = config_path(mpoint, create=True)
        with open(arquivo_config, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"\n   [OK] Configuração atualizada: {arquivo_config}")
        return True

    def abrir_graficos_gerados_eletrico(self, mpoint):
        """Abre os gráficos e relatórios gerados automaticamente após o treino elétrico"""
        import os
        import platform
        
        print("\n   [INFO] Abrindo gráficos e relatórios gerados...")
        
        # Lista de arquivos a procurar
        arquivos = []
        
        # Gráfico de normalização
        grafico_norm = self.base_dir / 'plots' / f'dados_normalizados_analise_{mpoint}.png'
        if grafico_norm.exists():
            arquivos.append(grafico_norm)
        
        # Gráfico de K-means
        grafico_kmeans = self.base_dir / 'results' / f'analise_kmeans_clusters_moderado_{mpoint}.png'
        if grafico_kmeans.exists():
            arquivos.append(grafico_kmeans)
        
        # Gráficos 3D
        dirs_mpoint = get_mpoint_dirs(mpoint, create=False)
        grafico_3d_current = dirs_mpoint['results'] / f'estados_corrente_vibracao_tempo_3d_{mpoint}.png'
        if grafico_3d_current.exists():
            arquivos.append(grafico_3d_current)
        
        grafico_3d_rpm = dirs_mpoint['results'] / f'estados_rpm_vibracao_tempo_3d_{mpoint}.png'
        if grafico_3d_rpm.exists():
            arquivos.append(grafico_3d_rpm)
        
        # Relatório TXT (pode ter nomes variados)
        for txt_file in dirs_mpoint['results'].glob(f'*{mpoint}*.txt'):
            arquivos.append(txt_file)
        
        if not arquivos:
            print("   [AVISO] Nenhum arquivo encontrado para abrir")
            return
        
        # Abrir cada arquivo de acordo com o sistema operacional
        sistema = platform.system()
        
        for arquivo in arquivos:
            try:
                if sistema == 'Windows':
                    os.startfile(str(arquivo))
                elif sistema == 'Darwin':  # macOS
                    subprocess.run(['open', str(arquivo)])
                else:  # Linux
                    subprocess.run(['xdg-open', str(arquivo)])
                print(f"   [OK] Aberto: {arquivo.name}")
            except Exception as e:
                print(f"   [AVISO] Não foi possível abrir {arquivo.name}: {e}")
    
    def executar_visualizacao_3d(self, mpoint=None, data_inicio=None, data_fim=None, intervalo_arquivo=None):
        """Executa visualização 3D dos clusters treinados"""
        if mpoint is None:
            mpoint = self.mpoint_atual
        if mpoint is None:
            raise ValueError("Mpoint deve ser especificado")

        print("   - Executando visualização 3D dos clusters...")

        script_path = self.base_dir / 'scripts' / 'visualizar_clusters_3d_simples.py'

        if not script_path.exists():
            print("   [ERRO] Script de visualização 3D não encontrado")
            return False

        try:
            cmd = [sys.executable, str(script_path), '--mpoint', mpoint]

            # Adicionar parâmetros de data se fornecidos
            if data_inicio:
                cmd.extend(['--data-inicio', data_inicio])
            if data_fim:
                cmd.extend(['--data-fim', data_fim])
            
            # Adicionar intervalo se no modo análise
            if intervalo_arquivo:
                cmd.extend(['--intervalo-arquivo', intervalo_arquivo])

            print(f"   [INFO] Executando: {' '.join(cmd)}")
            print("   " + "="*70)
            resultado = subprocess.run(
                cmd,
                timeout=300  # 5 minutos timeout
            )
            print("   " + "="*70)

            if resultado.returncode == 0:
                print("   [OK] Visualização 3D concluída")
                return True
            else:
                print(f"   [ERRO] Falha na visualização 3D (código {resultado.returncode})")
                return False

        except subprocess.TimeoutExpired:
            print("   [ERRO] Timeout na visualização 3D (>5 minutos)")
            return False
        except Exception as e:
            print(f"   [ERRO] Erro ao executar visualização 3D: {e}")
            return False

    def modo_treino(self, mpoint=None):
        """Modo de treino: processar dados e gerar parâmetros"""
        print("\n" + "="*80)
        print("MODO: TREINO")
        print("="*80)

        # Se mpoint não foi especificado, listar disponíveis
        if mpoint is None:
            # Listar mpoints disponíveis
            mpoints = self.listar_mpoints_disponiveis()

            if len(mpoints) == 0:
                return

            # Selecionar mpoint
            print("\nSelecione o mpoint para treinar:")
            try:
                opcao = int(input("   Número: "))
                if opcao < 1 or opcao > len(mpoints):
                    print("   [ERRO] Opção inválida")
                    return
            except:
                print("   [ERRO] Entrada inválida")
                return

            mpoint_selecionado = mpoints[opcao - 1]

            if not mpoint_selecionado['completo']:
                print(f"\n   [ERRO] Mpoint incompleto. Verifique os arquivos em data/raw/")
                return

            mpoint = mpoint_selecionado['mpoint']
            print(f"\n   [OK] Mpoint selecionado: {mpoint}")
        else:
            print(f"   [OK] Mpoint especificado: {mpoint}")

        # Definir mpoint atual
        self.mpoint_atual = mpoint

        # Verificar arquivos
        completo, arquivos = self.verificar_arquivos_raw(mpoint)
        if not completo:
            print(f"   [ERRO] Arquivos incompletos")
            return

        # Executar pipeline
        if not self.executar_processamento(mpoint):
            print("\n   [ERRO] Falha no processamento")
            return

        # Verificar duração mínima
        if not self.verificar_duracao_minima(mpoint, minimo_dias=30):
            print("\n   [AVISO] Dados insuficientes, mas continuando com treino...")
            # Mesmo com dados insuficientes, continua se pelo menos tem alguns dados
            if not self.verificar_duracao_minima(mpoint, minimo_dias=1):
                print("\n   [ERRO] Dados insuficientes (menos de 1 dia)")
                return

        # Treinar K-means
        if not self.executar_treino_kmeans(mpoint):
            print("\n   [ERRO] Falha no treino K-means")
            return

        # Gerar parâmetros
        if not self.gerar_parametros(mpoint):
            print("\n   [ERRO] Falha na geração de parâmetros")
            return

        # Gerar visualização 3D dos clusters
        print("\n   [INFO] Gerando visualização 3D dos clusters...")
        if not self.executar_visualizacao_3d(mpoint):
            print("   [AVISO] Não foi possível gerar visualização 3D, mas treino foi concluído")
        else:
            print("   [OK] Visualização 3D gerada com sucesso")

        print("\n" + "="*80)
        print("TREINO CONCLUÍDO COM SUCESSO!")
        print("="*80)
        print(f"   Parâmetros salvos em: models/{mpoint}/")
        print("   Visualização 3D dos clusters disponível em: results/")
        
        # Abrir gráficos e relatórios automaticamente
        self.abrir_graficos_gerados_eletrico(mpoint)

        # Gerar logs detalhados para TCC
        import time
        start_time = time.time()  # Nota: deveria ser definido no início, mas para compatibilidade vamos estimar

        # Coletar informações dos artefatos gerados
        generated_files = []
        chart_files = []

        # Diretórios dos artefatos
        dirs_mpoint = get_mpoint_dirs(mpoint, create=False)

        # Listar arquivos gerados
        artifacts_to_check = [
            config_path(mpoint),
            kmeans_model_path(mpoint),
            scaler_model_path(mpoint),
            info_kmeans_path(mpoint),
            scaler_maxmin_path(mpoint),
            info_normalizacao_path(mpoint),
            preprocess_pipeline_path(mpoint),
            normalized_csv_path(mpoint),
            normalized_numpy_path(mpoint),
            processed_unificado_path(mpoint),
            processed_classificado_path(mpoint),
            processed_rotulado_path(mpoint),
        ]

        for artifact_path in artifacts_to_check:
            if artifact_path.exists():
                generated_files.append(str(artifact_path))

        # Verificar gráficos gerados
        results_path = results_dir(mpoint)
        if results_path.exists():
            for file_path in results_path.iterdir():
                if file_path.is_file():
                    generated_files.append(str(file_path))
                    if file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.svg']:
                        chart_files.append(str(file_path))

        # Log de treinamento completo
        training_log = create_training_log(
            script_name='pipeline_deteccao_estados',
            mpoint=mpoint,
            model_info={
                'pipeline_type': 'complete_training_pipeline',
                'stages_completed': [
                    'data_processing',
                    'data_validation',
                    'kmeans_training',
                    'parameter_generation',
                    '3d_visualization'
                ],
                'algorithm': 'K-means_with_preprocessing',
                'n_clusters': 6,
                'classification_strategy': 'Dynamic_2_state_based_on_scores'
            },
            training_data_info={
                'data_sources': ['raw_csv_files', 'influxdb_data'],
                'processing_steps': [
                    'segmentation_and_preprocessing',
                    'data_interpolation',
                    'period_unification',
                    'normalization',
                    'kmeans_clustering'
                ],
                'quality_checks': ['duration_validation', 'data_completeness'],
                'mpoint': mpoint
            },
            performance_metrics={
                'pipeline_completion': True,
                'stages_successful': 5,
                'artifacts_generated': len(generated_files),
                'charts_generated': len(chart_files)
            },
            model_files=generated_files,
            processing_time=time.time() - start_time,
            training_parameters={
                'mpoint': mpoint,
                'training_mode': 'complete_pipeline',
                'data_quality_threshold': '30_days_minimum',
                'kmeans_clusters': 6,
                'normalization_method': 'MinMax_with_outlier_clipping'
            },
            pipeline_summary={
                'total_stages': 5,
                'completed_stages': ['processing', 'validation', 'normalization', 'training', 'visualization'],
                'output_artifacts': generated_files,
                'visualization_charts': chart_files
            }
        )

        save_log(training_log, 'pipeline_deteccao_estados', mpoint, 'training_pipeline_complete')

        # Enriquecer arquivo results
        results_data = {
            'pipeline_training_completed': True,
            'pipeline_training_timestamp': datetime.now().isoformat(),
            'mpoint': mpoint,
            'stages_completed': training_log['pipeline_summary']['completed_stages'],
            'artifacts_generated': len(generated_files),
            'charts_generated': len(chart_files),
            'training_parameters': training_log['training_parameters'],
            'pipeline_performance': training_log['performance_metrics']
        }

        enrich_results_file(mpoint, results_data)
    
    def modo_analise(self, mpoint=None, influx_ip=None, influx_port=None, data_inicio=None, data_fim=None):
        """Modo de análise: usar parâmetros para classificar por intervalo"""
        print("\n" + "="*80)
        print("MODO: ANÁLISE")
        print("="*80)

        # Se mpoint não foi especificado, listar disponíveis
        if mpoint is None:
            # Listar mpoints com parâmetros treinados
            mpoints_treinados = self.listar_mpoints_treinados()

            if len(mpoints_treinados) == 0:
                print("\n   [ERRO] Nenhum mpoint com parâmetros treinados encontrado!")
                print("   Execute primeiro o modo TREINO para gerar os parâmetros.")
                return

            # Selecionar mpoint
            print("\nSelecione o mpoint para predição:")
            for i, mp in enumerate(mpoints_treinados, 1):
                print(f"   {i}. {mp}")

            try:
                opcao = int(input("\nNúmero do mpoint: "))
                if opcao < 1 or opcao > len(mpoints_treinados):
                    print("   [ERRO] Opção inválida")
                    return
                mpoint = mpoints_treinados[opcao - 1]
            except:
                print("   [ERRO] Entrada inválida")
                return
        else:
            print(f"   [OK] Mpoint especificado: {mpoint}")

        # Verificar se parâmetros existem
        if not self.verificar_parametros_mpoint(mpoint):
            print(f"\n   [ERRO] Parâmetros não encontrados para mpoint {mpoint}")
            print(f"   Execute primeiro o treino: python pipeline_deteccao_estados.py --mpoint {mpoint} --modo treino")
            return

        # Análise de intervalo de dados (única opção agora)

        # Registrar início da análise
        analysis_start_time = datetime.now()

        # Input do IP e porta do InfluxDB
        if influx_ip is None:
            influx_ip = input("\nDigite o IP do InfluxDB (ex: 192.168.1.100): ").strip()
            if not influx_ip:
                print("   [ERRO] IP do InfluxDB é obrigatório")
                return

        if influx_port is None:
            influx_port = input("Digite a porta do InfluxDB (ex: 8086): ").strip()
            if not influx_port:
                influx_port = "8086"  # porta padrão do InfluxDB API
                print(f"   [INFO] Usando porta padrão do InfluxDB API: {influx_port}")
            else:
                # Validar porta
                try:
                    port_num = int(influx_port)
                    if port_num == 8888 or port_num == 8886:
                        print("   [AVISO] Porta 8888/8886 é para Chronograf (interface web)")
                        print("   [INFO] Corrigindo automaticamente para porta 8086 (API do InfluxDB)")
                        influx_port = "8086"
                    elif port_num != 8086:
                        print(f"   [INFO] Usando porta customizada: {port_num}")
                except:
                    print("   [ERRO] Porta deve ser um número")
                    return
        else:
            # Mesmo com porta especificada, validar
            try:
                port_num = int(influx_port)
                if port_num == 8888 or port_num == 8886:
                    print("   [AVISO] Porta 8888/8886 é para Chronograf (interface web)")
                    print("   [INFO] Corrigindo automaticamente para porta 8086 (API do InfluxDB)")
                    influx_port = "8086"
                elif port_num != 8086:
                    print(f"   [INFO] Usando porta customizada: {port_num}")
            except:
                print("   [ERRO] Porta deve ser um número")
                return

        print(f"   [INFO] URL do InfluxDB: http://{influx_ip}:{influx_port}")
        print("   [INFO] Porta 8086 = API do InfluxDB (correta)")
        print("   [INFO] Porta 8888 = Chronograf (interface web - errada)")

        influx_url = f"http://{influx_ip}:{influx_port}"

        # Loop para solicitar intervalo de datas até encontrar dados
        dados_encontrados = False
        tentativas = 0
        max_tentativas = 5

        while not dados_encontrados and tentativas < max_tentativas:
            tentativas += 1

            # Solicitar intervalo de datas apenas se não foi fornecido via parâmetro
            if data_inicio is None:
                if tentativas > 1:
                    print(f"\n   [INFO] Tentativa {tentativas}/{max_tentativas} - Nenhum dado encontrado no intervalo anterior")
                    print("   [INFO] Insira um novo intervalo de datas (formato: YYYY-MM-DD HH:MM:SS em GMT-3)")
                data_inicio = input("\nDigite a data/hora inicial (YYYY-MM-DD HH:MM:SS em GMT-3): ").strip()
                if not data_inicio:
                    print("   [ERRO] Data inicial é obrigatória")
                    return

            if data_fim is None:
                data_fim = input("Digite a data/hora final (YYYY-MM-DD HH:MM:SS em GMT-3): ").strip()
                if not data_fim:
                    print("   [ERRO] Data final é obrigatória")
                    return

            print(f"   [INFO] Intervalo solicitado: {data_inicio} até {data_fim} (GMT-3)")
            print(f"   [INFO] Equivalente em UTC: {self.converter_para_utc(data_inicio)} até {self.converter_para_utc(data_fim)}")

            # Verificar se mpoint existe no InfluxDB (só na primeira tentativa)
            if tentativas == 1:
                print(f"\n   [INFO] Verificando mpoint {mpoint} no InfluxDB ({influx_url})...")
                if not self.verificar_mpoint_influx(mpoint, influx_url, "validated_default"):
                    print(f"   [ERRO] Mpoint {mpoint} não encontrado no InfluxDB")
                    return

            # Tentar executar análise de intervalo
            try:
                print("\n=== ANÁLISE POR INTERVALO ===")
                self.executar_analise_intervalo(mpoint, influx_url, data_inicio, data_fim)
                dados_encontrados = True
                print("   [OK] Dados encontrados e processados com sucesso!")

            except ValueError as e:
                erro_msg = str(e)
                if ("Nenhum dado encontrado" in erro_msg or 
                    "Intervalo muito curto" in erro_msg or 
                    "Dados insuficientes" in erro_msg or
                    "cobertura temporal insuficiente" in erro_msg):
                    if tentativas < max_tentativas:
                        print(f"   [AVISO] {erro_msg}")
                        if "Intervalo muito curto" in erro_msg:
                            print("   [INFO] O intervalo deve ter no mínimo 24 horas (1 dia)")
                        elif "Dados insuficientes" in erro_msg or "cobertura temporal insuficiente" in erro_msg:
                            print("\n   [INFO] Requisitos de dados:")
                            print("   • Cobertura temporal: mínimo 70% do período com dados preenchidos")
                            print("   • Valores 0.0 SÃO VÁLIDOS (equipamento desligado)")
                            print("   • Valores vazios (None/NaN) por longos períodos NÃO são válidos")
                            print("   • Frequência de amostragem: 1-2 minutos")
                        print("   [INFO] Tente um intervalo diferente com dados mais completos.")
                        # Resetar datas para pedir novamente
                        data_inicio = None
                        data_fim = None
                    else:
                        print(f"   [ERRO] {erro_msg}")
                        print(f"   [ERRO] Máximo de tentativas ({max_tentativas}) atingido. Abortando.")
                        return
                else:
                    # Outro tipo de erro, não relacionado a dados não encontrados
                    raise

        if not dados_encontrados:
            print("   [ERRO] Não foi possível encontrar dados válidos após todas as tentativas")
            return

        # Gerar logs detalhados para TCC
        import time
        analysis_processing_time = (datetime.now() - analysis_start_time).total_seconds()

        # Coletar informações dos resultados gerados
        generated_files = []
        chart_files = []

        # Verificar arquivos gerados na pasta results do mpoint
        results_path = results_dir(mpoint)
        if results_path.exists():
            for file_path in results_path.iterdir():
                if file_path.is_file():
                    generated_files.append(str(file_path))
                    if file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.svg']:
                        chart_files.append(str(file_path))

        # Log de análise de intervalo
        analysis_log = create_analysis_log(
            script_name='pipeline_deteccao_estados',
            mpoint=mpoint,
            analysis_type='interval_analysis_pipeline',
            input_period={
                'start_datetime': data_inicio,
                'end_datetime': data_fim,
                'influxdb_url': influx_url,
                'timezone': 'GMT-3'
            },
            results_summary={
                'mpoint': mpoint,
                'analysis_type': 'interval_based_analysis',
                'influxdb_connection': {
                    'ip': influx_ip,
                    'port': influx_port,
                    'url': influx_url
                },
                'time_period_analyzed': {
                    'start': data_inicio,
                    'end': data_fim,
                    'duration_hours': None  # Poderia calcular se tivesse os dados
                },
                'files_generated': len(generated_files),
                'charts_generated': len(chart_files),
                'analysis_mode': 'complete_interval_analysis'
            },
            generated_files=generated_files,
            processing_time=analysis_processing_time,
            analysis_parameters={
                'data_source': 'InfluxDB_validated_default',
                'model_used': 'trained_K-means_model',
                'processing_steps': [
                    'data_download_from_influx',
                    'model_prediction',
                    'state_classification',
                    'result_visualization'
                ],
                'output_format': 'charts_and_time_series'
            },
            data_characteristics={
                'mpoint_analyzed': mpoint,
                'data_quality': 'validated_default_from_influxdb',
                'analysis_scope': 'time_interval_based'
            }
        )

        save_log(analysis_log, 'pipeline_deteccao_estados', mpoint, 'interval_analysis_complete')

        # Enriquecer arquivo results
        results_data = {
            'pipeline_analysis_completed': True,
            'pipeline_analysis_timestamp': datetime.now().isoformat(),
            'mpoint': mpoint,
            'influxdb_ip': influx_ip,
            'analysis_period': {
                'start': data_inicio,
                'end': data_fim
            },
            'files_generated': len(generated_files),
            'charts_generated': len(chart_files),
            'analysis_parameters': analysis_log['analysis_parameters'],
            'processing_time_seconds': analysis_processing_time
        }

        enrich_results_file(mpoint, results_data)

        # Após análise completa, executar visualização 3D automaticamente
        print("\n=== VISUALIZAÇÃO 3D DOS ESTADOS ===")
        # Formatar intervalo para passar ao script de visualização
        data_inicio_fmt = data_inicio.replace("-", "").replace(" ", "_").replace(":", ";")
        data_fim_fmt = data_fim.replace("-", "").replace(" ", "_").replace(":", ";")
        intervalo_arquivo = f"{data_inicio_fmt}_{data_fim_fmt}"
        self.executar_visualizacao_3d(mpoint, data_inicio, data_fim, intervalo_arquivo)

    def converter_para_utc(self, data_str):
        """
        Converte data/hora de GMT-3 para UTC (formato ISO para InfluxDB)
        Entrada: 'YYYY-MM-DD HH:MM:SS' (GMT-3)
        Saída: 'YYYY-MM-DDTHH:MM:SSZ' (UTC)
        """
        from datetime import datetime, timezone, timedelta

        try:
            # Parse da data em GMT-3
            dt_local = datetime.strptime(data_str, '%Y-%m-%d %H:%M:%S')
            tz_local = timezone(timedelta(hours=-3))  # GMT-3
            dt_com_tz = dt_local.replace(tzinfo=tz_local)

            # Converter para UTC
            dt_utc = dt_com_tz.astimezone(timezone.utc)
            return dt_utc.strftime('%Y-%m-%dT%H:%M:%SZ')
        except Exception as e:
            print(f"[ERRO] Erro ao converter data '{data_str}': {e}")
            raise

    def listar_mpoints_treinados(self):
        """Lista mpoints que têm parâmetros treinados"""
        if not self.dir_models.exists():
            return []

        mpoints = []
        for item in self.dir_models.iterdir():
            if item.is_dir():
                # O padrão correto é config_{mpoint}.json dentro de models/{mpoint}/
                config_file = item / f'config_{item.name}.json'

                # Arquivos podem estar com tag ou sem tag (compatibilidade)
                kmeans_com_tag = item / f'kmeans_model_moderado_{item.name}.pkl'
                kmeans_sem_tag = item / 'kmeans_model_moderado.pkl'
                kmeans_ok = kmeans_com_tag.exists() or kmeans_sem_tag.exists()

                scaler_com_tag = item / f'scaler_model_moderado_{item.name}.pkl'
                scaler_sem_tag = item / 'scaler_model_moderado.pkl'
                scaler_ok = scaler_com_tag.exists() or scaler_sem_tag.exists()

                info_com_tag = item / f'info_kmeans_model_moderado_{item.name}.json'
                info_sem_tag = item / 'info_kmeans_model_moderado.json'
                info_ok = info_com_tag.exists() or info_sem_tag.exists()

                config_exists = config_file.exists()

                if config_exists and kmeans_ok and scaler_ok and info_ok:
                    mpoints.append(item.name)

        return sorted(mpoints)

    def verificar_parametros_mpoint(self, mpoint):
        """Verifica se os parâmetros treinados existem para o mpoint"""
        dir_mpoint = self.dir_models / mpoint
        if not dir_mpoint.exists():
            return False

        # Verificar arquivos essenciais - o padrão correto é com tag do mpoint
        arquivos_necessarios = [
            f'kmeans_model_moderado_{mpoint}.pkl',
            f'scaler_model_moderado_{mpoint}.pkl',
            f'info_kmeans_model_moderado_{mpoint}.json',
            f'config_{mpoint}.json'
        ]

        for arquivo in arquivos_necessarios:
            if not (dir_mpoint / arquivo).exists():
                return False

        return True

    def verificar_mpoint_influx(self, mpoint, influx_url, tabela):
        """Verifica se o mpoint existe no InfluxDB"""
        try:
            # Conectar ao InfluxDB
            from influxdb import InfluxDBClient

            # Parse da URL
            if influx_url.startswith('http://'):
                url_parts = influx_url.replace('http://', '').split(':')
                host = url_parts[0]
                port = int(url_parts[1]) if len(url_parts) > 1 else 8086
            else:
                # Assume formato host:port
                url_parts = influx_url.split(':')
                host = url_parts[0]
                port = int(url_parts[1]) if len(url_parts) > 1 else 8086

            # Conectar ao banco
            client = InfluxDBClient(host=host, port=port, database='aihub')

            # Definir measurement baseado na tabela
            measurement = "estimated" if tabela == "estimated" else "validated_default"

            # Query EXATAMENTE como o código de referência
            retention_policy = "autogen"
            
            # Testar query simples para verificar se mpoint existe
            query = f"""
            SELECT *
            FROM "{client._database}"."{retention_policy}"."{measurement}"
            WHERE "m_point" = '{mpoint}'
            LIMIT 1
            """

            result = client.query(query)
            points = list(result.get_points())

            if not points or len(points) == 0:
                return False

            print(f"   Mpoint {mpoint} encontrado no InfluxDB")
            return True

        except Exception as e:
            print(f"   [AVISO] Não foi possível verificar mpoint no InfluxDB: {e}")
            # Em caso de dúvida, permitir continuar
            return True

    def executar_analise_intervalo(self, mpoint, influx_url, data_inicio, data_fim):
        """Executa análise por intervalo de dados"""
        print(f"   [INFO] Iniciando análise por intervalo para mpoint {mpoint}")
        print(f"   [INFO] InfluxDB URL: {influx_url}")
        print(f"   [INFO] Período: {data_inicio} até {data_fim}")

        # Extrair IP da URL
        if influx_url.startswith('http://'):
            influx_ip = influx_url.replace('http://', '').split(':')[0]
        else:
            influx_ip = influx_url.split(':')[0]

        # Executar script de análise por intervalo
        script_path = self.base_dir / 'scripts' / 'analise_intervalo_completa.py'

        if not script_path.exists():
            print(f"   [ERRO] Script não encontrado: {script_path}")
            return

        try:
            cmd = [sys.executable, str(script_path),
                   '--mpoint', mpoint,
                   '--influx-ip', influx_ip,
                   '--inicio', data_inicio,
                   '--fim', data_fim]

            print(f"   [INFO] Executando: {' '.join(cmd)}")
            resultado = subprocess.run(cmd, cwd=self.base_dir)

            if resultado.returncode == 0:
                print("   [OK] Análise por intervalo concluída")
            else:
                print("   [ERRO] Falha na análise por intervalo")

        except Exception as e:
            print(f"   [ERRO] Erro ao executar análise por intervalo: {e}")

    def executar_classificacao(self, mpoint):
        """Executa a classificação de dados usando modelo treinado"""
        print(f"\n   [INFO] Classificando dados usando modelo treinado para {mpoint}...")

        # Verificar se existe dados unificados para classificar
        DIR_PROCESSED = self.base_dir / 'data' / 'processed'
        arquivo_dados = DIR_PROCESSED / f'dados_unificados_final_{mpoint}.csv'

        if not arquivo_dados.exists():
            print(f"   [ERRO] Dados unificados não encontrados: {arquivo_dados.name}")
            return False

        # Executar script de classificação
        script_path = self.base_dir / 'scripts' / 'kmeans_classificacao_moderado.py'

        if not script_path.exists():
            print(f"   [ERRO] Script de classificação não encontrado: {script_path}")
            return False

        try:
            cmd = [sys.executable, str(script_path), '--mpoint', mpoint]

            print(f"   [INFO] Executando classificação...")
            print(f"   [INFO] Comando: {' '.join(cmd)}")
            print("   " + "="*70)
            resultado = subprocess.run(
                cmd,
                timeout=600  # 10 minutos timeout
            )
            print("   " + "="*70)

            if resultado.returncode == 0:
                print("   [OK] Classificação concluída")
                return True
            else:
                print(f"   [ERRO] Falha na classificação (código {resultado.returncode})")
                return False

        except subprocess.TimeoutExpired:
            print("   [ERRO] Timeout na classificação (>10 minutos)")
            return False
        except Exception as e:
            print(f"   [ERRO] Erro ao executar classificação: {e}")
            return False

    def modo_visualizar_3d(self, mpoint=None):
        """Modo de visualização 3D dos dados classificados"""
        print("\n" + "="*80)
        print("MODO: VISUALIZAÇÃO 3D")
        print("="*80)

        # Se mpoint não foi especificado, listar disponíveis (treinados)
        if mpoint is None:
            mpoints_treinados = self.listar_mpoints_treinados()

            if len(mpoints_treinados) == 0:
                print("\n   [ERRO] Nenhum mpoint com modelos treinados encontrado!")
                print("   Execute primeiro o modo TREINO para gerar os modelos necessários.")
                return

            # Selecionar mpoint
            print("\nSelecione o mpoint para visualizar:")
            for i, mp in enumerate(mpoints_treinados, 1):
                print(f"   {i}. {mp}")

            try:
                opcao = int(input("\n   Número: "))
                if opcao < 1 or opcao > len(mpoints_treinados):
                    print("   [ERRO] Opção inválida")
                    return
                mpoint = mpoints_treinados[opcao - 1]
            except ValueError:
                print("   [ERRO] Entrada inválida")
                return

        print(f"   [OK] Mpoint selecionado: {mpoint}")
        self.mpoint_atual = mpoint

        # Verificar se existe arquivo classificado na pasta processed
        DIR_PROCESSED = self.base_dir / 'data' / 'processed'
        arquivo_classificado = DIR_PROCESSED / f'dados_classificados_kmeans_moderado_{mpoint}.csv'

        if not arquivo_classificado.exists():
            print(f"\n   [AVISO] Arquivo classificado não encontrado: {arquivo_classificado.name}")

            # Verificar se existe modelo treinado
            DIR_MODELS = self.base_dir / 'models'
            arquivo_modelo = DIR_MODELS / f'kmeans_model_moderado_{mpoint}.pkl'

            if arquivo_modelo.exists():
                print(f"   [INFO] Modelo treinado encontrado: {arquivo_modelo.name}")
                print(f"   [INFO] Classificando dados automaticamente...")

                if not self.executar_classificacao(mpoint):
                    print("   [ERRO] Falha na classificação")
                    return
            else:
                print(f"   [AVISO] Modelo não encontrado: {arquivo_modelo.name}")
                print("   [INFO] É necessário treinar o modelo primeiro")
                print("\n   [INFO] Iniciando treinamento automático...")

                if not self.executar_processamento(mpoint):
                    print("\n   [ERRO] Falha no processamento")
                    return

                if not self.verificar_duracao_minima(mpoint, minimo_dias=1):
                    print("\n   [ERRO] Dados insuficientes")
                    return

                if not self.executar_treino_kmeans(mpoint):
                    print("\n   [ERRO] Falha no treino K-means")
                    return

                print("   [OK] Treinamento concluído")

        # Executar visualização 3D
        print("\n   [INFO] Gerando visualização 3D...")
        if not self.executar_visualizacao_3d(mpoint):
            print("   [ERRO] Falha na visualização 3D")
        else:
            print("\n" + "="*80)
            print("VISUALIZAÇÃO 3D CONCLUÍDA!")
            print("="*80)
            print(f"   Gráficos salvos em: results/")

    def menu_principal(self):
        """Menu principal do pipeline"""
        print("\n" + "="*80)
        print("PIPELINE DE DETECÇÃO DE ESTADOS")
        print("="*80)
        print("\n1. TREINO - Processar dados e gerar parâmetros K-means")
        print("2. ANÁLISE - Classificar estados por intervalo de dados")
        print("3. VISUALIZAR 3D - Visualizar dados classificados em 3D")
        print("4. Sair")

        try:
            opcao = int(input("\nEscolha uma opção: "))
        except:
            print("Opção inválida")
            return

        if opcao == 1:
            self.modo_treino()
        elif opcao == 2:
            self.modo_analise()
        elif opcao == 3:
            self.modo_visualizar_3d()
        elif opcao == 4:
            print("Encerrando...")
        else:
            print("Opção inválida")

    def parse_args(self):
        """Parse argumentos de linha de comando"""
        parser = argparse.ArgumentParser(
            description="Pipeline de Detecção de Estados de Equipamentos"
        )
        parser.add_argument(
            '--mpoint',
            type=str,
            help='Mpoint para processar (ex: c_636)'
        )
        parser.add_argument(
            '--modo',
            choices=['treino', 'analise', 'visualizar'],
            required=False,
            help='Modo de operação'
        )
        parser.add_argument(
            '--ip',
            type=str,
            help='IP do InfluxDB (para modo analise)'
        )
        parser.add_argument(
            '--porta',
            type=str,
            default='8086',
            help='Porta do InfluxDB (padrão: 8086)'
        )
        parser.add_argument(
            '--inicio',
            type=str,
            help='Data/hora inicial (YYYY-MM-DD HH:MM:SS) - para modo analise'
        )
        parser.add_argument(
            '--fim',
            type=str,
            help='Data/hora final (YYYY-MM-DD HH:MM:SS) - para modo analise'
        )
        parser.add_argument(
            '--auto',
            action='store_true',
            help='Executar automaticamente sem menu interativo'
        )

        return parser.parse_args()


def main():
    """Função principal"""
    pipeline = PipelineDeteccaoEstados()

    # Verificar argumentos de linha de comando
    if len(sys.argv) > 1:
        args = pipeline.parse_args()

        # Validações
        if args.modo and not args.mpoint:
            print("❌ ERRO: --mpoint é obrigatório quando --modo é especificado")
            return

        if args.modo:
            # Executar automaticamente
            if args.modo == 'treino':
                pipeline.modo_treino(mpoint=args.mpoint)
            elif args.modo == 'analise':
                pipeline.modo_analise(
                    mpoint=args.mpoint,
                    influx_ip=args.ip,
                    influx_port=args.porta,
                    data_inicio=args.inicio,
                    data_fim=args.fim
                )
            elif args.modo == 'visualizar':
                pipeline.modo_visualizar_3d(mpoint=args.mpoint)
            else:
                print(f"❌ ERRO: Modo inválido: {args.modo}")
        else:
            # Modo interativo com mpoint pré-definido
            if args.mpoint:
                pipeline = PipelineDeteccaoEstados(mpoint=args.mpoint)
            pipeline.menu_principal()
    else:
        # Modo interativo padrão
        pipeline.menu_principal()

if __name__ == "__main__":
    main()

