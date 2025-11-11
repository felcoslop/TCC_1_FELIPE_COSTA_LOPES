"""
Pipeline completo para equipamentos MECÂNICOS (sem estimated, sem current, sem RPM).
Análise baseada em TEMPERATURA e VIBRAÇÃO.

Equipamento MECÂNICO:
- Arquivos: dados_c_XXX.csv + dados_slip_XXX.csv (SEM estimated)
- Análise: Temperatura (object_temp) + Vibração (vel_rms, mag_x/y/z)
- Estados: DESLIGADO (temp ambiente + vibração zero/residual) vs LIGADO (temp alta + vibração significativa)
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
    scaler_maxmin_path,
    scaler_model_path,
    results_dir,
)
from utils.logging_utils import (
    save_log,
    create_training_log,
    enrich_results_file,
)

class PipelineDeteccaoEstadosMecanico:
    """Pipeline para equipamentos MECÂNICOS (temperatura + vibração)"""
    
    def __init__(self, mpoint=None):
        self.base_dir = Path(__file__).parent
        self.dir_raw = self.base_dir / 'data' / 'raw'
        self.dir_raw_preenchido = self.base_dir / 'data' / 'raw_preenchido'
        self.dir_models = self.base_dir / 'models'
        self.dir_scripts = self.base_dir / 'scripts'
        
        self.mpoint_atual = mpoint
        
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
    
    def verificar_arquivos_raw_mecanico(self, mpoint):
        """Verifica se existem os 2 arquivos necessários para equipamento MECÂNICO"""
        arquivos = {
            'dados_c': self.dir_raw / f'dados_{mpoint}.csv',
            'slip': self.dir_raw / f'dados_slip_{mpoint}.csv'
        }
        
        todos_existem = all(arq.exists() for arq in arquivos.values())
        
        # Verificar se NÃO existe estimated (característica de equipamento mecânico)
        estimated_existe = (self.dir_raw / f'dados_estimated_{mpoint}.csv').exists()
        
        return todos_existem and not estimated_existe, arquivos
    
    def listar_mpoints_disponiveis_mecanicos(self):
        """Lista mpoints de equipamentos MECÂNICOS"""
        print("\n" + "="*80)
        print("MPOINTS DISPONÍVEIS - EQUIPAMENTOS MECÂNICOS")
        print("="*80)
        
        import re
        
        arquivos_dados = list(self.dir_raw.glob('dados_c_*.csv'))
        mpoints = []
        
        for arq in arquivos_dados:
            nome = arq.stem
            match = re.match(r'dados_c_(\d+)$', nome)
            if match:
                numero = match.group(1)
                mpoint = f'c_{numero}'
                
                arq_slip = self.dir_raw / f'dados_slip_{mpoint}.csv'
                arq_estimated = self.dir_raw / f'dados_estimated_{mpoint}.csv'
                
                # MECÂNICO = tem dados_c + slip MAS NÃO TEM estimated
                if arq_slip.exists() and not arq_estimated.exists():
                    mpoints.append({
                        'mpoint': mpoint,
                        'completo': True,  # Para mecânico, só precisa de 2 arquivos
                        'dados_c': True,
                        'slip': arq_slip.exists(),
                        'tipo': 'MECANICO'
                    })
        
        if len(mpoints) == 0:
            print("   Nenhum equipamento MECÂNICO encontrado em data/raw/")
            print("   Equipamentos MECÂNICOS têm apenas: dados_c_XXX.csv + dados_slip_XXX.csv")
            print("   (SEM dados_estimated_XXX.csv)")
            return []
        
        print(f"\n   Total: {len(mpoints)} equipamento(s) MECÂNICO(s)\n")
        for i, mp in enumerate(mpoints, 1):
            print(f"   {i}. {mp['mpoint']} - MECÂNICO (temperatura + vibração)")
        
        return mpoints
    
    def executar_processamento_mecanico(self, mpoint=None):
        """Executa pipeline de processamento para equipamento MECÂNICO"""
        if mpoint is None:
            mpoint = self.mpoint_atual
        if mpoint is None:
            raise ValueError("Mpoint deve ser especificado")
        
        print("\n" + "="*80)
        print(f"FASE 1: PROCESSAMENTO DE DADOS MECÂNICOS - MPOINT: {mpoint}")
        print("="*80)
        
        pastas_mpoint = self.configurar_pastas_mpoint(mpoint)
        print(f"   [OK] Pastas prontas para {mpoint}")
        
        scripts = [
            ('processar_dados_simples_mecanico.py', 'Processamento e interpolação MECÂNICO'),
            ('unir_sincronizar_periodos_mecanico.py', 'União e sincronização MECÂNICO')
        ]
        
        for script, descricao in scripts:
            print(f"\n--> {descricao}: {script}")
            caminho_script = self.dir_scripts / script
            
            if not caminho_script.exists():
                print(f"   [ERRO] Script não encontrado: {caminho_script}")
                return False
            
            try:
                cmd = [sys.executable, str(caminho_script), '--mpoint', mpoint]
                
                print(f"   [INFO] Executando: {' '.join(cmd)}")
                print("   " + "="*70)
                resultado = subprocess.run(cmd, timeout=3600)
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
        """Verifica se há dados suficientes"""
        if mpoint is None:
            mpoint = self.mpoint_atual
        if mpoint is None:
            raise ValueError("Mpoint deve ser especificado")
        
        print("\n" + "="*80)
        print(f"VERIFICAÇÃO DE DADOS MÍNIMOS - MPOINT: {mpoint}")
        print("="*80)
        
        dir_periodos = self.dir_raw_preenchido
        arquivos_finais = list(dir_periodos.glob(f'periodo_*_final_{mpoint}.csv'))
        
        if len(arquivos_finais) == 0:
            print(f"   ✗ Nenhum arquivo de período processado encontrado para mpoint {mpoint}")
            return False
        
        print(f"   Encontrados {len(arquivos_finais)} períodos processados")
        
        if len(arquivos_finais) < 2:
            print(f"   ✗ Pelo menos 2 períodos são necessários (encontrado: {len(arquivos_finais)})")
            return False
        
        duracao_total_horas = 0
        total_registros = 0
        
        for arq in arquivos_finais:
            try:
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
        print(f"   - Mínimo necessário: {minimo_dias} dias")
        
        if duracao_total_dias < minimo_dias:
            print(f"   [ERRO] Dados insuficientes (faltam {minimo_dias - duracao_total_dias:.1f} dias)")
            return False
        
        print(f"   [OK] Dados suficientes para treino K-means")
        return True
    
    def executar_treino_kmeans_mecanico(self, mpoint=None):
        """Executa normalização e treino K-means para equipamento MECÂNICO"""
        if mpoint is None:
            mpoint = self.mpoint_atual
        if mpoint is None:
            raise ValueError("Mpoint deve ser especificado")
        
        print("\n" + "="*80)
        print(f"FASE 2: TREINO K-MEANS MECÂNICO - MPOINT: {mpoint}")
        print("="*80)
        
        scripts = [
            ('normalizar_dados_kmeans_mecanico.py', 'Normalização de dados MECÂNICO'),
            ('kmeans_classificacao_mecanico.py', 'Treino K-means MECÂNICO')
        ]
        
        for script, descricao in scripts:
            print(f"\n--> {descricao}: {script}")
            caminho_script = self.dir_scripts / script
            
            if not caminho_script.exists():
                print(f"   [ERRO] Script não encontrado: {caminho_script}")
                return False
            
            try:
                cmd = [sys.executable, str(caminho_script), '--mpoint', mpoint]
                
                print(f"   [INFO] Executando: {' '.join(cmd)}")
                print("   " + "="*70)
                resultado = subprocess.run(cmd, timeout=1800)
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
    
    def gerar_parametros_mecanico(self, mpoint=None):
        """Gera e salva parâmetros para equipamento MECÂNICO"""
        if mpoint is None:
            mpoint = self.mpoint_atual
        if mpoint is None:
            raise ValueError("Mpoint deve ser especificado")
        
        print("\n" + "="*80)
        print(f"FASE 3: GERAÇÃO DE PARÂMETROS MECÂNICO - MPOINT: {mpoint}")
        print("="*80)
        
        required_artifacts = {
            'kmeans_model': kmeans_model_path(mpoint),
            'scaler_model': scaler_model_path(mpoint),
            'info_kmeans': info_kmeans_path(mpoint),
        }
        
        missing_required = [nome for nome, path in required_artifacts.items() if not path.exists()]
        if missing_required:
            print("   [ERRO] Artefatos obrigatórios ausentes:")
            for nome in missing_required:
                print(f"      - {nome}: {required_artifacts[nome]}")
            return False
        
        config = {
            'mpoint': mpoint,
            'equipment_type': 'MECHANICAL',
            'data_sources': ['temperature', 'vibration'],
            'no_current_rpm': True,
            'data_treino': datetime.now().isoformat(),
            'artefatos': {
                nome: str(path.relative_to(self.base_dir))
                for nome, path in required_artifacts.items()
            },
            'colunas_temperatura': ['object_temp'],
            'colunas_vibracao': ['vel_rms_x', 'vel_rms_y', 'vel_rms_z', 'vel_max_x', 'vel_max_y', 'vel_max_z'],
            'colunas_magnetometro': ['mag_x', 'mag_y', 'mag_z'],
            'colunas_slip': ['fe_frequency', 'fe_magnitude_-_1', 'fe_magnitude_0', 'fe_magnitude_1', 'fr_frequency', 'rms']
        }
        
        arquivo_config = config_path(mpoint, create=True)
        with open(arquivo_config, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n   [OK] Configuração atualizada: {arquivo_config}")
        return True
    
    def executar_visualizacao_3d_mecanico(self, mpoint=None):
        """Executa visualização 3D para equipamento MECÂNICO"""
        if mpoint is None:
            mpoint = self.mpoint_atual
        if mpoint is None:
            raise ValueError("Mpoint deve ser especificado")
        
        print("\n   [INFO] Gerando visualização 3D (temperatura + vibração)...")
        
        script_path = self.dir_scripts / 'visualizar_clusters_3d_mecanico.py'
        
        if not script_path.exists():
            print("   [ERRO] Script de visualização 3D mecânico não encontrado")
            return False
        
        try:
            cmd = [sys.executable, str(script_path), '--mpoint', mpoint]
            
            print(f"   [INFO] Executando: {' '.join(cmd)}")
            print("   " + "="*70)
            resultado = subprocess.run(cmd, timeout=300)
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
    
    def abrir_graficos_gerados(self, mpoint):
        """Abre os gráficos e relatórios gerados automaticamente após o treino"""
        import os
        import platform
        
        print("\n   [INFO] Abrindo gráficos e relatórios gerados...")
        
        # Lista de arquivos a procurar
        arquivos = []
        
        # Gráfico de normalização
        grafico_norm = self.base_dir / 'plots' / f'dados_normalizados_analise_{mpoint}_mecanico.png'
        if grafico_norm.exists():
            arquivos.append(grafico_norm)
        
        # Gráfico de K-means
        grafico_kmeans = self.base_dir / 'results' / f'analise_kmeans_clusters_mecanico_{mpoint}.png'
        if grafico_kmeans.exists():
            arquivos.append(grafico_kmeans)
        
        # Gráfico 3D
        dirs_mpoint = get_mpoint_dirs(mpoint, create=False)
        grafico_3d = dirs_mpoint['results'] / f'estados_temperatura_vibracao_tempo_3d_{mpoint}.png'
        if grafico_3d.exists():
            arquivos.append(grafico_3d)
        
        # Relatório TXT
        relatorio_txt = dirs_mpoint['results'] / f'relatorio_treinamento_{mpoint}_mecanico.txt'
        if relatorio_txt.exists():
            arquivos.append(relatorio_txt)
        
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
    
    def modo_treino_mecanico(self, mpoint=None):
        """Modo de treino para equipamento MECÂNICO"""
        print("\n" + "="*80)
        print("MODO: TREINO - EQUIPAMENTO MECÂNICO")
        print("="*80)
        
        if mpoint is None:
            mpoints = self.listar_mpoints_disponiveis_mecanicos()
            
            if len(mpoints) == 0:
                return
            
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
            mpoint = mpoint_selecionado['mpoint']
            print(f"\n   [OK] Mpoint selecionado: {mpoint} (MECÂNICO)")
        else:
            print(f"   [OK] Mpoint especificado: {mpoint} (MECÂNICO)")
        
        self.mpoint_atual = mpoint
        
        # Verificar arquivos
        completo, arquivos = self.verificar_arquivos_raw_mecanico(mpoint)
        if not completo:
            print(f"   [ERRO] Arquivos incompletos ou não é equipamento MECÂNICO")
            print("   Equipamentos MECÂNICOS precisam de:")
            print("   - dados_c_XXX.csv (temperatura + vibração)")
            print("   - dados_slip_XXX.csv")
            print("   E NÃO devem ter: dados_estimated_XXX.csv")
            return
        
        # Executar pipeline
        if not self.executar_processamento_mecanico(mpoint):
            print("\n   [ERRO] Falha no processamento")
            return
        
        # Verificar duração mínima
        if not self.verificar_duracao_minima(mpoint, minimo_dias=30):
            print("\n   [AVISO] Dados insuficientes, mas continuando com treino...")
            if not self.verificar_duracao_minima(mpoint, minimo_dias=1):
                print("\n   [ERRO] Dados insuficientes (menos de 1 dia)")
                return
        
        # Treinar K-means
        if not self.executar_treino_kmeans_mecanico(mpoint):
            print("\n   [ERRO] Falha no treino K-means")
            return
        
        # Gerar parâmetros
        if not self.gerar_parametros_mecanico(mpoint):
            print("\n   [ERRO] Falha na geração de parâmetros")
            return
        
        # Gerar visualização 3D
        print("\n   [INFO] Gerando visualização 3D dos clusters...")
        if not self.executar_visualizacao_3d_mecanico(mpoint):
            print("   [AVISO] Não foi possível gerar visualização 3D, mas treino foi concluído")
        else:
            print("   [OK] Visualização 3D gerada com sucesso")
        
        print("\n" + "="*80)
        print("TREINO MECÂNICO CONCLUÍDO COM SUCESSO!")
        print("="*80)
        print(f"   Tipo: EQUIPAMENTO MECÂNICO (temperatura + vibração)")
        print(f"   Parâmetros salvos em: models/{mpoint}/")
        print(f"   Visualizações disponíveis em: results/ e plots/")
        
        # Abrir gráficos automaticamente
        self.abrir_graficos_gerados(mpoint)
        
        # Gerar logs
        training_log = create_training_log(
            script_name='pipeline_deteccao_estados_mecanico',
            mpoint=mpoint,
            model_info={
                'equipment_type': 'MECHANICAL',
                'data_sources': ['temperature', 'vibration'],
                'pipeline_type': 'complete_training_pipeline_mechanical',
                'algorithm': 'K-means_mechanical_temperature_vibration',
                'n_clusters': 6,
                'classification_strategy': 'Dynamic_2_state_mechanical'
            },
            training_data_info={
                'data_sources': ['dados_c_mechanical', 'slip_sensors'],
                'no_current_rpm': True,
                'processing_steps': [
                    'mechanical_data_processing',
                    'data_interpolation',
                    'period_unification',
                    'normalization_mechanical',
                    'kmeans_clustering_mechanical'
                ]
            },
            performance_metrics={
                'pipeline_completion': True,
                'equipment_type': 'MECHANICAL'
            },
            model_files=[],
            processing_time=0,
            training_parameters={
                'mpoint': mpoint,
                'equipment_type': 'MECHANICAL',
                'training_mode': 'complete_pipeline_mechanical'
            }
        )
        
        save_log(training_log, 'pipeline_deteccao_estados_mecanico', mpoint, 'training_pipeline_complete')
        
        enrich_results_file(mpoint, {
            'pipeline_training_completed': True,
            'equipment_type': 'MECHANICAL',
            'pipeline_training_timestamp': datetime.now().isoformat(),
            'mpoint': mpoint
        })
    
    def menu_principal(self):
        """Menu principal do pipeline MECÂNICO"""
        print("\n" + "="*80)
        print("PIPELINE DE DETECÇÃO DE ESTADOS - EQUIPAMENTOS MECÂNICOS")
        print("="*80)
        print("\n1. TREINO - Processar dados e gerar parâmetros K-means (MECÂNICO)")
        print("2. Sair")
        
        try:
            opcao = int(input("\nEscolha uma opção: "))
        except:
            print("Opção inválida")
            return
        
        if opcao == 1:
            self.modo_treino_mecanico()
        elif opcao == 2:
            print("Encerrando...")
        else:
            print("Opção inválida")
    
    def parse_args(self):
        """Parse argumentos de linha de comando"""
        parser = argparse.ArgumentParser(
            description="Pipeline de Detecção de Estados - EQUIPAMENTOS MECÂNICOS"
        )
        parser.add_argument('--mpoint', type=str, help='Mpoint para processar (ex: c_640)')
        parser.add_argument('--modo', choices=['treino'], required=False, help='Modo de operação')
        parser.add_argument('--auto', action='store_true', help='Executar automaticamente')
        
        return parser.parse_args()

def main():
    """Função principal"""
    pipeline = PipelineDeteccaoEstadosMecanico()
    
    if len(sys.argv) > 1:
        args = pipeline.parse_args()
        
        if args.modo and not args.mpoint:
            print("❌ ERRO: --mpoint é obrigatório quando --modo é especificado")
            return
        
        if args.modo:
            if args.modo == 'treino':
                pipeline.modo_treino_mecanico(mpoint=args.mpoint)
            else:
                print(f"❌ ERRO: Modo inválido: {args.modo}")
        else:
            if args.mpoint:
                pipeline = PipelineDeteccaoEstadosMecanico(mpoint=args.mpoint)
            pipeline.menu_principal()
    else:
        pipeline.menu_principal()

if __name__ == "__main__":
    main()

