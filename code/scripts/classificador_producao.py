#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para classificação em produção usando dados_c_636.csv
Usa modelo treinado com dados completos mas funciona com 12 features de produção
Permite filtrar dados por range de data e hora
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import json
import argparse
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class ClassificadorProducao:
    def __init__(self, modelo_path="models/cnn_model_robusto.h5", 
                 label_encoder_path="models/label_encoder_robusto.pkl",
                 scaler_path="models/scaler_maxmin.pkl", use_normalized_data=True):
        """
        Inicializa o classificador para produção
        
        Args:
            modelo_path (str): Caminho para o modelo CNN .h5
            label_encoder_path (str): Caminho para o label encoder .pkl
            scaler_path (str): Caminho para o scaler .pkl (usado apenas se use_normalized_data=False)
            use_normalized_data (bool): Se True, usa dados já normalizados (dados_kmeans.csv)
        """
        self.modelo_path = modelo_path
        self.label_encoder_path = label_encoder_path
        self.scaler_path = scaler_path
        self.use_normalized_data = use_normalized_data
        
        # Carregar modelo e componentes
        self.modelo = None
        self.label_encoder = None
        self.scaler = None
        self.feature_columns = None
        
        self.carregar_modelo()
    
    def carregar_modelo(self):
        """Carrega o modelo CNN, label encoder e scaler (se necessário)"""
        print("📦 Carregando modelo para produção...")
        
        try:
            # Carregar modelo CNN
            self.modelo = tf.keras.models.load_model(self.modelo_path)
            print(f"  - Modelo CNN carregado: {self.modelo_path}")
            
            # Carregar label encoder
            self.label_encoder = joblib.load(self.label_encoder_path)
            print(f"  - Label encoder carregado: {self.label_encoder_path}")
            
            # Carregar scaler apenas se necessário
            if not self.use_normalized_data:
                self.scaler = joblib.load(self.scaler_path)
                print(f"  - Scaler carregado: {self.scaler_path}")
            else:
                print("  - Scaler: Não necessário (dados já normalizados)")
            
            # Carregar informações do modelo
            if 'robusto' in self.modelo_path:
                info_file = 'models/info_modelo_robusto.json'
            elif 'conservador' in self.modelo_path:
                info_file = 'models/info_treinamento_conservador_otimizado.json'
            else:
                info_file = 'models/info_treinamento_cnn_convae_moderado.json'
            
            with open(info_file, 'r') as f:
                info = json.load(f)
                self.feature_columns = info['feature_columns']
            
            print(f"  - Features esperadas: {len(self.feature_columns)}")
            print("  - Modelo carregado com sucesso!")
            
        except Exception as e:
            print(f"❌ Erro ao carregar modelo: {str(e)}")
            raise
    
    def filtrar_por_data_hora(self, df, data_inicio, data_fim):
        """
        Filtra dados por range de data e hora
        
        Args:
            df (pd.DataFrame): DataFrame com dados
            data_inicio (str): Data/hora de início no formato 'YYYY-MM-DD HH:MM:SS'
            data_fim (str): Data/hora de fim no formato 'YYYY-MM-DD HH:MM:SS'
            
        Returns:
            pd.DataFrame: Dados filtrados
        """
        print(f"🕒 Filtrando dados por range de data/hora...")
        print(f"  - Início: {data_inicio}")
        print(f"  - Fim: {data_fim}")
        
        # Converter coluna time para datetime (mantém timezone se presente)
        df['time'] = pd.to_datetime(df['time'])
        
        # Converter strings de data para datetime com timezone UTC
        dt_inicio = pd.to_datetime(data_inicio, utc=True)
        dt_fim = pd.to_datetime(data_fim, utc=True)
        
        # Filtrar dados
        df_filtrado = df[(df['time'] >= dt_inicio) & (df['time'] <= dt_fim)]
        
        print(f"  - Dados originais: {len(df):,}")
        print(f"  - Dados filtrados: {len(df_filtrado):,}")
        print(f"  - Percentual mantido: {len(df_filtrado)/len(df)*100:.1f}%")
        
        return df_filtrado
    
    def preparar_dados_producao(self, df):
        """
        Prepara dados de produção (dados_unificados_final.csv) para classificação
        
        Args:
            df (pd.DataFrame): DataFrame com dados de produção
            
        Returns:
            np.array: Dados preparados para classificação
        """
        print("🔄 Preparando dados de produção...")
        
        # Features esperadas pelo modelo robusto (19 features)
        features_esperadas = [
            'mag_x', 'mag_y', 'mag_z', 'object_temp', 
            'vel_max_x', 'vel_max_y', 'vel_rms_x', 
            'vel_max_z', 'vel_rms_y', 'vel_rms_z',
            'estimated_current', 'estimated_rotational_speed', 'estimated_vel_rms',
            'slip_fe_frequency', 'slip_fe_magnitude_-_1', 'slip_fe_magnitude_0', 'slip_fe_magnitude_1',
            'slip_fr_frequency', 'slip_rms'
        ]
        
        # Verificar se todas as features estão disponíveis
        missing_features = [col for col in features_esperadas if col not in df.columns]
        if missing_features:
            print(f"⚠️ Features ausentes: {len(missing_features)}")
            print(f"  - Primeiras 10 ausentes: {missing_features[:10]}")
            raise ValueError(f"Features ausentes: {missing_features}")
        
        # Selecionar apenas as features esperadas pelo modelo
        df_features = df[features_esperadas].copy()
        
        # Normalizar dados usando o scaler treinado
        dados_normalizados = self.scaler.transform(df_features)
        
        print(f"  - Features esperadas: {len(features_esperadas)}")
        print(f"  - Dados preparados: {dados_normalizados.shape}")
        print(f"  - Normalização: Scaler treinado aplicado")
        
        return dados_normalizados
    
    def preparar_dados_normalizados(self, df):
        """
        Prepara dados já normalizados (dados_kmeans.csv) para classificação
        
        Args:
            df (pd.DataFrame): DataFrame com dados já normalizados
            
        Returns:
            np.array: Dados preparados para classificação
        """
        print("🔄 Preparando dados normalizados...")
        
        # Features esperadas pelo modelo robusto (19 features)
        features_esperadas = [
            'mag_x', 'mag_y', 'mag_z', 'object_temp', 
            'vel_max_x', 'vel_max_y', 'vel_rms_x', 
            'vel_max_z', 'vel_rms_y', 'vel_rms_z',
            'estimated_current', 'estimated_rotational_speed', 'estimated_vel_rms',
            'slip_fe_frequency', 'slip_fe_magnitude_-_1', 'slip_fe_magnitude_0', 'slip_fe_magnitude_1',
            'slip_fr_frequency', 'slip_rms'
        ]
        
        # Verificar se todas as features estão disponíveis
        missing_features = [col for col in features_esperadas if col not in df.columns]
        if missing_features:
            print(f"⚠️ Features ausentes: {len(missing_features)}")
            print(f"  - Primeiras 10 ausentes: {missing_features[:10]}")
            raise ValueError(f"Features ausentes: {missing_features}")
        
        # Selecionar apenas as features esperadas pelo modelo
        df_features = df[features_esperadas].copy()
        
        # Dados já estão normalizados, apenas converter para numpy
        dados_normalizados = df_features.values
        
        print(f"  - Features esperadas: {len(features_esperadas)}")
        print(f"  - Dados preparados: {dados_normalizados.shape}")
        print(f"  - Normalização: Dados já normalizados (0-1)")
        
        return dados_normalizados
    
    def criar_sequencias_producao(self, dados, window_size=30):
        """
        Cria sequências temporais para classificação
        
        Args:
            dados (np.array): Dados normalizados
            window_size (int): Tamanho da janela temporal
            
        Returns:
            np.array: Sequências para classificação
        """
        print(f"🔄 Criando sequências de {window_size} timesteps...")
        
        if len(dados) < window_size:
            print(f"⚠️ Dados insuficientes: {len(dados)} < {window_size}")
            return None
        
        # Criar sequências deslizantes
        sequencias = []
        for i in range(len(dados) - window_size + 1):
            sequencias.append(dados[i:i+window_size])
        
        sequencias = np.array(sequencias)
        print(f"  - Sequências criadas: {len(sequencias)}")
        
        return sequencias
    
    def classificar_sequencias(self, sequencias, detectar_incerteza=False, n_samples=100):
        """
        Classifica sequências usando o modelo CNN com opção de detecção de incerteza
        
        Args:
            sequencias (np.array): Sequências para classificação
            detectar_incerteza (bool): Se True, usa Monte Carlo Dropout para detectar incerteza
            n_samples (int): Número de amostras para Monte Carlo Dropout
            
        Returns:
            tuple: (predicoes, probabilidades, incertezas)
        """
        print("🧠 Classificando sequências...")
        
        if sequencias is None or len(sequencias) == 0:
            return None, None, None
        
        if detectar_incerteza:
            print(f"  - Detecção de incerteza ativada ({n_samples} amostras)")
            # Monte Carlo Dropout para detecção de incerteza
            predictions = []
            for _ in range(n_samples):
                pred = self.modelo(sequencias, training=True)  # Dropout ativo
                predictions.append(pred)
            
            predictions = np.array(predictions)
            probabilidades = np.mean(predictions, axis=0)
            incertezas = -np.sum(probabilidades * np.log(probabilidades + 1e-8), axis=1)
        else:
            # Predição normal
            probabilidades = self.modelo.predict(sequencias, verbose=0)
            incertezas = None
        
        predicoes = np.argmax(probabilidades, axis=1)
        
        # Converter para labels
        labels = self.label_encoder.inverse_transform(predicoes)
        
        print(f"  - Sequências classificadas: {len(labels)}")
        print(f"  - Distribuição: {pd.Series(labels).value_counts().to_dict()}")
        
        if incertezas is not None:
            alta_incerteza = np.sum(incertezas > 0.5)
            print(f"  - Amostras com alta incerteza: {alta_incerteza}/{len(incertezas)} ({alta_incerteza/len(incertezas)*100:.1f}%)")
            print(f"  - Incerteza média: {np.mean(incertezas):.4f}")
        
        return labels, probabilidades, incertezas
    
    def classificar_arquivo(self, arquivo_path, window_size=30, data_inicio=None, data_fim=None):
        """
        Classifica um arquivo de dados_c_636.csv com opção de filtrar por data/hora
        
        Args:
            arquivo_path (str): Caminho para o arquivo
            window_size (int): Tamanho da janela temporal
            data_inicio (str, optional): Data/hora de início no formato 'YYYY-MM-DD HH:MM:SS'
            data_fim (str, optional): Data/hora de fim no formato 'YYYY-MM-DD HH:MM:SS'
            
        Returns:
            pd.DataFrame: Resultados da classificação
        """
        print(f"📁 Classificando arquivo: {arquivo_path}")
        
        # Carregar dados
        df = pd.read_csv(arquivo_path)
        print(f"  - Linhas carregadas: {len(df):,}")
        
        # Filtrar por data/hora se especificado
        if data_inicio and data_fim:
            df = self.filtrar_por_data_hora(df, data_inicio, data_fim)
            if len(df) == 0:
                print("❌ Nenhum dado encontrado no range especificado")
                return None
        
        # Preparar dados
        dados_producao = self.preparar_dados_producao(df)
        
        # Criar sequências
        sequencias = self.criar_sequencias_producao(dados_producao, window_size)
        
        if sequencias is None:
            print("❌ Não foi possível criar sequências")
            return None
        
        # Classificar
        labels, probabilidades, incertezas = self.classificar_sequencias(sequencias)
        
        if labels is None:
            print("❌ Não foi possível classificar")
            return None
        
        # Criar DataFrame de resultados
        resultados = pd.DataFrame({
            'timestamp': df['time'].iloc[window_size-1:],  # Timestamp do último elemento da sequência
            'predicao': labels,
            'prob_ligado': probabilidades[:, 1],  # Probabilidade de LIGADO
            'prob_desligado': probabilidades[:, 0]  # Probabilidade de DESLIGADO
        })
        
        # Adicionar coluna de incerteza se disponível
        if incertezas is not None:
            resultados['incerteza'] = incertezas
            resultados['alta_incerteza'] = incertezas > 0.5
        
        print(f"  - Resultados gerados: {len(resultados):,}")
        
        return resultados
    
    def classificar_arquivo_normalizado(self, arquivo_path, window_size=30, data_inicio=None, data_fim=None):
        """
        Classifica um arquivo de dados já normalizados (dados_kmeans.csv) com opção de filtrar por data/hora
        
        Args:
            arquivo_path (str): Caminho para o arquivo
            window_size (int): Tamanho da janela temporal
            data_inicio (str, optional): Data/hora de início no formato 'YYYY-MM-DD HH:MM:SS'
            data_fim (str, optional): Data/hora de fim no formato 'YYYY-MM-DD HH:MM:SS'
            
        Returns:
            pd.DataFrame: Resultados da classificação
        """
        print(f"📁 Classificando arquivo normalizado: {arquivo_path}")
        
        # Carregar dados
        df = pd.read_csv(arquivo_path)
        print(f"  - Linhas carregadas: {len(df):,}")
        
        # Filtrar por data/hora se especificado
        if data_inicio and data_fim:
            df = self.filtrar_por_data_hora(df, data_inicio, data_fim)
            if len(df) == 0:
                print("❌ Nenhum dado encontrado no range especificado")
                return None
        
        # Preparar dados (já normalizados)
        dados_producao = self.preparar_dados_normalizados(df)
        
        # Criar sequências
        sequencias = self.criar_sequencias_producao(dados_producao, window_size)
        
        if sequencias is None:
            print("❌ Não foi possível criar sequências")
            return None
        
        # Classificar
        labels, probabilidades, incertezas = self.classificar_sequencias(sequencias)
        
        if labels is None:
            print("❌ Não foi possível classificar")
            return None
        
        # Criar DataFrame de resultados
        resultados = pd.DataFrame({
            'timestamp': df['time'].iloc[window_size-1:],  # Timestamp do último elemento da sequência
            'predicao': labels,
            'prob_ligado': probabilidades[:, 1],  # Probabilidade de LIGADO
            'prob_desligado': probabilidades[:, 0]  # Probabilidade de DESLIGADO
        })
        
        # Adicionar coluna de incerteza se disponível
        if incertezas is not None:
            resultados['incerteza'] = incertezas
            resultados['alta_incerteza'] = incertezas > 0.5
        
        print(f"  - Resultados gerados: {len(resultados):,}")
        
        return resultados
    
    def classificar_janela_3min(self, df, window_size=30):
        """
        Classifica uma janela de 3 minutos de dados
        
        Args:
            df (pd.DataFrame): Dados da janela
            window_size (int): Tamanho da janela temporal
            
        Returns:
            dict: Resultado da classificação
        """
        print("⏱️ Classificando janela de 3 minutos...")
        
        # Preparar dados
        dados_producao = self.preparar_dados_producao(df)
        
        # Criar sequências
        sequencias = self.criar_sequencias_producao(dados_producao, window_size)
        
        if sequencias is None:
            return {
                'status': 'ERRO',
                'motivo': 'Dados insuficientes para criar sequências',
                'predicao': None,
                'probabilidades': None
            }
        
        # Classificar
        labels, probabilidades, incertezas = self.classificar_sequencias(sequencias, detectar_incerteza=True)
        
        if labels is None:
            return {
                'status': 'ERRO',
                'motivo': 'Falha na classificação',
                'predicao': None,
                'probabilidades': None
            }
        
        # Calcular resultado final (maioria das predições)
        predicao_final = pd.Series(labels).mode()[0]
        prob_media = np.mean(probabilidades, axis=0)
        incerteza_media = np.mean(incertezas) if incertezas is not None else 0.0
        
        return {
            'status': 'SUCESSO',
            'predicao': predicao_final,
            'probabilidades': {
                'ligado': float(prob_media[1]),
                'desligado': float(prob_media[0])
            },
            'confianca': float(max(prob_media)),
            'incerteza': float(incerteza_media),
            'alta_incerteza': incerteza_media > 0.5,
            'total_sequencias': len(labels)
        }
    
    def salvar_resultados(self, resultados, arquivo_saida=None):
        """
        Salva resultados da classificação em arquivo CSV
        
        Args:
            resultados (pd.DataFrame): Resultados da classificação
            arquivo_saida (str, optional): Caminho do arquivo de saída
        """
        if resultados is None:
            print("❌ Nenhum resultado para salvar")
            return
        
        if arquivo_saida is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            arquivo_saida = f"results/classificacao_producao_{timestamp}.csv"
        
        # Criar diretório se não existir
        import os
        os.makedirs(os.path.dirname(arquivo_saida), exist_ok=True)
        
        # Salvar resultados
        resultados.to_csv(arquivo_saida, index=False)
        print(f"💾 Resultados salvos em: {arquivo_saida}")
        
        # Estatísticas resumidas
        print(f"\n📊 Resumo da classificação:")
        print(f"  - Total de predições: {len(resultados):,}")
        print(f"  - Distribuição:")
        for classe, count in resultados['predicao'].value_counts().items():
            print(f"    • {classe}: {count:,} ({count/len(resultados)*100:.1f}%)")
        print(f"  - Probabilidade média LIGADO: {resultados['prob_ligado'].mean():.3f}")
        print(f"  - Probabilidade média DESLIGADO: {resultados['prob_desligado'].mean():.3f}")
        
        # Estatísticas de incerteza se disponível
        if 'incerteza' in resultados.columns:
            alta_incerteza = resultados['alta_incerteza'].sum()
            print(f"  - Amostras com alta incerteza: {alta_incerteza:,} ({alta_incerteza/len(resultados)*100:.1f}%)")
            print(f"  - Incerteza média: {resultados['incerteza'].mean():.4f}")
            print(f"  - Incerteza máxima: {resultados['incerteza'].max():.4f}")

def parse_arguments():
    """Configura e parseia argumentos da linha de comando"""
    parser = argparse.ArgumentParser(
        description="Classificador de produção com filtro por data/hora",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:

1. Classificar arquivo completo:
   python classificador_producao.py

2. Classificar com filtro de data/hora:
   python classificador_producao.py --inicio "2025-02-18 16:30:00" --fim "2025-02-18 17:00:00"

3. Classificar com arquivo específico e salvar resultados:
   python classificador_producao.py --arquivo data/raw/dados_c_636.csv --saida results/meus_resultados.csv

4. Classificar com janela personalizada:
   python classificador_producao.py --janela 100 --inicio "2025-02-18 16:30:00" --fim "2025-02-18 16:45:00"
        """
    )
    
    parser.add_argument(
        '--arquivo', 
        type=str, 
        default='data/normalized/dados_kmeans.csv',
        help='Caminho para o arquivo de dados (padrão: data/normalized/dados_kmeans.csv)'
    )
    
    parser.add_argument(
        '--inicio', 
        type=str,
        help='Data/hora de início no formato "YYYY-MM-DD HH:MM:SS" (ex: "2025-02-18 16:30:00")'
    )
    
    parser.add_argument(
        '--fim', 
        type=str,
        help='Data/hora de fim no formato "YYYY-MM-DD HH:MM:SS" (ex: "2025-02-18 17:00:00")'
    )
    
    parser.add_argument(
        '--janela', 
        type=int, 
        default=30,
        help='Tamanho da janela temporal para classificação (padrão: 30)'
    )
    
    parser.add_argument(
        '--saida', 
        type=str,
        help='Caminho para salvar resultados (padrão: results/classificacao_producao_TIMESTAMP.csv)'
    )
    
    parser.add_argument(
        '--modelo', 
        type=str, 
        default='models/cnn_model_robusto.h5',
        help='Caminho para o modelo CNN (padrão: models/cnn_model_robusto.h5)'
    )
    
    parser.add_argument(
        '--label-encoder', 
        type=str, 
        default='models/label_encoder_robusto.pkl',
        help='Caminho para o label encoder (padrão: models/label_encoder_robusto.pkl)'
    )
    
    parser.add_argument(
        '--scaler', 
        type=str, 
        default='models/scaler_maxmin.pkl',
        help='Caminho para o scaler (padrão: models/scaler_maxmin.pkl)'
    )
    
    return parser.parse_args()

def solicitar_data_hora():
    """Solicita data e hora do usuário via teclado"""
    print("\n📅 CONFIGURAÇÃO DE DATA/HORA")
    print("-" * 40)
    
    while True:
        try:
            print("\nDigite as datas no formato: YYYY-MM-DD HH:MM:SS")
            print("Exemplo: 2025-02-18 16:30:00")
            
            # Solicitar data de início
            data_inicio = input("\n🗓️ Data/hora de INÍCIO: ").strip()
            if not data_inicio:
                print("❌ Data de início é obrigatória!")
                continue
            
            # Validar formato da data de início
            try:
                pd.to_datetime(data_inicio)
            except:
                print("❌ Formato inválido! Use: YYYY-MM-DD HH:MM:SS")
                continue
            
            # Solicitar data de fim
            data_fim = input("🗓️ Data/hora de FIM: ").strip()
            if not data_fim:
                print("❌ Data de fim é obrigatória!")
                continue
            
            # Validar formato da data de fim
            try:
                pd.to_datetime(data_fim)
            except:
                print("❌ Formato inválido! Use: YYYY-MM-DD HH:MM:SS")
                continue
            
            # Validar se data de fim é posterior à data de início
            dt_inicio = pd.to_datetime(data_inicio)
            dt_fim = pd.to_datetime(data_fim)
            
            if dt_fim <= dt_inicio:
                print("❌ Data de fim deve ser posterior à data de início!")
                continue
            
            # Confirmar com o usuário
            print(f"\n✅ Período selecionado:")
            print(f"  - Início: {data_inicio}")
            print(f"  - Fim: {data_fim}")
            print(f"  - Duração: {dt_fim - dt_inicio}")
            
            confirmar = input("\n🤔 Confirma este período? (s/n): ").strip().lower()
            if confirmar in ['s', 'sim', 'y', 'yes']:
                return data_inicio, data_fim
            else:
                print("🔄 Vamos tentar novamente...")
                continue
                
        except KeyboardInterrupt:
            print("\n\n❌ Operação cancelada pelo usuário.")
            return None, None
        except Exception as e:
            print(f"❌ Erro: {str(e)}")
            continue

def solicitar_configuracoes():
    """Solicita configurações adicionais do usuário"""
    print("\n⚙️ CONFIGURAÇÕES ADICIONAIS")
    print("-" * 40)
    
    # Solicitar janela temporal
    while True:
        try:
            janela_input = input("🔢 Janela temporal (padrão: 30): ").strip()
            if not janela_input:
                janela = 30
                break
            
            janela = int(janela_input)
            if janela < 10:
                print("❌ Janela muito pequena! Mínimo: 10")
                continue
            if janela > 200:
                print("❌ Janela muito grande! Máximo: 200")
                continue
            
            break
        except ValueError:
            print("❌ Digite um número válido!")
            continue
        except KeyboardInterrupt:
            print("\n❌ Operação cancelada.")
            return None
    
    # Solicitar arquivo de saída
    arquivo_saida = input("💾 Arquivo de saída (Enter para auto-gerar): ").strip()
    if not arquivo_saida:
        arquivo_saida = None
    
    return janela, arquivo_saida

def main():
    """Função principal com interface interativa"""
    print("=== CLASSIFICADOR DE PRODUÇÃO COM FILTRO DE DATA/HORA ===")
    print("=" * 60)
    
    try:
        # Verificar se foi chamado com argumentos de linha de comando
        import sys
        if len(sys.argv) > 1:
            # Modo linha de comando
            args = parse_arguments()
            
            # Validar argumentos de data/hora
            if (args.inicio and not args.fim) or (args.fim and not args.inicio):
                print("❌ Erro: --inicio e --fim devem ser especificados juntos")
                return
            
            data_inicio = args.inicio
            data_fim = args.fim
            janela = args.janela
            arquivo_saida = args.saida
            
            print(f"\n🔧 Configurações (linha de comando):")
            print(f"  - Arquivo: {args.arquivo}")
            print(f"  - Janela temporal: {janela}")
            if data_inicio and data_fim:
                print(f"  - Data/hora início: {data_inicio}")
                print(f"  - Data/hora fim: {data_fim}")
            else:
                print(f"  - Filtro de data/hora: Não aplicado")
            
            # Inicializar classificador
            classificador = ClassificadorProducao(
                modelo_path=args.modelo,
                label_encoder_path=args.label_encoder,
                scaler_path=args.scaler
            )
            
            # Classificar arquivo (usar método apropriado baseado no tipo de dados)
            if 'normalized' in args.arquivo:
                resultados = classificador.classificar_arquivo_normalizado(
                    arquivo_path=args.arquivo,
                    window_size=janela,
                    data_inicio=data_inicio,
                    data_fim=data_fim
                )
            else:
                resultados = classificador.classificar_arquivo(
                    arquivo_path=args.arquivo,
                    window_size=janela,
                    data_inicio=data_inicio,
                    data_fim=data_fim
                )
            
        else:
            # Modo interativo
            print("🎯 MODO INTERATIVO")
            print("Você será solicitado a inserir as configurações via teclado.")
            
            # Solicitar range de data/hora
            data_inicio, data_fim = solicitar_data_hora()
            if data_inicio is None:
                print("❌ Operação cancelada.")
                return
            
            # Solicitar configurações adicionais
            config = solicitar_configuracoes()
            if config is None:
                print("❌ Operação cancelada.")
                return
            
            janela, arquivo_saida = config
            
            print(f"\n🔧 Configurações:")
            print(f"  - Arquivo: data/normalized/dados_kmeans.csv")
            print(f"  - Janela temporal: {janela}")
            print(f"  - Data/hora início: {data_inicio}")
            print(f"  - Data/hora fim: {data_fim}")
            
            # Inicializar classificador
            classificador = ClassificadorProducao()
            
            # Classificar arquivo (usar dados normalizados por padrão)
            resultados = classificador.classificar_arquivo_normalizado(
                arquivo_path="data/normalized/dados_kmeans.csv",
                window_size=janela,
                data_inicio=data_inicio,
                data_fim=data_fim
            )
        
        # Salvar resultados
        if resultados is not None:
            classificador.salvar_resultados(resultados, arquivo_saida)
            print("\n✅ Classificação concluída com sucesso!")
        else:
            print("\n❌ Falha na classificação")
        
    except Exception as e:
        print(f"\n❌ Erro durante a execução: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
