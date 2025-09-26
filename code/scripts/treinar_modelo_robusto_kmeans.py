#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para treinar modelo robusto CNN + ConvAE usando dados rotulados do K-means
Com capacidade de detectar incerteza e classificar LIGADO/DESLIGADO
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import json
import time
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configurar TensorFlow
tf.config.set_visible_devices([], 'GPU')  # Usar CPU para estabilidade

class UncertaintyCNN:
    """CNN com detecção de incerteza usando dropout durante inferência"""
    
    def __init__(self, input_shape, num_classes=2):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.uncertainty_model = None
    
    def build_model(self):
        """Constrói modelo CNN com dropout para detecção de incerteza"""
        model = tf.keras.Sequential([
            # Primeira camada convolucional
            tf.keras.layers.Conv1D(64, 3, activation='relu', input_shape=self.input_shape, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Dropout(0.2),
            
            # Segunda camada convolucional
            tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Dropout(0.2),
            
            # Terceira camada convolucional
            tf.keras.layers.Conv1D(256, 3, activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.GlobalMaxPooling1D(),
            tf.keras.layers.Dropout(0.3),
            
            # Camadas densas
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),
            
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            
            # Camada de saída
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def build_uncertainty_model(self):
        """Constrói modelo para detecção de incerteza (com dropout ativo)"""
        if self.model is None:
            self.build_model()
        
        # Criar modelo de incerteza com dropout ativo
        uncertainty_model = tf.keras.Sequential([
            # Mesma arquitetura mas com dropout sempre ativo
            tf.keras.layers.Conv1D(64, 3, activation='relu', input_shape=self.input_shape, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Dropout(0.2),
            
            tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Dropout(0.2),
            
            tf.keras.layers.Conv1D(256, 3, activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.GlobalMaxPooling1D(),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),
            
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.uncertainty_model = uncertainty_model
        return uncertainty_model
    
    def predict_with_uncertainty(self, X, n_samples=100):
        """Prediz com detecção de incerteza usando Monte Carlo Dropout"""
        if self.uncertainty_model is None:
            self.build_uncertainty_model()
        
        # Copiar pesos do modelo treinado
        self.uncertainty_model.set_weights(self.model.get_weights())
        
        # Fazer múltiplas predições com dropout ativo
        predictions = []
        for _ in range(n_samples):
            pred = self.uncertainty_model.predict(X, verbose=0)
            predictions.append(pred)
        
        predictions = np.array(predictions)  # Shape: (n_samples, n_data, n_classes)
        
        # Calcular média e variância
        mean_pred = np.mean(predictions, axis=0)
        var_pred = np.var(predictions, axis=0)
        
        # Calcular incerteza (entropia)
        uncertainty = -np.sum(mean_pred * np.log(mean_pred + 1e-8), axis=1)
        
        return mean_pred, uncertainty

class ConvAE:
    """Convolutional Autoencoder para extração de features"""
    
    def __init__(self, input_shape, encoding_dim=64):
        self.input_shape = input_shape
        self.encoding_dim = encoding_dim
        self.encoder = None
        self.decoder = None
        self.autoencoder = None
    
    def build_encoder(self):
        """Constrói o encoder"""
        input_layer = tf.keras.layers.Input(shape=self.input_shape)
        
        x = tf.keras.layers.Conv1D(32, 3, activation='relu', padding='same')(input_layer)
        x = tf.keras.layers.MaxPooling1D(2, padding='same')(x)
        x = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling1D(2, padding='same')(x)
        x = tf.keras.layers.Conv1D(32, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling1D(2, padding='same')(x)
        
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        encoded = tf.keras.layers.Dense(self.encoding_dim, activation='relu')(x)
        
        self.encoder = tf.keras.Model(input_layer, encoded)
        return self.encoder
    
    def build_decoder(self):
        """Constrói o decoder"""
        encoded_input = tf.keras.layers.Input(shape=(self.encoding_dim,))
        
        pooled_size = self.input_shape[0] // 8
        features_after_pooling = 32
        
        x = tf.keras.layers.Dense(128, activation='relu')(encoded_input)
        x = tf.keras.layers.Dense(pooled_size * features_after_pooling, activation='relu')(x)
        x = tf.keras.layers.Reshape((pooled_size, features_after_pooling))(x)
        
        x = tf.keras.layers.Conv1D(32, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.UpSampling1D(2)(x)
        x = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.UpSampling1D(2)(x)
        x = tf.keras.layers.Conv1D(32, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.UpSampling1D(2)(x)
        
        # Ajustar para a dimensão correta
        if x.shape[1] != self.input_shape[0]:
            pad_size = self.input_shape[0] - x.shape[1]
            if pad_size > 0:
                x = tf.keras.layers.ZeroPadding1D(padding=(pad_size//2, pad_size-pad_size//2))(x)
            else:
                crop_size = (x.shape[1] - self.input_shape[0]) // 2
                x = tf.keras.layers.Cropping1D(cropping=(crop_size, crop_size))(x)
        
        decoded = tf.keras.layers.Conv1D(self.input_shape[1], 3, activation='sigmoid', padding='same')(x)
        
        self.decoder = tf.keras.Model(encoded_input, decoded)
        return self.decoder
    
    def build_autoencoder(self):
        """Constrói o autoencoder completo"""
        if self.encoder is None:
            self.build_encoder()
        if self.decoder is None:
            self.build_decoder()
        
        encoded = self.encoder(self.encoder.input)
        decoded = self.decoder(encoded)
        
        self.autoencoder = tf.keras.Model(self.encoder.input, decoded)
        return self.autoencoder

def carregar_dados_rotulados():
    """Carrega dados rotulados do K-means"""
    print("📁 Carregando dados rotulados limpos (apenas clusters de alta certeza)...")
    
    # Carregar dados rotulados (apenas clusters de alta certeza)
    df = pd.read_csv('data/normalized/dados_kmeans_rotulados_conservador.csv')
    
    # Todos os dados já estão classificados (não há nulos)
    df_classificados = df.copy()
    
    print(f"  - Total de amostras limpas: {len(df):,}")
    print(f"  - Amostras para treinamento: {len(df_classificados):,}")
    print(f"  - Estratégia: Apenas clusters com alta certeza")
    
    # Verificar distribuição das classes
    status_counts = df_classificados['equipamento_status'].value_counts()
    print(f"  - Distribuição das classes:")
    for status, count in status_counts.items():
        print(f"    * {status}: {count:,} ({count/len(df_classificados)*100:.1f}%)")
    
    # Separar features e labels
    feature_cols = [col for col in df_classificados.columns 
                   if col not in ['time', 'equipamento_status', 'cluster']]
    
    X = df_classificados[feature_cols].values
    y = df_classificados['equipamento_status'].values
    
    print(f"  - Features: {X.shape[1]}")
    print(f"  - Amostras: {len(X):,}")
    
    return X, y, feature_cols, df_classificados

def preparar_dados_sequencias(X, y, window_size=30, test_size=0.2, max_samples_per_class=25000):
    """Prepara dados em formato de sequências"""
    print(f"\n🔄 Preparando sequências de {window_size} timesteps...")
    
    # Balancear dados
    unique_classes = np.unique(y)
    X_balanced = []
    y_balanced = []
    
    print(f"  - Balanceando dados (máximo por classe: {max_samples_per_class:,})...")
    
    for class_label in unique_classes:
        class_indices = np.where(y == class_label)[0]
        
        if len(class_indices) > max_samples_per_class:
            np.random.seed(42)
            selected_indices = np.random.choice(class_indices, max_samples_per_class, replace=False)
        else:
            selected_indices = class_indices
            
        X_balanced.append(X[selected_indices])
        y_balanced.append(y[selected_indices])
        
        print(f"    * {class_label}: {len(selected_indices):,} amostras")
    
    X_balanced = np.vstack(X_balanced)
    y_balanced = np.hstack(y_balanced)
    
    print(f"  - Dados balanceados: {X_balanced.shape[0]:,} amostras")
    
    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=test_size, random_state=42, stratify=y_balanced
    )
    
    print(f"  - Treino: {X_train.shape[0]:,} amostras")
    print(f"  - Teste: {X_test.shape[0]:,} amostras")
    
    def create_sequences(X, y):
        X_seq = []
        y_seq = []
        
        for i in range(len(X) - window_size + 1):
            X_seq.append(X[i:i+window_size])
            y_seq.append(y[i+window_size-1])
            
        return np.array(X_seq), np.array(y_seq)
    
    # Criar sequências
    print(f"  - Criando sequências de {window_size} timesteps...")
    X_train_seq, y_train_seq = create_sequences(X_train, y_train)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test)
    
    print(f"  - Sequências de treino: {X_train_seq.shape}")
    print(f"  - Sequências de teste: {X_test_seq.shape}")
    
    return X_train_seq, X_test_seq, y_train_seq, y_test_seq

def treinar_convae(X_train, X_test, window_size, features, epochs=5):
    """Treina o ConvAE"""
    print(f"\n🤖 Treinando ConvAE ({epochs} épocas)...")
    
    # Criar ConvAE
    convae = ConvAE(input_shape=(window_size, features), encoding_dim=64)
    autoencoder = convae.build_autoencoder()
    
    # Compilar modelo
    autoencoder.compile(optimizer='adam', loss='mse')
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, 
                                        restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7, verbose=1),
        tf.keras.callbacks.ModelCheckpoint('models/convae_robusto.h5', 
                                         monitor='val_loss', save_best_only=True, 
                                         mode='min', verbose=0)
    ]
    
    # Treinar
    start_time = time.time()
    
    history = autoencoder.fit(
        X_train, X_train,
        epochs=epochs,
        batch_size=32,
        validation_data=(X_test, X_test),
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    print(f"  - Tempo de treinamento: {training_time:.2f} segundos")
    
    # Salvar modelos
    autoencoder.save('models/convae_model_robusto.h5')
    convae.encoder.save('models/convae_encoder_robusto.h5')
    convae.decoder.save('models/convae_decoder_robusto.h5')
    
    print("  - ConvAE salvo: models/convae_model_robusto.h5")
    
    return convae, history

def treinar_cnn_robusto(X_train, X_test, y_train, y_test, window_size, features, epochs=5):
    """Treina a CNN robusta com detecção de incerteza"""
    print(f"\n🧠 Treinando CNN robusta ({epochs} épocas)...")
    
    # Codificar labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    print(f"  - Classes: {le.classes_}")
    
    # Salvar encoder de labels
    joblib.dump(le, 'models/label_encoder_robusto.pkl')
    print("  - Label encoder salvo: models/label_encoder_robusto.pkl")
    
    # Criar CNN
    cnn = UncertaintyCNN(input_shape=(window_size, features), num_classes=2)
    model = cnn.build_model()
    
    # Compilar modelo
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=15, 
                                        restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=8, min_lr=1e-7, verbose=1),
        tf.keras.callbacks.ModelCheckpoint('models/cnn_best_robusto.h5', 
                                         monitor='val_accuracy', save_best_only=True, 
                                         mode='max', verbose=0)
    ]
    
    # Treinar
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train_encoded,
        epochs=epochs,
        batch_size=32,
        validation_data=(X_test, y_test_encoded),
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    print(f"  - Tempo de treinamento: {training_time:.2f} segundos")
    
    # Carregar melhor modelo
    model.load_weights('models/cnn_best_robusto.h5')
    
    # Salvar modelo final
    model.save('models/cnn_model_robusto.h5')
    print("  - CNN salva: models/cnn_model_robusto.h5")
    
    # Avaliar modelo
    test_loss, test_accuracy = model.evaluate(X_test, y_test_encoded, verbose=0)
    print(f"  - Acurácia no teste: {test_accuracy:.4f}")
    
    # Testar detecção de incerteza
    print("  - Testando detecção de incerteza...")
    mean_pred, uncertainty = cnn.predict_with_uncertainty(X_test[:100], n_samples=10)
    
    # Calcular estatísticas de incerteza
    high_uncertainty = np.sum(uncertainty > 0.5)  # Threshold para alta incerteza
    print(f"  - Amostras com alta incerteza: {high_uncertainty}/{len(uncertainty)} ({high_uncertainty/len(uncertainty)*100:.1f}%)")
    print(f"  - Incerteza média: {np.mean(uncertainty):.4f}")
    print(f"  - Incerteza máxima: {np.max(uncertainty):.4f}")
    
    # Relatório de classificação
    y_pred = np.argmax(mean_pred, axis=1)
    print("\n📊 Relatório de classificação:")
    print(classification_report(y_test_encoded[:100], y_pred, target_names=le.classes_))
    
    return cnn, history, le

def criar_visualizacoes_robustas(convae_history, cnn_history, le):
    """Cria visualizações dos resultados"""
    print("\n📊 Criando visualizações...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Loss do ConvAE
    axes[0,0].plot(convae_history.history['loss'], label='Treino', linewidth=2)
    axes[0,0].plot(convae_history.history['val_loss'], label='Validação', linewidth=2)
    axes[0,0].set_title('ConvAE - Loss', fontsize=12, fontweight='bold')
    axes[0,0].set_xlabel('Época')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Loss da CNN
    axes[0,1].plot(cnn_history.history['loss'], label='Treino', linewidth=2)
    axes[0,1].plot(cnn_history.history['val_loss'], label='Validação', linewidth=2)
    axes[0,1].set_title('CNN - Loss', fontsize=12, fontweight='bold')
    axes[0,1].set_xlabel('Época')
    axes[0,1].set_ylabel('Loss')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Accuracy da CNN
    axes[1,0].plot(cnn_history.history['accuracy'], label='Treino', linewidth=2)
    axes[1,0].plot(cnn_history.history['val_accuracy'], label='Validação', linewidth=2)
    axes[1,0].set_title('CNN - Accuracy', fontsize=12, fontweight='bold')
    axes[1,0].set_xlabel('Época')
    axes[1,0].set_ylabel('Accuracy')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Distribuição das classes
    class_counts = pd.Series(le.classes_).value_counts()
    colors = ['#ff9999', '#66b3ff']
    axes[1,1].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%', 
                  colors=colors, startangle=90)
    axes[1,1].set_title('Distribuição das Classes', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/treinamento_modelo_robusto.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("  - Visualizações salvas: results/treinamento_modelo_robusto.png")

def salvar_informacoes_robustas(X_train, X_test, y_train, y_test, feature_cols, window_size, 
                               convae_time, cnn_time, epochs):
    """Salva informações do treinamento"""
    print("\n💾 Salvando informações do treinamento...")
    
    info = {
        'timestamp': datetime.now().isoformat(),
        'modelo': 'CNN + ConvAE Robusto com Detecção de Incerteza',
        'dataset': 'dados_classificados_kmeans_moderado.csv',
        'window_size': window_size,
        'features': len(feature_cols),
        'feature_columns': feature_cols,
        'train_sequences': X_train.shape[0],
        'test_sequences': X_test.shape[0],
        'sequence_shape': X_train.shape[1:],
        'classes': ['DESLIGADO', 'LIGADO'],
        'epochs_treinamento': epochs,
        'training_times': {
            'convae_seconds': convae_time,
            'cnn_seconds': cnn_time,
            'total_seconds': convae_time + cnn_time
        },
        'modelos_salvos': [
            'models/convae_model_robusto.h5',
            'models/convae_encoder_robusto.h5',
            'models/convae_decoder_robusto.h5',
            'models/cnn_model_robusto.h5',
            'models/cnn_best_robusto.h5',
            'models/label_encoder_robusto.pkl'
        ],
        'caracteristicas_especiais': [
            'Detecção de incerteza com Monte Carlo Dropout',
            'CNN com arquitetura robusta',
            'ConvAE para extração de features',
            'Classificação LIGADO/DESLIGADO',
            'Capacidade de detectar quando não tem certeza'
        ]
    }
    
    with open('models/info_modelo_robusto.json', 'w') as f:
        json.dump(info, f, indent=2)
    
    print("  - Informações salvas: models/info_modelo_robusto.json")

def main():
    """Função principal"""
    print("=== TREINAMENTO MODELO ROBUSTO CNN + CONVAE (DADOS K-MEANS) ===")
    print("=" * 70)
    
    # Parâmetros
    EPOCHS = 100  # Treinamento completo com 100 épocas
    WINDOW_SIZE = 30
    
    try:
        # 1. Carregar dados rotulados
        X, y, feature_cols, df_classificados = carregar_dados_rotulados()
        
        # 2. Preparar sequências
        X_train_seq, X_test_seq, y_train_seq, y_test_seq = preparar_dados_sequencias(
            X, y, WINDOW_SIZE
        )
        
        # 3. Treinar ConvAE
        convae_start = time.time()
        convae, convae_history = treinar_convae(
            X_train_seq, X_test_seq, WINDOW_SIZE, len(feature_cols), EPOCHS
        )
        convae_time = time.time() - convae_start
        
        # Limpar memória
        del convae
        gc.collect()
        
        # 4. Treinar CNN robusta
        cnn_start = time.time()
        cnn, cnn_history, le = treinar_cnn_robusto(
            X_train_seq, X_test_seq, y_train_seq, y_test_seq, 
            WINDOW_SIZE, len(feature_cols), EPOCHS
        )
        cnn_time = time.time() - cnn_start
        
        # 5. Criar visualizações
        criar_visualizacoes_robustas(convae_history, cnn_history, le)
        
        # 6. Salvar informações
        salvar_informacoes_robustas(
            X_train_seq, X_test_seq, y_train_seq, y_test_seq, 
            feature_cols, WINDOW_SIZE, convae_time, cnn_time, EPOCHS
        )
        
        print("\n=== TREINAMENTO ROBUSTO CONCLUÍDO COM SUCESSO ===")
        print(f"\n⏱️  Tempos de treinamento:")
        print(f"  - ConvAE: {convae_time:.2f} segundos")
        print(f"  - CNN: {cnn_time:.2f} segundos")
        print(f"  - Total: {convae_time + cnn_time:.2f} segundos")
        
        print("\n📁 Modelos treinados e salvos:")
        print("  - ConvAE: models/convae_model_robusto.h5")
        print("  - CNN: models/cnn_model_robusto.h5")
        print("  - Label Encoder: models/label_encoder_robusto.pkl")
        
        print("\n🎯 Características especiais:")
        print("  - Detecção de incerteza com Monte Carlo Dropout")
        print("  - Classificação LIGADO/DESLIGADO")
        print("  - Capacidade de detectar quando não tem certeza")
        print("  - Arquitetura robusta CNN + ConvAE")
        
        print(f"\n✅ Teste com {EPOCHS} épocas concluído!")
        print("💡 Se funcionou bem, execute novamente com 100 épocas para treinamento completo")
        
    except Exception as e:
        print(f"\n❌ Erro durante o treinamento: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
