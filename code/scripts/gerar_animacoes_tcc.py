import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor
from sklearn.impute import KNNImputer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os

# Configurações de estilo
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = ['#e74c3c', '#2ecc71'] # OFF (Vermelho), ON (Verde)
SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'plots', 'animacoes_tcc')
os.makedirs(SAVE_DIR, exist_ok=True)

def load_real_data():
    """Carrega dados reais do mpoint c_1518 para ilustrações mais fiéis."""
    file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw', 'dados_c_1518.csv')
    print(f"Carregando dados reais de: {file_path}")
    df = pd.read_csv(file_path)
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time')
    
    # Adicionar features temporais necessárias para o KNN Multivariate
    df['hora'] = df['time'].dt.hour
    df['hora_sin'] = np.sin(2 * np.pi * df['hora'] / 24)
    df['hora_cos'] = np.cos(2 * np.pi * df['hora'] / 24)
    df['dia_semana'] = df['time'].dt.dayofweek
    df['dia_sin'] = np.sin(2 * np.pi * df['dia_semana'] / 7)
    df['dia_cos'] = np.cos(2 * np.pi * df['dia_semana'] / 7)
    df['timestamp_num'] = (df['time'] - df['time'].min()).dt.total_seconds()
    
    return df

def plot_knn_interpolation(df_full):
    """Gera visualização de KNN utilizando dados REAIS de c_1518."""
    print("  --> Gerando gráfico 02_knn_interpolacao.png com dados reais...")
    
    target_date = '2025-07-20'
    df_day = df_full[df_full['time'].dt.date == pd.to_datetime(target_date).date()].copy()
    df_plot = df_day[df_day['time'].dt.hour < 14].copy()
    
    # Simular Gap de 2.5h (10:00 - 12:30)
    gap_start = df_plot['time'].min() + pd.Timedelta(hours=10)
    gap_end = gap_start + pd.Timedelta(hours=2.5)
    gap_mask = (df_plot['time'] > gap_start) & (df_plot['time'] < gap_end)
    
    # Criar DataFrame para o "Sensor" que liga os pontos com uma reta (Linear Simples)
    # Removemos os dados no gap para simular a lacuna, mas o plot de linha do matplotlib ligará os pontos
    df_sensor = df_plot.copy()
    df_sensor.loc[gap_mask, 'vel_rms_x'] = np.nan
    df_sensor_clean = df_sensor.dropna(subset=['vel_rms_x'])
    
    # KNN Multivariate (Técnica do Pipeline)
    features_knn = ['vel_rms_x', 'mag_x', 'mag_y', 'mag_z', 'object_temp', 'timestamp_num', 'hora_sin', 'hora_cos']
    imputer = KNNImputer(n_neighbors=10, weights='distance')
    df_train = df_plot.dropna(subset=['vel_rms_x'])
    imputer.fit(df_train[features_knn])
    
    df_filled = pd.DataFrame(imputer.transform(df_plot[features_knn]), columns=features_knn, index=df_plot.index)
    y_knn = df_filled['vel_rms_x']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 1. Reta do Sensor (Ligando o buraco com uma reta simples - O cenário "ruim")
    ax.plot(df_sensor_clean['time'], df_sensor_clean['vel_rms_x'], color='blue', alpha=0.3, linestyle='--', label='Conexão Linear (Sem dados)', linewidth=1)
    # Plotar os pontos reais fora do gap
    ax.scatter(df_sensor_clean['time'], df_sensor_clean['vel_rms_x'], color='blue', s=5, alpha=0.6, label='Dados Reais (Sensor)')
    
    # 2. Reconstrução KNN (Apenas DENTRO do Gap)
    gap_idx = np.where(gap_mask)[0]
    ax.scatter(df_plot['time'].iloc[gap_idx], y_knn.iloc[gap_idx], 
               color='cyan', marker='x', s=10, alpha=0.9, label='Reconstrução KNN Multivariate (Técnica Pipeline)')
    
    # Linha de tendência da reconstrução para mostrar o padrão temporal
    ax.plot(df_plot['time'].iloc[gap_idx], y_knn.iloc[gap_idx], color='cyan', alpha=0.4, linewidth=1, label='Assinatura Reconstruída')
    
    data_fmt = df_plot['time'].iloc[0].strftime('%d/%m/%Y')
    ax.set_title(f'Reconstrução de Sinal via KNN Multivariate (Gap 2.5h) - {data_fmt}', fontsize=13, fontweight='bold')
    ax.set_xlabel('Horário da Medição')
    ax.set_ylabel('Vibração RMS X (mm/s)')
    
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.xticks(rotation=45)
    ax.legend(loc='upper right', frameon=True, shadow=True)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, '02_knn_interpolacao.png'), dpi=300)
    print("Salvo: 02_knn_interpolacao.png")
    plt.close()

def plot_spline_interpolation(df_full):
    """Gera visualização de Spline utilizando dados REAIS (Gap de até 1h)."""
    from scipy.interpolate import PchipInterpolator
    print("  --> Gerando gráfico 02_spline_interpolacao.png com dados reais...")
    
    target_date = '2025-07-20'
    df_day = df_full[df_full['time'].dt.date == pd.to_datetime(target_date).date()].copy()
    df_plot = df_day[df_day['time'].dt.hour < 14].copy()
    
    # Simular Gap de 30 minutos (0.5h) - Reduzido para evitar instabilidade matemática extrema
    gap_start = df_plot['time'].min() + pd.Timedelta(hours=9)
    gap_end = gap_start + pd.Timedelta(minutes=30)
    gap_mask = (df_plot['time'] > gap_start) & (df_plot['time'] < gap_end)
    
    # Dados Reais fora do Gap
    df_sensor = df_plot.copy()
    df_sensor.loc[gap_mask, 'vel_rms_x'] = np.nan
    df_sensor_clean = df_sensor.dropna(subset=['vel_rms_x'])
    
    # Interpolação Pchip (Cubic Hermite) - Mais estável que a Spline Cúbica padrão (evita o "buraco" ou overshoot)
    df_train = df_plot[~gap_mask].dropna(subset=['vel_rms_x'])
    x_train = (df_train['time'] - df_train['time'].min()).dt.total_seconds()
    # Pchip garante que a curva não oscile descontroladamente em gaps grandes
    f_stable = PchipInterpolator(x_train, df_train['vel_rms_x'], extrapolate=True)
    
    x_pred = (df_plot['time'] - df_train['time'].min()).dt.total_seconds()
    y_stable = f_stable(x_pred)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 1. Reta do Sensor (Cenário sem reconstrução)
    ax.plot(df_sensor_clean['time'], df_sensor_clean['vel_rms_x'], color='blue', alpha=0.3, linestyle='--', label='Conexão Linear (Sem dados)', linewidth=1)
    ax.scatter(df_sensor_clean['time'], df_sensor_clean['vel_rms_x'], color='blue', s=5, alpha=0.6, label='Dados Reais (Sensor)')
    
    # 2. Preenchimento Spline Estável (Apenas DENTRO do Gap)
    ax.plot(df_plot['time'][gap_mask], y_stable[gap_mask], color='orange', linewidth=2, label='Interpolação Spline (Hermitiana)')
    
    data_fmt = df_plot['time'].iloc[0].strftime('%d/%m/%Y')
    ax.set_title(f'Interpolação via Spline Hermitiana (Gap 30min) - {data_fmt}', fontsize=13, fontweight='bold')
    ax.set_xlabel('Horário da Medição')
    ax.set_ylabel('Vibração RMS X (mm/s)')
    
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.xticks(rotation=45)
    ax.legend(loc='upper right', frameon=True, shadow=True)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, '02_spline_interpolacao.png'), dpi=300)
    print("Salvo: 02_spline_interpolacao.png")
    plt.close()

def plot_lstm_interpolation(df):
    """Gera visualização utilizando Rede Neural LSTM para preencher o gap."""
    print("\n  --> Iniciando treinamento da Rede Neural LSTM (pode demorar um pouco)...")
    df_work = df.copy()
    
    # Configuração do Gap
    gap_start_idx = 300
    gap_end_idx = 750
    
    # Normalização para a Rede Neural (Crucial para LSTM)
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    features = ['current', 'rpm', 'timestamp_num', 'hora_sin', 'hora_cos']
    target = 'vel_rms'
    
    # Dados de treino (exclui o gap)
    df_train = df_work.drop(df_work.index[gap_start_idx:gap_end_idx])
    
    X_train_raw = scaler_x.fit_transform(df_train[features])
    y_train_raw = scaler_y.fit_transform(df_train[[target]])
    
    # Preparar sequências para LSTM (window size = 10)
    def create_sequences(X, y, window=10):
        Xs, ys = [], []
        for i in range(len(X) - window):
            Xs.append(X[i:(i + window)])
            ys.append(y[i + window])
        return np.array(Xs), np.array(ys)
    
    window_size = 15
    X_train, y_train = create_sequences(X_train_raw, y_train_raw, window_size)
    
    # Criar Modelo LSTM Simples
    model = Sequential([
        LSTM(32, activation='relu', input_shape=(window_size, len(features)), return_sequences=False),
        Dropout(0.1),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    # Treino rápido (15 épocas)
    model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)
    
    # Predição do Gap
    # Precisamos dos dados completos para alimentar a janela deslizante
    X_all_raw = scaler_x.transform(df_work[features])
    v_reconstructed = df_work[target].values.copy()
    
    print("  --> Reconstruindo sinal com LSTM...")
    # Predição iterativa (usando o contexto anterior para prever o próximo ponto)
    for i in range(gap_start_idx, gap_end_idx):
        # Pega a janela anterior (que pode conter predições anteriores)
        window = X_all_raw[i-window_size:i].reshape(1, window_size, len(features))
        pred_norm = model.predict(window, verbose=0)
        v_reconstructed[i] = scaler_y.inverse_transform(pred_norm)[0, 0]
        
        # Adicionar um pequeno ruído branco para realismo (variabilidade da máquina)
        v_reconstructed[i] += np.random.normal(0, 0.05)
    
    # Gráfico de Comparação
    fig, ax = plt.subplots(figsize=(12, 6))
    
    zoom_start = gap_start_idx - 150
    zoom_end = gap_end_idx + 150
    plot_mask = (df_work.index >= zoom_start) & (df_work.index <= zoom_end)
    
    # Dados Reais
    ax.plot(df_work.loc[plot_mask, 'time'], df_work.loc[plot_mask, 'vel_rms'], 
            'bo-', alpha=0.3, label='Dados Originais', markersize=3)
    
    # Dados LSTM
    gap_range = range(gap_start_idx, gap_end_idx)
    ax.scatter(df_work.loc[gap_range, 'time'], v_reconstructed[gap_range], 
               c='magenta', marker='.', s=15, alpha=0.6, label='Reconstrução LSTM (Deep Learning)')
    
    # Linha de tendência LSTM
    ax.plot(df_work.loc[plot_mask, 'time'], v_reconstructed[plot_mask], 
            'm--', alpha=0.5, linewidth=1.5, label='Assinatura LSTM')
    
    ax.set_title('Reconstrução de Sinais via Redes Neurais LSTM (Deep Learning)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Data e Hora')
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.xticks(rotation=45)
    ax.set_ylabel('Vibração RMS (mm/s)')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 0.05), ncol=3, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, '02_lstm_interpolacao.png'), dpi=300)
    print("Salvo: 02_lstm_interpolacao.png")
    plt.close()

def plot_outlier_removal(df):
    """Gera visualização da remoção de outliers utilizando o método IQR."""
    df_outliers = df.copy()
    
    # Injetar outliers artificiais aleatórios MUITO altos para cruzar o IQR de 3.0 num sinal bimodal
    np.random.seed(99)
    outlier_indices = np.random.choice(df_outliers.index, size=15, replace=False)
    # Adiciona picos absurdos (30 a 60 mm/s)
    df_outliers.loc[outlier_indices, 'vel_rms'] = df_outliers.loc[outlier_indices, 'vel_rms'] + np.random.uniform(30, 60, size=15)
    
    # Aplicar IQR
    Q1 = df_outliers['vel_rms'].quantile(0.25)
    Q3 = df_outliers['vel_rms'].quantile(0.75)
    IQR = Q3 - Q1
    limite_superior = Q3 + 3.0 * IQR
    
    # Identificar
    is_outlier = df_outliers['vel_rms'] > limite_superior
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot com outliers
    ax1.plot(df_outliers['time'], df_outliers['vel_rms'], label='Sinal (com ruído da máquina)', color='gray', alpha=0.6)
    ax1.scatter(df_outliers.loc[is_outlier, 'time'], df_outliers.loc[is_outlier, 'vel_rms'], color='red', marker='o', s=50, label=f'Outliers Detectados (Acima do limite)', zorder=5)
    ax1.axhline(limite_superior, color='orange', linestyle='--', linewidth=2, label=f'Limite IQR (3.0): {limite_superior:.2f} mm/s')
    ax1.set_title('Sinal Bruto Contendo Picos Anômalos (Outliers Físicos)')
    ax1.set_ylabel('Vibração RMS (mm/s)')
    ax1.legend(loc='upper right', frameon=True, shadow=True)
    
    # Plot sem outliers
    df_cleaned = df_outliers.copy()
    df_cleaned.loc[is_outlier, 'vel_rms'] = np.nan # Removidos
    
    ax2.plot(df_cleaned['time'], df_cleaned['vel_rms'], label='Sinal Tratado e Contínuo', color='blue', alpha=0.8)
    ax2.set_title('Sinal após Limpeza Automática pelo IQR')
    ax2.set_xlabel('Data e Hora')
    ax2.set_ylabel('Vibração RMS (mm/s)')
    ax2.legend(loc='upper right', frameon=True, shadow=True)
    
    import matplotlib.dates as mdates
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%y %H:%M'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, '04_remocao_outliers.png'), dpi=300)
    print("Salvo: 04_remocao_outliers.png")
    plt.close()

def animate_kmeans(df):
    """Cria uma animação GIF do K-Means convergindo com 6 clusters e cores customizadas."""
    scaler = MinMaxScaler()
    # Usamos Corrente e Vibração para o gráfico 2D
    X = scaler.fit_transform(df[['current', 'vel_rms']])
    
    n_clusters = 6
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Inicialização dos centróides
    # Vamos espalhar os centróides iniciais para garantir uma convergência visual bonita
    centroids_history = []
    labels_history = []
    
    # Rodar o K-Means passo a passo para a animação
    # Inicialização manual para garantir 6 clusters bem distribuídos na animação
    current_centroids = np.array([
        [0.05, 0.05], [0.15, 0.15], [0.4, 0.4], 
        [0.6, 0.6], [0.8, 0.8], [0.95, 0.95]
    ])
    
    for i in range(12):
        # Atribuição (E-step)
        distances = np.linalg.norm(X[:, np.newaxis] - current_centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        centroids_history.append(current_centroids.copy())
        labels_history.append(labels.copy())
        
        # Atualização (M-step)
        new_centroids = []
        for k in range(n_clusters):
            if np.any(labels == k):
                new_centroids.append(X[labels == k].mean(axis=0))
            else:
                new_centroids.append(current_centroids[k]) # Mantém se vazio
        current_centroids = np.array(new_centroids)

    # Cores solicitadas pelo usuário:
    # 2 para desligado (vermelhos), 4 para ligado (verdes)
    final_centroids = current_centroids
    magnitudes = np.sum(final_centroids, axis=1)
    sorted_idx = np.argsort(magnitudes)
    
    # Mapeamento de Cores (6 tons):
    cluster_colors = {}
    # Vermelhos (OFF): sorted_idx[0], sorted_idx[1]
    cluster_colors[sorted_idx[0]] = '#4d0000' # Vermelho escuro (mais próximo de 0)
    cluster_colors[sorted_idx[1]] = '#ff4d4d' # Vermelho claro
    
    # Verdes (ON): 2, 3, 4, 5 -> (Gradiante para o escuro conforme aumenta)
    cluster_colors[sorted_idx[2]] = '#ccffcc' # Verde bem claro
    cluster_colors[sorted_idx[3]] = '#66ff66' # Verde claro
    cluster_colors[sorted_idx[4]] = '#00cc00' # Verde médio
    cluster_colors[sorted_idx[5]] = '#004d00' # Verde escuro (maiores valores)

    def update(frame):
        ax.clear()
        labels = labels_history[frame]
        centroids = centroids_history[frame]
        
        # Plot dos pontos com suas cores de cluster
        for k in range(n_clusters):
            mask = (labels == k)
            color = cluster_colors[k]
            ax.scatter(X[mask, 0], X[mask, 1], c=color, alpha=0.6, edgecolors='none', s=30)
        
        # Plot dos centróides
        ax.scatter(centroids[:, 0], centroids[:, 1], c='blue', marker='+', s=150, linewidths=3, label='Centróides')
        
        ax.set_title(f'Convergência do K-Means (K=6): Iteração {frame + 1}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Corrente Normalizada (Current)')
        ax.set_ylabel('Vibração Normalizada (Vel_RMS)')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        
        # Criar legenda customizada com as bolinhas
        from matplotlib.lines import Line2D
        legend_elements = []
        for idx_pos in range(n_clusters):
            k = sorted_idx[idx_pos]
            color = cluster_colors[k]
            name = "Desligado" if idx_pos < 2 else "Ligado"
            legend_elements.append(Line2D([0], [0], marker='o', color='w', label=f'{name} Cluster {idx_pos+1}',
                                        markerfacecolor=color, markersize=10))
        
        legend_elements.append(Line2D([0], [0], marker='+', color='black', label='Centróides',
                                    markersize=10, linestyle='None'))
            
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), title="Clusters de Estados")
        plt.tight_layout()
        
    ani = animation.FuncAnimation(fig, update, frames=len(centroids_history), repeat=True, interval=1000)
    
    try:
        ani.save(os.path.join(SAVE_DIR, '03_kmeans_animacao.gif'), writer='pillow', fps=1)
        print("Salvo: 03_kmeans_animacao.gif")
    except Exception as e:
        print(f"Erro ao salvar GIF: {e}. Salvando frame final.")
        update(len(centroids_history)-1)
        plt.savefig(os.path.join(SAVE_DIR, '03_kmeans_final.png'), dpi=300, bbox_inches='tight')
        
    plt.close()

if __name__ == "__main__":
    print("Gerando gráficos ilustrativos para o TCC...")
    
    # Dados Reais para Interpolação (c_1518)
    df_real = load_real_data()
    plot_knn_interpolation(df_real)
    plot_spline_interpolation(df_real)
    
    # Dados Sintéticos para ilustrações conceituais (c_636/Genérico)
    # Recriar a função de dados sintéticos para o resto do script
    def generate_synthetic_context(n_samples=1500):
        np.random.seed(42)
        start_time = pd.Timestamp('2025-10-15 08:00:00')
        t = pd.date_range(start=start_time, periods=n_samples, freq='20s')
        idx = np.arange(n_samples)
        samples_per_state = 75 
        states = np.where((idx % (2 * samples_per_state)) < samples_per_state, 0, 1)
        current = np.where(states == 1, np.random.normal(32, 1.2, n_samples), np.random.normal(2, 0.2, n_samples))
        rpm = np.where(states == 1, np.random.normal(1190, 8, n_samples), np.random.normal(10, 2, n_samples))
        vib = np.where(states == 1, np.random.normal(4.5, 0.5, n_samples), np.random.normal(0.5, 0.1, n_samples))
        df = pd.DataFrame({'time': t, 'current': current, 'rpm': rpm, 'vel_rms': vib, 'estado_real': states})
        # Features temporais para LSTM sintética
        df['hora'] = df['time'].dt.hour
        df['hora_sin'] = np.sin(2 * np.pi * df['hora'] / 24)
        df['hora_cos'] = np.cos(2 * np.pi * df['hora'] / 24)
        df['timestamp_num'] = (df['time'] - df['time'].min()).dt.total_seconds()
        return df

    df_synthetic = generate_synthetic_context(1500)
    
    # Ilustrações Conceituais
    def plot_minmax_scaler_v2(df):
        scaler = MinMaxScaler()
        df_scaled = df.copy()
        features = ['current', 'rpm', 'vel_rms']
        df_scaled[features] = scaler.fit_transform(df[features])
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.scatter(df['current'], df['vel_rms'], c=[COLORS[s] for s in df['estado_real']], alpha=0.6, edgecolors='k')
        ax1.set_title('Dados Originais (Escalas Diferentes)')
        ax1.set_xlabel('Corrente (A)')
        ax1.set_ylabel('Vibração (mm/s)')
        ax2.scatter(df_scaled['current'], df_scaled['vel_rms'], c=[COLORS[s] for s in df_scaled['estado_real']], alpha=0.6, edgecolors='k')
        ax2.set_title('Dados Normalizados (MinMaxScaler)')
        ax2.set_xlabel('Corrente Normalizada [0, 1]')
        ax2.set_ylabel('Vibração Normalizada [0, 1]')
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, '01_minmax_scaler.png'), dpi=300)
        print("Salvo: 01_minmax_scaler.png")
        plt.close()

    plot_minmax_scaler_v2(df_synthetic)
    plot_lstm_interpolation(df_synthetic)
    plot_outlier_removal(df_synthetic)
    animate_kmeans(df_synthetic)
    
    print(f"Todas as ilustrações foram salvas em: {SAVE_DIR}")
