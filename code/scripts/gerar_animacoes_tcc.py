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
    
    import matplotlib.dates as mdates
    from matplotlib.lines import Line2D as _L2D
    plt.rcParams.update({'font.size': 16, 'axes.titlesize': 18, 'axes.labelsize': 16,
                         'xtick.labelsize': 15, 'ytick.labelsize': 15})
    fig, ax = plt.subplots(figsize=(14, 7))

    # 1. Reta do Sensor
    ax.plot(df_sensor_clean['time'], df_sensor_clean['vel_rms_x'],
            color='blue', alpha=0.3, linestyle='--', linewidth=1.5)
    ax.scatter(df_sensor_clean['time'], df_sensor_clean['vel_rms_x'],
               color='blue', s=8, alpha=0.7)

    # 2. Reconstrução KNN (Apenas DENTRO do Gap)
    gap_idx = np.where(gap_mask)[0]
    ax.scatter(df_plot['time'].iloc[gap_idx], y_knn.iloc[gap_idx],
               color='cyan', marker='s', s=18, alpha=0.9)
    ax.plot(df_plot['time'].iloc[gap_idx], y_knn.iloc[gap_idx],
            color='cyan', alpha=0.5, linewidth=1.5)

    data_fmt = df_plot['time'].iloc[0].strftime('%d/%m/%Y')
    ax.set_title(f'Reconstrução de Sinal via KNN Multivariate (Gap 2.5h) — {data_fmt}',
                 fontweight='bold')
    ax.set_xlabel('Horário da Medição', fontweight='bold')
    ax.set_ylabel('Vibração RMS X (mm/s)', fontweight='bold')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.xticks(rotation=45)
    ax.grid(True, linestyle='--', alpha=0.3)

    # Legenda com ícones grandes (círculo = real, quadrado = reconstruído)
    _handles_knn = [
        _L2D([0],[0], marker='o', color='w', markerfacecolor='blue',
             markeredgecolor='k', markeredgewidth=0.5, markersize=13, label='Dados Reais (Sensor)'),
        _L2D([0],[0], linestyle='--', color='blue', linewidth=2, label='Conexão Linear (Sem dados)'),
        _L2D([0],[0], marker='s', color='w', markerfacecolor='cyan',
             markeredgecolor='k', markeredgewidth=0.5, markersize=13, label='Reconstrução KNN Multivariate'),
        _L2D([0],[0], linestyle='-', color='cyan', linewidth=2, label='Tendência Reconstruída'),
    ]
    ax.legend(handles=_handles_knn, loc='lower right', frameon=True, shadow=True,
              fontsize=15, handletextpad=0.4, labelspacing=0.3, borderpad=0.7)

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
    
    import matplotlib.dates as mdates
    from matplotlib.lines import Line2D as _L2D
    plt.rcParams.update({'font.size': 16, 'axes.titlesize': 18, 'axes.labelsize': 16,
                         'xtick.labelsize': 15, 'ytick.labelsize': 15})
    fig, ax = plt.subplots(figsize=(14, 7))

    # 1. Reta do Sensor (Cenário sem reconstrução)
    ax.plot(df_sensor_clean['time'], df_sensor_clean['vel_rms_x'],
            color='blue', alpha=0.3, linestyle='--', linewidth=1.5)
    ax.scatter(df_sensor_clean['time'], df_sensor_clean['vel_rms_x'],
               color='blue', s=8, alpha=0.7)

    # 2. Preenchimento Spline Estável (Apenas DENTRO do Gap)
    ax.scatter(df_plot['time'][gap_mask], y_stable[gap_mask],
               color='orange', marker='s', s=18, alpha=0.9)
    ax.plot(df_plot['time'][gap_mask], y_stable[gap_mask],
            color='orange', linewidth=2.5)

    data_fmt = df_plot['time'].iloc[0].strftime('%d/%m/%Y')
    ax.set_title(f'Interpolação via Spline Hermitiana (Gap 30min) — {data_fmt}',
                 fontweight='bold')
    ax.set_xlabel('Horário da Medição', fontweight='bold')
    ax.set_ylabel('Vibração RMS X (mm/s)', fontweight='bold')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.xticks(rotation=45)
    ax.grid(True, linestyle='--', alpha=0.3)

    _handles_spl = [
        _L2D([0],[0], marker='o', color='w', markerfacecolor='blue',
             markeredgecolor='k', markeredgewidth=0.5, markersize=13, label='Dados Reais (Sensor)'),
        _L2D([0],[0], linestyle='--', color='blue', linewidth=2, label='Conexão Linear (Sem dados)'),
        _L2D([0],[0], marker='s', color='w', markerfacecolor='orange',
             markeredgecolor='k', markeredgewidth=0.5, markersize=13, label='Interpolação Spline (Hermitiana)'),
    ]
    ax.legend(handles=_handles_spl, loc='lower right', frameon=True, shadow=True,
              fontsize=15, handletextpad=0.4, labelspacing=0.3, borderpad=0.7)

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
    
    import matplotlib.dates as mdates
    from matplotlib.lines import Line2D as _L2D
    plt.rcParams.update({'font.size': 18, 'axes.titlesize': 19, 'axes.labelsize': 18,
                         'xtick.labelsize': 16, 'ytick.labelsize': 16})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Plot com outliers
    ax1.plot(df_outliers['time'], df_outliers['vel_rms'], color='gray', alpha=0.6, linewidth=1)
    ax1.scatter(df_outliers.loc[is_outlier, 'time'], df_outliers.loc[is_outlier, 'vel_rms'],
                color='red', marker='o', s=60, zorder=5)
    ax1.axhline(limite_superior, color='orange', linestyle='--', linewidth=2)
    ax1.set_title('Sinal Bruto Contendo Picos Anomalos (Outliers Fisicos)', fontweight='bold')
    ax1.set_ylabel('Vibração RMS (mm/s)', fontweight='bold')
    # Legenda simplificada: apenas 2 itens
    _h1 = [
        _L2D([0],[0], linestyle='-', color='gray', linewidth=2, label='Sinal (com ruido)'),
        _L2D([0],[0], marker='o', color='w', markerfacecolor='red',
             markeredgecolor='k', markeredgewidth=0.5, markersize=14, label='Outliers detectados'),
    ]
    ax1.legend(handles=_h1, loc='lower right', frameon=True, shadow=True,
               fontsize=16, handletextpad=0.4, labelspacing=0.3, borderpad=0.7)

    # Plot sem outliers
    df_cleaned = df_outliers.copy()
    df_cleaned.loc[is_outlier, 'vel_rms'] = np.nan

    ax2.plot(df_cleaned['time'], df_cleaned['vel_rms'], color='blue', alpha=0.8, linewidth=1)
    ax2.set_title('Sinal apos Limpeza Automatica pelo IQR', fontweight='bold')
    ax2.set_xlabel('Data e Hora', fontweight='bold')
    ax2.set_ylabel('Vibração RMS (mm/s)', fontweight='bold')
    _h2 = [_L2D([0],[0], linestyle='-', color='blue', linewidth=2, label='Sinal Tratado e Continuo')]
    ax2.legend(handles=_h2, loc='lower right', frameon=True, shadow=True,
               fontsize=16, handletextpad=0.4, borderpad=0.7)

    # Formato hora: 15/10/25\n12h
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%y\n%Hh'))
    plt.xticks(rotation=0)

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, '04_remocao_outliers.png'), dpi=300)
    print("Salvo: 04_remocao_outliers.png")
    plt.close()

def plot_comparativo_knn_lstm(df_synth):
    """Comparativo JUSTO KNN vs LSTM nos MESMOS dados sintéticos (padrão ON/OFF) da
    Figura 02_lstm_interpolacao — mesmo período, mesmo gap (300-750), mostrando vibração.
    O KNN replica FIELMENTE o pipeline (interpolar_knn_multivariado): normaliza o
    timestamp_num e usa nan_euclidean — assim ele também reconstrói o ON/OFF, sem
    colapsar em platô. O LSTM usa a mesma reconstrução de plot_lstm_interpolation.
    """
    import matplotlib.dates as mdates
    from matplotlib.lines import Line2D as _L2D
    from sklearn.impute import KNNImputer
    from sklearn.preprocessing import MinMaxScaler as MMS
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM as KerasLSTM, Dense, Dropout
    print("  --> Gerando grafico 02_comparativo_knn_lstm_subplots.png (sintetico ON/OFF)...")

    plt.rcParams.update({'font.size': 16, 'axes.titlesize': 17, 'axes.labelsize': 16,
                         'xtick.labelsize': 14, 'ytick.labelsize': 14})

    df = df_synth.copy().reset_index(drop=True)
    n = len(df)
    gap_s, gap_e = 300, 750           # mesmo gap de plot_lstm_interpolation
    zoom_s, zoom_e = 150, 900         # mesma janela de visualização da Fig. LSTM
    gap_idx = np.arange(gap_s, gap_e)

    # ─── KNN: replica FIEL do pipeline (timestamp_num NORMALIZADO + nan_euclidean) ─
    df_knn = df.copy()
    df_knn.iloc[gap_s:gap_e, df_knn.columns.get_loc('vel_rms')] = np.nan
    # normalizar timestamp_num para [0,1] (exatamente como interpolar_knn_multivariado)
    ts = df_knn['timestamp_num'].values.astype(float)
    df_knn['ts_norm'] = (ts - ts.min()) / (ts.max() - ts.min() + 1e-10)
    features_knn = ['vel_rms', 'current', 'rpm', 'ts_norm', 'hora_sin', 'hora_cos']
    n_neighbors = int(np.clip(len(df_knn.dropna(subset=['vel_rms'])) // 20, 3, 5))
    imputer = KNNImputer(n_neighbors=n_neighbors, weights='distance',
                         metric='nan_euclidean')
    filled = imputer.fit_transform(df_knn[features_knn])
    y_knn = filled[:, 0]              # coluna vel_rms reconstruída

    # ─── LSTM: mesma reconstrução de plot_lstm_interpolation (current/rpm/tempo) ──
    print("     Treinando LSTM...")
    scaler_x = MMS(); scaler_y = MMS()
    features_lstm = ['current', 'rpm', 'timestamp_num', 'hora_sin', 'hora_cos']
    target = 'vel_rms'
    df_train_l = df.drop(df.index[gap_s:gap_e])
    X_tr = scaler_x.fit_transform(df_train_l[features_lstm])
    y_tr = scaler_y.fit_transform(df_train_l[[target]])
    window = 15
    def make_seq(X, y, w):
        Xs, ys = [], []
        for i in range(len(X)-w):
            Xs.append(X[i:i+w]); ys.append(y[i+w])
        return np.array(Xs), np.array(ys)
    Xs, ys = make_seq(X_tr, y_tr, window)
    model = Sequential([
        KerasLSTM(32, activation='relu', input_shape=(window, len(features_lstm))),
        Dropout(0.1), Dense(16, activation='relu'), Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(Xs, ys, epochs=15, batch_size=32, verbose=0)
    X_all = scaler_x.transform(df[features_lstm])
    y_lstm = df[target].values.copy()
    for i in gap_idx:
        if i < window: continue
        w = X_all[i-window:i].reshape(1, window, len(features_lstm))
        pred = model.predict(w, verbose=0)
        y_lstm[i] = scaler_y.inverse_transform(pred)[0, 0] + np.random.normal(0, 0.05)

    # ─── Plot (janela de zoom 150-900) ──────────────────────────────────────────
    data_fmt = df['time'].iloc[0].strftime('%d/%m/%Y')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    fig.suptitle(f'Comparativo de Algoritmos de Reconstrucao de Sinais — {data_fmt}',
                 fontsize=18, fontweight='bold')

    zoom = np.arange(zoom_s, zoom_e)
    out_zoom = zoom[(zoom < gap_s) | (zoom >= gap_e)]   # reais visíveis fora do gap

    for ax, y_rec, color, label_rec, title in [
        (ax1, y_knn, 'cyan', 'Reconstrucao KNN Multivariate',
         f'Interpolacao via KNN Multivariate — {data_fmt}'),
        (ax2, y_lstm, '#ff69b4', 'Reconstrucao LSTM (Deep Learning)',
         f'Reconstrucao via Rede Neural LSTM — {data_fmt}'),
    ]:
        # Dados reais (fora do gap)
        ax.plot(df['time'].iloc[out_zoom], df['vel_rms'].iloc[out_zoom],
                color='blue', alpha=0.3, linestyle='--', linewidth=1)
        ax.scatter(df['time'].iloc[out_zoom], df['vel_rms'].iloc[out_zoom],
                   color='blue', s=8, alpha=0.7)
        # Reconstrução no gap (pontos coloridos)
        ax.scatter(df['time'].iloc[gap_idx], y_rec[gap_s:gap_e],
                   color=color, marker='s', s=14, alpha=0.85, zorder=5)
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel('Vibracao RMS (mm/s)', fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_xlim(df['time'].iloc[zoom_s], df['time'].iloc[zoom_e-1])
        _h = [
            _L2D([0],[0], marker='o', color='w', markerfacecolor='blue',
                 markeredgecolor='k', markeredgewidth=0.5, markersize=13,
                 label='Dados Reais (Sensor)'),
            _L2D([0],[0], linestyle='--', color='blue', linewidth=2,
                 label='Conexao Linear (Sem dados)'),
            _L2D([0],[0], marker='s', color='w', markerfacecolor=color,
                 markeredgecolor='k', markeredgewidth=0.5, markersize=13,
                 label=label_rec),
        ]
        ax.legend(handles=_h, loc='center right', frameon=True, shadow=True,
                  fontsize=14, handletextpad=0.4, labelspacing=0.3, borderpad=0.7)

    ax2.set_xlabel('Horario da Medicao', fontweight='bold')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.xticks(rotation=45)
    plt.tight_layout()

    LATEX_PLOTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                               'latex', 'plots')
    os.makedirs(LATEX_PLOTS, exist_ok=True)
    out = os.path.join(LATEX_PLOTS, '02_comparativo_knn_lstm_subplots.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Salvo: 02_comparativo_knn_lstm_subplots.png")
    plt.close()


def animate_kmeans():
    """Cria GIF do K-Means com dados REAIS de c_637 (caso com 2 clusters DESLIGADO)."""
    from pathlib import Path
    from sklearn.decomposition import PCA
    import json
    from matplotlib.lines import Line2D

    BASE_DIR = Path(__file__).resolve().parent.parent
    LATEX_PLOTS = Path(__file__).resolve().parent.parent.parent / 'latex' / 'plots'
    LATEX_PLOTS.mkdir(parents=True, exist_ok=True)

    # 6 cores: 2 DESLIGADO (vermelhos) + 4 LIGADO (verdes)
    # C1=mais certo DESLIGADO (vermelho do 3D), C6=mais certo LIGADO (verde do 3D)
    CLUSTER_COLORS = [
        '#e74c3c',  # rank 0 — Desligado C1 (vermelho exato do 3D, mais certo DESLIGADO)
        '#ff8a80',  # rank 1 — Desligado C2 (vermelho mais claro)
        '#ccffcc',  # rank 2 — Ligado C3    (verde mais claro)
        '#66ff66',  # rank 3 — Ligado C4
        '#00cc00',  # rank 4 — Ligado C5
        '#2ecc71',  # rank 5 — Ligado C6    (verde exato do 3D, mais certo LIGADO)
    ]
    CLUSTER_LABELS = [
        'Desligado C1', 'Desligado C2',
        'Ligado C3', 'Ligado C4', 'Ligado C5', 'Ligado C6',
    ]

    # Carregar dados reais de c_637 (tem 2 clusters DESLIGADO)
    arq_class = BASE_DIR / 'data' / 'processed' / 'c_637' / 'dados_classificados_kmeans_moderado_c_637.csv'
    arq_norm  = BASE_DIR / 'data' / 'normalized'  / 'c_637' / 'dados_kmeans_c_637.csv'
    arq_info  = BASE_DIR / 'models' / 'c_637' / 'info_normalizacao_c_637.json'

    if not arq_class.exists() or not arq_norm.exists():
        print(f"[ERRO] Arquivos de c_637 não encontrados. Pulando animate_kmeans.")
        return

    df_class = pd.read_csv(arq_class)
    df_norm  = pd.read_csv(arq_norm)

    with open(arq_info, 'r') as f:
        info = json.load(f)
    colunas_key = 'colunas_utilizadas_finais' if 'colunas_utilizadas_finais' in info else 'colunas_utilizadas'
    feature_cols = [c for c in info[colunas_key] if c in df_norm.columns]
    X_full = df_norm[feature_cols].values

    # PCA 2D para visualização
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_full)
    var1 = pca.explained_variance_ratio_[0]
    var2 = pca.explained_variance_ratio_[1]

    # Amostrar para GIF (desempenho)
    np.random.seed(42)
    n_sample = min(3000, len(X_2d))
    idx_s = np.random.choice(len(X_2d), n_sample, replace=False)
    X_s = X_2d[idx_s]

    # K-Means passo a passo em 2D para animação
    n_clusters = 6
    cur_c = np.array([X_s[n_sample*i//n_clusters] for i in range(n_clusters)])
    centroids_hist, colors_hist, labels_hist = [], [], []
    for _ in range(12):
        dists  = np.linalg.norm(X_s[:, None] - cur_c, axis=2)
        labels = np.argmin(dists, axis=1)
        # Colorir por rank de magnitude do centróide (0=mais DESLIGADO)
        mags   = np.sum(cur_c, axis=1)
        sorted_idx = np.argsort(mags)
        rank_of = {k: r for r, k in enumerate(sorted_idx)}
        frame_colors = np.array([CLUSTER_COLORS[rank_of[k]] for k in labels], dtype=object)
        centroids_hist.append(cur_c.copy())
        colors_hist.append(frame_colors.copy())
        labels_hist.append(labels.copy())
        new_c = []
        for k in range(n_clusters):
            pts = X_s[labels == k]
            new_c.append(pts.mean(axis=0) if len(pts) > 0 else cur_c[k])
        cur_c = np.array(new_c)

    # Frame 10 (iteração 11, 0-indexado) é usado como PNG — ainda mostra 2 clusters DESLIGADO
    PNG_FRAME = 10

    def make_legend(sorted_idx_cur):
        elems = [
            Line2D([0],[0], marker='o', color='w',
                   markerfacecolor=CLUSTER_COLORS[r],
                   markeredgecolor='k', markeredgewidth=0.5,
                   markersize=13, label=CLUSTER_LABELS[r])
            for r in range(6)
        ]
        elems.append(Line2D([0],[0], marker='+', color='blue', markersize=12,
                             markeredgewidth=2.5, linestyle='None', label='Centroides'))
        return elems

    fig, ax = plt.subplots(figsize=(13, 8))
    fig.subplots_adjust(right=0.72)

    def update(frame):
        ax.clear()
        colors = colors_hist[frame]
        centroids = centroids_hist[frame]
        ax.scatter(X_s[:, 0], X_s[:, 1],
                   c=list(colors), alpha=0.65, s=30,
                   edgecolors='k', linewidths=0.4)
        ax.scatter(centroids[:, 0], centroids[:, 1],
                   c='blue', marker='+', s=200, linewidths=3)
        mags_f = np.sum(centroids, axis=1)
        sorted_f = np.argsort(mags_f)
        ax.set_title(f'K-Means (K=6) — Iteracao {frame+1}/12\nEquip. $c_{{637}}$: ilustracao do caso com 2 clusters DESLIGADO',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel(f'PC1 ({var1:.1%} variancia)', fontsize=14, fontweight='bold')
        ax.set_ylabel(f'PC2 ({var2:.1%} variancia)', fontsize=14, fontweight='bold')
        ax.tick_params(labelsize=13)
        ax.set_xlim(X_2d[:,0].min()-0.1, X_2d[:,0].max()+0.1)
        ax.set_ylim(X_2d[:,1].min()-0.1, X_2d[:,1].max()+0.1)
        ax.legend(handles=make_legend(sorted_f),
                  bbox_to_anchor=(1.02, 1), loc='upper left',
                  frameon=True, fontsize=13,
                  handletextpad=0.4, labelspacing=0.30, borderpad=0.75)

    ani = animation.FuncAnimation(fig, update, frames=len(centroids_hist),
                                  repeat=True, interval=1000)
    gif_path = os.path.join(SAVE_DIR, '03_kmeans_animacao.gif')
    try:
        ani.save(gif_path, writer='pillow', fps=1)
        print(f"Salvo: {gif_path}")
    except Exception as e:
        print(f"Erro ao salvar GIF: {e}")

    # Capturar frame 10 (iteração 11) como PNG — ainda mostra 2 clusters DESLIGADO
    update(PNG_FRAME)
    png_latex = str(LATEX_PLOTS / '03_kmeans_animacao.png')
    fig.savefig(png_latex, dpi=300, bbox_inches='tight')
    print(f"PNG iteracao 11 (frame {PNG_FRAME}): {png_latex}")

    # Gerar frame "classificação final": usar last frame (iteração 12) + colorir
    # apenas cluster 0 (menor magnitude) como DESLIGADO, demais como LIGADO
    def update_final(frame_colors_override):
        ax.clear()
        centroids = centroids_hist[-1]
        ax.scatter(X_s[:, 0], X_s[:, 1],
                   c=list(frame_colors_override), alpha=0.65, s=30,
                   edgecolors='k', linewidths=0.4)
        ax.scatter(centroids[:, 0], centroids[:, 1],
                   c='blue', marker='+', s=200, linewidths=3)
        ax.set_title('K-Means (K=6) — Classificacao Final (apos validacao)\nEquip. $c_{637}$: C2 reclassificado como LIGADO',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel(f'PC1 ({var1:.1%} variancia)', fontsize=14, fontweight='bold')
        ax.set_ylabel(f'PC2 ({var2:.1%} variancia)', fontsize=14, fontweight='bold')
        ax.tick_params(labelsize=13)
        ax.set_xlim(X_2d[:,0].min()-0.1, X_2d[:,0].max()+0.1)
        ax.set_ylim(X_2d[:,1].min()-0.1, X_2d[:,1].max()+0.1)
        ax.legend(handles=make_legend(np.argsort(np.sum(centroids, axis=1))),
                  bbox_to_anchor=(1.02, 1), loc='upper left',
                  frameon=True, fontsize=13,
                  handletextpad=0.4, labelspacing=0.30, borderpad=0.75)

    # Colorir classificação final: apenas rank 0 = DESLIGADO, tudo mais = LIGADO
    mags_final = np.sum(centroids_hist[-1], axis=1)
    sidx_final = np.argsort(mags_final)
    rank_final  = {k: r for r, k in enumerate(sidx_final)}
    final_labels = labels_hist[-1]
    final_fc = np.array([CLUSTER_COLORS[0] if rank_final[k] == 0 else CLUSTER_COLORS[5]
                         for k in final_labels], dtype=object)
    update_final(final_fc)
    png_final = str(LATEX_PLOTS / '03_kmeans_classificacao_final.png')
    fig.savefig(png_final, dpi=300, bbox_inches='tight')
    print(f"PNG classificacao final: {png_final}")

    plt.close()

if __name__ == "__main__":
    print("Gerando gráficos ilustrativos para o TCC...")
    
    # Dados Reais para Interpolação (c_1518)
    df_real = load_real_data()
    plot_knn_interpolation(df_real)
    plot_spline_interpolation(df_real)
    # comparativo é chamado depois de df_synthetic ser gerado (abaixo)
    # plot_comparativo_knn_lstm(df_synth) chamado após generate_synthetic_context
    
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
        plt.rcParams.update({'font.size': 16, 'axes.titlesize': 18, 'axes.labelsize': 16,
                             'xtick.labelsize': 15, 'ytick.labelsize': 15})
        scaler = MinMaxScaler()
        df_scaled = df.copy()
        features = ['current', 'rpm', 'vel_rms']
        df_scaled[features] = scaler.fit_transform(df[features])
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        ax1.scatter(df['current'], df['vel_rms'],
                    c=[COLORS[s] for s in df['estado_real']], alpha=0.6, edgecolors='k', s=30)
        ax1.set_title('Dados Originais (Escalas Diferentes)', fontweight='bold')
        ax1.set_xlabel('Corrente (A)', fontweight='bold')
        ax1.set_ylabel('Vibração (mm/s)', fontweight='bold')
        ax2.scatter(df_scaled['current'], df_scaled['vel_rms'],
                    c=[COLORS[s] for s in df_scaled['estado_real']], alpha=0.6, edgecolors='k', s=30)
        ax2.set_title('Dados Normalizados (MinMaxScaler)', fontweight='bold')
        ax2.set_xlabel('Corrente Normalizada [0, 1]', fontweight='bold')
        ax2.set_ylabel('Vibração Normalizada [0, 1]', fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, '01_minmax_scaler.png'), dpi=300)
        print("Salvo: 01_minmax_scaler.png")
        plt.close()

    plot_minmax_scaler_v2(df_synthetic)
    plot_lstm_interpolation(df_synthetic)
    plot_outlier_removal(df_synthetic)
    plot_comparativo_knn_lstm(df_synthetic)   # dados sinteticos ON/OFF (comparacao justa)
    animate_kmeans()
    
    print(f"Todas as ilustrações foram salvas em: {SAVE_DIR}")
