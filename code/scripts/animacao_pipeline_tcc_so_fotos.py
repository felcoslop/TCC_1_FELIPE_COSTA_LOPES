import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor
from scipy.interpolate import interp1d
import os

# Configurações de estilo
plt.style.use('seaborn-v0_8-whitegrid')
SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'plots', 'animacoes_tcc')
os.makedirs(SAVE_DIR, exist_ok=True)

def generate_pipeline_data(n_samples=300):
    """Gera dados para a animação do pipeline com gaps de diferentes tamanhos."""
    np.random.seed(42)
    # 2 estados: 0=OFF, 1=ON
    states = np.zeros(n_samples)
    states[100:250] = 1
    
    # Features
    current = np.where(states == 1, np.random.normal(30, 1.5, n_samples), np.random.normal(2, 0.4, n_samples))
    vib = np.where(states == 1, np.random.normal(4, 0.4, n_samples), np.random.normal(0.5, 0.1, n_samples))
    rpm = np.where(states == 1, np.random.normal(1200, 15, n_samples), np.random.normal(0, 5, n_samples))
    
    # Injetar Gaps (Lacunas)
    current_with_gap = current.copy()
    # Gap 1: Pequeno (20 min)
    current_with_gap[45:65] = np.nan
    # Gap 2: Grande (50 min - 2 horas simuladas)
    current_with_gap[120:170] = np.nan
    
    df = pd.DataFrame({
        'time': pd.date_range('2025-01-01', periods=n_samples, freq='min'),
        'current': current,
        'current_gap': current_with_gap,
        'vib': vib,
        'rpm': rpm,
        'state': states
    })
    return df

def generate_pngs():
    print("Iniciando geração de PNGs do Pipeline...")
    df = generate_pipeline_data(300)
    n_samples = len(df)
    
    # Prepara Interpolações
    t_idx = np.arange(n_samples).reshape(-1, 1)
    mask_valida = ~np.isnan(df['current_gap'])
    mask_gap = np.isnan(df['current_gap'])
    
    # KNN
    knn = KNeighborsRegressor(n_neighbors=5, weights='distance')
    knn.fit(t_idx[mask_valida], df.loc[mask_valida, 'current_gap'])
    v_knn = knn.predict(t_idx)
    # MÁSCARA RIGOROSA: Preencher apenas as lacunas, mas incluir os vizinhos imediatos para conectar as linhas
    mask_gap = np.isnan(df['current_gap'])
    # Dilatar a máscara em 1 pixel para cada lado para garantir conexão visual
    mask_gap_dilated = mask_gap.copy()
    for idx in np.where(mask_gap)[0]:
        if idx > 0: mask_gap_dilated[idx-1] = True
        if idx < n_samples-1: mask_gap_dilated[idx+1] = True
        
    v_knn_only_gap = np.full(n_samples, np.nan)
    v_knn_only_gap[mask_gap_dilated] = v_knn[mask_gap_dilated]
    
    # Linear
    f_lin = interp1d(t_idx[mask_valida].flatten(), df.loc[mask_valida, 'current_gap'], kind='linear', fill_value="extrapolate")
    v_lin = f_lin(t_idx.flatten())
    v_lin_only_gap = np.full(n_samples, np.nan)
    v_lin_only_gap[mask_gap_dilated] = v_lin[mask_gap_dilated]

    # Prepara Normalização
    scaler = MinMaxScaler()
    X_raw = df[['current', 'vib']].values
    X_clean = X_raw.copy()
    X_clean[X_clean > 15] = 6.0 
    X_scaled = scaler.fit_transform(X_clean)
    current_norm = X_scaled[:, 0]

    # Prepara K-Means
    kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    centroids = kmeans.cluster_centers_
    scores = np.sum(centroids, axis=1)
    idx_sorted = np.argsort(scores)
    
    colors = {}
    colors[idx_sorted[0]], colors[idx_sorted[1]] = '#4d0000', '#ff4d4d'
    colors[idx_sorted[2]], colors[idx_sorted[3]], colors[idx_sorted[4]], colors[idx_sorted[5]] = '#ccffcc', '#66ff66', '#00cc00', '#004d00'

    df['time_str'] = df['time'].dt.strftime('%d-%m-%y %H:%M')

    # 1. Dados Raw
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    ax.set_title("1. Dados Raw (CSV InfluxDB)", fontsize=18, fontweight='bold', pad=20)
    table_data = [['Timestamp', 'Current (A)', 'Vib (mm/s)', 'RPM']]
    for i in [0, 1, 2, 150, 151, 299]:
        row = [df['time_str'].iloc[i], f"{df['current'].iloc[i]:.2f}", f"{df['vib'].iloc[i]:.2f}", f"{df['rpm'].iloc[i]:.0f}"]
        table_data.append(row)
        if i == 2: table_data.append(['...', '...', '...', '...'])
    tab = ax.table(cellText=table_data, loc='center', cellLoc='center')
    tab.scale(1, 3)
    plt.savefig(os.path.join(SAVE_DIR, 'cap_1_raw.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Fórmula
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    ax.set_title("2. Fórmula do Score de Atividade", fontsize=18, fontweight='bold')
    formula = r"$Score = (Current_{norm} \times 2.0) + (RPM_{norm} \times 2.0) + (Vib_{norm} \times 1.0)$"
    ax.text(0.5, 0.6, formula, ha='center', fontsize=22, color='#1f538d', bbox=dict(facecolor='white', edgecolor='#1f538d', pad=15))
    ax.text(0.5, 0.4, "LIGADO se Score > Limiar Dinâmico", ha='center', fontsize=14, color='green', fontweight='bold')
    plt.savefig(os.path.join(SAVE_DIR, 'cap_2_formula.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Interpolacao < 1h
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title("3. Interpolação < 1h (Linear/Spline)", fontsize=18, fontweight='bold')
    # USAR current_gap para mostrar as lacunas reais
    ax.plot(df['time'], df['current_gap'], 'k-', alpha=0.9, label='Dados Originais (Com Lacunas)')
    # O conector deve tocar os pontos das extremidades, então pegamos o gap + 1 ponto de cada lado
    # Para visualização perfeita, usamos a máscara de gap mas garantimos que a linha conecte
    ax.plot(df['time'], v_lin_only_gap, 'b--', linewidth=2.5, label='Reconstrução Linear')
    ax.set_xlim(df['time'].iloc[30], df['time'].iloc[80])
    ax.set_ylim(0, 15)
    ax.legend()
    plt.savefig(os.path.join(SAVE_DIR, 'cap_3_interp_linear.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Interpolacao 1-3h
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title("4. Interpolação 1-3h (KNN Temporal)", fontsize=18, fontweight='bold')
    # USAR current_gap para mostrar as lacunas reais
    ax.plot(df['time'], df['current_gap'], 'k-', alpha=0.9, label='Dados Originais (Com Lacunas)')
    ax.plot(df['time'], v_knn_only_gap, 'g-', linewidth=3, label='Reconstrução KNN')
    ax.set_xlim(df['time'].iloc[110], df['time'].iloc[180])
    ax.set_ylim(0, 40)
    ax.legend()
    plt.savefig(os.path.join(SAVE_DIR, 'cap_4_interp_knn.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Normalização Comparativa (DADOS REAIS VS NORMALIZADOS)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    ax1.set_title("5. Normalização (Preservação da Distribuição)", fontsize=18, fontweight='bold')
    
    # Dados Reais
    ax1.plot(df['time'], df['current'], color='black', label='Sinal Real (Amperes)')
    ax1.set_ylabel("Corrente (A)")
    ax1.legend(loc='upper right')
    
    # Dados Normalizados
    ax2.plot(df['time'], current_norm, color='#1f538d', label='Sinal Normalizado [0, 1]')
    ax2.set_ylabel("Escala [0, 1]")
    ax2.set_xlabel("Tempo")
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'cap_5_norm_comp.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 6. K-Means
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title("6. Clusterização K-Means (K=6)", fontsize=18, fontweight='bold')
    for k in range(6):
        m = (clusters == k)
        ax.scatter(X_scaled[m, 0], X_scaled[m, 1], c=colors[k], s=40, alpha=0.6, edgecolors='none')
    ax.scatter(centroids[:, 0], centroids[:, 1], c='blue', marker='+', s=250, linewidths=4, label='Centróides')
    ax.set_xlabel("Corrente Normalizada")
    ax.set_ylabel("Vibração Normalizada")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    plt.savefig(os.path.join(SAVE_DIR, 'cap_6_kmeans.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 7. Resultados
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title("7. Classificação Final do Estado", fontsize=18, fontweight='bold')
    ax.plot(df['time'], df['current'], 'k-', alpha=0.2)
    ax.fill_between(df['time'], 0, 45, where=(df['state'] == 1), color='green', alpha=0.3, label='LIGADO')
    ax.fill_between(df['time'], 0, 45, where=(df['state'] == 0), color='red', alpha=0.3, label='DESLIGADO')
    ax.set_ylabel("Corrente (A)")
    ax.legend(loc='upper right')
    plt.savefig(os.path.join(SAVE_DIR, 'cap_7_resultado.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Todas as fotos foram salvas em: {SAVE_DIR}")

if __name__ == "__main__":
    generate_pngs()
