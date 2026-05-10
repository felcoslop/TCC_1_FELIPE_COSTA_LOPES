import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor
from scipy.interpolate import CubicSpline
import os

# Configurações de estilo
plt.style.use('seaborn-v0_8-whitegrid')
SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'plots', 'animacoes_tcc')
os.makedirs(SAVE_DIR, exist_ok=True)

def generate_pipeline_data(n_samples=300):
    """Gera dados para a animação do pipeline com gaps de diferentes tamanhos."""
    np.random.seed(42)
    t = np.arange(n_samples)
    
    # 2 estados: 0=OFF, 1=ON
    states = np.zeros(n_samples)
    states[100:250] = 1
    
    # Features
    current = np.where(states == 1, np.random.normal(30, 1.5, n_samples), np.random.normal(2, 0.4, n_samples))
    vib = np.where(states == 1, np.random.normal(4, 0.4, n_samples), np.random.normal(0.5, 0.1, n_samples))
    rpm = np.where(states == 1, np.random.normal(1200, 15, n_samples), np.random.normal(0, 5, n_samples))
    
    # Injetar Outliers
    vib[180] = 45.0
    current[40] = 12.0
    
    # Injetar Gaps (Lacunas)
    current_with_gap = current.copy()
    
    # Gap 1: Pequeno (30 min) -> Perto do frame 50
    current_with_gap[45:65] = np.nan
    
    # Gap 2: Grande (2 horas) -> Perto do frame 140 (na transição para ON)
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

def create_pipeline_animation(df):
    """Cria a animação completa do pipeline."""
    n_samples = len(df)
    fig = plt.figure(figsize=(14, 10))
    
    # Prepara dados interpolados para a animação
    t_idx = np.arange(n_samples).reshape(-1, 1)
    mask = ~np.isnan(df['current_gap'])
    gap_mask = np.isnan(df['current_gap'])
    
    # KNN
    knn = KNeighborsRegressor(n_neighbors=5, weights='distance')
    knn.fit(t_idx[mask], df.loc[mask, 'current_gap'])
    current_knn = knn.predict(t_idx)
    # MASCARAR: Mostrar apenas nos gaps
    current_knn_only_gap = np.full(n_samples, np.nan)
    current_knn_only_gap[gap_mask] = current_knn[gap_mask]
    
    # Linear
    from scipy.interpolate import interp1d
    f_lin = interp1d(t_idx[mask].flatten(), df.loc[mask, 'current_gap'], kind='linear', fill_value="extrapolate")
    current_linear = f_lin(t_idx.flatten())
    # MASCARAR: Mostrar apenas nos gaps
    current_linear_only_gap = np.full(n_samples, np.nan)
    current_linear_only_gap[gap_mask] = current_linear[gap_mask]

    # Prepara dados normalizados
    scaler = MinMaxScaler()
    X_raw = df[['current', 'vib']].values
    X_clean = X_raw.copy()
    X_clean[X_clean > 15] = 6.0 
    X_scaled = scaler.fit_transform(X_clean)
    current_normalized = X_scaled[:, 0]

    # Prepara K-Means
    kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    final_centroids = kmeans.cluster_centers_
    
    centroid_scores = np.sum(final_centroids, axis=1)
    sorted_centroid_idx = np.argsort(centroid_scores)
    
    cluster_colors = {}
    cluster_colors[sorted_centroid_idx[0]] = '#4d0000'
    cluster_colors[sorted_centroid_idx[1]] = '#ff4d4d'
    cluster_colors[sorted_centroid_idx[2]] = '#ccffcc'
    cluster_colors[sorted_centroid_idx[3]] = '#66ff66'
    cluster_colors[sorted_centroid_idx[4]] = '#00cc00'
    cluster_colors[sorted_centroid_idx[5]] = '#004d00'

    chapters = [
        "1. Dados Raw (CSV InfluxDB)",
        "2. Fórmula do Score de Atividade",
        "3. Interpolação < 1h (Linear)",
        "4. Interpolação 1-3h (KNN)",
        "5. Normalização (Escalabilidade)",
        "6. Clusterização K-Means (K=6)",
        "7. Classificação Final (Ligado/Desligado)"
    ]
    
    frames_per_chapter = 50
    total_frames = len(chapters) * frames_per_chapter
    df['time_str'] = df['time'].dt.strftime('%d-%m-%y %H:%M')

    def update(frame):
        plt.clf()
        chapter_idx = min(frame // frames_per_chapter, len(chapters) - 1)
        chapter_frame = frame % frames_per_chapter
        
        ax = fig.add_subplot(111)
        ax.set_title(f"Pipeline TCC: {chapters[chapter_idx]}", fontsize=18, fontweight='bold', pad=25)
        
        if chapter_idx == 0: # Raw Data
            ax.axis('off')
            table_data = [['Timestamp', 'Current (A)', 'Vib (mm/s)', 'RPM']]
            for i in [0, 1, 2, 150, 151, 299]:
                row = [df['time_str'].iloc[i], f"{df['current'].iloc[i]:.2f}", f"{df['vib'].iloc[i]:.2f}", f"{df['rpm'].iloc[i]:.0f}"]
                table_data.append(row)
                if i == 2: table_data.append(['...', '...', '...', '...'])
            the_table = ax.table(cellText=table_data, loc='center', cellLoc='center')
            the_table.auto_set_font_size(False)
            the_table.set_fontsize(13)
            the_table.scale(1.2, 3.5)

        elif chapter_idx == 1: # Formula
            ax.axis('off')
            formula = r"$Score = (Current_{norm} \times 2.0) + (RPM_{norm} \times 2.0) + (Vib_{norm} \times 1.0)$"
            ax.text(0.5, 0.75, "Cálculo de Importância Ponderada:", ha='center', fontsize=18)
            ax.text(0.5, 0.60, formula, ha='center', fontsize=22, color='#1f538d', bbox=dict(facecolor='white', edgecolor='#1f538d', pad=15))
            ax.text(0.1, 0.40, "Exemplo DESLIGADO:", fontsize=14, fontweight='bold', color='red')
            ax.text(0.1, 0.33, r"$(0.04 \times 2) + (0.01 \times 2) + (0.12 \times 1) = \mathbf{0.22}$", fontsize=14)
            ax.text(0.6, 0.40, "Exemplo LIGADO:", fontsize=14, fontweight='bold', color='green')
            ax.text(0.6, 0.33, r"$(0.88 \times 2) + (0.92 \times 2) + (0.75 \times 1) = \mathbf{4.35}$", fontsize=14)

        elif chapter_idx == 2: # Interpolation < 1h
            # Mostrar os dados originais
            ax.plot(df['time'], df['current'], 'k-', alpha=0.8, label='Dados Originais')
            # Mostrar a linha interpolada APENAS onde há gaps
            ax.plot(df['time'], current_linear_only_gap, 'b--', linewidth=2, label='Conector Linear (Gap < 1h)')
            ax.set_xlim(df['time'].iloc[30], df['time'].iloc[80])
            ax.set_ylabel("Corrente (A)")
            ax.set_ylim(0, 40)
            ax.legend(loc='upper right')

        elif chapter_idx == 3: # Interpolation 1-3h
            # Mostrar os dados originais
            ax.plot(df['time'], df['current'], 'k-', alpha=0.8, label='Dados Originais')
            # Mostrar a linha interpolada APENAS onde há gaps
            ax.plot(df['time'], current_knn_only_gap, 'g-', linewidth=3, label='Conector KNN (Gap 1-3h)')
            ax.set_xlim(df['time'].iloc[110], df['time'].iloc[180])
            ax.set_ylabel("Corrente (A)")
            ax.set_ylim(0, 40)
            ax.legend(loc='upper right')

        elif chapter_idx == 4: # Normalization Comparison (Time Series)
            # Mostrar que o sinal é idêntico em formato, mudando apenas o eixo Y
            progress = min(chapter_frame / 40.0, 1.0)
            
            if progress < 0.5:
                # Sinal Original
                ax.plot(df['time'], df['current'], color='gray', label='Sinal Original (A)')
                ax.set_ylabel("Corrente (Amperes)", color='gray', fontsize=14)
                ax.set_ylim(0, 40)
            else:
                # Sinal Normalizado
                ax.plot(df['time'], current_normalized * 40, color='#1f538d', label='Sinal Normalizado [0, 1]')
                ax.set_ylabel("Escala Normalizada (0.0 a 1.0)", color='#1f538d', fontsize=14)
                ax.set_ylim(0, 40)
                # Adicionar ticks secundários para mostrar a escala 0-1
                ax2 = ax.twinx()
                ax2.set_ylim(0, 1)
                ax2.set_ylabel("Valor Normalizado", color='#1f538d')
            
            ax.set_xlim(df['time'].iloc[0], df['time'].iloc[299])
            ax.legend(loc='upper right')

        elif chapter_idx == 5: # K-Means
            for k in range(6):
                mask = (clusters == k)
                ax.scatter(X_scaled[mask, 0], X_scaled[mask, 1], c=cluster_colors[k], s=40, alpha=0.6, edgecolors='none')
            for k in range(6):
                ax.scatter(final_centroids[k, 0], final_centroids[k, 1], c='blue', marker='+', s=250, linewidths=4)
            ax.set_xlabel("Corrente Normalizada")
            ax.set_ylabel("Vibração Normalizada")
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)

        elif chapter_idx == 6: # Final Result
            ax.plot(df['time'], df['current'], 'k-', alpha=0.15)
            ax.fill_between(df['time'], 0, 45, where=(df['state'] == 1), color='green', alpha=0.25, label='Status: LIGADO')
            ax.fill_between(df['time'], 0, 45, where=(df['state'] == 0), color='red', alpha=0.25, label='Status: DESLIGADO')
            ax.set_ylabel("Corrente (A) / Classificação")
            ax.legend(loc='upper right')
            ax.set_ylim(0, 45)
            import matplotlib.dates as mdates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%y %H:%M'))

        plt.tight_layout()

    # Criar animação
    print(f"Iniciando geração de {total_frames} frames...")
    ani = animation.FuncAnimation(fig, update, frames=range(total_frames), interval=250)
    
    save_path = os.path.join(SAVE_DIR, '05_pipeline_completo.gif')
    try:
        # Tentar salvar o GIF
        ani.save(save_path, writer='pillow', fps=5) 
        print(f"Salvo: {save_path}")
    except Exception as e:
        print(f"Erro ao salvar GIF: {e}")
    
    # Gerar PNGs de cada capítulo (pegando o último frame de cada um)
    print("Gerando PNGs dos capítulos...")
    for i in range(len(chapters)):
        try:
            update(i * frames_per_chapter + (frames_per_chapter - 1))
            png_path = os.path.join(SAVE_DIR, f'capitulo_{i+1}_{chapters[i][:15].lower().replace(" ", "_").replace(".", "")}.png')
            plt.savefig(png_path, dpi=300, bbox_inches='tight')
            print(f"Salvo: {png_path}")
        except Exception as e:
            print(f"Erro ao salvar PNG {i}: {e}")
        
    plt.close()

if __name__ == "__main__":
    print("Gerando Animação do Pipeline Completo (Versão Refinada Final)...")
    data = generate_pipeline_data(300)
    create_pipeline_animation(data)
    print("Concluído!")
