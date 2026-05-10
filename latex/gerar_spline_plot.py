import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime, timedelta
from scipy.interpolate import CubicSpline

# Dados originais (mesmo padrão do gráfico KNN)
# Pontos ANTES do gap
datas_antes = [
    datetime(2025, 10, 15, 8, 12, 0),
    datetime(2025, 10, 15, 8, 14, 0),
    datetime(2025, 10, 15, 8, 16, 0),
    datetime(2025, 10, 15, 8, 18, 0),
    datetime(2025, 10, 15, 8, 20, 0),
    datetime(2025, 10, 15, 8, 22, 0),
]
valores_antes = [0.55, 0.58, 0.45, 0.40, 0.50, 0.70]

# Pontos DEPOIS do gap
datas_depois = [
    datetime(2025, 10, 15, 8, 30, 0),
    datetime(2025, 10, 15, 8, 32, 0),
    datetime(2025, 10, 15, 8, 34, 0),
    datetime(2025, 10, 15, 8, 35, 0),
    datetime(2025, 10, 15, 8, 36, 0),
    datetime(2025, 10, 15, 8, 37, 0),
    datetime(2025, 10, 15, 8, 38, 0),
    datetime(2025, 10, 15, 8, 40, 0),
    datetime(2025, 10, 15, 8, 42, 0),
]
valores_depois = [4.55, 5.05, 5.60, 5.70, 5.30, 4.90, 4.50, 3.95, 3.85]

# Todos os dados originais para a spline
todas_datas = datas_antes + datas_depois
todos_valores = valores_antes + valores_depois

# Converter para numérico para interpolação
t_ref = datas_antes[0]
t_num_todos = np.array([(d - t_ref).total_seconds() for d in todas_datas])
v_todos = np.array(todos_valores)

# Pontos do gap onde vamos interpolar (entre 08:22 e 08:30)
gap_inicio = datetime(2025, 10, 15, 8, 23, 0)
gap_fim = datetime(2025, 10, 15, 8, 29, 0)

datas_gap = [gap_inicio + timedelta(minutes=i) for i in range(0, int((gap_fim - gap_inicio).total_seconds() / 60) + 1, 1)]
t_num_gap = np.array([(d - t_ref).total_seconds() for d in datas_gap])

# Interpolação Spline Cúbica
cs = CubicSpline(t_num_todos, v_todos)
valores_interpolados = cs(t_num_gap)

# ======================== PLOT ========================
fig, ax = plt.subplots(figsize=(12, 5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Dados originais (antes + depois) — linha azul com círculos
ax.plot(datas_antes + datas_depois, valores_antes + valores_depois,
        color='#4444FF', marker='o', markersize=6, linewidth=1.2,
        linestyle='-', label='Dados Originais (Antes e Depois do Gap)',
        zorder=3)

# Valores interpolados — marcadores X vermelhos
ax.plot(datas_gap, valores_interpolados,
        color='red', marker='X', markersize=10, linewidth=0,
        label='Valores Interpolados (Spline Cúbica)',
        zorder=4)

# Fluxo da interpolação — linha tracejada vermelha conectando tudo
fluxo_datas = [datas_antes[-1]] + datas_gap + [datas_depois[0]]
fluxo_valores = [valores_antes[-1]] + list(valores_interpolados) + [valores_depois[0]]
ax.plot(fluxo_datas, fluxo_valores,
        color='red', linestyle='--', linewidth=1.2,
        label='Fluxo da Interpolação',
        zorder=2)

# Formatação dos eixos
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%y %H:%M'))
ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
plt.xticks(rotation=45, ha='right')

ax.set_xlabel('Data e Hora', fontsize=11)
ax.set_ylabel('Vibração RMS (mm/s)', fontsize=11)
ax.set_title('Interpolação Temporal de Lacuna utilizando Spline Cúbica', fontsize=13, fontweight='bold')

ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
ax.grid(False)

plt.tight_layout()
plt.savefig(r'c:\Users\manu_\Downloads\TCC_1_FELIPE_COSTA_LOPES-master\latex\02_spline_interpolacao.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("Salvo: 02_spline_interpolacao.png")
