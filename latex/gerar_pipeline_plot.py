import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(16, 7))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
ax.set_xlim(0, 16)
ax.set_ylim(0, 7.5)
ax.axis('off')

ax.text(8, 7.2, 'Pipeline Completo — Cinco Fases do Processamento',
        ha='center', va='center', fontsize=15, fontweight='bold', color='#222222')

border_color = '#333333'
fill_color = '#f5f5f5'
accent = '#2c3e50'
text_color = '#222222'
bullet_color = '#555555'

fases = [
    {'num': '1', 'titulo': 'Validação de\nRequisitos',
     'itens': ['Intervalo mínimo 24h', 'Modelo treinado',
               'Scaler persistido', 'Acesso ao InfluxDB']},
    {'num': '2', 'titulo': 'Extração e\nValidação',
     'itens': ['Conversão GMT-3 → UTC', 'Download de dados',
               'Cobertura temporal > 70%']},
    {'num': '3', 'titulo': 'Processamento',
     'itens': ['Segmentação temporal', 'Interpolação adaptativa',
               'Sincronização\nmultifrequência']},
    {'num': '4', 'titulo': 'Classificação\nAvançada',
     'itens': ['Predição K-Means', 'Correção por limiares\nfísicos',
               'Filtro de outliers\ntemporais']},
    {'num': '5', 'titulo': 'Análise de\nMétricas',
     'itens': ['Tempo em cada estado', 'Durações temporais reais',
               'Relatório final']},
]

box_w = 2.5
box_h = 4.6
gap = 0.55
start_x = (16 - (5 * box_w + 4 * gap)) / 2
y_center = 3.6

for i, fase in enumerate(fases):
    x = start_x + i * (box_w + gap)

    rect = FancyBboxPatch((x, y_center - box_h/2), box_w, box_h,
                           boxstyle="round,pad=0.12",
                           facecolor=fill_color, edgecolor=border_color, linewidth=1.8)
    ax.add_patch(rect)

    header_h = 1.5
    header = FancyBboxPatch((x, y_center + box_h/2 - header_h), box_w, header_h,
                             boxstyle="round,pad=0.12",
                             facecolor=accent, edgecolor=border_color, linewidth=1.8)
    ax.add_patch(header)
    ax.fill_between([x, x + box_w],
                    y_center + box_h/2 - header_h,
                    y_center + box_h/2 - header_h + 0.18,
                    color=accent, zorder=2)

    ax.text(x + box_w/2, y_center + box_h/2 - 0.3, f'Fase {fase["num"]}',
            ha='center', va='center', fontsize=10, color='#bbbbbb', zorder=3)
    ax.text(x + box_w/2, y_center + box_h/2 - 0.95, fase['titulo'],
            ha='center', va='center', fontsize=11, fontweight='bold', color='white',
            linespacing=1.2, zorder=3)

    y_item = y_center + box_h/2 - header_h - 0.35
    for item in fase['itens']:
        n_lines = item.count('\n') + 1
        ax.text(x + 0.25, y_item, '▸', ha='left', va='top',
                fontsize=10, color=bullet_color)
        ax.text(x + 0.50, y_item, item, ha='left', va='top',
                fontsize=10.5, color=text_color, linespacing=1.15)
        y_item -= 0.45 * n_lines

    if i < 4:
        arrow_x = x + box_w + 0.08
        arrow_end = arrow_x + gap - 0.16
        ax.annotate('',
                    xy=(arrow_end, y_center), xytext=(arrow_x, y_center),
                    arrowprops=dict(arrowstyle='-|>',
                                    color='#444444',
                                    lw=2.5,
                                    mutation_scale=22))

plt.tight_layout()
plt.savefig(r'c:\Users\manu_\Downloads\TCC_1_FELIPE_COSTA_LOPES-master\latex\05_pipeline_cinco_fases.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("Salvo: 05_pipeline_cinco_fases.png")
