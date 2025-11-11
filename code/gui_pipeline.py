"""
Interface gráfica pra rodar o sistema de detecção de estados dos equipamentos.
Tem calendario pra escolher datas, botoes pra executar as analises, e mostra resultados.
Feito pra facilitar o uso sem precisar mexer no terminal o tempo todo.
"""

import sys
import threading
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import json
import customtkinter as ctk
from tkcalendar import DateEntry
from PIL import Image

        # Deixa a interface com tema escuro e azul
ctk.set_appearance_mode("dark")  # "dark", "light", "system"
ctk.set_default_color_theme("blue")  # "blue", "green", "dark-blue"

class PipelineGUI:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.dir_raw = self.base_dir / 'data' / 'raw'
        self.dir_models = self.base_dir / 'models'

        # Lista para controlar processos filhos
        self.processos_filhos = []

        # Cria a janela principal da aplicacao
        self.root = ctk.CTk()

        self.root.title("Pipeline de Detecção de Estados - Sistema Versátil de ML")

        # Configurar para fullscreen
        self.root.attributes('-fullscreen', True)
        # Permitir sair do fullscreen com F11
        self.root.bind('<F11>', lambda e: self.toggle_fullscreen())
        self.root.bind('<Escape>', lambda e: self.root.attributes('-fullscreen', False))

        # Configurar fechamento da aplicação
        self.root.protocol("WM_DELETE_WINDOW", self.fechar_aplicacao)

        # Configurar grid responsivo
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Criar interface
        self.criar_interface()

    def criar_frame(self, parent, **kwargs):
        """Cria CTkFrame"""
        return ctk.CTkFrame(parent, **kwargs)

    def criar_label(self, parent, text, **kwargs):
        """Cria CTkLabel"""
        return ctk.CTkLabel(parent, text=text, **kwargs)

    def criar_button(self, parent, text, command, **kwargs):
        """Cria CTkButton"""
        return ctk.CTkButton(parent, text=text, command=command, **kwargs)

    def criar_entry(self, parent, **kwargs):
        """Cria CTkEntry"""
        return ctk.CTkEntry(parent, **kwargs)

    def criar_tabview(self, parent):
        """Cria CTkTabview"""
        return ctk.CTkTabview(parent)

    def criar_combo(self, parent, values, **kwargs):
        """Cria CTkComboBox"""
        return ctk.CTkComboBox(parent, values=values, **kwargs)

    def criar_interface(self):
        """Cria a interface principal com abas"""
        # Container principal
        main_container = self.criar_frame(self.root)
        main_container.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        main_container.grid_rowconfigure(0, weight=1)
        main_container.grid_columnconfigure(0, weight=1)

        # Título
        title_label = self.criar_label(
            main_container,
            text="Sistema de Detecção de Estados Operacionais",
            font=("Arial", 24, "bold")
        )
        title_label.grid(row=0, column=0, pady=(20, 10), sticky="ew")

        # Logo UFMG
        try:
            logo_path = self.base_dir.parent / 'latex' / 'logo-ufmg.png'
            if logo_path.exists():
                logo_image = Image.open(logo_path)
                
                print(f"[DEBUG] Logo mode: {logo_image.mode}, size: {logo_image.size}")
                
                # Converter para RGB com fundo branco (CRÍTICO para dark mode)
                # A logo está em modo LA (Luminance + Alpha) = escala de cinza com transparência
                if logo_image.mode in ('RGBA', 'LA', 'PA'):
                    # Criar fundo branco
                    background = Image.new('RGB', logo_image.size, (255, 255, 255))
                    # Converter para RGB se necessário
                    if logo_image.mode == 'LA':
                        logo_image = logo_image.convert('RGBA')
                    # Colar logo sobre fundo branco usando canal alpha
                    background.paste(logo_image, (0, 0), logo_image)
                    logo_image = background
                elif logo_image.mode != 'RGB':
                    # Outros modos: converter direto para RGB
                    logo_image = logo_image.convert('RGB')
                
                print(f"[DEBUG] Logo convertida para: {logo_image.mode}")
                
                # Redimensionar mantendo proporção (altura 100px)
                aspect_ratio = logo_image.width / logo_image.height
                new_height = 100
                new_width = int(new_height * aspect_ratio)
                
                # Criar CTkImage - funciona melhor com RGB
                logo_ctk = ctk.CTkImage(
                    light_image=logo_image,
                    dark_image=logo_image,
                    size=(new_width, new_height)
                )
                
                # Frame com fundo branco para garantir visibilidade
                logo_frame = ctk.CTkFrame(
                    main_container,
                    fg_color="white",
                    corner_radius=10
                )
                logo_frame.grid(row=1, column=0, pady=10, padx=20)
                
                # Label para exibir a logo dentro do frame branco
                logo_label = ctk.CTkLabel(
                    logo_frame,
                    image=logo_ctk,
                    text="",
                    fg_color="white"
                )
                logo_label.pack(padx=10, pady=10)
                
                print(f"[OK] Logo UFMG carregada e exibida: {new_width}x{new_height}px")
            else:
                print(f"[AVISO] Logo não encontrada: {logo_path}")
        except Exception as e:
            print(f"[ERRO] Falha ao carregar logo: {e}")
            import traceback
            traceback.print_exc()

        # Subtítulo
        subtitle_label = self.criar_label(
            main_container,
            text="Algoritmo K-means Versátil com Thresholds Dinâmicos",
            font=("Arial", 14)
        )
        subtitle_label.grid(row=2, column=0, pady=(10, 20), sticky="ew")

        # Criar abas
        self.tabview = self.criar_tabview(main_container)
        self.tabview.grid(row=3, column=0, sticky="nsew", padx=10, pady=10)
        main_container.grid_rowconfigure(3, weight=1)

        # Adicionar abas
        tab_treino = self.tabview.add("Treino")
        tab_analise = self.tabview.add("Analise de Intervalo")
        tab_visualizar = self.tabview.add("Visualizacao 3D")
        
        # Flag para controlar execução de tarefas
        self.tarefa_em_execucao = False
        
        # Configurar callback para bloquear mudança de aba
        self.tabview.configure(command=self.verificar_mudanca_aba)
        
        # Preencher cada aba
        self.criar_aba_treino(tab_treino)
        self.criar_aba_analise(tab_analise)
        self.criar_aba_visualizar(tab_visualizar)
        
        # Rodapé com informações do autor
        footer = self.criar_label(
            main_container,
            text="FELIPE COSTA LOPES | Matrícula: 2018019648",
            font=("Arial", 10, "bold")
        )
        footer.grid(row=4, column=0, pady=10, sticky="ew")
    
    def criar_aba_treino(self, parent):
        """Cria interface da aba de treino"""
        parent.grid_rowconfigure(2, weight=1)
        parent.grid_columnconfigure(0, weight=1)

        # Header
        header_frame = self.criar_frame(parent)
        header_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)

        self.criar_label(
            header_frame,
            text="Treinar Modelo K-means",
            font=("Arial", 18, "bold")
        ).pack(pady=10)

        self.criar_label(
            header_frame,
            text="Processar dados historicos e gerar parametros dinamicos",
            font=("Arial", 12)
        ).pack(pady=5)
        
        # Container de mpoints
        mpoints_frame = self.criar_frame(parent)
        mpoints_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        mpoints_frame.grid_rowconfigure(1, weight=1)
        mpoints_frame.grid_columnconfigure(0, weight=1)

        self.criar_label(
            mpoints_frame,
            text="Selecione o Equipamento (Mpoint):",
            font=("Arial", 14, "bold")
        ).grid(row=0, column=0, pady=(10, 5), sticky="w", padx=10)
        
        # Frame para lista com scrollbar
        list_frame = self.criar_frame(mpoints_frame)
        list_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        list_frame.grid_rowconfigure(0, weight=1)
        list_frame.grid_columnconfigure(0, weight=1)

        # Lista de mpoints
        self.mpoint_listbox = tk.Listbox(
            list_frame,
            font=("Courier", 12),
            selectmode=tk.SINGLE,
            bg="#2b2b2b",
            fg="white",
            selectbackground="#1f6aa5",
            selectforeground="white",
            height=10
        )
        self.mpoint_listbox.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Scrollbar
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.mpoint_listbox.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.mpoint_listbox.configure(yscrollcommand=scrollbar.set)

        # Botão de atualizar lista
        btn_frame = self.criar_frame(mpoints_frame)
        btn_frame.grid(row=2, column=0, pady=10, sticky="ew", padx=10)

        self.criar_button(
            btn_frame,
            text="Atualizar Lista",
            command=self.atualizar_lista_mpoints,
            width=150,
            height=35
        ).pack(side="left", padx=5)

        self.criar_button(
            btn_frame,
            text="Iniciar Treino",
            command=self.executar_treino,
            width=200,
            height=35,
            fg_color="green",
            hover_color="darkgreen"
        ).pack(side="right", padx=5)

        # Console de saída TREINO
        console_frame = self.criar_frame(parent)
        console_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=10)
        parent.grid_rowconfigure(2, weight=1)

        self.criar_label(console_frame, text="Console de Saida (Treino):", font=("Arial", 12, "bold")).pack(
            anchor="w", padx=10, pady=(10, 5)
        )

        self.console_treino = scrolledtext.ScrolledText(
            console_frame,
            height=15,
            font=("Courier", 9),
            bg="#1e1e1e",
            fg="lime",
            insertbackground="white"
        )
        self.console_treino.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # Carregar lista inicial
        self.atualizar_lista_mpoints()
    
    def atualizar_display_datas(self, event=None):
        """Atualiza os labels de exibição das datas no formato DD-MM-AAAA"""
        try:
            # Data início
            data_inicio = self.data_inicio_cal.get_date()
            self.data_inicio_display.configure(text=data_inicio.strftime('%d-%m-%Y'))
            
            # Data fim
            data_fim = self.data_fim_cal.get_date()
            self.data_fim_display.configure(text=data_fim.strftime('%d-%m-%Y'))
        except Exception as e:
            print(f"[AVISO] Erro ao atualizar display de datas: {e}")
    
    def criar_aba_analise(self, parent):
        """Cria interface da aba de análise por intervalo"""
        parent.grid_rowconfigure(5, weight=1)
        parent.grid_columnconfigure(0, weight=1)

        # Header
        header_frame = self.criar_frame(parent)
        header_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)

        self.criar_label(
            header_frame,
            text="Analise por Intervalo",
            font=("Arial", 18, "bold")
        ).pack(pady=10)

        self.criar_label(
            header_frame,
            text="Classificar estados usando modelo treinado e thresholds dinamicos",
            font=("Arial", 12)
        ).pack(pady=5)

        # Configurações de conexão InfluxDB
        config_frame = self.criar_frame(parent)
        config_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=10)

        self.criar_label(config_frame, text="Configuracao InfluxDB", font=("Arial", 14, "bold")).grid(
            row=0, column=0, columnspan=4, pady=10, sticky="w", padx=10
        )
        
        # IP
        self.criar_label(config_frame, text="IP:", font=("Arial", 12)).grid(row=1, column=0, padx=(10, 5), pady=5, sticky="e")
        self.ip_entry = self.criar_entry(config_frame, width=150)
        self.ip_entry.insert(0, "10.8.0.121")
        self.ip_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        # Porta
        self.criar_label(config_frame, text="Porta:", font=("Arial", 12)).grid(row=1, column=2, padx=(20, 5), pady=5, sticky="e")
        self.porta_entry = self.criar_entry(config_frame, width=80)
        self.porta_entry.insert(0, "8086")
        self.porta_entry.grid(row=1, column=3, padx=5, pady=5, sticky="w")
        
        # Seleção de Mpoint
        mpoint_frame = self.criar_frame(parent)
        mpoint_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=10)

        self.criar_label(mpoint_frame, text="Equipamento (Mpoint):", font=("Arial", 14, "bold")).grid(
            row=0, column=0, pady=10, sticky="w", padx=10
        )

        self.mpoint_analise_var = tk.StringVar()
        self.mpoint_analise_combo = self.criar_combo(
            mpoint_frame,
            values=self.listar_mpoints_treinados(),
            width=200
        )
        self.mpoint_analise_combo.grid(row=0, column=1, padx=10, pady=10, sticky="w")

        self.criar_button(
            mpoint_frame,
            text="Atualizar",
            command=self.atualizar_combo_mpoints,
            width=80
        ).grid(row=0, column=2, padx=5)
        
        # Seleção de datas
        datas_frame = self.criar_frame(parent)
        datas_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=10)

        self.criar_label(datas_frame, text="Intervalo de Analise (GMT-3):", font=("Arial", 14, "bold")).grid(
            row=0, column=0, columnspan=4, pady=10, sticky="w", padx=10
        )

        # Data início
        self.criar_label(datas_frame, text="Início:", font=("Arial", 12)).grid(row=1, column=0, padx=(10, 5), pady=5, sticky="e")

        # Label para mostrar data selecionada em formato DD-MM-AAAA (MAIOR)
        self.data_inicio_display = self.criar_label(datas_frame, text="", font=("Arial", 16, "bold"), text_color="cyan")
        self.data_inicio_display.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        
        # Calendário com apenas o botão visível (esconder data)
        self.data_inicio_cal = DateEntry(
            datas_frame,
            width=2,  # Muito pequeno para esconder a data
            background='darkblue',
            foreground='white',
            borderwidth=2,
            date_pattern='yyyy-mm-dd',
            font=("Arial", 1),  # Fonte minúscula para esconder texto
            # Configurações do calendário popup
            headersbackground='darkblue',
            headersforeground='white',
            normalbackground='white',
            normalforeground='black',
            selectbackground='#1f6aa5',
            selectforeground='white',
            weekendbackground='lightgray',
            weekendforeground='black',
            othermonthforeground='gray',
            othermonthbackground='white',
            othermonthweforeground='gray',
            othermonthwebackground='lightgray',
            bordercolor='darkblue',
            showweeknumbers=False,
            firstweekday='sunday',
            maxdate=datetime.now().date(),  # Não permitir futuro
            mindate=datetime(2025, 1, 1).date()  # Não permitir antes de 2025
        )
        self.data_inicio_cal.grid(row=1, column=2, padx=(0, 10), pady=5, sticky="w")
        # Aumentar tamanho da fonte do calendário popup
        self.data_inicio_cal._calendar.configure(font=("Arial", 12))
        self.data_inicio_cal.bind("<<DateEntrySelected>>", self.atualizar_display_datas)

        self.criar_label(datas_frame, text="Hora:", font=("Arial", 12)).grid(row=1, column=3, padx=(10, 5), pady=5, sticky="e")
        self.hora_inicio_entry = self.criar_entry(datas_frame, width=100, placeholder_text="HH:MM:SS")
        self.hora_inicio_entry.insert(0, "00:00:00")
        self.hora_inicio_entry.grid(row=1, column=4, padx=5, pady=5, sticky="w")

        # Data fim
        self.criar_label(datas_frame, text="Fim:", font=("Arial", 12)).grid(row=2, column=0, padx=(10, 5), pady=5, sticky="e")

        # Label para mostrar data selecionada em formato DD-MM-AAAA (MAIOR)
        self.data_fim_display = self.criar_label(datas_frame, text="", font=("Arial", 16, "bold"), text_color="cyan")
        self.data_fim_display.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        
        # Calendário com apenas o botão visível (esconder data)
        self.data_fim_cal = DateEntry(
            datas_frame,
            width=2,  # Muito pequeno para esconder a data
            background='darkblue',
            foreground='white',
            borderwidth=2,
            date_pattern='yyyy-mm-dd',
            font=("Arial", 1),  # Fonte minúscula para esconder texto
            # Configurações do calendário popup
            headersbackground='darkblue',
            headersforeground='white',
            normalbackground='white',
            normalforeground='black',
            selectbackground='#1f6aa5',
            selectforeground='white',
            weekendbackground='lightgray',
            weekendforeground='black',
            othermonthforeground='gray',
            othermonthbackground='white',
            othermonthweforeground='gray',
            othermonthwebackground='lightgray',
            bordercolor='darkblue',
            showweeknumbers=False,
            firstweekday='sunday',
            maxdate=datetime.now().date(),  # Não permitir futuro
            mindate=datetime(2025, 1, 1).date()  # Não permitir antes de 2025
        )
        self.data_fim_cal.grid(row=2, column=2, padx=(0, 10), pady=5, sticky="w")
        # Aumentar tamanho da fonte do calendário popup
        self.data_fim_cal._calendar.configure(font=("Arial", 12))
        self.data_fim_cal.bind("<<DateEntrySelected>>", self.atualizar_display_datas)

        self.criar_label(datas_frame, text="Hora:", font=("Arial", 12)).grid(row=2, column=3, padx=(10, 5), pady=5, sticky="e")
        self.hora_fim_entry = self.criar_entry(datas_frame, width=100, placeholder_text="HH:MM:SS")
        self.hora_fim_entry.insert(0, "23:59:59")
        self.hora_fim_entry.grid(row=2, column=4, padx=5, pady=5, sticky="w")
        
        # Atualizar displays iniciais das datas
        self.root.after(100, self.atualizar_display_datas)
        
        # Botão de análise
        btn_analise_frame = self.criar_frame(parent)
        btn_analise_frame.grid(row=4, column=0, pady=20, sticky="ew", padx=10)

        self.criar_button(
            btn_analise_frame,
            text="Executar Analise",
            command=self.executar_analise,
            width=250,
            height=45,
            fg_color="green",
            hover_color="darkgreen"
        ).pack(pady=10)

        # Console de saída
        console_frame = self.criar_frame(parent)
        console_frame.grid(row=5, column=0, sticky="nsew", padx=10, pady=10)
        parent.grid_rowconfigure(5, weight=1)

        # Forçar atualização do layout
        parent.update_idletasks()

        self.criar_label(console_frame, text="Console de Saida:", font=("Arial", 12, "bold")).pack(
            anchor="w", padx=10, pady=(10, 5)
        )

        self.console_analise = scrolledtext.ScrolledText(
            console_frame,
            height=10,
            font=("Courier", 9),
            bg="#1e1e1e",
            fg="lime",
            insertbackground="white"
        )
        self.console_analise.pack(fill="both", expand=True, padx=10, pady=(0, 10))
    
    def criar_aba_visualizar(self, parent):
        """Cria interface da aba de visualização 3D"""
        parent.grid_rowconfigure(2, weight=1)
        parent.grid_columnconfigure(0, weight=1)

        # Header
        header_frame = self.criar_frame(parent)
        header_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)

        self.criar_label(
            header_frame,
            text="Visualizacao 3D dos Clusters",
            font=("Arial", 18, "bold")
        ).pack(pady=10)

        self.criar_label(
            header_frame,
            text="Gerar graficos 3D: Corrente x Vibracao x Tempo e RPM x Vibracao x Tempo",
            font=("Arial", 12)
        ).pack(pady=5)

        # Seleção de mpoint
        select_frame = self.criar_frame(parent)
        select_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=10)

        self.criar_label(select_frame, text="Equipamento:", font=("Arial", 14, "bold")).grid(
            row=0, column=0, padx=10, pady=10, sticky="e"
        )

        self.mpoint_viz_var = tk.StringVar()
        self.mpoint_viz_combo = self.criar_combo(
            select_frame,
            values=self.listar_mpoints_treinados(),
            width=200
        )
        self.mpoint_viz_combo.grid(row=0, column=1, padx=10, pady=10, sticky="w")

        # Botões
        btn_frame = self.criar_frame(parent)
        btn_frame.grid(row=2, column=0, pady=20, sticky="n")

        self.criar_button(
            btn_frame,
            text="Gerar Visualizacao 3D",
            command=self.executar_visualizacao,
            width=250,
            height=45,
            fg_color="#1f6aa5",
            hover_color="#144870"
        ).pack(pady=10)

        # Console
        console_frame = self.criar_frame(parent)
        console_frame.grid(row=3, column=0, sticky="nsew", padx=10, pady=10)
        parent.grid_rowconfigure(3, weight=1)

        self.criar_label(console_frame, text="Console de Saida:", font=("Arial", 12, "bold")).pack(
            anchor="w", padx=10, pady=(10, 5)
        )

        self.console_viz = scrolledtext.ScrolledText(
            console_frame,
            height=10,
            font=("Courier", 9),
            bg="#1e1e1e",
            fg="cyan",
            insertbackground="white"
        )
        self.console_viz.pack(fill="both", expand=True, padx=10, pady=(0, 10))
    
    def atualizar_lista_mpoints(self):
        """Atualiza lista de mpoints disponíveis para treino (ELÉTRICOS e MECÂNICOS)"""
        self.mpoint_listbox.delete(0, tk.END)

        # Buscar apenas mpoints completos (com todos os dados necessários)
        mpoints_completos = [mp for mp in self.listar_mpoints_disponiveis() if mp['completo']]

        if not mpoints_completos:
            self.mpoint_listbox.insert(tk.END, "  Nenhum equipamento com dados completos encontrado")
            self.mpoint_listbox.insert(tk.END, "")
            self.mpoint_listbox.insert(tk.END, "  Arquivos necessários:")
            self.mpoint_listbox.insert(tk.END, "  ELÉTRICO: dados_c + dados_estimated + dados_slip")
            self.mpoint_listbox.insert(tk.END, "  MECÂNICO: dados_c + dados_slip (SEM estimated)")
            return

        self.mpoint_listbox.insert(tk.END, f"  {len(mpoints_completos)} equipamento(s) pronto(s) para treino:")
        self.mpoint_listbox.insert(tk.END, "  " + "="*50)

        for mp in mpoints_completos:
            # Adicionar indicador de tipo ao exibir
            tipo_label = f"[{mp['tipo']}]"
            linha = f"{mp['mpoint']} {tipo_label}"
            self.mpoint_listbox.insert(tk.END, linha)
    
    def atualizar_combo_mpoints(self):
        """Atualiza combo de mpoints treinados"""
        mpoints = self.listar_mpoints_treinados()
        self.mpoint_analise_combo.configure(values=mpoints)
        self.mpoint_viz_combo.configure(values=mpoints)
        
        if mpoints:
            self.mpoint_analise_var.set(mpoints[0])
            self.mpoint_viz_var.set(mpoints[0])
    
    def listar_mpoints_disponiveis(self):
        """Lista mpoints com dados disponíveis (ELÉTRICOS e MECÂNICOS)"""
        if not self.dir_raw.exists():
            return []
        
        import re

        # Usar regex para encontrar apenas arquivos que são exatamente 'dados_c_' + dígitos + '.csv'
        arquivos_dados = list(self.dir_raw.glob('dados_c_*.csv'))
        mpoints = []

        for arq in arquivos_dados:
            nome = arq.stem
            # Verificar se corresponde exatamente ao padrão: dados_c_ + apenas números
            match = re.match(r'dados_c_(\d+)$', nome)
            if match:
                numero = match.group(1)
                mpoint = f'c_{numero}'

                arq_estimated = self.dir_raw / f'dados_estimated_{mpoint}.csv'
                arq_slip = self.dir_raw / f'dados_slip_{mpoint}.csv'

                # Determinar tipo de equipamento:
                # ELÉTRICO: tem estimated (current + RPM)
                # MECÂNICO: NÃO tem estimated (só temperatura + vibração)
                if arq_estimated.exists() and arq_slip.exists():
                    # EQUIPAMENTO ELÉTRICO (tem todas as 3 tabelas)
                    mpoints.append({
                        'mpoint': mpoint,
                        'completo': True,
                        'estimated': True,
                        'slip': True,
                        'tipo': 'ELETRICO'
                    })
                elif not arq_estimated.exists() and arq_slip.exists():
                    # EQUIPAMENTO MECÂNICO (só tem dados_c + slip, sem estimated)
                    mpoints.append({
                        'mpoint': mpoint,
                        'completo': True,
                        'estimated': False,
                        'slip': True,
                        'tipo': 'MECANICO'
                    })
                else:
                    # Incompleto
                    mpoints.append({
                        'mpoint': mpoint,
                        'completo': False,
                        'estimated': arq_estimated.exists(),
                        'slip': arq_slip.exists(),
                        'tipo': 'INCOMPLETO'
                    })
        
        return mpoints
    
    def detectar_tipo_equipamento(self, mpoint):
        """Detecta se o equipamento é MECÂNICO ou ELÉTRICO baseado nos arquivos raw ou config"""
        # Método 1: Verificar config do modelo (mais confiável se já treinou)
        config_file = self.dir_models / mpoint / f'config_{mpoint}.json'
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    equipment_type = config.get('equipment_type', None)
                    if equipment_type == 'MECHANICAL':
                        return 'MECANICO'
                    elif equipment_type:
                        return 'ELETRICO'
            except:
                pass
        
        # Método 2: Verificar arquivos raw (fallback)
        arq_estimated = self.dir_raw / f'dados_estimated_{mpoint}.csv'
        if arq_estimated.exists():
            return 'ELETRICO'
        else:
            return 'MECANICO'
    
    def listar_mpoints_treinados(self):
        """Lista mpoints que têm modelo treinado"""
        if not self.dir_models.exists():
            return []
        
        mpoints = []
        for item in self.dir_models.iterdir():
            if item.is_dir():
                config_file = item / f'config_{item.name}.json'
                kmeans_file = item / f'kmeans_model_moderado_{item.name}.pkl'
                info_file = item / f'info_kmeans_model_moderado_{item.name}.json'
                
                if config_file.exists() and kmeans_file.exists() and info_file.exists():
                    mpoints.append(item.name)
        
        return sorted(mpoints)
    
    def executar_treino(self):
        """Executa treino do modelo (ELÉTRICO ou MECÂNICO)"""
        # Pegar seleção
        selection = self.mpoint_listbox.curselection()
        if not selection:
            messagebox.showwarning("Aviso", "Selecione um mpoint da lista")
            return

        # Extrair mpoint da linha selecionada (formato: "c_XXX [TIPO]")
        linha_selecionada = self.mpoint_listbox.get(selection[0]).strip()
        
        # Separar mpoint do tipo
        partes = linha_selecionada.split()
        if not partes or not partes[0].startswith('c_'):
            messagebox.showwarning("Aviso", "Selecione um mpoint válido da lista")
            return
        
        mpoint = partes[0]
        
        # Detectar tipo de equipamento
        mpoints_disponiveis = self.listar_mpoints_disponiveis()
        mpoint_info = next((mp for mp in mpoints_disponiveis if mp['mpoint'] == mpoint), None)
        
        if not mpoint_info:
            messagebox.showwarning("Aviso", "Informações do mpoint não encontradas")
            return
        
        tipo_equipamento = mpoint_info.get('tipo', 'DESCONHECIDO')
        
        # Mensagem de confirmação adaptada ao tipo
        if tipo_equipamento == 'MECANICO':
            mensagem_tipo = "MECÂNICO (temperatura + vibração)"
        elif tipo_equipamento == 'ELETRICO':
            mensagem_tipo = "ELÉTRICO (current + RPM + vibração)"
        else:
            mensagem_tipo = "DESCONHECIDO"
        
        # Confirmar
        resposta = messagebox.askyesno(
            "Confirmar Treino",
            f"Iniciar treino do modelo para {mpoint}?\n\n"
            f"Tipo: {mensagem_tipo}\n"
            f"Isso pode demorar 5-15 minutos.\n"
            f"Thresholds dinâmicos serão calculados automaticamente."
        )

        if not resposta:
            return

        # Limpar console da aba TREINO
        self.console_treino.delete('1.0', tk.END)
        self.console_treino.insert(tk.END, f"Iniciando treino do modelo para {mpoint}...\n")
        self.console_treino.insert(tk.END, f"Tipo de equipamento: {mensagem_tipo}\n")
        self.console_treino.insert(tk.END, "Isso pode demorar alguns minutos...\n\n")
        self.console_treino.see(tk.END)

        # Bloquear mudança de aba
        self.tarefa_em_execucao = True

        # Executar em thread separada com o tipo correto
        thread = threading.Thread(
            target=self.thread_treino,
            args=(mpoint, tipo_equipamento),
            daemon=True
        )
        thread.start()
    
    def executar_analise(self):
        """Executa análise por intervalo"""
        # Validar mpoint
        mpoint = self.mpoint_analise_combo.get()
        if not mpoint:
            messagebox.showwarning("Aviso", "Selecione um equipamento (mpoint)")
            return
        
        # Validar IP e porta
        ip = self.ip_entry.get().strip()
        porta = self.porta_entry.get().strip()
        
        if not ip:
            messagebox.showwarning("Aviso", "Digite o IP do InfluxDB")
            return
        
        # Validar datas
        try:
            data_inicio = self.data_inicio_cal.get_date().strftime('%Y-%m-%d')
            data_fim = self.data_fim_cal.get_date().strftime('%Y-%m-%d')
            
            hora_inicio = self.hora_inicio_entry.get().strip()
            hora_fim = self.hora_fim_entry.get().strip()
            
            datetime_inicio = f"{data_inicio} {hora_inicio}"
            datetime_fim = f"{data_fim} {hora_fim}"
            
            # Validar formato
            dt_inicio = datetime.strptime(datetime_inicio, '%Y-%m-%d %H:%M:%S')
            dt_fim = datetime.strptime(datetime_fim, '%Y-%m-%d %H:%M:%S')
            
            # Validar restrições de data
            data_minima = datetime(2025, 1, 1)
            data_maxima = datetime.now()
            
            if dt_inicio < data_minima or dt_fim < data_minima:
                messagebox.showerror("Erro", "As datas não podem ser anteriores a 01/01/2025")
                return
            
            if dt_inicio > data_maxima or dt_fim > data_maxima:
                messagebox.showerror("Erro", "As datas não podem ser futuras")
                return
            
            if dt_inicio >= dt_fim:
                messagebox.showerror("Erro", "A data inicial deve ser anterior à data final")
                return
            
        except ValueError as e:
            messagebox.showerror("Erro", f"Data/hora inválida:\n{e}")
            return
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao validar datas:\n{e}")
            return
        
        # Limpar console da aba ANALISE
        self.console_analise.delete('1.0', tk.END)
        self.console_analise.insert(tk.END, f"Iniciando analise para {mpoint}...\n")
        self.console_analise.insert(tk.END, f"   InfluxDB: {ip}:{porta}\n")
        self.console_analise.insert(tk.END, f"   Periodo: {datetime_inicio} ate {datetime_fim} (GMT-3)\n")
        self.console_analise.insert(tk.END, "   Carregando thresholds dinamicos do equipamento...\n\n")
        self.console_analise.see(tk.END)
        
        # Bloquear mudança de aba
        self.tarefa_em_execucao = True
        
        # Executar em thread separada
        thread = threading.Thread(
            target=self.thread_analise,
            args=(mpoint, ip, porta, datetime_inicio, datetime_fim),
            daemon=True
        )
        thread.start()
    
    def executar_visualizacao(self):
        """Executa visualização 3D"""
        mpoint = self.mpoint_viz_combo.get()
        if not mpoint:
            messagebox.showwarning("Aviso", "Selecione um equipamento (mpoint)")
            return
        
        # Limpar console da aba VISUALIZACAO
        self.console_viz.delete('1.0', tk.END)
        self.console_viz.insert(tk.END, f"Gerando visualizacao 3D para {mpoint}...\n")
        self.console_viz.insert(tk.END, "Isso pode levar alguns segundos...\n\n")
        self.console_viz.see(tk.END)
        
        # Bloquear mudança de aba
        self.tarefa_em_execucao = True
        
        # Executar em thread separada
        thread = threading.Thread(
            target=self.thread_visualizacao,
            args=(mpoint,),
            daemon=True
        )
        thread.start()
    
    def thread_treino(self, mpoint, tipo_equipamento='ELETRICO'):
        """Thread para executar treino sem travar a GUI (ELÉTRICO ou MECÂNICO)"""
        try:
            # Mensagem inicial NO CONSOLE TREINO
            self.root.after(0, lambda: self.atualizar_console_treino_seguro("=" * 80 + "\n"))
            self.root.after(0, lambda: self.atualizar_console_treino_seguro("INICIANDO TREINO DO MODELO\n"))
            self.root.after(0, lambda: self.atualizar_console_treino_seguro("=" * 80 + "\n"))
            self.root.after(0, lambda: self.atualizar_console_treino_seguro(f"Mpoint: {mpoint}\n"))

            # Escolher pipeline correto baseado no tipo de equipamento
            if tipo_equipamento == 'MECANICO':
                pipeline_script = 'pipeline_deteccao_estados_mecanico.py'
                self.root.after(0, lambda: self.atualizar_console_treino_seguro("Tipo: MECANICO (temperatura + vibracao)\n\n"))
            else:  # ELETRICO ou padrão
                pipeline_script = 'pipeline_deteccao_estados.py'
                self.root.after(0, lambda: self.atualizar_console_treino_seguro("Tipo: ELETRICO (current + RPM + vibracao)\n\n"))

            # Comando com output unbuffered e encoding correto
            cmd = [
                sys.executable, 
                '-u',  # unbuffered output (CRÍTICO!)
                str(self.base_dir / pipeline_script),
                '--mpoint', mpoint,
                '--modo', 'treino',
                '--auto'
            ]

            self.root.after(0, lambda: self.atualizar_console_treino_seguro(f"Comando: {' '.join(cmd)}\n\n"))

            # Criar ambiente com encoding UTF-8
            import os
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            env['PYTHONIOENCODING'] = 'utf-8'

            # Executar processo
            processo = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Redirecionar stderr para stdout (mais simples)
                text=True,
                encoding='utf-8',
                errors='replace',  # Substituir caracteres inválidos
                bufsize=1,  # Line buffered
                universal_newlines=True,
                cwd=str(self.base_dir),
                env=env
            )

            # Registrar processo filho
            self.processos_filhos.append(processo)

            # Ler output em tempo real
            linha_num = 0
            for linha in iter(processo.stdout.readline, ''):
                if linha:
                    linha_num += 1
                    # Adicionar quebra de linha se não tiver
                    if not linha.endswith('\n'):
                        linha += '\n'
                    
                    # Atualizar console TREINO COM quebra de linha
                    self.root.after(0, lambda l=linha: self.atualizar_console_treino_seguro(l))
                    
                    # Forçar atualização da GUI a cada 10 linhas
                    if linha_num % 10 == 0:
                        self.root.after(0, lambda: self.root.update_idletasks())

            # Aguardar processo terminar
            returncode = processo.wait()

            # Fechar stream
            processo.stdout.close()

            # Mensagem final
            self.root.after(0, lambda: self.atualizar_console_treino_seguro("\n" + "=" * 80 + "\n"))
            
            # Finalizar
            self.root.after(0, lambda rc=returncode: self.finalizar_treino(rc))

        except Exception as e:
            import traceback
            erro_completo = traceback.format_exc()
            self.root.after(0, lambda: self.atualizar_console_treino_seguro(f"\n\nERRO FATAL:\n{erro_completo}\n"))
            self.root.after(0, lambda: self.desbloquear_abas())


    def finalizar_treino(self, returncode):
        """Finaliza o processo de treino após garantir que todas as mensagens foram processadas"""
        if returncode == 0:
            self.root.after(0, lambda: self.atualizar_console_treino_seguro("\n" + "="*50 + "\n"))
            self.root.after(0, lambda: self.atualizar_console_treino_seguro("TREINO CONCLUÍDO COM SUCESSO!\n"))
            self.root.after(0, lambda: self.atualizar_console_treino_seguro("="*50 + "\n"))
            self.root.after(100, self.atualizar_combo_mpoints)
        else:
            self.root.after(0, lambda: self.atualizar_console_treino_seguro("\nERRO durante o treino!\n"))
        
        # Desbloquear mudança de aba
        self.root.after(0, lambda: self.desbloquear_abas())

    def thread_analise(self, mpoint, ip, porta, inicio, fim):
        """Thread para executar análise sem travar a GUI"""
        try:
            # Detectar tipo de equipamento
            tipo_equipamento = self.detectar_tipo_equipamento(mpoint)
            
            # Mensagem inicial
            self.root.after(0, lambda: self.atualizar_console_analise_seguro("=" * 80 + "\n"))
            self.root.after(0, lambda: self.atualizar_console_analise_seguro("INICIANDO ANALISE POR INTERVALO\n"))
            self.root.after(0, lambda: self.atualizar_console_analise_seguro("=" * 80 + "\n"))
            self.root.after(0, lambda t=tipo_equipamento: self.atualizar_console_analise_seguro(f"Tipo de equipamento: {t}\n"))
            self.root.after(0, lambda: self.atualizar_console_analise_seguro("=" * 80 + "\n"))
            
            # Criar ambiente com encoding UTF-8
            import os
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            env['PYTHONIOENCODING'] = 'utf-8'
            
            # Escolher script correto baseado no tipo
            if tipo_equipamento == 'MECANICO':
                script_analise = 'analise_intervalo_completa_mecanico.py'
            else:
                script_analise = 'analise_intervalo_completa.py'
            
            cmd = [
                sys.executable,
                '-u',  # unbuffered
                str(self.base_dir / 'scripts' / script_analise),
                '--mpoint', mpoint,
                '--influx-ip', ip,
                '--inicio', inicio,
                '--fim', fim
            ]

            processo = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                bufsize=1,
                universal_newlines=True,
                cwd=str(self.base_dir),
                env=env
            )

            # Registrar processo filho
            self.processos_filhos.append(processo)

            # Ler output em tempo real
            linha_num = 0
            for linha in iter(processo.stdout.readline, ''):
                if linha:
                    linha_num += 1
                    if not linha.endswith('\n'):
                        linha += '\n'
                    self.root.after(0, lambda l=linha: self.atualizar_console_analise_seguro(l))
                    if linha_num % 10 == 0:
                        self.root.after(0, lambda: self.root.update_idletasks())
            
            returncode = processo.wait()
            processo.stdout.close()
            
            self.root.after(0, lambda: self.atualizar_console_analise_seguro("\n" + "=" * 80 + "\n"))
            
            if returncode == 0:
                self.root.after(0, lambda: self.atualizar_console_analise_seguro("ANALISE CONCLUIDA COM SUCESSO!\n"))
                
                # Abrir arquivos gerados
                self.root.after(0, lambda: self.abrir_arquivos_analise(mpoint, tipo_equipamento))
                
                self.root.after(0, lambda: messagebox.showinfo("Sucesso", "Analise concluida!\nResultados salvos em: results/"))
            else:
                self.root.after(0, lambda: self.atualizar_console_analise_seguro("ERRO DURANTE A ANALISE\n"))
            
            # Desbloquear mudança de aba
            self.root.after(0, lambda: self.desbloquear_abas())

        except Exception as e:
            import traceback
            erro_completo = traceback.format_exc()
            self.root.after(0, lambda: self.atualizar_console_analise_seguro(f"\n\nERRO FATAL:\n{erro_completo}\n"))
            self.root.after(0, lambda: self.desbloquear_abas())
    
    def abrir_arquivos_analise(self, mpoint, tipo_equipamento):
        """Abre os arquivos .txt e gráfico 3D HTML gerados pela análise"""
        import os
        import webbrowser
        
        try:
            # Diretório de resultados
            results_dir = self.base_dir / 'results' / mpoint
            
            # Arquivo .txt principal de resultados
            arquivo_txt = results_dir / f'estados_intervalo_{mpoint}.txt'
            
            # Abrir arquivo .txt
            if arquivo_txt.exists():
                self.atualizar_console_analise_seguro(f"\nAbrindo arquivo de resultados: {arquivo_txt.name}\n")
                os.startfile(str(arquivo_txt))
            else:
                self.atualizar_console_analise_seguro(f"\n[AVISO] Arquivo .txt não encontrado: {arquivo_txt}\n")
            
            # Gráficos HTML não são mais gerados - visualização 3D agora é feita via matplotlib em janela separada
            
        except Exception as e:
            import traceback
            erro = traceback.format_exc()
            self.atualizar_console_analise_seguro(f"\n[ERRO] Falha ao abrir arquivos: {e}\n{erro}\n")
    
    def thread_visualizacao(self, mpoint):
        """Thread para executar visualização sem travar a GUI"""
        try:
            # Detectar tipo de equipamento
            tipo_equipamento = self.detectar_tipo_equipamento(mpoint)
            
            # Mensagem inicial
            self.root.after(0, lambda: self.atualizar_console_viz_seguro("=" * 80 + "\n"))
            self.root.after(0, lambda: self.atualizar_console_viz_seguro("GERANDO VISUALIZACAO 3D\n"))
            self.root.after(0, lambda: self.atualizar_console_viz_seguro("=" * 80 + "\n"))
            self.root.after(0, lambda t=tipo_equipamento: self.atualizar_console_viz_seguro(f"Tipo de equipamento: {t}\n"))
            self.root.after(0, lambda: self.atualizar_console_viz_seguro("=" * 80 + "\n"))
            
            # Criar ambiente com encoding UTF-8
            import os
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            env['PYTHONIOENCODING'] = 'utf-8'
            
            # Escolher script correto baseado no tipo
            if tipo_equipamento == 'MECANICO':
                script_viz = 'visualizar_clusters_3d_mecanico.py'
            else:
                script_viz = 'visualizar_clusters_3d_simples.py'
            
            cmd = [
                sys.executable,
                '-u',  # unbuffered
                str(self.base_dir / 'scripts' / script_viz),
                '--mpoint', mpoint
            ]

            processo = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                bufsize=1,
                universal_newlines=True,
                cwd=str(self.base_dir),
                env=env
            )

            # Registrar processo filho
            self.processos_filhos.append(processo)

            # Ler output em tempo real
            linha_num = 0
            for linha in iter(processo.stdout.readline, ''):
                if linha:
                    linha_num += 1
                    if not linha.endswith('\n'):
                        linha += '\n'
                    self.root.after(0, lambda l=linha: self.atualizar_console_viz_seguro(l))
                    if linha_num % 10 == 0:
                        self.root.after(0, lambda: self.root.update_idletasks())
            
            returncode = processo.wait()
            processo.stdout.close()
            
            self.root.after(0, lambda: self.atualizar_console_viz_seguro("\n" + "=" * 80 + "\n"))
            
            if returncode == 0:
                self.root.after(0, lambda: self.atualizar_console_viz_seguro("VISUALIZACAO CONCLUIDA!\n"))
                self.root.after(0, lambda: messagebox.showinfo("Sucesso", "Visualizacao 3D gerada!\nGraficos salvos em: results/"))
            else:
                self.root.after(0, lambda: self.atualizar_console_viz_seguro("ERRO DURANTE A VISUALIZACAO\n"))
            
            # Desbloquear mudança de aba
            self.root.after(0, lambda: self.desbloquear_abas())

        except Exception as e:
            import traceback
            erro_completo = traceback.format_exc()
            self.root.after(0, lambda: self.atualizar_console_viz_seguro(f"\n\nERRO FATAL:\n{erro_completo}\n"))
            self.root.after(0, lambda: self.desbloquear_abas())
    
    def criar_janela_progresso(self, titulo):
        """Cria janela de progresso"""
        # Fechar janela anterior se existir
        if hasattr(self, 'janela_progresso') and self.janela_progresso:
            try:
                self.janela_progresso.destroy()
            except:
                pass

        self.janela_progresso = ctk.CTkToplevel(self.root)
        self.janela_progresso.title(titulo)
        self.janela_progresso.geometry("600x400")
        self.janela_progresso.transient(self.root)
        self.janela_progresso.resizable(False, False)

        # Label
        ctk.CTkLabel(
            self.janela_progresso,
            text=titulo,
            font=("Arial", 16, "bold")
        ).pack(pady=20)

        # Console
        self.console_progresso = scrolledtext.ScrolledText(
            self.janela_progresso,
            font=("Courier", 9),
            bg="#1e1e1e",
            fg="yellow",
            height=15
        )
        self.console_progresso.pack(fill="both", expand=True, padx=20, pady=(0, 10))

        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(self.janela_progresso, mode="indeterminate")
        self.progress_bar.pack(pady=(0, 20), padx=20, fill="x")
        self.progress_bar.start()

        # Forçar atualização da interface
        self.janela_progresso.update()
    
    def atualizar_janela_progresso_seguro(self, texto):
        """Atualiza texto da janela de progresso de forma segura"""
        try:
            if hasattr(self, 'console_progresso') and self.console_progresso.winfo_exists():
                self.console_progresso.insert(tk.END, texto)
                self.console_progresso.see(tk.END)
                # Forçar atualização visual
                self.console_progresso.update_idletasks()
        except:
            pass  # Widget foi destruído, ignorar

    def atualizar_console_treino_seguro(self, texto):
        """Atualiza console de treino de forma segura"""
        try:
            if hasattr(self, 'console_treino') and self.console_treino.winfo_exists():
                self.console_treino.insert(tk.END, texto)
                self.console_treino.see(tk.END)
                # Forçar atualização visual
                self.console_treino.update_idletasks()
        except:
            pass  # Widget foi destruído, ignorar
    
    def atualizar_console_analise_seguro(self, texto):
        """Atualiza console de análise de forma segura"""
        try:
            if hasattr(self, 'console_analise') and self.console_analise.winfo_exists():
                self.console_analise.insert(tk.END, texto)
                self.console_analise.see(tk.END)
                # Forçar atualização visual
                self.console_analise.update_idletasks()
        except:
            pass  # Widget foi destruído, ignorar

    def atualizar_console_viz_seguro(self, texto):
        """Atualiza console de visualização de forma segura"""
        try:
            if hasattr(self, 'console_viz') and self.console_viz.winfo_exists():
                self.console_viz.insert(tk.END, texto)
                self.console_viz.see(tk.END)
                # Forçar atualização visual
                self.console_viz.update_idletasks()
        except:
            pass  # Widget foi destruído, ignorar

    def atualizar_janela_progresso(self, texto):
        """Atualiza texto da janela de progresso"""
        self.atualizar_janela_progresso_seguro(texto)
    
    def verificar_mudanca_aba(self):
        """Verifica se pode mudar de aba (bloquear se tarefa em execução)"""
        if self.tarefa_em_execucao:
            messagebox.showwarning(
                "Tarefa em Execução",
                "Aguarde a tarefa atual terminar antes de mudar de aba!"
            )
            return False
        return True
    
    def desbloquear_abas(self):
        """Desbloqueia mudança de abas após conclusão da tarefa"""
        self.tarefa_em_execucao = False
    
    def toggle_fullscreen(self):
        """Alterna entre modo fullscreen e janela normal"""
        current_state = self.root.attributes('-fullscreen')
        self.root.attributes('-fullscreen', not current_state)

    def fechar_aplicacao(self):
        """Fecha a aplicação e mata todos os processos filhos"""
        # Matar todos os processos filhos
        for processo in self.processos_filhos:
            try:
                if processo.poll() is None:  # Se ainda está rodando
                    processo.terminate()
                    # Esperar um pouco e forçar kill se necessário
                    try:
                        processo.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        processo.kill()
            except Exception:
                pass  # Ignorar erros ao matar processos

        # Fechar a aplicação
        self.root.quit()
        self.root.destroy()

    
    def run(self):
        """Inicia a aplicação"""
        self.root.mainloop()


def main():
    """Função principal"""
    # Iniciar GUI
    app = PipelineGUI()
    app.run()


if __name__ == "__main__":
    main()

