# Visualizador Web de Estados Operacionais

Interface web desenvolvida para possibilitar a visualização offline e interativa das séries temporais de sensores e a correspondente classificação dos estados operacionais (LIGADO/DESLIGADO) dos equipamentos industriais.

## Visão Geral

O visualizador funciona de forma 100% offline diretamente no navegador. Ele renderiza gráficos sincronizados no eixo temporal, permitindo correlacionar as variáveis físicas do equipamento com as decisões de classificação geradas pelo algoritmo K-Means.

### Recursos Principais
- **Gráficos Sincronizados**: Zoom e navegação integrados no tempo para todas as variáveis plotadas.
- **Diferenciação por Tipo de Equipamento**:
  - **Elétricos**: Apresenta gráficos de vibração, corrente, RPM e temperatura.
  - **Mecânicos**: Apresenta gráficos de vibração e temperatura.
- **Identificação Visual de Estados**:
  - **Verde**: Equipamento classificado como **LIGADO**.
  - **Vermelho**: Equipamento classificado como **DESLIGADO**.
- **Indicação de Confiabilidade dos Dados**:
  - **Círculo**: Dado real/original (confiabilidade de 1.00).
  - **Quadrado**: Dado interpolado via PCHIP (confiabilidade de 0.95).
  - **Losango**: Dado interpolado via KNN Temporal (confiabilidade de 0.75).
  - **Faixa Cinza**: Períodos sem dados (lacunas maiores que 3 horas, não interpoladas).

---

## Estrutura de Arquivos

A pasta do visualizador é estruturada da seguinte forma:

- **index.html**: Estrutura principal da página web.
- **style.css**: Estilização visual da interface.
- **app.js**: Lógica da aplicação web, incluindo carregamento de dados e renderização dos gráficos (utilizando Plotly).
- **iniciar_visualizador.py** e **iniciar_visualizador.bat**: Utilitários para inicializar o servidor web local e abrir o navegador automaticamente.
- **comprimir_dados.py**: Script para comprimir os dados em formato `.csv.gz`, reduzindo o volume de armazenamento para deploy.
- **lib/**: Contém a biblioteca Plotly local (`plotly.min.js`) para garantir o funcionamento sem internet.
- **dados/**: Diretório que contém os arquivos CSV correspondentes aos dados exportados dos equipamentos.
- **Dockerfile** e **default.conf**: Configurações para deploy automatizado via Docker (ex. Nginx com compressão estática).
- **DEPLOY.md**: Guia com instruções para implantação (deploy) em produção.
- **LEIA-ME.txt**: Instruções de uso rápido em formato de texto.

---

## Modo de Uso Local

### 1. Preparação dos Dados
Os dados a serem carregados pelo visualizador devem ser colocados em `visualizador/dados/`. 
Gere os arquivos necessários executando o treinamento geral dos modelos no diretório principal:
```bash
cd code
python treinar_todos.py
```

### 2. Inicialização do Visualizador
Para rodar o visualizador localmente no Windows, você pode dar um clique duplo no arquivo:
`iniciar_visualizador.bat`

Ou executar via terminal na pasta `visualizador`:
```bash
python iniciar_visualizador.py
```

O script irá:
1. Iniciar um servidor HTTP local.
2. Identificar e alocar automaticamente uma porta livre (padrão: `8000`).
3. Abrir automaticamente a URL `http://localhost:8000/` no navegador padrão.

> **Importante**: Mantenha o terminal/janela aberta enquanto estiver utilizando o visualizador. Para encerrar, pressione `Ctrl+C` no terminal.

---

## Deploy em Produção

Para instruções detalhadas de como realizar o deploy do visualizador em um servidor remoto (utilizando EasyPanel, Dockerfile ou volumes compartilhados), consulte o guia [DEPLOY.md](DEPLOY.md).
