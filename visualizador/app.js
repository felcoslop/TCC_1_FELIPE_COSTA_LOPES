/* ============================================================================
   Visualizador de Estados Operacionais - app offline (Plotly WebGL + LOD)
   Tela 1: escolha do equipamento.
   Tela 2: graficos sincronizados no eixo do tempo (X).
           Eletricos: 4 graficos (vibracao, corrente, RPM, temperatura)
           Mecanicos: 2 graficos (vibracao, temperatura)

   Cor do ponto   = estado    -> LIGADO verde / DESLIGADO vermelho
   Forma do ponto = origem    -> real circulo / PCHIP quadrado / KNN losango
   Faixas cinza   = lacunas sem dados (> 3h)

   DESEMPENHO: nivel de detalhe dinamico (LOD). Em vez de desenhar os ~1,4
   milhoes de pontos de uma vez, renderiza no maximo ~MAX_PTS pontos da janela
   visivel (dizimacao por min/max, que preserva picos). Ao dar zoom, recalcula
   apenas a janela; quando a janela tem poucos pontos, mostra TODOS (20 em 20s).
============================================================================ */

const COR_LIGADO = "#2ecc71";
const COR_DESLIGADO = "#e74c3c";
const COR_LACUNA = "#bdc3c7";

// forma: 0=real(circulo) 1=spline/PCHIP(quadrado) 2=KNN(losango)
const SIMBOLO = ["circle", "square", "diamond"];
const NOME_FORMA = ["real", "PCHIP", "KNN"];

const GAP_MS = 3 * 3600 * 1000; // 3 horas
const MAX_PTS = 3500;           // pontos maximos renderizados por atualizacao (SVG)

const META = {
    vib: { titulo: "Vibração RMS (mm/s)" },
    cur: { titulo: "Corrente (A)" },
    rpm: { titulo: "Rotação (RPM)" },
    temp: { titulo: "Temperatura (°C)" },
};

const EQUIP_INFO = {
    "c_636": {
        nome: "Motobomba SH 2",
        tipo: "Motor - Elétrico",
        descr: "Motobomba SH 2 – Motor",
        detalhes: "Motor 90 kW, 440 V, 60 Hz, 155 A, 6 polos, rolamentos 6316 / 6316, bomba A4VSO125DR/30R-PPB13N, 9 pistões. Localização: Motor SH 2 (PLTCM Ipatinga).",
        horas: { ligado: 3933.98, desligado: 195.34, sem_dados: 430.38, total: 4559.69 }
    },
    "c_637": {
        nome: "Motobomba SH 3",
        tipo: "Motor - Elétrico",
        descr: "Motobomba SH 3 – Motor",
        detalhes: "Motor 90 kW, 440 V, 60 Hz, 155 A, 6 polos, rolamentos 6316 / 6316, bomba A4VSO125DR/30R-PPB13N, 9 pistões. Localização: Motor SH 3 (PLTCM Ipatinga).",
        horas: { ligado: 5668.32, desligado: 268.26, sem_dados: 1093.58, total: 7030.16 }
    },
    "c_640": {
        nome: "SHAP Recirculação 1",
        tipo: "Motor REC 1 - Elétrico",
        descr: "SHAP Recirculação 1 – Motor REC 1 SHAP",
        detalhes: "Motobomba de recirculação da área SHAP (Motor REC 1 SHAP). Propriedades não cadastradas no modelo. Localização: Área SHAP - Oil Cellar - PLTCM Ipatinga.",
        horas: { ligado: 6259.16, desligado: 373.83, sem_dados: 446.40, total: 7079.38 }
    },
    "c_670": {
        nome: "Bridle 4 R1",
        tipo: "Redutora - Mecânico",
        descr: "Bridle 4 R1 – Redutora",
        detalhes: "Bridle 4 R1 – monitoramento da redutora. Modelo sem propriedades cadastradas. Localização: Área Centro - PLTCM Ipatinga.",
        horas: { ligado: 6047.99, desligado: 1606.76, sem_dados: 963.66, total: 8618.40 }
    },
    "c_1518": {
        nome: "CT-M1 Tambor 01 Descarga",
        tipo: "Mancal Direito - Mecânico",
        descr: "CT-M1 Tambor 01 Descarga – Mancal Direito",
        detalhes: "Tambor de descarga da correia transportadora CT-M1. Monitoramento do mancal lado direito do tambor. Modelo Conveyor Drum. Localização: CT-M1 - Alto Forno 3 - Ipatinga.",
        horas: { ligado: 3097.38, desligado: 154.63, sem_dados: 247.00, total: 3499.00 }
    }
};

const el = (id) => document.getElementById(id);

// Estado global do equipamento carregado
const G = {
    T: null, Y: null, ESTADO: null, FORMA: null, N: 0,
    charts: [], plotDiv: null, timer: null, viewT0: 0, viewT1: 0, appliedT0: 0, appliedT1: 0,
    ignoreUntil: 0, yFull: null,
};

const isoUTC = (ms) => new Date(ms).toISOString();

/* ----------------------------- Tela de selecao ---------------------------- */
async function iniciar() {
    try {
        const resp = await fetch("dados/manifest.json", { cache: "no-store" });
        if (!resp.ok) throw new Error("manifest.json não encontrado");
        const dados = await resp.json();
        renderCards(dados.equipamentos || []);
    } catch (e) {
        el("msg-selecao").textContent =
            "Não foi possível carregar a lista de equipamentos. " +
            "Gere os dados com 'python exportar_visualizador_web.py'. (" + e.message + ")";
    }
}

function renderCards(lista) {
    const cont = el("lista-equipamentos");
    cont.innerHTML = "";
    if (!lista.length) {
        el("msg-selecao").textContent = "Nenhum equipamento exportado ainda.";
        return;
    }
    el("msg-selecao").textContent = "Selecione um equipamento:";
    for (const eq of lista) {
        const pctOn = eq.pct_ligado != null ? eq.pct_ligado : 0;
        const card = document.createElement("div");
        card.className = "card";
        const info = EQUIP_INFO[eq.mpoint] || { nome: eq.mpoint, tipo: eq.tipo, descr: eq.mpoint, detalhes: "", horas: { ligado: 0, desligado: 0, sem_dados: 0 } };
        
        // Define ícone com base no tipo (elétrico = raio, mecânico = engrenagem)
        const tagIcon = eq.tipo === "eletrico"
            ? `<i data-lucide="zap"></i>`
            : `<i data-lucide="settings"></i>`;

        card.innerHTML = `
            <span class="tag ${eq.tipo}">${tagIcon}${info.tipo}</span>
            <h3>${info.nome}</h3>
            <p class="card-mpoint">${eq.mpoint}</p>
            <div class="info">
                ${Number(eq.n_pontos).toLocaleString("pt-BR")} pontos &middot; ${eq.graficos.length} gráficos<br>
                <span style="color:#2ecc71;font-weight:600">LIGADO:</span> ${pctOn}% (${info.horas.ligado.toFixed(0)}h)<br>
                <span style="color:#e74c3c;font-weight:600">DESLIGADO:</span> ${(100 - pctOn).toFixed(1)}% (${info.horas.desligado.toFixed(0)}h)<br>
                <span style="color:#7f8c8d;font-weight:600">SEM CONEXÃO:</span> ${info.horas.sem_dados.toFixed(0)}h
            </div>
            <div class="bar"><span style="width:${pctOn}%"></span></div>`;
        card.onclick = () => abrirEquipamento(eq);
        cont.appendChild(card);
    }
    
    // Inicializa todos os ícones Lucide no documento (tanto os botões quanto as tags)
    if (window.lucide) {
        lucide.createIcons();
    }
}

/* --------------------------- Carregar e desenhar --------------------------- */
async function abrirEquipamento(eq) {
    el("tela-selecao").classList.add("hidden");
    el("tela-graficos").classList.remove("hidden");
    
    const info = EQUIP_INFO[eq.mpoint] || { nome: eq.mpoint, tipo: eq.tipo, descr: eq.mpoint, detalhes: "", horas: { ligado: 0, desligado: 0, sem_dados: 0, total: 0 } };
    el("titulo-equip").textContent = `${info.nome} (${eq.mpoint})`;
    
    // Popula a legenda com as horas
    el("legenda-horas-ligado").textContent = info.horas.ligado.toFixed(0);
    el("legenda-horas-desligado").textContent = info.horas.desligado.toFixed(0);
    el("legenda-horas-semconexao").textContent = info.horas.sem_dados.toFixed(0);

    // Configura o clique do botão de informações
    el("btn-info").onclick = () => {
        el("info-nome").textContent = info.nome;
        el("info-mpoint").textContent = eq.mpoint;
        el("info-tipo").textContent = info.tipo;
        el("info-descr").textContent = info.descr;
        el("info-detalhes").textContent = info.detalhes;
        el("modal-info").classList.remove("hidden");
    };

    // Configura o clique do botão de horas totais
    el("btn-horas").onclick = () => {
        el("horas-equip-nome").textContent = info.nome;
        el("horas-equip-mpoint").textContent = eq.mpoint;
        
        const h = info.horas;
        const total = h.total || (h.ligado + h.desligado + h.sem_dados);
        el("horas-periodo-total").textContent = `${total.toFixed(0)} horas`;
        
        el("horas-val-ligado").textContent = `${h.ligado.toFixed(0)}h (${(h.ligado/total*100).toFixed(1)}%)`;
        el("horas-val-desligado").textContent = `${h.desligado.toFixed(0)}h (${(h.desligado/total*100).toFixed(1)}%)`;
        el("horas-val-semconexao").textContent = `${h.sem_dados.toFixed(0)}h (${(h.sem_dados/total*100).toFixed(1)}%)`;
        
        el("horas-bar-ligado").style.width = `${(h.ligado/total*100)}%`;
        el("horas-bar-desligado").style.width = `${(h.desligado/total*100)}%`;
        el("horas-bar-semconexao").style.width = `${(h.sem_dados/total*100)}%`;
        
        el("modal-horas").classList.remove("hidden");
    };

    const msg = el("msg-graficos");
    msg.classList.remove("hidden");
    msg.textContent = "Carregando dados…";
    Plotly.purge("plot");

    try {
        const resp = await fetch(eq.arquivo, { cache: "no-store" });
        if (!resp.ok) throw new Error(eq.arquivo + " não encontrado");
        const texto = await resp.text();
        msg.textContent = "Processando pontos…";
        await new Promise((r) => setTimeout(r, 30));
        prepararDados(eq, texto);
        desenharInicial(eq);
        msg.classList.add("hidden");
    } catch (e) {
        msg.textContent = "Erro ao carregar: " + e.message;
    }
}

function parseCSV(texto) {
    const fimCab = texto.indexOf("\n");
    const cab = texto.slice(0, fimCab).trim().split(",");
    const corpo = texto.slice(fimCab + 1);
    const linhas = corpo.split("\n");
    let n = linhas.length;
    while (n > 0 && linhas[n - 1].trim() === "") n--;

    const col = {};
    for (const c of cab) col[c] = new Float64Array(n);
    for (let i = 0; i < n; i++) {
        const p = linhas[i].split(",");
        for (let j = 0; j < cab.length; j++) {
            const v = p[j];
            col[cab[j]][i] = v === "" || v === undefined ? NaN : +v;
        }
    }
    return { col, n };
}

function prepararDados(eq, texto) {
    const { col, n } = parseCSV(texto);
    G.N = n;
    G.T = col.t;
    G.Y = {};
    for (const k of eq.graficos) G.Y[k] = col[k];
    G.ESTADO = new Int8Array(n);
    G.FORMA = new Int8Array(n);
    for (let i = 0; i < n; i++) { G.ESTADO[i] = col.estado[i]; G.FORMA[i] = col.forma[i]; }
    G.charts = eq.graficos.slice();

    // Prefixo de lacunas REAIS (>3h) para quebrar a linha apenas nelas
    // (e nao entre pontos distantes por causa da dizimacao).
    G.gapPrefix = new Int32Array(n);
    for (let i = 1; i < n; i++) {
        G.gapPrefix[i] = G.gapPrefix[i - 1] + (G.T[i] - G.T[i - 1] > GAP_MS ? 1 : 0);
    }

    // Faixa Y do periodo completo (pre-calculada para o reset ser instantaneo).
    G.yFull = {};
    for (const k of G.charts) G.yFull[k] = faixaY(0, n, k);
}

/* --------- LOD: dizimacao por min/max preservando picos e transicoes -------- */
function indicesJanela(i0, i1, yArr) {
    const span = i1 - i0;
    if (span <= MAX_PTS) {
        const idx = new Array(span);
        for (let k = 0; k < span; k++) idx[k] = i0 + k;
        return idx;
    }
    const buckets = Math.floor(MAX_PTS / 2);
    const step = span / buckets;
    const idx = [];
    for (let b = 0; b < buckets; b++) {
        const s = i0 + Math.floor(b * step);
        const e = i0 + Math.floor((b + 1) * step);
        let mn = -1, mx = -1, vmn = Infinity, vmx = -Infinity;
        for (let i = s; i < e; i++) {
            const v = yArr[i];
            if (v !== v) continue; // NaN
            if (v < vmn) { vmn = v; mn = i; }
            if (v > vmx) { vmx = v; mx = i; }
        }
        if (mn < 0) { idx.push(s); continue; }
        if (mn <= mx) { idx.push(mn); if (mx !== mn) idx.push(mx); }
        else { idx.push(mx); idx.push(mn); }
    }
    return idx;
}

function buscaInf(T, alvo) {
    let lo = 0, hi = T.length;
    while (lo < hi) { const m = (lo + hi) >> 1; if (T[m] < alvo) lo = m + 1; else hi = m; }
    return lo;
}

function montarSerie(idx) {
    // Constroi as series ja com a linha conectora; insere um ponto nulo nas
    // lacunas > 3h para a linha NAO atravessar o vazio (quebra a serie ali).
    const x = [];
    const cores = [];
    const simbolos = [];
    const ys = G.charts.map(() => []);
    for (let k = 0; k < idx.length; k++) {
        const i = idx[k];
        x.push(isoUTC(G.T[i]));   // data ISO -> Plotly trata o eixo como DATA (e nao numero)
        cores.push(G.ESTADO[i] === 1 ? COR_LIGADO : COR_DESLIGADO);
        simbolos.push(SIMBOLO[G.FORMA[i]]);
        for (let c = 0; c < G.charts.length; c++) ys[c].push(G.Y[G.charts[c]][i]);

        if (k < idx.length - 1 && G.gapPrefix[idx[k + 1]] - G.gapPrefix[i] > 0) {
            // ponto de quebra: x VALIDO (meio da lacuna) e y nulo -> quebra a linha,
            // nao desenha marcador e nao gera hover com data NaN.
            x.push(isoUTC((G.T[i] + G.T[idx[k + 1]]) / 2));
            cores.push(COR_LACUNA);
            simbolos.push("circle");
            for (let c = 0; c < G.charts.length; c++) ys[c].push(null);
        }
    }
    return { x, cores, simbolos, ys };
}

// Faixa [min,max] de uma coluna na janela [i0,i1), com folga de 5%.
function faixaY(i0, i1, key) {
    const src = G.Y[key];
    let mn = Infinity, mx = -Infinity;
    for (let i = i0; i < i1; i++) {
        const v = src[i];
        if (v !== v) continue;
        if (v < mn) mn = v;
        if (v > mx) mx = v;
    }
    if (mn === Infinity) return null;
    if (mn === mx) { mn -= 1; mx += 1; }
    const pad = (mx - mn) * 0.05;
    return [mn - pad, mx + pad];
}

function desenharInicial(eq) {
    const div = el("plot");
    G.plotDiv = div;

    const t0 = G.T[0], t1 = G.T[G.N - 1];
    const idx = indicesJanela(0, G.N, G.Y[G.charts[0]]);
    const s = montarSerie(idx);

    const traces = G.charts.map((key, ci) => ({
        type: "scatter",
        mode: "lines+markers",
        x: s.x,
        y: s.ys[ci],
        xaxis: "x",
        yaxis: ci === 0 ? "y" : "y" + (ci + 1),
        connectgaps: false,
        line: { color: "rgba(120,130,140,0.45)", width: 1 },
        marker: { size: 4, color: s.cores, symbol: s.simbolos, line: { width: 0 } },
        showlegend: false,
        hovertemplate: `%{x}<br>${META[key].titulo}: %{y}<extra></extra>`,
    }));

    const layout = montarLayout(G.charts);
    layout.xaxis.autorange = true;        // Plotly ajusta o X aos dados (data ISO) -> eixo de DATA
    G.charts.forEach((key, ci) => {       // Y fixo no periodo completo (pontos nao somem)
        const r = G.yFull[key];
        if (r) layout[ci === 0 ? "yaxis" : "yaxis" + (ci + 1)].range = r;
    });

    Plotly.newPlot(div, traces, layout, {
        responsive: true,
        scrollZoom: true,
        displaylogo: false,
        modeBarButtonsToRemove: ["lasso2d", "select2d"],
    });

    div.removeAllListeners && div.removeAllListeners("plotly_relayout");
    div.on("plotly_relayout", aoRelayout);
    atualizarDataTopo(t0, t1);
}

function montarLayout(charts) {
    const k = charts.length;
    const gap = 0.07;
    const h = (1 - (k - 1) * gap) / k;

    const layout = {
        margin: { l: 70, r: 20, t: 12, b: 46 },
        hovermode: "closest",
        dragmode: "zoom",
        uirevision: "fixa",
        plot_bgcolor: "#ffffff",
        paper_bgcolor: "#ffffff",
        shapes: faixasLacuna(),
    };
    charts.forEach((key, i) => {
        const top = 1 - i * (h + gap);
        const bot = top - h;
        const id = i === 0 ? "yaxis" : "yaxis" + (i + 1);
        layout[id] = {
            domain: [Math.max(0, bot), top],
            title: { text: META[key].titulo, font: { size: 12 } },
            zeroline: false,
            gridcolor: "#eef1f4",
            fixedrange: true,   // Y controlado pelo app (ajusta a janela); zoom so no X
            autorange: false,
        };
    });
    layout.xaxis = {
        type: "date",
        anchor: "y" + (k === 1 ? "" : k),
        domain: [0, 1],
        showgrid: true,
        gridcolor: "#dfe6ee",
        ticks: "outside",
        // formatacao automatica do Plotly (data quando afastado, hora quando proximo).
        // A data explicita fica no cabecalho do topo (#data-topo).
    };
    return layout;
}

// Espacamento das linhas de grade (em ms) conforme o tamanho da janela visivel.
// Quanto mais zoom, mais finas: chega em linhas de hora em hora (e menores).
function escolherDtick(span) {
    const M = 60000, H = 3600000, D = 86400000;
    if (span > 60 * D) return null;     // auto (meses)
    if (span > 12 * D) return 2 * D;
    if (span > 3 * D) return D;         // dia em dia
    if (span > 30 * H) return 6 * H;
    if (span > 10 * H) return 3 * H;
    if (span > 3 * H) return H;         // hora em hora
    if (span > 80 * M) return 30 * M;
    if (span > 25 * M) return 10 * M;
    if (span > 6 * M) return 2 * M;
    return null;                        // auto (segundos)
}

function fmtDataUTC(ms) {
    const d = new Date(ms);
    const p = (n) => String(n).padStart(2, "0");
    return `${p(d.getUTCDate())}/${p(d.getUTCMonth() + 1)}/${d.getUTCFullYear()}`;
}

// Cabecalho de data: 1 data se a janela esta no mesmo dia; 2 datas se cruza dias.
function atualizarDataTopo(t0, t1) {
    const dia0 = Math.floor(t0 / 86400000);
    const dia1 = Math.floor(t1 / 86400000);
    const texto = dia0 === dia1
        ? fmtDataUTC(t0)
        : `${fmtDataUTC(t0)} — ${fmtDataUTC(t1)}`;
    el("data-topo").textContent = texto + "  (UTC)";
}

function faixasLacuna() {
    const shapes = [];
    const T = G.T, n = G.N;
    for (let i = 0; i < n - 1; i++) {
        if (T[i + 1] - T[i] > GAP_MS) {
            shapes.push({
                type: "rect", xref: "x", yref: "paper",
                // coordenadas em data ISO (mesmo sistema do eixo) -> acompanham o zoom
                x0: isoUTC(T[i]), x1: isoUTC(T[i + 1]), y0: 0, y1: 1,
                fillcolor: COR_LACUNA, opacity: 0.35, line: { width: 0 }, layer: "below",
            });
        }
    }
    return shapes;
}

/* --------------------------- Zoom -> recalcula LOD ------------------------- */
function rangeToMs(v) {
    if (v == null) return null;
    if (typeof v === "number") return v;
    const s = String(v).trim();
    if (/^-?\d+(\.\d+)?$/.test(s)) return parseFloat(s); // numero em string (ms)
    // formato Plotly de data: "YYYY-MM-DD HH:MM:SS.sss"
    let iso = s.replace(" ", "T");
    if (iso.indexOf("T") < 0) iso += "T00:00:00";
    if (!/[Zz]$|[+\-]\d\d:?\d\d$/.test(iso)) iso += "Z"; // assume UTC (eixo do Plotly e UTC)
    let ms = Date.parse(iso);
    if (!isNaN(ms)) return ms;
    ms = Date.parse(s);
    return isNaN(ms) ? null : ms;
}

function aoRelayout(ev) {
    let t0, t1;
    if (ev["xaxis.autorange"]) { t0 = G.T[0]; t1 = G.T[G.N - 1]; }
    else if (ev["xaxis.range[0]"] !== undefined) { t0 = rangeToMs(ev["xaxis.range[0]"]); t1 = rangeToMs(ev["xaxis.range[1]"]); }
    else if (ev["xaxis.range"]) { t0 = rangeToMs(ev["xaxis.range"][0]); t1 = rangeToMs(ev["xaxis.range"][1]); }
    else return; // evento sem mudanca no eixo do tempo -> ignora
    if (t0 == null || t1 == null) return;

    clearTimeout(G.timer);
    G.timer = setTimeout(() => restyleJanela(t0, t1), 40);
}

// Atualiza SO os dados (Plotly.restyle nao mexe no layout -> nao dispara relayout,
// nao reseta o zoom, nao troca o tipo do eixo, nao fica branco). O range do X fica
// por conta do Plotly (o zoom nativo do usuario); o Y fica fixo no periodo completo.
function restyleJanela(t0, t1) {
    if (t1 < t0) { const tmp = t0; t0 = t1; t1 = tmp; }
    t0 = Math.max(t0, G.T[0]);
    t1 = Math.min(t1, G.T[G.N - 1]);
    if (!(t1 > t0)) return;

    let i0 = buscaInf(G.T, t0);
    let i1 = buscaInf(G.T, t1);
    i0 = Math.max(0, i0 - 1);
    i1 = Math.min(G.N, i1 + 1);
    if (i1 <= i0) return;

    const idx = indicesJanela(i0, i1, G.Y[G.charts[0]]);
    const s = montarSerie(idx);
    const dataUpdate = {
        x: G.charts.map(() => s.x),
        y: s.ys,
        "marker.color": G.charts.map(() => s.cores),
        "marker.symbol": G.charts.map(() => s.simbolos),
    };
    try {
        Plotly.restyle(G.plotDiv, dataUpdate, G.charts.map((_, i) => i));
    } catch (e) {
        console.error("[Visualizador] falha ao redesenhar:", e);
    }
    atualizarDataTopo(t0, t1);
}

el("btn-voltar").onclick = () => {
    clearTimeout(G.timer);
    Plotly.purge("plot");
    el("tela-graficos").classList.add("hidden");
    el("tela-selecao").classList.remove("hidden");
};

el("btn-reset").onclick = () => {
    if (!G.plotDiv) return;
    clearTimeout(G.timer);
    restyleJanela(G.T[0], G.T[G.N - 1]);                       // carrega dados do periodo todo
    Plotly.relayout(G.plotDiv, { "xaxis.autorange": true });   // e ajusta o X ao periodo todo (nativo)
};

// Controles do modal de informações
el("btn-fechar-modal").onclick = () => {
    el("modal-info").classList.add("hidden");
};

// Controles do modal de horas
el("btn-fechar-modal-horas").onclick = () => {
    el("modal-horas").classList.add("hidden");
};

window.addEventListener("click", (event) => {
    const modalInfo = el("modal-info");
    if (event.target === modalInfo) {
        modalInfo.classList.add("hidden");
    }
    const modalHoras = el("modal-horas");
    if (event.target === modalHoras) {
        modalHoras.classList.add("hidden");
    }
});

// Marcador de versao: confira no console (F12) e no rodape se carregou o codigo novo.
const BUILD = "2026-06-04-m3";
console.log("%c[Visualizador] build " + BUILD, "color:#2980b9;font-weight:bold");
window.addEventListener("DOMContentLoaded", () => {
    const d = document.querySelector(".dica");
    if (d) d.insertAdjacentText("beforeend", "  ·  build " + BUILD);
});

iniciar();
