"""
Protein-Conservation v2
========================
App Streamlit para análisis de conservación de secuencias proteicas.

Algoritmos implementados:
  1. Jensen-Shannon Divergence (JSD)       — Capra & Singh 2007
  2. Shannon Entropy                        — Shannon 1948
  3. Property Entropy (Williamson)          — Taylor 1986 groups
  4. Kullback-Leibler (Relative Entropy)    — Wang & Samudrala 2006
  5. Valdar01 (Sum-of-Pairs BLOSUM62)       — Valdar & Thornton 2001
  6. SMERFS (sliding window JSD, w=7)       — Manning et al. 2008

Nuevas features v2:
  - Logos paginados (50 pos/página) con alturas proporcionales a la entropía
  - Logo de frecuencias + logo de conservación (bits)
  - Visor 3D py3Dmol embebido + scripts PyMOL/ChimeraX
  - Estructura secundaria desde DSSP (si hay PDB) con fallback a predicción
  - Mapeo doble: numeración MSA ↔ numeración PDB
  - Formulita + descripción colapsable por algoritmo
"""

import streamlit as st
import math, io, re
import numpy as np
import pandas as pd
from Bio import SeqIO
from collections import Counter
import streamlit.components.v1 as components

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Protein-Conservation",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600;700&family=Lora:wght@400;500;600&display=swap');

:root {
    --bg:       #0b0f14;
    --s1:       #131820;
    --s2:       #1c2330;
    --border:   #2a3444;
    --accent:   #4fc3f7;
    --green:    #56c596;
    --orange:   #f5a623;
    --red:      #f07070;
    --purple:   #b39ddb;
    --text:     #d8e4f0;
    --muted:    #657a8e;
    --mono:     'IBM Plex Mono', monospace;
    --serif:    'Lora', serif;
}

html, body, [class*="css"] {
    font-family: var(--mono);
    background-color: var(--bg);
    color: var(--text);
}

/* ── sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--s1);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }
section[data-testid="stSidebar"] label { font-size: 0.78rem !important; }

/* ── header ── */
.pc-header {
    padding: 16px 0 20px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 24px;
}
.pc-title {
    font-family: var(--serif);
    font-size: 2rem;
    font-weight: 600;
    color: var(--accent);
    letter-spacing: 0.02em;
    margin: 0;
}
.pc-sub {
    font-size: 0.73rem;
    color: var(--muted);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin: 4px 0 0;
}

/* ── section headers ── */
.sh {
    font-size: 0.68rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    border-bottom: 1px solid var(--border);
    padding-bottom: 5px;
    margin-bottom: 12px;
    margin-top: 20px;
}

/* ── metric cards ── */
.mcards { display:flex; gap:10px; margin-bottom:20px; flex-wrap:wrap; }
.mcard {
    background: var(--s1);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 12px 16px;
    min-width: 130px;
    flex: 1;
}
.mcard-label { font-size:0.65rem; color:var(--muted); text-transform:uppercase; letter-spacing:.1em; }
.mcard-value { font-size:1.3rem; font-weight:700; color:var(--accent); margin-top:3px; }

/* ── algo card ── */
.algo-card {
    background: var(--s1);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 12px 16px;
    margin-bottom: 8px;
    font-size: 0.8rem;
}
.algo-name { font-weight:700; color:var(--accent); font-size:0.85rem; }
.algo-ref  { color:var(--muted); font-size:0.65rem; }
.algo-formula {
    background: var(--s2);
    border-left: 3px solid var(--accent);
    border-radius: 4px;
    padding: 6px 10px;
    margin: 8px 0 6px;
    font-size: 0.75rem;
    color: var(--orange);
    white-space: pre;
}
.algo-desc { color: var(--text); font-size:0.75rem; line-height:1.5; }

/* ── info box ── */
.info-box {
    background: var(--s1);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 6px;
    padding: 12px 16px;
    font-size: 0.8rem;
    color: var(--muted);
    margin-bottom: 16px;
    line-height: 1.6;
}
.info-box b { color: var(--text); }

/* ── badge ── */
.badge {
    display:inline-block;
    background:var(--s2);
    border:1px solid var(--border);
    border-radius:4px;
    padding:2px 8px;
    font-size:0.68rem;
    color:var(--accent);
    margin-bottom:8px;
}

/* ── buttons ── */
.stButton > button {
    background: var(--accent) !important;
    color: #0b0f14 !important;
    font-family: var(--mono) !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 5px !important;
    font-size: 0.78rem !important;
    letter-spacing: .05em;
}
.stButton > button:hover { background: #81d4fa !important; }

/* ── tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 3px;
    background: var(--s1);
    border-radius: 7px;
    padding: 4px;
    border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    font-family: var(--mono) !important;
    font-size: 0.72rem !important;
    border-radius: 5px !important;
    color: var(--muted) !important;
}
.stTabs [aria-selected="true"] {
    background: var(--s2) !important;
    color: var(--accent) !important;
}

/* ── misc overrides ── */
.stSelectbox > div > div { background: var(--s2) !important; border-color: var(--border) !important; }
div[data-testid="stFileUploader"] { border: 1px dashed var(--border) !important; background: var(--s1) !important; }
hr { border-color: var(--border) !important; }
.stDataFrame { border: 1px solid var(--border) !important; border-radius: 6px !important; overflow: hidden; }

/* ── secondary structure bar ── */
.ss-bar { display:flex; height:14px; border-radius:3px; overflow:hidden; margin:6px 0 2px; }
.ss-helix  { background:#4fc3f7; }
.ss-sheet  { background:#56c596; }
.ss-loop   { background:#2a3444; }
.ss-legend { display:flex; gap:14px; font-size:0.65rem; color:var(--muted); margin-top:4px; }
.ss-dot { width:10px; height:10px; border-radius:2px; display:inline-block; margin-right:4px; vertical-align:middle; }

/* ── pdb info ── */
.pdb-note {
    background: var(--s1);
    border: 1px solid var(--border);
    border-left: 3px solid var(--green);
    border-radius: 6px;
    padding: 12px 16px;
    font-size: 0.76rem;
    color: var(--muted);
}
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
AMINO_ACIDS = list("ARNDCQEGHILKMFPSTWYV")
AA_TO_IDX   = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

BLOSUM_BG = [0.078,0.051,0.041,0.052,0.024,0.034,0.059,0.083,0.025,0.062,
             0.092,0.056,0.024,0.044,0.043,0.059,0.055,0.014,0.034,0.072]

# BLOSUM62 upper-triangle (20×20) — valores de la matriz para Valdar/SoP
BLOSUM62 = {
    ('A','A'):4,('A','R'):-1,('A','N'):-2,('A','D'):-2,('A','C'):0,
    ('A','Q'):-1,('A','E'):-1,('A','G'):0,('A','H'):-2,('A','I'):-1,
    ('A','L'):-1,('A','K'):-1,('A','M'):-1,('A','F'):-2,('A','P'):-1,
    ('A','S'):1,('A','T'):0,('A','W'):-3,('A','Y'):-2,('A','V'):0,
    ('R','R'):5,('R','N'):-1,('R','D'):-2,('R','C'):-3,('R','Q'):1,
    ('R','E'):0,('R','G'):-2,('R','H'):0,('R','I'):-3,('R','L'):-2,
    ('R','K'):2,('R','M'):-1,('R','F'):-3,('R','P'):-2,('R','S'):-1,
    ('R','T'):-1,('R','W'):-3,('R','Y'):-2,('R','V'):-3,
    ('N','N'):6,('N','D'):1,('N','C'):-3,('N','Q'):0,('N','E'):0,
    ('N','G'):0,('N','H'):1,('N','I'):-3,('N','L'):-3,('N','K'):0,
    ('N','M'):-2,('N','F'):-3,('N','P'):-2,('N','S'):1,('N','T'):0,
    ('N','W'):-4,('N','Y'):-2,('N','V'):-3,
    ('D','D'):6,('D','C'):-3,('D','Q'):0,('D','E'):2,('D','G'):-1,
    ('D','H'):-1,('D','I'):-3,('D','L'):-4,('D','K'):-1,('D','M'):-3,
    ('D','F'):-3,('D','P'):-1,('D','S'):0,('D','T'):-1,('D','W'):-4,
    ('D','Y'):-3,('D','V'):-3,
    ('C','C'):9,('C','Q'):-3,('C','E'):-4,('C','G'):-3,('C','H'):-3,
    ('C','I'):-1,('C','L'):-1,('C','K'):-3,('C','M'):-1,('C','F'):-2,
    ('C','P'):-3,('C','S'):-1,('C','T'):-1,('C','W'):-2,('C','Y'):-2,
    ('C','V'):-1,
    ('Q','Q'):5,('Q','E'):2,('Q','G'):-2,('Q','H'):0,('Q','I'):-3,
    ('Q','L'):-2,('Q','K'):1,('Q','M'):0,('Q','F'):-3,('Q','P'):-1,
    ('Q','S'):0,('Q','T'):-1,('Q','W'):-2,('Q','Y'):-1,('Q','V'):-2,
    ('E','E'):5,('E','G'):-2,('E','H'):0,('E','I'):-3,('E','L'):-3,
    ('E','K'):1,('E','M'):-2,('E','F'):-3,('E','P'):-1,('E','S'):0,
    ('E','T'):-1,('E','W'):-3,('E','Y'):-2,('E','V'):-2,
    ('G','G'):6,('G','H'):-2,('G','I'):-4,('G','L'):-4,('G','K'):-2,
    ('G','M'):-3,('G','F'):-3,('G','P'):-2,('G','S'):0,('G','T'):-2,
    ('G','W'):-2,('G','Y'):-3,('G','V'):-3,
    ('H','H'):8,('H','I'):-3,('H','L'):-3,('H','K'):-1,('H','M'):-2,
    ('H','F'):-1,('H','P'):-2,('H','S'):-1,('H','T'):-2,('H','W'):-2,
    ('H','Y'):2,('H','V'):-3,
    ('I','I'):4,('I','L'):2,('I','K'):-1,('I','M'):1,('I','F'):0,
    ('I','P'):-3,('I','S'):-2,('I','T'):-1,('I','W'):-3,('I','Y'):-1,
    ('I','V'):3,
    ('L','L'):4,('L','K'):-2,('L','M'):2,('L','F'):0,('L','P'):-3,
    ('L','S'):-2,('L','T'):-1,('L','W'):-2,('L','Y'):-1,('L','V'):1,
    ('K','K'):5,('K','M'):-1,('K','F'):-3,('K','P'):-1,('K','S'):0,
    ('K','T'):-1,('K','W'):-3,('K','Y'):-2,('K','V'):-2,
    ('M','M'):5,('M','F'):0,('M','P'):-2,('M','S'):-1,('M','T'):-1,
    ('M','W'):-1,('M','Y'):-1,('M','V'):1,
    ('F','F'):6,('F','P'):-4,('F','S'):-2,('F','T'):-2,('F','W'):1,
    ('F','Y'):3,('F','V'):-1,
    ('P','P'):7,('P','S'):-1,('P','T'):-1,('P','W'):-4,('P','Y'):-3,
    ('P','V'):-2,
    ('S','S'):4,('S','T'):1,('S','W'):-3,('S','Y'):-2,('S','V'):-2,
    ('T','T'):5,('T','W'):-2,('T','Y'):-2,('T','V'):0,
    ('W','W'):11,('W','Y'):2,('W','V'):-3,
    ('Y','Y'):7,('Y','V'):-1,
    ('V','V'):4,
}

def blosum62(a, b):
    key = (a,b) if (a,b) in BLOSUM62 else (b,a)
    return BLOSUM62.get(key, -4)

AA_COLORS = {
    'K':'#4e9af1','R':'#4e9af1','H':'#4e9af1',
    'D':'#f07070','E':'#f07070',
    'C':'#f5a623','M':'#f5a623',
    'G':'#657a8e','P':'#657a8e',
    'A':'#4fc3f7','V':'#4fc3f7','L':'#4fc3f7',
    'I':'#4fc3f7','F':'#b39ddb','W':'#b39ddb','Y':'#b39ddb',
    'S':'#56c596','T':'#56c596','N':'#56c596','Q':'#56c596',
}

# ─── PIPELINE CORE ────────────────────────────────────────────────────────────

def calc_weights(msa):
    """Henikoff & Henikoff 1994 position-based sequence weighting."""
    n, L = len(msa), len(msa[0])
    w = [0.0]*n
    for i in range(L):
        col = [msa[j][i] for j in range(n)]
        cnt = Counter(col)
        r = len([c for c in cnt if c in AA_TO_IDX or c=='-'])
        for j in range(n):
            aa = msa[j][i]
            if aa in cnt and cnt[aa]*r > 0:
                w[j] += 1.0/(cnt[aa]*r)
    total = sum(w) or 1
    norm = [x/L for x in w]
    return norm

def gap_penalty(col, w):
    gs = sum(w[i] for i,aa in enumerate(col) if aa=='-')
    return 1-(gs/sum(w)) if sum(w)>0 else 1.0

def wfreq(col, w, pc=1e-6):
    """Frecuencias ponderadas con pseudoconteo configurable."""
    f = [pc]*20
    tw = sum(w)
    for i,aa in enumerate(AMINO_ACIDS):
        for j,res in enumerate(col):
            if res==aa: f[i]+=w[j]
    d = tw + 20*pc
    return [x/d for x in f]

# 1. Jensen-Shannon Divergence
def jsd(col, w, gp=True, pc=1e-6, lam=0.5):
    fc = wfreq(col, w, pc)
    r  = [lam*fc[i] + (1-lam)*BLOSUM_BG[i] for i in range(20)]
    d  = 0.
    for i in range(20):
        if r[i]>0:
            t1 = fc[i]*math.log(fc[i]/r[i],2) if fc[i]>0 else 0
            t2 = BLOSUM_BG[i]*math.log(BLOSUM_BG[i]/r[i],2) if BLOSUM_BG[i]>0 else 0
            d += t1+t2
    d /= 2
    return d*gap_penalty(col,w) if gp else d

# 2. Shannon Entropy (inverted)
def shannon(col, w, gp=True, pc=1e-6, lam=0.5):
    fc = wfreq(col, w, pc)
    h  = sum(f*math.log(f) for f in fc if f>0)
    denom = math.log(min(20,len(col)))
    h /= denom if denom!=0 else 1
    s = 1-(-h)
    return s*gap_penalty(col,w) if gp else s

# 3. Property Entropy (Williamson groups, Taylor 1986)
def prop_entropy(col, w, gp=True, pc=1e-6, lam=0.5):
    groups = [['V','L','I','M'],['F','W','Y'],['S','T'],['N','Q'],
              ['H','K','R'],['D','E'],['A','G'],['P'],['C']]
    fc = wfreq(col, w, pc)
    pf = [sum(fc[AA_TO_IDX[aa]] for aa in g if aa in AA_TO_IDX) for g in groups]
    h  = sum(f*math.log(f) for f in pf if f>0)
    denom = math.log(min(len(groups),len(col)))
    h /= denom if denom!=0 else 1
    s = 1-(-h)
    return s*gap_penalty(col,w) if gp else s

# 4. Kullback-Leibler (Relative Entropy)
def kl_div(col, w, gp=True, pc=1e-6, lam=0.5):
    fc = wfreq(col, w, pc)
    d  = sum(fc[i]*math.log(fc[i]/BLOSUM_BG[i],2) for i in range(20) if fc[i]>0 and BLOSUM_BG[i]>0)
    d_norm = min(d/4.322, 1.0) if d>0 else 0.0
    return d_norm*gap_penalty(col,w) if gp else d_norm

# 5. Valdar01 — Sum-of-Pairs with BLOSUM62, normalized
def valdar(col, w, gp=True, pc=1e-6, lam=0.5):
    aas = [aa for aa in col if aa in AA_TO_IDX]
    ws  = [w[i] for i,aa in enumerate(col) if aa in AA_TO_IDX]
    if len(aas)<2:
        return 0.0
    score = 0.
    max_s = 0.
    for i in range(len(aas)):
        for j in range(i+1,len(aas)):
            s = blosum62(aas[i],aas[j])
            score += ws[i]*ws[j]*s
            max_s += ws[i]*ws[j]*max(blosum62(aas[i],aas[i]),blosum62(aas[j],aas[j]))
    if max_s<=0:
        return 0.0
    result = score/max_s
    result = max(0., min(1., (result+1)/2))
    return result*gap_penalty(col,w) if gp else result

# 6. SMERFS — sliding window JSD (window=7)
def smerfs_score(scores_jsd, window=7):
    """
    Post-processing: suaviza JSD con una ventana deslizante de tamaño `window`.
    Equivale al algoritmo SMERFS de Manning et al. 2008.
    """
    n = len(scores_jsd)
    half = window//2
    smoothed = []
    for i in range(n):
        start = max(0, i-half)
        end   = min(n, i+half+1)
        smoothed.append(float(np.mean(scores_jsd[start:end])))
    return smoothed

METRICS_META = {
    'Jensen-Shannon': {
        'fn': jsd,
        'ref': 'Capra & Singh, Bioinformatics 2007',
        'formula': 'JSD(P,Q) = ½·KL(P‖M) + ½·KL(Q‖M)\nM = ½(P+Q)',
        'desc': 'Mide la divergencia simétrica entre la distribución observada (P) '
                'y la distribución de fondo BLOSUM62 (Q). Acotada en [0,1]. '
                'La mejor métrica práctica para identificar núcleos catalíticos. '
                'Valores altos = columna muy diferente del "fondo evolutivo" = conservación funcional.',
    },
    'Shannon Entropy': {
        'fn': shannon,
        'ref': 'Shannon, Bell System Tech. J. 1948',
        'formula': 'H = -Σ pᵢ·log₂(pᵢ)   →   score = 1 - H/H_max',
        'desc': 'Mide la variabilidad pura de una columna. Sin distribución de fondo. '
                'Simple e interpretable: score=1 = un solo residuo en todas las secuencias. '
                'Limitación: asume todos los aminoácidos equiprobables (no captura "rareza" evolutiva).',
    },
    'Property Entropy': {
        'fn': prop_entropy,
        'ref': 'Williamson 1995; grupos Taylor 1986',
        'formula': 'H_prop = -Σ f_g·log(f_g)   sobre 9 grupos fisicoquímicos\nscore = 1 - H_prop/H_max',
        'desc': 'Como Shannon pero sobre grupos fisicoquímicos (alifáticos, aromáticos, polares, etc.). '
                'Detecta conservación funcional aunque cambien los residuos específicos: '
                'Val→Leu cuenta como conservado dentro del grupo hidrofóbico.',
    },
    'Kullback-Leibler': {
        'fn': kl_div,
        'ref': 'Wang & Samudrala, Bioinformatics 2006',
        'formula': 'KL(P‖Q) = Σ pᵢ·log₂(pᵢ/qᵢ)',
        'desc': 'Entropía Relativa: mide cuánta información extra se necesita para '
                'codificar P asumiendo Q (BLOSUM62 background). '
                'Asimétrica (KL(P‖Q) ≠ KL(Q‖P)) y no acotada — se normaliza a [0,1]. '
                'Asigna mayor score a residuos conservados que son intrínsecamente raros.',
    },
    'Valdar01': {
        'fn': valdar,
        'ref': 'Valdar & Thornton, Proteins 2001',
        'formula': 'C = Σᵢ<ⱼ wᵢ·wⱼ·BLOSUM62(aᵢ,aⱼ) / max_SoP',
        'desc': 'Sum-of-Pairs ponderado con BLOSUM62. Evalúa la similitud fisicoquímica '
                'entre todos los pares de residuos en la columna. '
                'El método por defecto de SnapGene. Robusto pero tiende a sobre-puntuar '
                'el núcleo hidrofóbico vs sitios catalíticos puros.',
    },
    'SMERFS': {
        'fn': None,  # post-processing sobre JSD
        'ref': 'Manning et al., Bioinformatics 2008',
        'formula': 'SMERFS(i) = mean(JSD(i-3)...JSD(i+3))   ventana=7',
        'desc': 'Suavizado con ventana deslizante sobre JSD (ventana=7). '
                'Basado en la observación de que los sitios funcionales no son aislados: '
                'están rodeados de restricciones estructurales locales. '
                'Superior en ROC para interfaces proteína-proteína y bolsas de ligandos.',
    },
}

# ─── PIPELINE RUN ─────────────────────────────────────────────────────────────

def run_pipeline(msa_records, selected_metrics, gap_pen=True,
                 use_weights=True, pc=1e-6, lam=0.5, smerfs_win=7):
    msa     = [str(r.seq).upper() for r in msa_records]
    ref_seq = msa[0]
    n_seqs  = len(msa)

    # pesos: Henikoff o uniforme
    if use_weights:
        weights = calc_weights(msa)
    else:
        weights = [1.0/n_seqs] * n_seqs

    rows    = []
    pos_ref = 0
    col_jsd = []

    for i in range(len(ref_seq)):
        col = [seq[i] for seq in msa]
        if ref_seq[i] != '-':
            pos_ref += 1
            row = {'pos_msa': i+1, 'pos_ref': pos_ref, 'residuo': ref_seq[i]}
            for mname in selected_metrics:
                if mname == 'SMERFS':
                    continue
                fn = METRICS_META[mname]['fn']
                row[mname] = round(fn(col, weights, gp=gap_pen, pc=pc, lam=lam), 4)
            # siempre calculamos JSD base si hace falta para SMERFS
            if 'Jensen-Shannon' in selected_metrics or 'SMERFS' in selected_metrics:
                col_jsd.append(jsd(col, weights, gp=gap_pen, pc=pc, lam=lam))
            rows.append(row)

    df = pd.DataFrame(rows)

    # SMERFS post-processing con ventana configurable
    if 'SMERFS' in selected_metrics and col_jsd:
        sm = smerfs_score(col_jsd, window=smerfs_win)
        df['SMERFS'] = [round(v,4) for v in sm]

    return df, weights


# ─── LOGO BUILDER ─────────────────────────────────────────────────────────────

def build_logo_svg(msa_records, page=0, page_size=50, logo_type='freq'):
    """
    logo_type = 'freq'   → alturas proporcionales a frecuencia (clásico)
    logo_type = 'bits'   → alturas proporcionales a información (bits = H_max - H)
                           La columna total refleja la conservación real
    """
    msa     = [str(r.seq).upper() for r in msa_records]
    ref_seq = msa[0]

    # recolectar todas las columnas (solo posiciones no-gap en ref)
    all_cols = []
    for i in range(len(ref_seq)):
        if ref_seq[i] != '-':
            all_cols.append((len(all_cols)+1, [seq[i] for seq in msa]))

    total_pos = len(all_cols)
    start     = page * page_size
    end       = min(start + page_size, total_pos)
    cols      = all_cols[start:end]

    W = 16; H = 70; pad_x = 28; pad_y = 8; label_h = 16
    total_w = len(cols)*W + pad_x*2
    total_h = H + pad_y*2 + label_h + 4

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}" height="{total_h}" '
        f'style="background:#131820;font-family:\'IBM Plex Mono\',monospace">',
        f'<rect width="{total_w}" height="{total_h}" fill="#131820"/>',
    ]

    H_MAX = math.log(20)  # max Shannon entropy (nats)

    for ci, (pos, col) in enumerate(cols):
        counts = Counter(aa for aa in col if aa in AA_TO_IDX)
        total  = sum(counts.values()) or 1
        freqs  = {aa: counts[aa]/total for aa in counts}
        sorted_aas = sorted(freqs.items(), key=lambda x: x[1])

        # entropía en nats para escalar la columna (bits logo)
        H_col = -sum(f*math.log(f) for f in freqs.values() if f>0)
        info  = max(0, H_MAX - H_col)  # bits de información (nats)
        col_height_frac = info / H_MAX  # [0,1]

        x      = pad_x + ci*W
        y_base = pad_y + H

        if logo_type == 'bits':
            col_h = int(col_height_frac * H)
            y_base = pad_y + H  # alineado al fondo
        else:
            col_h = H

        y_cursor = y_base
        for aa, freq in sorted_aas:
            if logo_type == 'bits':
                bh = max(1, int(freq * col_h))
            else:
                bh = max(1, int(freq * H))

            col_color = AA_COLORS.get(aa, '#657a8e')
            y_cursor -= bh
            fs = max(6, int(W * 0.82))

            if logo_type == 'bits':
                scale_y = bh / max(fs, 1)
                ty      = y_cursor * (1/max(scale_y,0.01) - 1)
                svg.append(
                    f'<text x="{x+W//2}" y="{y_cursor+bh}" text-anchor="middle" '
                    f'font-size="{fs}" font-weight="bold" fill="{col_color}" '
                    f'transform="scale(1,{scale_y:.3f}) translate(0,{ty:.1f})">{aa}</text>'
                )
            else:
                scale_y = (freq*H) / max(fs, 1)
                ty      = (y_cursor+bh) * (1/max(scale_y,0.01) - 1)
                svg.append(
                    f'<text x="{x+W//2}" y="{y_cursor+bh}" text-anchor="middle" '
                    f'font-size="{fs}" font-weight="bold" fill="{col_color}" '
                    f'transform="scale(1,{scale_y:.3f}) translate(0,{ty:.1f})">{aa}</text>'
                )

        # posición label
        if pos % 10 == 0 or pos == 1:
            svg.append(
                f'<text x="{x+W//2}" y="{pad_y+H+label_h-2}" text-anchor="middle" '
                f'font-size="7" fill="#657a8e">{pos}</text>'
            )

    svg.append('</svg>')
    return '\n'.join(svg), total_pos


# ─── SECONDARY STRUCTURE PARSING ──────────────────────────────────────────────

def parse_dssp_from_pdb(pdb_bytes, chain_id='A'):
    """
    Intenta correr DSSP sobre el PDB subido.
    Retorna (ss_map, source) donde source es 'DSSP', 'PDB records' o None.
    ss_map = {resnum: ss_type} donde ss_type in {'H','E','C'}
    """
    # Intento 1: DSSP via Biopython
    try:
        from Bio.PDB import PDBParser, DSSP
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as tmp:
            tmp.write(pdb_bytes)
            tmp_path = tmp.name
        parser = PDBParser(QUIET=True)
        struct  = parser.get_structure('prot', tmp_path)
        model   = struct[0]
        dssp    = DSSP(model, tmp_path)
        ss_map  = {}
        for key in dssp.property_keys:
            res    = dssp[key]
            resnum = key[1][1]
            raw_ss = res[2]
            if raw_ss in ('H','G','I'):
                ss_map[resnum] = 'H'
            elif raw_ss in ('B','E'):
                ss_map[resnum] = 'E'
            else:
                ss_map[resnum] = 'C'
        os.unlink(tmp_path)
        if ss_map:
            return ss_map, 'DSSP (experimental)'
    except Exception:
        pass

    # Intento 2: registros HELIX/SHEET del PDB
    ss_map = {}
    try:
        text = pdb_bytes.decode('utf-8', errors='ignore')
        for line in text.splitlines():
            if line.startswith('HELIX'):
                try:
                    start = int(line[21:25].strip())
                    end   = int(line[33:37].strip())
                    for r in range(start, end+1):
                        ss_map[r] = 'H'
                except: pass
            elif line.startswith('SHEET'):
                try:
                    start = int(line[22:26].strip())
                    end   = int(line[33:37].strip())
                    for r in range(start, end+1):
                        ss_map[r] = 'E'
                except: pass
    except: pass
    if ss_map:
        return ss_map, 'Registros HELIX/SHEET del PDB'

    return {}, None


def predict_ss_from_sequence(seq):
    """
    Predicción heurística de estructura secundaria desde secuencia primaria.
    Basada en las propensidades de Chou-Fasman (1974/1978).
    Fuente: Chou & Fasman, Advances in Enzymology 47:45–148, 1978.

    Retorna (ss_map, source) donde ss_map = {pos_1based: 'H'|'E'|'C'}
    IMPORTANTE: esto es una estimación de baja resolución — siempre se indica
    como "predicción" en la UI. No reemplaza DSSP ni estructuras experimentales.
    """
    # Propensidades Chou-Fasman: (P_helix, P_sheet)
    CF = {
        'A':(1.42,0.83),'R':(0.98,0.93),'N':(0.67,0.89),'D':(1.01,0.54),
        'C':(0.70,1.19),'Q':(1.11,1.10),'E':(1.51,0.37),'G':(0.57,0.75),
        'H':(1.00,0.87),'I':(1.08,1.60),'L':(1.21,1.30),'K':(1.16,0.74),
        'M':(1.45,1.05),'F':(1.13,1.38),'P':(0.57,0.55),'S':(0.77,0.75),
        'T':(0.83,1.19),'W':(1.08,1.37),'Y':(0.69,1.47),'V':(1.06,1.70),
    }
    # Umbral: si P_helix>1.0 → helix, P_sheet>1.0 → sheet (simplificado)
    raw = []
    for aa in seq.upper():
        ph, pe = CF.get(aa, (1.0, 1.0))
        if ph >= pe and ph > 1.0:
            raw.append('H')
        elif pe > ph and pe > 1.0:
            raw.append('E')
        else:
            raw.append('C')

    # Suavizado con ventana=5: el estado de mayoría gana
    n = len(raw)
    smooth = []
    for i in range(n):
        window = raw[max(0,i-2):min(n,i+3)]
        cnt = Counter(window)
        smooth.append(cnt.most_common(1)[0][0])

    ss_map = {i+1: s for i, s in enumerate(smooth)}
    return ss_map, 'Predicción Chou-Fasman (heurística, no experimental)'


def build_ss_bar_html(df, ss_map, pdb_offset=0, source_label=""):
    """
    Genera un visor HTML de estructura secundaria con:
    - Barra de color por segmento
    - Iconografía animada: hélice (espiral SVG) y lámina β (flecha)
    - Etiquetas de posición
    """
    if not ss_map:
        return ""
    positions = list(df['pos_ref'])
    n = len(positions)
    if n == 0:
        return ""

    # Agrupar posiciones en segmentos continuos
    segs = []
    if positions:
        cur_ss  = ss_map.get(positions[0] + pdb_offset, 'C')
        seg_start_idx = 0
        for ci in range(1, n):
            pdb_res = positions[ci] + pdb_offset
            ss = ss_map.get(pdb_res, 'C')
            if ss != cur_ss:
                segs.append((seg_start_idx, ci-1, cur_ss))
                seg_start_idx = ci
                cur_ss = ss
        segs.append((seg_start_idx, n-1, cur_ss))

    BAR_H   = 28   # altura de la barra de color
    ICON_H  = 44   # altura de la zona de iconos animados
    TICK_H  = 14   # altura para ticks de posición
    TOTAL_H = BAR_H + ICON_H + TICK_H + 8

    W = 900
    cell = W / n

    # SVG principal
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{TOTAL_H}" '
        f'style="background:#131820;font-family:\'IBM Plex Mono\',monospace;display:block">',
        '<defs>',
        # animación de hélice: clip path + onda senoidal
        '''<style>
          @keyframes sheetPulse  { 0%,100%{opacity:0.7} 50%{opacity:1} }
          .sheet-arr  { animation: sheetPulse  1.8s ease-in-out infinite; }
        </style>''',
        '</defs>',
    ]

    # ── barra de color (fila superior) ──
    for si, ei, ss in segs:
        x1 = si * cell
        w_seg = (ei - si + 1) * cell
        color = {'H':'#4fc3f7','E':'#56c596','C':'#1c2330'}.get(ss, '#1c2330')
        parts.append(f'<rect x="{x1:.1f}" y="0" width="{w_seg:.1f}" height="{BAR_H}" fill="{color}" rx="0"/>')

    # ── iconografía animada por segmento ──
    y_icon = BAR_H + 4
    for si, ei, ss in segs:
        if ss == 'C':
            continue
        x1    = si * cell
        w_seg = (ei - si + 1) * cell
        cx    = x1 + w_seg / 2
        min_w = 12

        if ss == 'H' and w_seg > min_w:
            # Hélice: onda senoidal animada (doble vuelta)
            n_seg = ei - si + 1
            amp   = min(10, BAR_H * 0.38)
            freq  = max(2, n_seg // 3)
            pts1, pts2 = [], []
            steps = max(n_seg * 3, 30)
            for k in range(steps + 1):
                t  = k / steps
                px = x1 + t * w_seg
                py = y_icon + ICON_H/2 + amp * math.sin(t * freq * 2 * math.pi)
                py2= y_icon + ICON_H/2 + amp * math.sin(t * freq * 2 * math.pi + math.pi)
                pts1.append(f"{px:.1f},{py:.1f}")
                pts2.append(f"{px:.1f},{py2:.1f}")
            stroke_w = max(1.5, min(3.0, w_seg / 60))
            parts.append(
                f'<polyline points="{" ".join(pts1)}" fill="none" stroke="#4fc3f7" '
                f'stroke-width="{stroke_w}" stroke-linecap="round" opacity="0.9"/>'
            )
            parts.append(
                f'<polyline points="{" ".join(pts2)}" fill="none" stroke="#81d4fa" '
                f'stroke-width="{stroke_w * 0.7:.1f}" stroke-linecap="round" opacity="0.45"/>'
            )

        elif ss == 'E' and w_seg > min_w:
            # Lámina β: flecha horizontal animada
            ay  = y_icon + ICON_H / 2
            ah  = min(14, ICON_H * 0.55)
            aw  = min(w_seg - 4, w_seg * 0.85)
            ax  = cx - aw / 2
            tip = ax + aw
            sw  = ah * 0.45
            # cuerpo
            parts.append(
                f'<rect x="{ax:.1f}" y="{ay - sw/2:.1f}" width="{aw * 0.72:.1f}" '
                f'height="{sw:.1f}" fill="#56c596" rx="1" class="sheet-arr"/>'
            )
            # cabeza de flecha
            pts = (f"{ax + aw*0.72:.1f},{ay - ah/2:.1f} "
                   f"{tip:.1f},{ay:.1f} "
                   f"{ax + aw*0.72:.1f},{ay + ah/2:.1f}")
            parts.append(f'<polygon points="{pts}" fill="#56c596" class="sheet-arr"/>')

    # ── ticks de posición ──
    y_tick = BAR_H + ICON_H + 10
    for ci, pos_ref in enumerate(positions):
        if pos_ref == 1 or pos_ref % 10 == 0:
            x = ci * cell + cell / 2
            parts.append(
                f'<text x="{x:.1f}" y="{y_tick}" text-anchor="middle" '
                f'font-size="8" fill="#657a8e">{pos_ref}</text>'
            )

    parts.append('</svg>')
    return ''.join(parts)


# ─── PY3DMOL VIEWER ───────────────────────────────────────────────────────────

def py3dmol_viewer(pdb_text, df, score_col, pdb_offset=0):
    """
    Visor py3Dmol.
    Coloreado en JS con addPropertyLabels + setStyle usando un dict de colores
    pasado como JSON — UNA sola llamada por residuo en JS, sin loop de setStyle
    previo al render que traba el viewer.
    """
    def score_to_hex(v):
        v = max(0, min(1, v))
        if v < 0.5:
            t = v*2;  r = int(t*255); g = int(t*255); b = 255
        else:
            t = (v-0.5)*2; r = 255; g = int((1-t)*255); b = int((1-t)*255)
        return f'#{r:02x}{g:02x}{b:02x}'

    import json
    res_data = {}
    for _, row in df.iterrows():
        if score_col in row:
            pdb_res = int(row['pos_ref']) + pdb_offset
            sc      = round(float(row[score_col]), 4)
            res_data[pdb_res] = {
                'resn':    row['residuo'],
                'pos_ref': int(row['pos_ref']),
                'score':   sc,
                'color':   score_to_hex(sc),
            }

    res_data_js = json.dumps(res_data)
    pdb_escaped = (pdb_text
                   .replace('\\', '\\\\')
                   .replace('`',  '\\`')
                   .replace('$',  '\\$'))

    html = f"""<!DOCTYPE html><html><head>
<script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.4/3Dmol-min.js"></script>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#0b0f14;font-family:'IBM Plex Mono',monospace;
        display:flex;flex-direction:column;height:100vh;overflow:hidden}}
  #toolbar{{display:flex;align-items:center;gap:6px;padding:6px 10px;
            background:#131820;border-bottom:1px solid #2a3444;flex-shrink:0}}
  #toolbar button{{background:#1c2330;color:#4fc3f7;border:1px solid #2a3444;
    border-radius:4px;padding:3px 11px;font:11px 'IBM Plex Mono',monospace;cursor:pointer}}
  #toolbar button:hover{{background:#2a3444}}
  #toolbar button.active{{background:#4fc3f7;color:#0b0f14}}
  #sel-count{{margin-left:auto;font-size:10px;color:#657a8e}}
  #main{{display:flex;flex:1;overflow:hidden}}
  #vwrap{{position:relative;flex:1;min-width:0}}
  #viewer{{width:100%;height:100%}}
  #tooltip{{position:absolute;pointer-events:none;display:none;
    background:rgba(11,15,20,.93);border:1px solid #2a3444;border-radius:6px;
    padding:8px 12px;font-size:11px;color:#d8e4f0;min-width:160px;
    z-index:100;line-height:1.7}}
  .t-res{{font-size:14px;font-weight:700;color:#4fc3f7}}
  .t-score{{color:#f5a623}}
  .t-hint{{font-size:9px;color:#657a8e;margin-top:4px}}
  #legend{{position:absolute;bottom:10px;right:10px;
    background:rgba(11,15,20,.88);border:1px solid #2a3444;
    border-radius:6px;padding:8px 12px;font-size:10px;color:#d8e4f0}}
  .lgbar{{width:110px;height:10px;
    background:linear-gradient(to right,#0000ff,#ffffff,#ff0000);
    border-radius:3px;margin:4px 0}}
  #sel-panel{{width:200px;background:#131820;border-left:1px solid #2a3444;
    display:flex;flex-direction:column;flex-shrink:0}}
  #sel-header{{padding:8px 10px;font-size:10px;color:#657a8e;
    text-transform:uppercase;letter-spacing:.08em;
    border-bottom:1px solid #2a3444;display:flex;
    align-items:center;justify-content:space-between}}
  #sel-header button{{background:none;border:none;color:#f07070;
    cursor:pointer;font:10px 'IBM Plex Mono',monospace;padding:0}}
  #sel-list{{flex:1;overflow-y:auto;padding:4px 0}}
  .sel-item{{display:flex;align-items:center;gap:6px;padding:5px 10px;
    font-size:11px;cursor:pointer;border-bottom:1px solid #1c2330}}
  .sel-item:hover{{background:#1c2330}}
  .si-dot{{width:8px;height:8px;border-radius:50%;flex-shrink:0}}
  .si-name{{color:#d8e4f0;flex:1}}
  .si-score{{color:#f5a623;font-size:10px}}
  .si-rm{{background:none;border:none;color:#657a8e;cursor:pointer;
    font-size:12px;padding:0 2px;line-height:1}}
  .si-rm:hover{{color:#f07070}}
  #sel-empty{{padding:16px 10px;font-size:10px;color:#657a8e;
    text-align:center;line-height:1.6}}
  #sel-export{{padding:8px 10px;border-top:1px solid #2a3444;display:flex;gap:6px}}
  #sel-export button{{flex:1;background:#1c2330;color:#4fc3f7;
    border:1px solid #2a3444;border-radius:4px;padding:4px 0;
    font:10px 'IBM Plex Mono',monospace;cursor:pointer}}
  #sel-export button:hover{{background:#2a3444}}
</style></head><body>

<div id="toolbar">
  <button onclick="setRep('cartoon')" class="active" id="btn-cartoon">Cartoon</button>
  <button onclick="setRep('surface')"                id="btn-surface">Surface</button>
  <button onclick="setRep('stick')"                  id="btn-stick">Sticks</button>
  <button onclick="viewer.zoomTo();viewer.render()">⌂ Reset</button>
  <button onclick="zoomSel()">⊙ Zoom sel.</button>
  <span id="sel-count"></span>
</div>

<div id="main">
  <div id="vwrap">
    <div id="viewer"></div>
    <div id="tooltip">
      <div class="t-res"   id="t-res">—</div>
      <div                 id="t-chain"></div>
      <div class="t-score" id="t-score"></div>
      <div class="t-hint">Click para seleccionar / deseleccionar</div>
    </div>
    <div id="legend">
      <div style="color:#8fa8bc;font-size:9px;margin-bottom:2px">Conservation</div>
      <div class="lgbar"></div>
      <div style="display:flex;justify-content:space-between;font-size:8px;color:#657a8e">
        <span>Baja</span><span>Alta</span>
      </div>
      <div style="margin-top:5px;font-size:9px;color:#657a8e">{score_col}</div>
    </div>
  </div>

  <div id="sel-panel">
    <div id="sel-header">
      <span>Seleccionados</span>
      <button onclick="clearAll()">✕ Limpiar</button>
    </div>
    <div id="sel-list">
      <div id="sel-empty">Hacé click en un<br>residuo para<br>seleccionarlo</div>
    </div>
    <div id="sel-export">
      <button onclick="exportTxt()">↓ .txt</button>
      <button onclick="exportCsv()">↓ .csv</button>
    </div>
  </div>
</div>

<script>
// ── datos pasados desde Python ──
const RES_DATA   = {res_data_js};
const PDB_OFFSET = {pdb_offset};

// ── init viewer ──
const viewer = $3Dmol.createViewer(
  document.getElementById("viewer"),
  {{backgroundColor: "0x0b0f14", antialias: true}}
);
viewer.addModel(`{pdb_escaped}`, "pdb");

// ── función de coloreado: UNA sola llamada setStyle por residuo ──
// 3Dmol acepta una función callback en el campo "color" — la llamamos
// una vez con setStyle global + colorFunction, sin loop de setStyle.
function applyConservationColors(rep) {{
  // Primero pintamos todo en gris oscuro
  viewer.setStyle({{}}, {{[rep]: {{color: "0x1c2330"}}}});
  // Luego por cada residuo con datos sobreescribimos el color
  // Usamos un objeto de selección individual — es más rápido que N setStyle
  // porque 3Dmol internamente agrupa por modelo
  Object.entries(RES_DATA).forEach(([resi, d]) => {{
    viewer.setStyle(
      {{resi: parseInt(resi)}},
      {{[rep]: {{color: d.color}}}}
    );
  }});
}}

applyConservationColors("cartoon");
viewer.zoomTo();
viewer.render();

// ── estado ──
let selected = {{}};
let curRep   = "cartoon";

// ── tooltip ──
const tooltip = document.getElementById("tooltip");
document.getElementById("vwrap").addEventListener("mousemove", e => {{
  tooltip.style.left = (e.offsetX + 16) + "px";
  tooltip.style.top  = Math.max(0, e.offsetY - 60) + "px";
}});

// ── hover ──
viewer.setHoverable({{}}, true,
  function onHover(atom) {{
    if (!atom) return;
    const d = RES_DATA[atom.resi];
    document.getElementById("t-res").textContent =
      (atom.resn || "?") + " " + atom.resi;
    document.getElementById("t-chain").textContent =
      "Chain " + (atom.chain || "A");
    document.getElementById("t-score").textContent =
      d ? "Score: " + d.score.toFixed(4) : "Sin datos de score";
    tooltip.style.display = "block";
    // highlight blanco solo si no está seleccionado
    if (!selected[atom.resi]) {{
      viewer.setStyle({{resi: atom.resi}},
        {{[curRep]: {{color: "0xffffff", opacity: 0.85}}}});
      viewer.render();
    }}
  }},
  function onUnhover(atom) {{
    tooltip.style.display = "none";
    if (atom && !selected[atom.resi]) restoreColor(atom.resi);
  }}
);

// ── click: seleccionar / deseleccionar ──
viewer.setClickable({{}}, true, function(atom) {{
  if (!atom) return;
  const resi = atom.resi;
  if (selected[resi]) {{
    delete selected[resi];
    restoreColor(resi);
    // eliminar label — 3Dmol no soporta removeLabel por id fácilmente,
    // re-renderizamos todos los labels
    rebuildLabels();
  }} else {{
    const d = RES_DATA[resi];
    selected[resi] = {{
      resi,
      resn:    atom.resn  || "?",
      chain:   atom.chain || "A",
      score:   d ? d.score   : null,
      pos_ref: d ? d.pos_ref : resi - PDB_OFFSET,
    }};
    viewer.setStyle({{resi}}, {{
      [curRep]: {{color: "0xf5a623"}},
      stick:    {{color: "0xf5a623", radius: 0.25}},
    }});
    rebuildLabels();
  }}
  viewer.render();
  updatePanel();
}});

function restoreColor(resi) {{
  const d = RES_DATA[resi];
  viewer.setStyle(
    {{resi}},
    {{[curRep]: {{color: d ? d.color : "0x1c2330"}}}}
  );
  viewer.render();
}}

function rebuildLabels() {{
  viewer.removeAllLabels();
  Object.values(selected).forEach(d => {{
    viewer.addLabel(
      d.resn + d.resi,
      {{fontSize: 12, fontColor: "0xf5a623",
        backgroundOpacity: 0.6, backgroundColor: "0x0b0f14",
        borderColor: "0xf5a623", borderThickness: 0.5}},
      {{resi: d.resi, atom: "CA"}}
    );
  }});
}}

// ── panel lateral ──
function updatePanel() {{
  const list   = document.getElementById("sel-list");
  const empty  = document.getElementById("sel-empty");
  const sorted = Object.values(selected).sort((a, b) => a.resi - b.resi);
  document.getElementById("sel-count").textContent =
    sorted.length ? sorted.length + " sel." : "";
  if (!sorted.length) {{
    list.innerHTML = "";
    list.appendChild(empty);
    empty.style.display = "block";
    return;
  }}
  empty.style.display = "none";
  list.innerHTML = sorted.map(d => {{
    const hex = d.score != null ? d.score >= 0.5
      ? `rgb(${{Math.round((d.score-0.5)*2*255)}},0,255)`
      : `rgb(0,${{Math.round((1-d.score*2)*255)}},255)`
      : "#657a8e";
    const sc = d.score != null ? d.score.toFixed(3) : "—";
    return `<div class="sel-item"
        onclick="viewer.zoomTo({{resi:${{d.resi}}}});viewer.render()">
      <div class="si-dot" style="background:${{d.score!=null?scoreToHex(d.score):'#657a8e'}}"></div>
      <div class="si-name">${{d.resn}}${{d.resi}}</div>
      <div class="si-score">${{sc}}</div>
      <button class="si-rm"
        onclick="event.stopPropagation();rmRes(${{d.resi}})">✕</button>
    </div>`;
  }}).join("");
}}

function rmRes(resi) {{
  delete selected[resi];
  restoreColor(resi);
  rebuildLabels();
  viewer.render();
  updatePanel();
}}

function clearAll() {{
  Object.keys(selected).forEach(r => restoreColor(+r));
  selected = {{}};
  viewer.removeAllLabels();
  viewer.render();
  updatePanel();
}}

function zoomSel() {{
  const keys = Object.keys(selected).map(Number);
  if (keys.length) {{ viewer.zoomTo({{resi: keys}}); viewer.render(); }}
}}

// ── cambiar representación ──
function setRep(rep) {{
  curRep = rep;
  applyConservationColors(rep);
  // restaurar seleccionados
  Object.values(selected).forEach(d => {{
    viewer.setStyle({{resi: d.resi}}, {{
      [rep]: {{color: "0xf5a623"}},
      stick: {{color: "0xf5a623", radius: 0.25}},
    }});
  }});
  viewer.render();
  ["cartoon","surface","stick"].forEach(r =>
    document.getElementById("btn-"+r)?.classList.remove("active")
  );
  document.getElementById("btn-"+rep)?.classList.add("active");
}}

// ── helpers ──
function scoreToHex(v) {{
  v = Math.max(0, Math.min(1, v));
  let r, g, b;
  if (v < 0.5) {{ const t=v*2; r=Math.round(t*255); g=Math.round(t*255); b=255; }}
  else         {{ const t=(v-0.5)*2; r=255; g=Math.round((1-t)*255); b=Math.round((1-t)*255); }}
  return "#"+[r,g,b].map(x=>x.toString(16).padStart(2,"0")).join("");
}}

// ── exportar ──
function exportTxt() {{
  const s = Object.values(selected).sort((a,b)=>a.resi-b.resi);
  dl("selected_residues.txt",
    s.map(d => d.resn+d.resi+(d.score!=null?" score="+d.score:"")).join("\\n"));
}}
function exportCsv() {{
  const s = Object.values(selected).sort((a,b)=>a.resi-b.resi);
  dl("selected_residues.csv",
    "pos_pdb,pos_ref,resname,chain,score\\n" +
    s.map(d => [d.resi,d.pos_ref,d.resn,d.chain,d.score??""]).join("\\n"));
}}
function dl(name, txt) {{
  const a = document.createElement("a");
  a.href = "data:text/plain;charset=utf-8," + encodeURIComponent(txt);
  a.download = name; a.click();
}}
</script>
</body></html>"""
    return html


def generate_pymol_script(df, score_col, pdb_name, pdb_offset=0):
    lines = [
        f"# Protein-Conservation — PyMOL script",
        f"# Metric: {score_col}  |  PDB offset: {pdb_offset}",
        f"load {pdb_name}.pdb, protein",
        f"bg_color white",
        f"hide everything",
        f"show cartoon, protein",
        f"alter protein, b=0.0",
    ]
    for _, r in df.iterrows():
        pdb_res = int(r['pos_ref']) + pdb_offset
        lines.append(f"alter protein and resi {pdb_res}, b={r[score_col]}")
    lines += [
        "spectrum b, blue_white_red, protein, minimum=0, maximum=1",
        f"# Top-10 as sticks",
    ]
    top10 = df.nlargest(10, score_col)['pos_ref'].tolist()
    resi  = '+'.join(str(int(x)+pdb_offset) for x in top10)
    lines += [f"show sticks, protein and resi {resi}",
              "zoom protein", f"# ray 1200,900", f"# png conservation_{score_col}.png,dpi=300"]
    return '\n'.join(lines)


def generate_chimerax_script(df, score_col, pdb_name, pdb_offset=0):
    lines = [f"# Protein-Conservation — ChimeraX script", f"# Metric: {score_col}",
             f"open {pdb_name}.pdb", "cartoon"]
    for _, r in df.iterrows():
        pdb_res = int(r['pos_ref']) + pdb_offset
        lines.append(f"setattr /A:{pdb_res} res conservation {r[score_col]}")
    lines += ["color byattribute conservation protein :* palette bluered", "view"]
    return '\n'.join(lines)


# ─── PLOTLY CHART ─────────────────────────────────────────────────────────────

def conservation_chart(df, score_cols):
    import plotly.graph_objects as go
    PAL = ['#4fc3f7','#56c596','#f5a623','#f07070','#b39ddb','#81d4fa']
    fig = go.Figure()
    for i,col in enumerate(score_cols):
        fig.add_trace(go.Scatter(
            x=df['pos_ref'], y=df[col], mode='lines', name=col,
            line=dict(color=PAL[i%len(PAL)], width=1.5),
            hovertemplate='Pos %{x} (%{customdata})<br>Score: %{y:.4f}<extra>'+col+'</extra>',
            customdata=df['residuo'],
        ))
    fig.update_layout(
        template='plotly_dark', paper_bgcolor='#0b0f14', plot_bgcolor='#131820',
        font=dict(family='IBM Plex Mono, monospace', size=11, color='#d8e4f0'),
        xaxis=dict(title='Posición (ref)', gridcolor='#1c2330', zeroline=False),
        yaxis=dict(title='Score de conservación', gridcolor='#1c2330', zeroline=False, range=[0,1.05]),
        legend=dict(bgcolor='#1c2330', bordercolor='#2a3444', borderwidth=1),
        margin=dict(l=40,r=20,t=20,b=40), height=300,
    )
    return fig


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🧬 Protein-Conservation")
    st.markdown("---")

    st.markdown("**Alineamiento (MSA)**")
    uploaded_msa = st.file_uploader("FASTA / ALN", type=['fasta','fa','aln','txt'])

    st.markdown("---")
    st.markdown("**Métricas**")
    sel_metrics = st.multiselect(
        "Seleccionar algoritmos",
        options=list(METRICS_META.keys()),
        default=['Jensen-Shannon', 'Shannon Entropy'],
    )

    st.markdown("---")
    st.markdown("**Parámetros del pipeline**")

    use_weights = st.toggle(
        "Pesar secuencias (Henikoff & Henikoff)",
        value=True,
        help="Asigna pesos inversamente proporcionales a la redundancia en cada columna. "
             "Recomendado si el MSA tiene familias sobre-representadas. "
             "Desactivar equivale a dar peso uniforme 1/N a todas las secuencias.",
    )

    gap_pen = st.toggle(
        "Penalización por gaps",
        value=True,
        help="Multiplica cada score por (1 − fracción_de_gaps_en_la_columna). "
             "Reduce el score de posiciones con muchos gaps en el alineamiento.",
    )

    pseudocount = st.select_slider(
        "Pseudoconteo",
        options=[1e-9, 1e-6, 1e-3, 0.01, 0.05, 0.1, 0.5],
        value=1e-6,
        format_func=lambda x: f"{x:.0e}" if x < 0.01 else str(x),
        help="Valor añadido a cada frecuencia antes de calcular logaritmos. "
             "Evita log(0) y suaviza distribuciones en MSAs pequeños. "
             "Valores mayores = más suavizado (útil con <20 secuencias).",
    )

    jsd_lambda = st.slider(
        "λ (mezcla JSD / KL)",
        min_value=0.0, max_value=1.0, value=0.5, step=0.05,
        help="Peso de la distribución observada vs fondo BLOSUM62 en JSD y KL.\n"
             "λ=0.5 → mezcla clásica 50/50 (Capra & Singh 2007).\n"
             "λ→1.0 → más peso a la secuencia observada.\n"
             "λ→0.0 → más peso al fondo BLOSUM62.",
    )

    smerfs_win = st.slider(
        "Ventana SMERFS",
        min_value=1, max_value=21, value=7, step=2,
        help="Tamaño de la ventana deslizante para suavizar JSD (debe ser impar). "
             "w=7 es el valor óptimo reportado por Manning et al. 2008. "
             "Ventanas más grandes → más suavizado, mejor para detectar dominios; "
             "ventanas más chicas → más resolución posicional.",
    )

    st.markdown("---")
    st.markdown("**Logo**")
    logo_page_size = 50  # fijo para mayor densidad
    st.caption("Paginado automático — 50 posiciones/página")

    st.markdown("---")
    st.markdown("**Estructura 3D (opcional)**")
    uploaded_pdb = st.file_uploader("Subir PDB", type=['pdb','ent'])
    pdb_offset   = st.number_input("Offset numeración PDB", value=0, step=1,
                                    help="pos_pdb = pos_ref + offset. Útil si tu PDB no empieza en 1.")
    pdb_name_inp = st.text_input("Nombre PDB (sin extensión)", value="protein")

    run_btn = st.button("▶  Calcular", use_container_width=True)


# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="pc-header">
  <p class="pc-title">Protein-Conservation</p>
  <p class="pc-sub">Multiple Sequence Alignment · Conservation Scoring · Structure Mapping</p>
</div>
""", unsafe_allow_html=True)

# ─── GUARD ────────────────────────────────────────────────────────────────────
if not uploaded_msa:
    st.markdown("""
    <div class="info-box">
    <b>Cómo usar:</b><br><br>
    1. Subí tu MSA en formato FASTA (sidebar).<br>
    2. Elegí uno o más algoritmos de scoring.<br>
    3. Opcionalmente subí un PDB para visualización 3D y estructura secundaria.<br>
    4. Hacé click en <b>Calcular</b>.<br><br>
    La <b>primera secuencia</b> es la referencia para la numeración.
    </div>
    """, unsafe_allow_html=True)

    # Show algo cards
    st.markdown('<p class="sh">Algoritmos disponibles</p>', unsafe_allow_html=True)
    for mname, meta in METRICS_META.items():
        with st.expander(f"**{mname}** — {meta['ref']}"):
            st.markdown(f"""
<div class="algo-card">
  <div class="algo-formula">{meta['formula']}</div>
  <div class="algo-desc">{meta['desc']}</div>
</div>
""", unsafe_allow_html=True)
    st.stop()

# ─── PARSE MSA ────────────────────────────────────────────────────────────────
try:
    msa_bytes   = uploaded_msa.read()
    msa_records = list(SeqIO.parse(io.StringIO(msa_bytes.decode('utf-8')), 'fasta'))
    if len(msa_records) < 2:
        st.error("Necesitás al menos 2 secuencias.")
        st.stop()
except Exception as e:
    st.error(f"Error leyendo MSA: {e}")
    st.stop()

if not sel_metrics:
    st.warning("Seleccioná al menos un método.")
    st.stop()

# ─── PARSE PDB (si hay) / PREDECIR SS (si no) ────────────────────────────────
pdb_bytes = None
ss_map    = {}
ss_source = None   # texto que se muestra en la UI
pdb_text  = ""
if uploaded_pdb:
    pdb_bytes         = uploaded_pdb.read()
    pdb_text          = pdb_bytes.decode('utf-8', errors='ignore')
    ss_map, ss_source = parse_dssp_from_pdb(pdb_bytes)

# ─── COMPUTE ──────────────────────────────────────────────────────────────────
if run_btn or 'df' not in st.session_state:
    with st.spinner("Calculando scores..."):
        df, wts = run_pipeline(
            msa_records, sel_metrics,
            gap_pen=gap_pen,
            use_weights=use_weights,
            pc=pseudocount,
            lam=jsd_lambda,
            smerfs_win=smerfs_win,
        )

        # Si no hay ss_map todavía (sin PDB o PDB sin registros SS),
        # predecir desde la secuencia de referencia (primera del MSA)
        if not ss_map:
            ref_seq_clean = str(msa_records[0].seq).upper().replace('-','')
            ss_map, ss_source = predict_ss_from_sequence(ref_seq_clean)

        st.session_state['df']          = df
        st.session_state['msa_records'] = msa_records
        st.session_state['ss_map']      = ss_map
        st.session_state['ss_source']   = ss_source
        st.session_state['pdb_text']    = pdb_text
        # guardar config activa para mostrar en el resumen
        st.session_state['run_config'] = {
            'use_weights': use_weights,
            'gap_pen':     gap_pen,
            'pc':          pseudocount,
            'lam':         jsd_lambda,
            'smerfs_win':  smerfs_win,
        }

df          = st.session_state['df']
ss_map      = st.session_state.get('ss_map', {})
ss_source   = st.session_state.get('ss_source', None)
pdb_text    = st.session_state.get('pdb_text', '')
score_cols  = [c for c in df.columns if c not in ('pos_msa','pos_ref','residuo')]

# ─── SUMMARY METRICS ─────────────────────────────────────────────────────────
c1,c2,c3,c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="mcard"><div class="mcard-label">Secuencias</div><div class="mcard-value">{len(msa_records)}</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="mcard"><div class="mcard-label">Posiciones (ref)</div><div class="mcard-value">{len(df)}</div></div>', unsafe_allow_html=True)
with c3:
    if score_cols:
        sc = score_cols[0]
        tp = int(df.loc[df[sc].idxmax(),'pos_ref'])
        tr = df.loc[df[sc].idxmax(),'residuo']
        st.markdown(f'<div class="mcard"><div class="mcard-label">Top conservado ({sc[:3]})</div><div class="mcard-value">{tp} ({tr})</div></div>', unsafe_allow_html=True)
with c4:
    if ss_source:
        is_pred = 'Chou-Fasman' in ss_source or 'predicc' in ss_source.lower()
        ss_label = "⚠ Predicha" if is_pred else "✓ Experimental"
        ss_color = "#f5a623" if is_pred else "#56c596"
    else:
        ss_label, ss_color = "—", "#657a8e"
    pdb_status = "✓ cargado" if pdb_text else "—"
    st.markdown(f'<div class="mcard"><div class="mcard-label">PDB / SS</div>'
                f'<div class="mcard-value" style="font-size:0.9rem">{pdb_status} &nbsp;'
                f'<span style="color:{ss_color};font-size:0.75rem">{ss_label}</span></div></div>',
                unsafe_allow_html=True)

st.markdown("---")

# ── Resumen de configuración activa ──
cfg = st.session_state.get('run_config', {})
if cfg:
    w_label = "Henikoff pesos" if cfg.get('use_weights') else "Pesos uniformes 1/N"
    pc_val  = cfg.get('pc', 1e-6)
    lam_val = cfg.get('lam', 0.5)
    sw_val  = cfg.get('smerfs_win', 7)
    pill = ("display:inline-block;background:#1c2330;border:1px solid #2a3444;"
            "border-radius:4px;padding:2px 9px;font-size:0.67rem;"
            "color:#8fa8bc;margin:2px 3px 4px 0;font-family:'IBM Plex Mono',monospace")
    hi   = pill.replace('#2a3444','#4fc3f7').replace('#8fa8bc','#4fc3f7')
    st.markdown(
        f'<div style="margin-bottom:14px">'
        f'<span style="{pill}">⚖ {w_label}</span>'
        f'<span style="{pill}">{"✓" if cfg.get("gap_pen") else "✗"} Gap penalty</span>'
        f'<span style="{pill}">pc = {pc_val:.0e}</span>'
        f'<span style="{hi}">λ = {lam_val:.2f}</span>'
        f'<span style="{pill}">SMERFS w = {sw_val}</span>'
        f'</div>',
        unsafe_allow_html=True
    )

# ─── TABS ─────────────────────────────────────────────────────────────────────
tabs = st.tabs(["📊 Gráfico", "🔤 Sequence Logo", "🗂 MSA + Estructura", "📋 Tabla", "🧪 Estructura 3D", "ℹ Algoritmos"])

# ══════════════════════════════════════════════════════
# TAB 1 — GRÁFICO
# ══════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<p class="sh">Score de conservación por posición</p>', unsafe_allow_html=True)
    fig = conservation_chart(df, score_cols)
    st.plotly_chart(fig, use_container_width=True)

    # Estructura secundaria bajo el gráfico
    if ss_map:
        src_label = ss_source or "desconocida"
        is_pred   = 'Chou-Fasman' in src_label or 'predicc' in src_label.lower()
        badge_color = '#f5a623' if is_pred else '#56c596'
        badge_icon  = '⚠ Predicción' if is_pred else '✓ Experimental'
        st.markdown(f'<p class="sh">Estructura secundaria</p>', unsafe_allow_html=True)
        st.markdown(
            f'<span style="background:{badge_color}22;border:1px solid {badge_color};'
            f'border-radius:4px;padding:2px 8px;font-size:0.68rem;color:{badge_color};">'
            f'{badge_icon} — {src_label}</span>',
            unsafe_allow_html=True
        )
        if is_pred:
            st.caption("⚠ La predicción Chou-Fasman es una aproximación heurística de baja resolución. "
                       "Subí un PDB para obtener la estructura secundaria experimental (DSSP).")
        ss_svg = build_ss_bar_html(df, ss_map, pdb_offset)
        st.markdown(
            f'<div style="overflow-x:auto;border:1px solid #2a3444;border-radius:6px">{ss_svg}</div>',
            unsafe_allow_html=True
        )
        st.markdown("""
<div style="display:flex;gap:16px;font-size:0.7rem;margin-top:6px;font-family:'IBM Plex Mono',monospace">
  <span><span style="display:inline-block;width:12px;height:12px;border-radius:2px;background:#4fc3f7;vertical-align:middle;margin-right:4px"></span>Hélice α</span>
  <span><span style="display:inline-block;width:12px;height:12px;border-radius:2px;background:#56c596;vertical-align:middle;margin-right:4px"></span>Lámina β</span>
  <span><span style="display:inline-block;width:12px;height:12px;border-radius:2px;background:#2a3444;vertical-align:middle;margin-right:4px"></span>Loop / coil</span>
</div>""", unsafe_allow_html=True)

    # Top residuos
    st.markdown('<p class="sh">Top 10 posiciones más conservadas</p>', unsafe_allow_html=True)
    for sc in score_cols:
        top = df.nlargest(10, sc)[['pos_ref','residuo',sc]].reset_index(drop=True)
        st.markdown(f'<span class="badge">{sc}</span>', unsafe_allow_html=True)
        st.dataframe(top, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════
# TAB 2 — SEQUENCE LOGO (zoomable, Canvas-based)
# ══════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<p class="sh">Sequence Logo</p>', unsafe_allow_html=True)

    total_pos = len(df)
    n_pages   = math.ceil(total_pos / logo_page_size)

    st.markdown("""
<div class="info-box">
<b>Frecuencia:</b> todas las columnas igual de altas — letras proporcionales a su frecuencia ponderada.<br>
<b>Información (bits):</b> columna alta = posición conservada (igual que logos de HMM/Pfam/InterPro).<br>
<b>Zoom:</b> rueda del mouse sobre el logo · <b>Pan:</b> click + arrastrar.
</div>""", unsafe_allow_html=True)

    logo_col1, logo_col2 = st.columns([2, 1])
    with logo_col1:
        logo_type = st.radio("Tipo de logo", ['freq', 'bits'],
                             format_func=lambda x: 'Frecuencia' if x == 'freq' else 'Información (bits)',
                             horizontal=True)
    with logo_col2:
        logo_height = st.slider("Altura (px)", min_value=80, max_value=300, value=140, step=20)

    st.caption(f"{total_pos} posiciones · scroll para zoom · drag para deslizar")

    # Build logo data for JS rendering — ALL positions, no pagination
    msa_recs  = st.session_state['msa_records']
    msa_seqs  = [str(r.seq).upper() for r in msa_recs]
    ref_seq   = msa_seqs[0]
    logo_wts  = calc_weights(msa_seqs)
    H_MAX_LN  = math.log(20)

    logo_cols_data = []
    all_ref_cols   = []
    for i in range(len(ref_seq)):
        if ref_seq[i] != '-':
            all_ref_cols.append((len(all_ref_cols)+1, i))

    for pos_ref, msa_i in all_ref_cols:
        col = [seq[msa_i] for seq in msa_seqs]
        counts = Counter(aa for aa in col if aa in AA_TO_IDX)
        total  = sum(counts.values()) or 1
        freqs  = {aa: counts[aa]/total for aa in counts}
        H_col  = -sum(f*math.log(f) for f in freqs.values() if f > 0)
        info   = max(0.0, H_MAX_LN - H_col) / H_MAX_LN  # [0,1]
        sorted_aas = sorted(freqs.items(), key=lambda x: x[1])
        logo_cols_data.append({
            'pos': pos_ref,
            'info': round(info, 4),
            'aas': [[aa, round(f, 4)] for aa, f in sorted_aas],
        })

    AA_COLOR_JS = str(AA_COLORS).replace("'", '"')
    import json
    logo_json = json.dumps(logo_cols_data)

    logo_html = f"""<!DOCTYPE html><html><head>
<style>
  body {{margin:0;background:#131820;overflow:hidden;user-select:none;}}
  canvas {{display:block;cursor:grab;}}
  canvas.grabbing {{cursor:grabbing;}}
  #info {{position:fixed;bottom:8px;left:12px;font:11px 'IBM Plex Mono',monospace;
          color:#657a8e;pointer-events:none;}}
  #controls {{position:fixed;top:8px;right:10px;display:flex;gap:6px;}}
  button {{background:#1c2330;color:#4fc3f7;border:1px solid #2a3444;border-radius:4px;
           padding:3px 10px;font:11px monospace;cursor:pointer;}}
  button:hover {{background:#2a3444;}}
</style></head><body>
<canvas id="c"></canvas>
<div id="info">Scroll: zoom · Drag: pan</div>
<div id="controls">
  <button onclick="resetView()">Reset</button>
  <button onclick="zoomIn()">＋</button>
  <button onclick="zoomOut()">－</button>
</div>
<script>
const DATA      = {logo_json};
const COLORS    = {AA_COLOR_JS};
const MODE      = "{logo_type}";
const COL_H     = {logo_height};
const PAD_TOP   = 10;
const PAD_BOT   = 22;
const BASE_W    = 20;  // px per column at zoom=1

const canvas = document.getElementById('c');
const ctx    = canvas.getContext('2d');

let zoom    = 1.0;
let offsetX = 0;
let isDrag  = false;
let dragStartX = 0;
let dragStartOffset = 0;

function resize() {{
  canvas.width  = window.innerWidth;
  canvas.height = COL_H + PAD_TOP + PAD_BOT;
  draw();
}}

function draw() {{
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#131820';
  ctx.fillRect(0, 0, W, H);

  const colW = BASE_W * zoom;
  const n    = DATA.length;

  for (let ci = 0; ci < n; ci++) {{
    const x     = ci * colW + offsetX;
    if (x + colW < 0 || x > W) continue;

    const col   = DATA[ci];
    const colH  = MODE === 'bits' ? col.info * COL_H : COL_H;
    let y       = PAD_TOP + COL_H;  // start from bottom

    for (let ai = 0; ai < col.aas.length; ai++) {{
      const [aa, freq] = col.aas[ai];
      const letterH = Math.max(1, freq * colH);
      y -= letterH;
      const color = COLORS[aa] || '#657a8e';

      if (letterH >= 4) {{
        ctx.save();
        ctx.translate(x + colW/2, y + letterH);
        ctx.scale(1, letterH / Math.max(colW * 0.85, 4));
        ctx.font = `bold ${{Math.max(8, colW * 0.85)}}px 'IBM Plex Mono',monospace`;
        ctx.fillStyle = color;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'alphabetic';
        ctx.fillText(aa, 0, 0);
        ctx.restore();
      }} else {{
        ctx.fillStyle = color;
        ctx.fillRect(x + 1, y, colW - 2, letterH);
      }}
    }}

    // position tick
    const pos = col.pos;
    if (pos === 1 || pos % 10 === 0) {{
      ctx.fillStyle = '#657a8e';
      ctx.font = `9px 'IBM Plex Mono',monospace`;
      ctx.textAlign = 'center';
      ctx.fillText(pos, x + colW/2, PAD_TOP + COL_H + PAD_BOT - 4);
    }}
  }}

  // axis line
  ctx.strokeStyle = '#2a3444';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, PAD_TOP + COL_H + 1);
  ctx.lineTo(W, PAD_TOP + COL_H + 1);
  ctx.stroke();
}}

function resetView() {{
  zoom    = 1.0;
  offsetX = 0;
  draw();
}}
function zoomIn()  {{ applyZoom(1.4, canvas.width/2); }}
function zoomOut() {{ applyZoom(0.7, canvas.width/2); }}

function applyZoom(factor, cx) {{
  const oldZoom = zoom;
  zoom = Math.max(0.2, Math.min(20, zoom * factor));
  offsetX = cx - (cx - offsetX) * (zoom / oldZoom);
  draw();
}}

canvas.addEventListener('wheel', e => {{
  e.preventDefault();
  applyZoom(e.deltaY < 0 ? 1.15 : 0.87, e.clientX);
}}, {{passive:false}});

canvas.addEventListener('mousedown', e => {{
  isDrag = true;
  dragStartX = e.clientX;
  dragStartOffset = offsetX;
  canvas.classList.add('grabbing');
}});
window.addEventListener('mousemove', e => {{
  if (!isDrag) return;
  offsetX = dragStartOffset + (e.clientX - dragStartX);
  draw();
}});
window.addEventListener('mouseup', () => {{
  isDrag = false;
  canvas.classList.remove('grabbing');
}});

// touch support
canvas.addEventListener('touchstart', e => {{
  if (e.touches.length === 1) {{
    isDrag = true;
    dragStartX = e.touches[0].clientX;
    dragStartOffset = offsetX;
  }}
}}, {{passive:true}});
canvas.addEventListener('touchmove', e => {{
  if (isDrag && e.touches.length === 1) {{
    offsetX = dragStartOffset + (e.touches[0].clientX - dragStartX);
    draw();
  }}
}}, {{passive:true}});
canvas.addEventListener('touchend', () => {{ isDrag = false; }});

window.addEventListener('resize', resize);
resize();
</script></body></html>"""

    components.html(logo_html, height=logo_height + 60, scrolling=False)

    # Color legend
    st.markdown("""
<div style="margin-top:8px;font-size:0.7rem;display:flex;gap:14px;flex-wrap:wrap;font-family:'IBM Plex Mono',monospace">
  <span><b style="color:#4e9af1">■</b> Básicos (K,R,H)</span>
  <span><b style="color:#f07070">■</b> Ácidos (D,E)</span>
  <span><b style="color:#56c596">■</b> Polares (S,T,N,Q)</span>
  <span><b style="color:#4fc3f7">■</b> Hidrofóbicos (A,V,L,I)</span>
  <span><b style="color:#b39ddb">■</b> Aromáticos (F,W,Y)</span>
  <span><b style="color:#f5a623">■</b> Sulfurados (C,M)</span>
  <span><b style="color:#657a8e">■</b> Especiales (G,P)</span>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# TAB 3 — MSA VIEWER + SECONDARY STRUCTURE
# ══════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<p class="sh">Alineamiento + scores + estructura secundaria</p>', unsafe_allow_html=True)

    st.markdown("""
<div class="info-box">
<b>Izquierda:</b> numeración en el MSA (incluyendo gaps). 
<b>pos_ref:</b> numeración de la secuencia de referencia (primera secuencia, sin contar gaps). 
<b>pos_pdb:</b> numeración en tu PDB = pos_ref + offset configurado.
</div>
""", unsafe_allow_html=True)

    # construir tabla MSA completa
    msa_all = [str(r.seq).upper() for r in st.session_state['msa_records']]
    ref_seq = msa_all[0]

    msa_rows = []
    pos_ref  = 0
    for i, res in enumerate(ref_seq):
        if res != '-':
            pos_ref += 1
        pos_pdb = pos_ref + pdb_offset if res != '-' else None
        ss = ss_map.get(pos_pdb, '—') if pos_pdb and ss_map else '—'
        row = {
            'pos_msa': i+1,
            'pos_ref': pos_ref if res != '-' else '—',
            'pos_pdb': pos_pdb if pos_pdb else '—',
            'ref_res': res,
            'SS': ss,
        }
        # agregar scores
        if res != '-':
            match = df[df['pos_ref']==pos_ref]
            for sc in score_cols:
                row[sc] = float(match[sc].values[0]) if not match.empty else None
        else:
            for sc in score_cols: row[sc] = None
        # consensus (most common non-gap aa)
        col = [seq[i] for seq in msa_all]
        aa_counts = Counter(aa for aa in col if aa != '-')
        row['consensus'] = aa_counts.most_common(1)[0][0] if aa_counts else '-'
        msa_rows.append(row)

    df_msa = pd.DataFrame(msa_rows)

    # Estructura secundaria lineal sobre toda la ref
    if ss_map:
        src_label = ss_source or "desconocida"
        is_pred   = 'Chou-Fasman' in src_label or 'predicc' in src_label.lower()
        badge_color = '#f5a623' if is_pred else '#56c596'
        badge_icon  = '⚠ Predicción' if is_pred else '✓ Experimental'
        st.markdown('<p class="sh">Estructura secundaria lineal (numeración ref)</p>', unsafe_allow_html=True)
        st.markdown(
            f'<span style="background:{badge_color}22;border:1px solid {badge_color};'
            f'border-radius:4px;padding:2px 8px;font-size:0.68rem;color:{badge_color};">'
            f'{badge_icon} — {src_label}</span>',
            unsafe_allow_html=True
        )
        if is_pred:
            st.caption("⚠ Predicción heurística Chou-Fasman. Subí un PDB para estructura experimental.")
        ss_svg = build_ss_bar_html(df, ss_map, pdb_offset)
        st.markdown(
            f'<div style="overflow-x:auto;border:1px solid #2a3444;border-radius:6px">{ss_svg}</div>',
            unsafe_allow_html=True
        )
        st.markdown("""
<div style="display:flex;gap:16px;font-size:0.7rem;margin-top:6px;font-family:'IBM Plex Mono',monospace">
  <span><span style="display:inline-block;width:12px;height:12px;border-radius:2px;background:#4fc3f7;vertical-align:middle;margin-right:4px"></span>Hélice α</span>
  <span><span style="display:inline-block;width:12px;height:12px;border-radius:2px;background:#56c596;vertical-align:middle;margin-right:4px"></span>Lámina β</span>
  <span><span style="display:inline-block;width:12px;height:12px;border-radius:2px;background:#2a3444;vertical-align:middle;margin-right:4px"></span>Loop / coil</span>
</div>""", unsafe_allow_html=True)

    st.markdown('<p class="sh">Tabla MSA completa</p>', unsafe_allow_html=True)
    # paginado para no saturar
    page_msa = st.number_input("Página de la tabla (100 filas/pág)", min_value=1,
                                max_value=max(1, math.ceil(len(df_msa)/100)), value=1, step=1)
    slice_s, slice_e = (page_msa-1)*100, page_msa*100
    st.dataframe(df_msa.iloc[slice_s:slice_e], use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════
# TAB 4 — TABLA
# ══════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<p class="sh">Dataset de scores (solo posiciones no-gap en ref)</p>', unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        st.download_button("⬇ Descargar CSV", csv_buf.getvalue(),
                           "protein_conservation_scores.csv", "text/csv")
    with c2:
        xl_buf = io.BytesIO()
        with pd.ExcelWriter(xl_buf, engine='openpyxl') as w:
            df.to_excel(w, index=False, sheet_name='Conservation')
        st.download_button("⬇ Descargar Excel", xl_buf.getvalue(),
                           "protein_conservation_scores.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# ══════════════════════════════════════════════════════
# TAB 5 — ESTRUCTURA 3D
# ══════════════════════════════════════════════════════
with tabs[4]:
    if not pdb_text:
        st.markdown("""
<div class="info-box">
Subí un archivo <b>.pdb</b> en el sidebar para activar la visualización 3D y el mapeo a estructura.
</div>""", unsafe_allow_html=True)
    else:
        score_for_3d = st.selectbox("Métrica para colorear", options=score_cols)

        st.markdown('<p class="sh">Visor 3D interactivo (py3Dmol)</p>', unsafe_allow_html=True)
        st.markdown("""<div class="pdb-note">
Coloreado: <b style="color:#0000ff">azul</b> = baja conservación → <b style="color:#ffffff">blanco</b> = media → <b style="color:#ff0000">rojo</b> = alta conservación.<br>
Usá los botones para cambiar la representación. El offset de numeración aplicado es <b>""" + str(pdb_offset) + """</b>.
</div>""", unsafe_allow_html=True)

        viewer_html = py3dmol_viewer(pdb_text, df, score_for_3d, pdb_offset)
        components.html(viewer_html, height=500, scrolling=False)

        st.markdown('<p class="sh">Descargar scripts</p>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            pymol_scr = generate_pymol_script(df, score_for_3d, pdb_name_inp, pdb_offset)
            st.download_button("⬇ Script PyMOL (.pml)", pymol_scr,
                               f"conservation_{score_for_3d.replace(' ','_')}.pml", "text/plain")
        with c2:
            chim_scr = generate_chimerax_script(df, score_for_3d, pdb_name_inp, pdb_offset)
            st.download_button("⬇ Script ChimeraX (.cxc)", chim_scr,
                               f"conservation_{score_for_3d.replace(' ','_')}.cxc", "text/plain")


# ══════════════════════════════════════════════════════
# TAB 6 — ALGORITMOS (con fórmulas LaTeX via MathJax)
# ══════════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<p class="sh">Descripción de los algoritmos implementados</p>', unsafe_allow_html=True)

    METRICS_LATEX = {
        'Jensen-Shannon': {
            'ref': 'Capra &amp; Singh, <i>Bioinformatics</i> 2007',
            'latex': r'JSD(P,Q) = \frac{1}{2} D_{KL}(P \| M) + \frac{1}{2} D_{KL}(Q \| M), \quad M = \frac{P+Q}{2}',
            'desc': (
                'Divergencia simétrica entre la distribución observada <b>P</b> '
                'y la distribución de fondo BLOSUM62 <b>Q</b>. '
                'Acotada en [0,1] — resuelve la asimetría de KL. '
                'Score alto = columna diverge del "ruido evolutivo" = conservación funcional. '
                'Mejor métrica práctica para sitios catalíticos (benchmark CSA, &gt;50% de verdaderos positivos a FPR=2%).'
            ),
        },
        'Shannon Entropy': {
            'ref': 'Shannon, <i>Bell System Tech. J.</i> 1948',
            'latex': r'H = -\sum_{i=1}^{20} p_i \log_2 p_i \quad \Rightarrow \quad \text{score} = 1 - \frac{H}{H_{\max}}',
            'desc': (
                'Mide la variabilidad pura de una columna. '
                'score = 1 cuando un solo residuo ocupa todas las posiciones. '
                '<b>Limitación:</b> asume todos los aminoácidos equiprobables — no captura la "rareza evolutiva" '
                'de conservar un residuo intrínsecamente poco frecuente como Trp o Cys.'
            ),
        },
        'Property Entropy': {
            'ref': 'Williamson 1995; grupos Taylor 1986',
            'latex': r'H_{\text{prop}} = -\sum_{g=1}^{9} f_g \log f_g, \quad f_g = \sum_{a \in g} p_a \quad \Rightarrow \quad \text{score} = 1 - \frac{H_{\text{prop}}}{H_{\max}}',
            'desc': (
                'Igual que Shannon pero sobre <b>9 grupos fisicoquímicos</b> '
                '(alifáticos, aromáticos, polares, básicos, ácidos, hidroxílicos, amídicos, Gly/Ala, Pro, Cys). '
                'Val→Leu cuenta como conservado dentro del grupo hidrofóbico. '
                'Detecta conservación funcional aunque cambien los residuos específicos.'
            ),
        },
        'Kullback-Leibler': {
            'ref': 'Wang &amp; Samudrala, <i>Bioinformatics</i> 2006',
            'latex': r'D_{KL}(P \| Q) = \sum_{i=1}^{20} p_i \log_2 \frac{p_i}{q_i}',
            'desc': (
                'Entropía Relativa: mide la ganancia de información al conocer P vs asumir Q (fondo BLOSUM62). '
                'Asigna mayor score a residuos raros conservados (Trp, His, Met). '
                '<b>Limitación matemática:</b> asimétrica (D<sub>KL</sub>(P‖Q) ≠ D<sub>KL</sub>(Q‖P)) y no acotada — se normaliza a [0,1].'
            ),
        },
        'Valdar01': {
            'ref': 'Valdar &amp; Thornton, <i>Proteins</i> 2001',
            'latex': r'C = \frac{\sum_{i<j} w_i w_j \cdot S_{aa}(r_i, r_j)}{\sum_{i<j} w_i w_j \cdot \max(S_{aa})}',
            'desc': (
                'Sum-of-Pairs ponderado (Henikoff weights) con BLOSUM62. '
                'Evalúa la similitud fisicoquímica entre todos los pares de residuos en la columna. '
                'Método por defecto de SnapGene. Robusto y estable (correlación &gt;0.95 con otros métodos matriciales), '
                'pero tiende a sobre-puntuar el núcleo hidrofóbico estructural vs sitios catalíticos puros.'
            ),
        },
        'SMERFS': {
            'ref': 'Manning et al., <i>Bioinformatics</i> 2008',
            'latex': r'\text{SMERFS}(i) = \frac{1}{w} \sum_{k=i-\lfloor w/2 \rfloor}^{i+\lfloor w/2 \rfloor} \text{JSD}(k), \quad w=7',
            'desc': (
                'Suavizado con <b>ventana deslizante w=7</b> sobre JSD. '
                'Los sitios funcionales no son aislados: están rodeados de restricciones estructurales locales. '
                'Superior en ROC para interfaces proteína-proteína y bolsas de ligandos, '
                'sin el costo computacional de métodos filogenéticos como Rate4Site.'
            ),
        },
    }

    algo_cards_html = ""
    for mname, meta in METRICS_LATEX.items():
        latex_escaped = meta['latex'].replace('\\', '\\\\').replace('`', '\\`')
        algo_cards_html += f"""
<div class="ac" id="ac-{mname.replace(' ','-')}">
  <div class="ac-head" onclick="toggle('{mname.replace(' ','-')}')">
    <span class="ac-name">{mname}</span>
    <span class="ac-ref">{meta['ref']}</span>
    <span class="ac-arrow" id="arr-{mname.replace(' ','-')}">▸</span>
  </div>
  <div class="ac-body" id="body-{mname.replace(' ','-')}" style="display:none">
    <div class="formula" id="f-{mname.replace(' ','-')}">\\({meta['latex']}\\)</div>
    <div class="desc">{meta['desc']}</div>
  </div>
</div>"""

    algo_html = f"""<!DOCTYPE html><html><head>
<script>
  window.MathJax = {{
    tex: {{inlineMath: [['\\\\(','\\\\)']]}},
    options: {{skipHtmlTags: ['script','noscript','style','textarea']}},
    startup: {{typeset: false}}
  }};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js" id="MathJax-script" async></script>
<style>
  body {{margin:0;padding:8px;background:#0b0f14;font-family:'IBM Plex Mono',monospace;}}
  .ac {{border:1px solid #2a3444;border-radius:6px;margin-bottom:8px;overflow:hidden;}}
  .ac-head {{display:flex;align-items:center;gap:12px;padding:12px 16px;
             background:#131820;cursor:pointer;transition:background .15s;}}
  .ac-head:hover {{background:#1c2330;}}
  .ac-name {{color:#4fc3f7;font-weight:700;font-size:0.88rem;flex:0 0 auto;}}
  .ac-ref  {{color:#657a8e;font-size:0.68rem;flex:1;}}
  .ac-arrow{{color:#657a8e;font-size:0.8rem;flex:0 0 auto;transition:transform .2s;}}
  .ac-body {{padding:14px 18px 16px;background:#0b0f14;border-top:1px solid #2a3444;}}
  .formula {{background:#131820;border-left:3px solid #4fc3f7;border-radius:4px;
             padding:12px 16px;margin-bottom:12px;color:#f5a623;
             font-size:1rem;overflow-x:auto;line-height:1.8;}}
  .desc    {{color:#8fa8bc;font-size:0.78rem;line-height:1.65;}}
  .desc b  {{color:#d8e4f0;}}
  .ref-box {{margin-top:16px;background:#131820;border:1px solid #2a3444;border-left:3px solid #4fc3f7;
             border-radius:6px;padding:12px 16px;font-size:0.75rem;color:#657a8e;line-height:1.6;}}
  .ref-box b {{color:#d8e4f0;}}
  .ref-box i {{color:#4fc3f7;}}
</style></head><body>
{algo_cards_html}
<div class="ref-box">
  <b>Referencia clave:</b> Capra &amp; Singh (2007), <i>Bioinformatics</i> 23(15):1875–1882.<br>
  Benchmark sobre el Catalytic Site Atlas (CSA). JSD con ventana supera a todos los métodos matriciales
  en identificación de sitios catalíticos a igualdad de costo computacional.
</div>
<script>
function toggle(id) {{
  const body = document.getElementById('body-'+id);
  const arr  = document.getElementById('arr-'+id);
  if (body.style.display === 'none') {{
    body.style.display = 'block';
    arr.textContent = '▾';
    arr.style.transform = 'rotate(0deg)';
    // trigger MathJax typeset on this element
    if (window.MathJax && MathJax.typesetPromise) {{
      MathJax.typesetPromise([body]);
    }}
  }} else {{
    body.style.display = 'none';
    arr.textContent = '▸';
  }}
}}
</script>
</body></html>"""

    components.html(algo_html, height=480, scrolling=True)
