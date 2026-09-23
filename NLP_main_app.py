# ============================================================
# EAFIT — Maestría en Ciencia de Datos
# NLP & LLM Interactive Lab
# Prof. Jorge Iván Padilla-Buriticá | linkedin.com/in/jipadilla
# ============================================================
# Módulos cubiertos:
#   1. Tokenización  (BPE, WordPiece, NLTK, spaCy)
#   2. Embeddings    (TF-IDF, Word2Vec, Sentence-BERT, similitud coseno)
#   3. NLP Clásico   (POS, NER, n-gramas, sentimientos)
#   4. LLM Lab       (temperatura, top-p, top-k, max_tokens, stop, system prompt)
#   5. Comparador    (múltiples modelos Groq, misma query, tabla de resultados)
#   6. Tokenómetro   (contador de tokens BPE interactivo)
#   7. Attention Viz (heat-map conceptual de atención)
#   8. Benchmark     (latencia, tokens/s, calidad subjetiva)
# ============================================================

import os
import re
import time
import json
import math
import hashlib
import textwrap
from collections import Counter
from io import StringIO

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Optional imports with graceful fallback ──────────────────
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.util import ngrams
    from nltk.corpus import stopwords
    from nltk import pos_tag, ne_chunk
    from nltk.sentiment import SentimentIntensityAnalyzer
    # Silent download of required NLTK data
    for pkg in ['punkt', 'averaged_perceptron_tagger', 'maxent_ne_chunker',
                'words', 'stopwords', 'vader_lexicon', 'punkt_tab',
                'averaged_perceptron_tagger_eng']:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.decomposition import TruncatedSVD
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False

try:
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer
    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False

try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

# ════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="NLP & LLM Lab — EAFIT",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "NLP & LLM Interactive Lab — EAFIT Maestría en Ciencia de Datos\nProf. Jorge Iván Padilla-Buriticá"
    }
)

# ════════════════════════════════════════════════════════════
# CUSTOM CSS — Dark Academic / Research Lab aesthetic
# ════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600&family=Crimson+Pro:ital,wght@0,300;0,400;0,600;1,400&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:         #0d1117;
    --surface:    #161b22;
    --surface2:   #1c2333;
    --border:     #30363d;
    --accent:     #f0883e;
    --accent2:    #58a6ff;
    --accent3:    #3fb950;
    --danger:     #f85149;
    --text:       #e6edf3;
    --text-muted: #8b949e;
    --teal:       #39d353;
    --purple:     #bc8cff;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

/* Hide default Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem; max-width: 100%; }

/* Top header bar */
.lab-header {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #1a1f2e 100%);
    border-bottom: 2px solid var(--accent);
    padding: 1.2rem 2rem;
    margin: -1rem -1rem 1.5rem -1rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.lab-header h1 {
    font-family: 'Crimson Pro', serif;
    font-size: 1.8rem;
    font-weight: 600;
    color: var(--text);
    margin: 0;
    letter-spacing: -0.02em;
}
.lab-header .subtitle {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: var(--accent);
    margin-top: 0.15rem;
}
.badge {
    background: var(--accent);
    color: #000;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    font-weight: 600;
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    white-space: nowrap;
}

/* Section headers */
.section-title {
    font-family: 'Crimson Pro', serif;
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--text);
    border-left: 4px solid var(--accent);
    padding-left: 0.8rem;
    margin: 1.5rem 0 1rem 0;
}

/* Info boxes */
.info-box {
    background: var(--surface2);
    border: 1px solid var(--accent2);
    border-left: 4px solid var(--accent2);
    border-radius: 6px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.88rem;
}
.warn-box {
    background: #2d1f00;
    border: 1px solid var(--accent);
    border-left: 4px solid var(--accent);
    border-radius: 6px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.88rem;
}
.success-box {
    background: #0d2818;
    border: 1px solid var(--accent3);
    border-left: 4px solid var(--accent3);
    border-radius: 6px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.88rem;
}
.formula-box {
    background: #0d1117;
    border: 1px solid var(--border);
    border-left: 4px solid var(--purple);
    border-radius: 6px;
    padding: 0.8rem 1.2rem;
    margin: 0.5rem 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: var(--purple);
}
.code-snippet {
    background: #0d1117;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.8rem 1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: var(--accent3);
    white-space: pre-wrap;
    overflow-x: auto;
}

/* Metric cards */
.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: var(--accent); }
.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.8rem;
    font-weight: 600;
    color: var(--accent);
}
.metric-label {
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-top: 0.2rem;
}

/* Token chips */
.token-container { display: flex; flex-wrap: wrap; gap: 4px; margin: 0.5rem 0; }
.token-chip {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    padding: 2px 8px;
    border-radius: 4px;
    border: 1px solid;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stTextInput label {
    color: var(--text-muted);
    font-size: 0.82rem;
    font-family: 'JetBrains Mono', monospace;
}

/* Streamlit overrides */
.stTextArea textarea {
    background: var(--surface2) !important;
    color: var(--text) !important;
    border-color: var(--border) !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stButton > button {
    background: var(--accent) !important;
    color: #000 !important;
    border: none !important;
    font-weight: 600 !important;
    border-radius: 6px !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stButton > button:hover {
    background: #e07830 !important;
    transform: translateY(-1px);
}
div[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: var(--text-muted);
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}
.stDataFrame { border: 1px solid var(--border) !important; border-radius: 6px; }

/* Response output area */
.llm-response {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.2rem;
    font-size: 0.92rem;
    line-height: 1.7;
    white-space: pre-wrap;
    max-height: 500px;
    overflow-y: auto;
}

/* Comparison table */
.compare-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
    font-family: 'DM Sans', sans-serif;
}
.compare-table th {
    background: var(--surface2);
    color: var(--accent);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    padding: 0.6rem 0.8rem;
    border: 1px solid var(--border);
    text-align: left;
}
.compare-table td {
    padding: 0.6rem 0.8rem;
    border: 1px solid var(--border);
    vertical-align: top;
}
.compare-table tr:nth-child(even) td { background: var(--surface); }

/* Progress / attention cells */
.attn-cell {
    display: inline-block;
    padding: 2px 6px;
    border-radius: 3px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    margin: 2px;
}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# CONSTANTS & MODEL CATALOG
# ════════════════════════════════════════════════════════════

GROQ_MODELS = {
    # ── Production Models ──
    "llama-3.3-70b-versatile": {
        "family": "LLaMA 3.3", "params": "70B", "context": 128_000,
        "type": "Decoder-only", "license": "Meta Community",
        "strengths": "Razonamiento general, instrucciones complejas, multilingual",
        "color": "#f0883e"
    },
    "llama-3.1-8b-instant": {
        "family": "LLaMA 3.1", "params": "8B", "context": 128_000,
        "type": "Decoder-only", "license": "Meta Community",
        "strengths": "Velocidad máxima, bajo costo, buen rendimiento general",
        "color": "#58a6ff"
    },
    "llama-3.1-70b-versatile": {
        "family": "LLaMA 3.1", "params": "70B", "context": 128_000,
        "type": "Decoder-only", "license": "Meta Community",
        "strengths": "Razonamiento avanzado, contexto largo, instrucciones",
        "color": "#3fb950"
    },
    "mixtral-8x7b-32768": {
        "family": "Mixtral MoE", "params": "8x7B (47B total, 13B activos)", "context": 32_768,
        "type": "Decoder MoE", "license": "Apache 2.0",
        "strengths": "Mixture of Experts: 8 expertos, activa 2 por token. Eficiencia vs calidad",
        "color": "#bc8cff"
    },
    "gemma2-9b-it": {
        "family": "Gemma 2", "params": "9B", "context": 8_192,
        "type": "Decoder-only", "license": "Gemma Terms",
        "strengths": "Google architecture, eficiente, bueno en instrucciones cortas",
        "color": "#39d353"
    },
    "llama-guard-3-8b": {
        "family": "LLaMA Guard", "params": "8B", "context": 8_192,
        "type": "Clasificador de seguridad", "license": "Meta Community",
        "strengths": "Clasificación de contenido dañino, guardrails de seguridad",
        "color": "#f85149"
    },
    # ── Preview / Experimental ──
    "deepseek-r1-distill-llama-70b": {
        "family": "DeepSeek R1 Distill", "params": "70B", "context": 128_000,
        "type": "Reasoning (CoT interno)", "license": "MIT",
        "strengths": "Razonamiento extendido, matemáticas, cadenas de pensamiento visibles",
        "color": "#ffa657"
    },
    "qwen-qwq-32b": {
        "family": "Qwen QwQ", "params": "32B", "context": 128_000,
        "type": "Reasoning", "license": "Apache 2.0",
        "strengths": "Razonamiento matemático y lógico, Alibaba Cloud",
        "color": "#d2a8ff"
    },
}

# Default sample texts for each module
SAMPLE_TEXTS = {
    "es": """La inteligencia artificial está transformando profundamente la educación superior en Colombia. 
Las universidades como EAFIT están adoptando modelos de lenguaje grande para personalizar el aprendizaje 
y automatizar la evaluación formativa. Sin embargo, los docentes señalan que el pensamiento crítico 
y la creatividad humana siguen siendo irreemplazables. El Ministerio de Educación analiza marcos 
regulatorios para garantizar el uso ético de estas tecnologías en el aula.""",
    
    "en": """Large language models have revolutionized natural language processing by demonstrating 
emergent capabilities that arise at scale. Models like GPT-4, Claude, and LLaMA can perform 
complex reasoning, generate code, and engage in nuanced conversations without task-specific training. 
The key architectural innovation — the Transformer's self-attention mechanism — allows each token 
to attend to all other tokens simultaneously, enabling parallelization impossible with recurrent networks.""",
    
    "mixed": """El Transformer architecture introduced by Vaswani et al. (2017) propone que 
"Attention is all you need." Esta arquitectura utiliza self-attention con matrices Q, K, V 
para calcular: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V. 
Los LLMs como LLaMA-3.1 tienen 70B parámetros y contextos de 128k tokens."""
}

# ════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="⚙️ Cargando modelo de embeddings…")
def load_sbert_model(model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
    """Load Sentence-BERT model (cached)."""
    if not SBERT_AVAILABLE:
        return None
    try:
        return SentenceTransformer(model_name)
    except Exception as e:
        st.warning(f"No se pudo cargar SBERT: {e}")
        return None

@st.cache_resource(show_spinner="⚙️ Cargando tokenizador BERT…")
def load_bert_tokenizer(model_name: str = "bert-base-multilingual-cased"):
    if not TOKENIZERS_AVAILABLE:
        return None
    try:
        return AutoTokenizer.from_pretrained(model_name)
    except Exception:
        return None

@st.cache_resource(show_spinner="⚙️ Cargando tokenizador GPT-2…")
def load_gpt2_tokenizer():
    if not TOKENIZERS_AVAILABLE:
        return None
    try:
        return AutoTokenizer.from_pretrained("gpt2")
    except Exception:
        return None

def count_tokens_tiktoken(text: str, model: str = "gpt-3.5-turbo") -> int:
    if not TIKTOKEN_AVAILABLE:
        return len(text.split())
    try:
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except Exception:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            return len(text.split())

def tokenize_with_tiktoken(text: str) -> list:
    if not TIKTOKEN_AVAILABLE:
        return text.split()
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        token_ids = enc.encode(text)
        return [enc.decode([t]) for t in token_ids]
    except Exception:
        return text.split()

def get_groq_client(api_key: str):
    if not GROQ_AVAILABLE:
        return None
    try:
        return Groq(api_key=api_key)
    except Exception:
        return None

def call_groq(
    client,
    model: str,
    messages: list,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    top_p: float = 1.0,
    frequency_penalty: float = 0.0,   # NOTE: Groq ignores this (OpenAI-only); kept for UI parity
    presence_penalty: float = 0.0,    # NOTE: Groq ignores this (OpenAI-only); kept for UI parity
    stop: list = None,
    stream: bool = False,
    seed: int = None,
) -> dict:
    """
    Wrapper around Groq API with full parameter exposure.
    NOTE: Groq API does NOT support frequency_penalty / presence_penalty.
    Those params are silently dropped to avoid 400/401 errors.
    Returns dict with: content, usage, latency, model, params_used
    """
    if client is None:
        return {"success": False, "error": "Cliente Groq no inicializado. Verifica tu API Key.", "latency": 0}

    # Groq-supported params only (no frequency_penalty / presence_penalty)
    params = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "top_p": float(top_p),
    }
    if stop:
        params["stop"] = stop
    if seed is not None:
        params["seed"] = int(seed)

    t0 = time.perf_counter()
    try:
        response = client.chat.completions.create(**params)
        latency = time.perf_counter() - t0
        content = response.choices[0].message.content or ""
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
        tokens_per_sec = usage["completion_tokens"] / latency if latency > 0 else 0
        finish_reason = response.choices[0].finish_reason
        return {
            "success": True,
            "content": content,
            "usage": usage,
            "latency": latency,
            "tokens_per_sec": tokens_per_sec,
            "finish_reason": finish_reason,
            "model": model,
            "params": params,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "latency": time.perf_counter() - t0,
            "content": "",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }


def compute_tfidf_matrix(docs: list) -> tuple:
    """Compute TF-IDF matrix and return (matrix, feature_names, vectorizer)."""
    if not SKLEARN_AVAILABLE:
        return None, [], None
    vec = TfidfVectorizer(
        max_features=50,
        stop_words=None,
        ngram_range=(1, 2),
        min_df=1
    )
    try:
        X = vec.fit_transform(docs)
        return X, vec.get_feature_names_out(), vec
    except Exception:
        return None, [], None

def cosine_sim_matrix(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    normed = embeddings / norms
    return normed @ normed.T

def color_for_value(val: float, low_color=(255,100,100), high_color=(50,200,100)) -> str:
    """Return CSS rgba string interpolated between two colors."""
    val = max(0.0, min(1.0, float(val)))
    r = int(low_color[0] + val * (high_color[0] - low_color[0]))
    g = int(low_color[1] + val * (high_color[1] - low_color[1]))
    b = int(low_color[2] + val * (high_color[2] - low_color[2]))
    return f"rgba({r},{g},{b},{0.25 + val * 0.6})"

def token_color(idx: int, total: int) -> str:
    """Cycle through a palette of token chip colors."""
    colors = [
        "#f0883e", "#58a6ff", "#3fb950", "#bc8cff",
        "#ffa657", "#79c0ff", "#7ee787", "#d2a8ff",
        "#ff7b72", "#39d353",
    ]
    return colors[idx % len(colors)]

def render_token_chips(tokens: list, show_ids: bool = False) -> str:
    """Render tokens as colored HTML chips."""
    chips = []
    for i, tok in enumerate(tokens):
        color = token_color(i, len(tokens))
        display = repr(tok).strip("'") if tok.strip() == "" else tok
        chip = (
            f'<span class="token-chip" '
            f'style="background:{color}22;border-color:{color};color:{color}" '
            f'title="Token {i}: {repr(tok)}">'
            f'{display}</span>'
        )
        chips.append(chip)
    return f'<div class="token-container">{"".join(chips)}</div>'

def explain_parameter(param: str) -> str:
    """Return a pedagogical explanation for each LLM parameter."""
    explanations = {
        "temperature": """
**🌡️ Temperature** — Controla la *aleatoriedad* de la distribución de probabilidades.

- **Fórmula:** `P(wᵢ) = softmax(logits / T)ᵢ`
- **T → 0:** Determinístico. Siempre elige el token más probable (greedy). Respuestas repetibles y conservadoras.
- **T = 1:** Distribución original del modelo sin modificar.
- **T > 1:** Distribución más plana. Mayor diversidad, mayor riesgo de incoherencia.
- **Rango práctico:** 0.0–2.0. Para código/datos: 0.0–0.3. Para creatividad: 0.7–1.2.
""",
        "top_p": """
**🎯 Top-P (Nucleus Sampling)** — Muestrea solo del conjunto mínimo de tokens cuyas probabilidades acumuladas alcanzan P.

- **Algoritmo:** Ordena tokens por probabilidad ↓, suma hasta llegar a P, muestrea de ese núcleo.
- **top_p = 1.0:** Considera todos los tokens (sin filtro).
- **top_p = 0.9:** Considera el 90% más probable de la masa de probabilidad.
- **top_p = 0.1:** Solo los tokens más probables (muy conservador).
- **Interacción con temperature:** Se aplica DESPUÉS de escalar los logits. No usar ambos muy bajos simultáneamente.
""",
        "top_k": """
**🔢 Top-K** — Restringe el muestreo a los K tokens más probables.

- **Algoritmo:** Ordena tokens por probabilidad, descarta todos excepto los K primeros, renormaliza, muestrea.
- **top_k = 1:** Equivalente a greedy decoding (siempre el más probable).
- **top_k = 50:** Estándar de facto. Balance entre calidad y diversidad.
- **top_k = 0 / -1:** Deshabilitado (usa todos los tokens).
- **vs top_p:** top_k es fijo; top_p es adaptativo según la distribución real.
""",
        "max_tokens": """
**📏 Max Tokens** — Límite duro del número máximo de tokens en la respuesta.

- No afecta la *calidad*, solo la *longitud*.
- El modelo puede terminar antes (token EOS) o ser truncado aquí.
- **finish_reason = "length"**: fue truncado. **finish_reason = "stop"**: terminó naturalmente.
- **Costo:** Los tokens de output cuestan más que los de input en la mayoría de APIs.
- Regla: 1 token ≈ 0.75 palabras (EN) ≈ 0.5 palabras (ES).
""",
        "frequency_penalty": """
**🔁 Frequency Penalty** — Penaliza tokens según cuántas veces ya aparecieron en el output.

- **Fórmula:** `logit(t) -= frequency_penalty × count(t)`
- **0.0:** Sin penalización.
- **> 0:** Reduce repetición de tokens específicos.
- **> 2.0:** Puede forzar vocabulario inusual.
- Diferencia con presence_penalty: este penaliza *frecuencia acumulada*, no solo presencia.
""",
        "presence_penalty": """
**🆕 Presence Penalty** — Penaliza cualquier token que ya haya aparecido en el output (binario).

- **Fórmula:** `logit(t) -= presence_penalty × (1 if t appeared else 0)`
- **0.0:** Sin penalización.
- **> 0:** Incentiva al modelo a hablar de temas nuevos.
- Útil para: generación de contenido variado, evitar loops temáticos.
- Diferencia: no importa cuántas veces apareció, solo si apareció.
""",
        "seed": """
**🌱 Seed** — Semilla del generador de números aleatorios para reproducibilidad.

- Con el mismo seed, temperatura y parámetros: el output debería ser idéntico.
- **Limitación:** No todos los backends garantizan reproducibilidad perfecta (paralelismo de GPU).
- Útil para: experimentos reproducibles, debugging, comparaciones justas entre modelos.
- `seed=None`: Aleatoriedad total en cada llamada.
""",
        "stop": """
**🛑 Stop Sequences** — Lista de strings que detienen la generación cuando aparecen.

- El modelo genera hasta encontrar cualquiera de estos strings (o max_tokens).
- Ejemplos: `["\\n\\n", "###", "<|end|>", "Usuario:"]`
- Útil para: controlar el formato de output, implementar turn-taking en diálogos.
- El stop token NO se incluye en el output final.
""",
    }
    return explanations.get(param, f"Parámetro: **{param}**")

# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1rem 0 0.5rem 0;">
        <div style="font-family:'Crimson Pro',serif;font-size:1.3rem;color:#e6edf3;font-weight:600;">
            🧠 NLP & LLM Lab
        </div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#8b949e;margin-top:0.2rem;">
            EAFIT · Maestría Ciencia de Datos
        </div>
    </div>
    <hr style="border-color:#30363d;margin:0.5rem 0;">
    """, unsafe_allow_html=True)

    # API Key
    st.markdown('<p style="font-family:\'JetBrains Mono\',monospace;font-size:0.75rem;color:#8b949e;">🔑 GROQ API KEY</p>', unsafe_allow_html=True)
    
    api_key_env = os.environ.get("GROQ_API_KEY", "")
    api_key_input = st.text_input(
        "Groq API Key",
        value=api_key_env,
        type="password",
        placeholder="gsk_...",
        label_visibility="collapsed",
        help="Obtén tu clave gratis en console.groq.com"
    )
    api_key = api_key_input or api_key_env
    
    if api_key:
        st.markdown('<div class="success-box" style="font-size:0.75rem;">✅ API Key detectada</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="warn-box" style="font-size:0.75rem;">⚠️ Sin API Key — módulos LLM deshabilitados</div>', unsafe_allow_html=True)

    st.markdown('<hr style="border-color:#30363d;margin:0.8rem 0;">', unsafe_allow_html=True)
    
    # Navigation
    st.markdown('<p style="font-family:\'JetBrains Mono\',monospace;font-size:0.75rem;color:#8b949e;">📍 MÓDULO ACTIVO</p>', unsafe_allow_html=True)
    module = st.selectbox(
        "Módulo",
        options=[
            "🏠  Inicio & Teoría",
            "🔤  Tokenización",
            "📐  Embeddings & Similitud",
            "🏷️  NLP Clásico (POS, NER, Sentimientos)",
            "⚡  LLM Lab — Parámetros",
            "⚖️  Comparador de Modelos",
            "📊  Benchmark de Velocidad",
            "🎯  Attention Visualizer",
            "🧪  Playground Libre",
        ],
        label_visibility="collapsed"
    )

    st.markdown('<hr style="border-color:#30363d;margin:0.8rem 0;">', unsafe_allow_html=True)

    # Global model selector
    st.markdown('<p style="font-family:\'JetBrains Mono\',monospace;font-size:0.75rem;color:#8b949e;">🤖 MODELO ACTIVO</p>', unsafe_allow_html=True)
    selected_model = st.selectbox(
        "Modelo",
        options=list(GROQ_MODELS.keys()),
        format_func=lambda m: f"{GROQ_MODELS[m]['family']} ({GROQ_MODELS[m]['params']})",
        label_visibility="collapsed"
    )
    
    model_info = GROQ_MODELS[selected_model]
    st.markdown(f"""
    <div style="background:#161b22;border:1px solid #30363d;border-radius:6px;padding:0.6rem;margin-top:0.3rem;">
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#8b949e;">
            <b style="color:{model_info['color']};">■</b> {model_info['type']}<br>
            📦 {model_info['params']} params<br>
            📚 ctx: {model_info['context']:,} tokens<br>
            ⚖️ {model_info['license']}
        </div>
        <div style="font-size:0.7rem;color:#8b949e;margin-top:0.4rem;font-style:italic;">
            {model_info['strengths']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr style="border-color:#30363d;margin:0.8rem 0;">', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#30363d;text-align:center;">
        Prof. Jorge I. Padilla-Buriticá<br>
        <a href="https://www.linkedin.com/in/jipadilla" style="color:#58a6ff;">linkedin/jipadilla</a>
        · Oficina 19-603
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════
st.markdown("""
<div class="lab-header">
    <div>
        <h1>🧠 NLP & LLM Interactive Lab</h1>
        <div class="subtitle">EAFIT · Maestría en Ciencia de Datos · Prof. Jorge Iván Padilla-Buriticá</div>
    </div>
    <span class="badge">Groq API</span>
    <span class="badge" style="background:#58a6ff;">Python</span>
    <span class="badge" style="background:#3fb950;">Streamlit</span>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# MODULE 0: INICIO & TEORÍA
# ════════════════════════════════════════════════════════════
if module == "🏠  Inicio & Teoría":
    st.markdown('<div class="section-title">Mapa del Laboratorio</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">8</div>
            <div class="metric-label">Módulos interactivos</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value" style="color:#58a6ff;">8</div>
            <div class="metric-label">Modelos Groq disponibles</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value" style="color:#3fb950;">12+</div>
            <div class="metric-label">Parámetros configurables</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value" style="color:#bc8cff;">∞</div>
            <div class="metric-label">Grados de libertad</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    
    tab_theory, tab_models, tab_arch = st.tabs(
        ["📖 Teoría NLP", "🤖 Catálogo de Modelos", "🏗️ Arquitectura Transformer"]
    )
    
    with tab_theory:
        col_left, col_right = st.columns([1, 1])
        with col_left:
            st.markdown("""
            <div class="section-title" style="font-size:1.1rem;">Pipeline NLP Completo</div>
            
            <div class="info-box">
            <b>1. Tokenización</b><br>
            Convertir texto crudo en unidades mínimas (tokens). Algoritmos: BPE, WordPiece, SentencePiece, Unigram LM.
            </div>
            
            <div class="info-box">
            <b>2. Normalización</b><br>
            Minúsculas, eliminación de acentos, caracteres especiales, URLs, stopwords.
            </div>
            
            <div class="info-box">
            <b>3. Representación vectorial</b><br>
            BoW → TF-IDF → Word2Vec → Embeddings contextuales (BERT, GPT).
            </div>
            
            <div class="info-box">
            <b>4. Modelos de secuencia</b><br>
            n-gramas → HMM → CRF → RNN/LSTM → Transformer.
            </div>
            
            <div class="info-box">
            <b>5. Tareas downstream</b><br>
            Clasificación, NER, Sentiment, QA, Resumen, Traducción, Generación.
            </div>
            """, unsafe_allow_html=True)
            
        with col_right:
            st.markdown('<div class="section-title" style="font-size:1.1rem;">Ecuaciones Clave</div>', unsafe_allow_html=True)
            
            st.markdown("""
            <div class="formula-box">
            TF-IDF(t,d) = tf(t,d) × log(N / df(t))
            
            tf(t,d)  = f(t,d) / Σ f(k,d)    # frecuencia relativa
            idf(t)   = log(N / |{d: t∈d}|)   # rareza en corpus
            </div>
            
            <div class="formula-box">
            Similitud Coseno:
            cos(θ) = (u·v) / (‖u‖ · ‖v‖)  ∈ [-1, 1]
            </div>
            
            <div class="formula-box">
            Softmax con temperatura T:
            P(wᵢ) = exp(logitᵢ / T) / Σⱼ exp(logitⱼ / T)
            
            T→0: determinístico (greedy)
            T=1: distribución original
            T→∞: uniforme (caótico)
            </div>
            
            <div class="formula-box">
            Scaled Dot-Product Attention:
            A(Q,K,V) = softmax(QKᵀ / √dₖ) · V
            
            Q,K,V ∈ ℝⁿˣᵈₖ  (proyecciones aprendidas)
            √dₖ  = factor de escala (estabilidad numérica)
            </div>
            
            <div class="formula-box">
            BLEU Score:
            BLEU = BP · exp(Σ wₙ · log pₙ)
            
            BP = min(1, exp(1 - r/c))  # brevity penalty
            pₙ = precisión de n-gramas modificada
            </div>
            """, unsafe_allow_html=True)

    with tab_models:
        st.markdown('<div class="section-title" style="font-size:1.1rem;">Modelos Disponibles en Groq</div>', unsafe_allow_html=True)
        rows = []
        for m, info in GROQ_MODELS.items():
            rows.append({
                "Modelo": m,
                "Familia": info["family"],
                "Parámetros": info["params"],
                "Contexto (tokens)": f"{info['context']:,}",
                "Tipo": info["type"],
                "Licencia": info["license"],
                "Fortalezas": info["strengths"],
            })
        df_models = pd.DataFrame(rows)
        st.dataframe(df_models, use_container_width=True, height=320)
        
        st.markdown("""
        <div class="info-box">
        <b>💡 ¿Por qué Groq?</b><br>
        Groq usa hardware <b>LPU (Language Processing Unit)</b> especializado que logra velocidades de 
        inferencia 10–100x superiores a GPUs convencionales. Esto permite experimentar con múltiples 
        modelos en tiempo real sin esperas. Velocidades típicas: 200–800 tokens/segundo vs ~20–50 tok/s en GPUs estándar.
        </div>
        """, unsafe_allow_html=True)
        
        # Architecture comparison chart
        fig = go.Figure()
        model_names = [GROQ_MODELS[m]["family"] for m in list(GROQ_MODELS.keys())[:6]]
        params_raw = ["70", "8", "70", "47", "9", "70"]
        params_num = [float(p) for p in params_raw]
        contexts = [GROQ_MODELS[m]["context"] for m in list(GROQ_MODELS.keys())[:6]]
        colors = [GROQ_MODELS[m]["color"] for m in list(GROQ_MODELS.keys())[:6]]
        
        fig.add_trace(go.Scatter(
            x=params_num, y=[c/1000 for c in contexts],
            mode="markers+text",
            text=model_names,
            textposition="top center",
            marker=dict(size=[p/3+8 for p in params_num], color=colors, opacity=0.85,
                       line=dict(color="#30363d", width=1)),
            hovertemplate="<b>%{text}</b><br>Params: %{x}B<br>Contexto: %{y}k tokens<extra></extra>"
        ))
        fig.update_layout(
            title="Parámetros vs Contexto — Modelos Groq",
            xaxis_title="Parámetros (B)", yaxis_title="Contexto (k tokens)",
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            font=dict(color="#e6edf3", family="DM Sans"),
            title_font=dict(size=14),
        )
        fig.update_xaxes(gridcolor="#30363d"); fig.update_yaxes(gridcolor="#30363d")
        st.plotly_chart(fig, use_container_width=True)

    with tab_arch:
        st.markdown("""
        <div class="section-title" style="font-size:1.1rem;">Arquitectura Transformer: Anatomía Completa</div>
        
        <div class="info-box">
        <b>Paper original:</b> "Attention Is All You Need" — Vaswani et al., Google Brain, 2017 (>120,000 citas)<br>
        <b>Revolución:</b> Eliminó la recurrencia (RNN/LSTM) y la convolución, usando solo mecanismos de atención.
        </div>
        """, unsafe_allow_html=True)
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
            <div class="warn-box">
            <b>🔴 Encoder (Solo-Encoder = BERT)</b><br>
            • Input Embedding + Positional Encoding<br>
            • Multi-Head Self-Attention (bidireccional)<br>
            • Add & Layer Norm (conexión residual)<br>
            • Feed Forward Network (MLP posición a posición)<br>
            • Add & Layer Norm<br>
            <br>
            → Objetivo: MLM (Masked Language Modeling)<br>
            → Uso: comprensión, clasificación, NER<br>
            → Ejemplos: BERT, RoBERTa, DeBERTa
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="formula-box">
            Positional Encoding (sinusoidal):
            PE(pos, 2i)   = sin(pos / 10000^(2i/d))
            PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
            
            Propiedades:
            • Único por posición
            • Distancias relativas consistentes
            • Generaliza a secuencias nuevas
            • Sin parámetros adicionales
            </div>
            """, unsafe_allow_html=True)
            
        with col_b:
            st.markdown("""
            <div class="success-box">
            <b>🟢 Decoder (Solo-Decoder = GPT)</b><br>
            • Output Embedding + Positional Encoding<br>
            • Masked Multi-Head Self-Attention (causal)<br>
            • Add & Layer Norm<br>
            • Cross-Attention (en Encoder-Decoder)<br>
            • Add & Layer Norm<br>
            • Feed Forward Network<br>
            • Add & Layer Norm<br>
            • Linear + Softmax → distribución sobre vocab<br>
            <br>
            → Objetivo: CLM (Causal Language Modeling)<br>
            → Uso: generación, diálogo, código<br>
            → Ejemplos: GPT-4, Claude, LLaMA, Mistral
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="formula-box">
            Multi-Head Attention:
            MHA(Q,K,V) = Concat(h₁,...,hₕ) Wᴼ
            
            hᵢ = Attention(Q·WᵢQ, K·WᵢK, V·WᵢV)
            
            GPT-3:  h=96 heads, d=12288, dₖ=128
            LLaMA: h=32 heads, d=4096,  dₖ=128
            BERT:   h=12 heads, d=768,   dₖ=64
            
            Feed Forward:
            FFN(x) = GELU(xW₁ + b₁)W₂ + b₂
            d_ff = 4 × d_model  (típico)
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# MODULE 1: TOKENIZACIÓN
# ════════════════════════════════════════════════════════════
elif module == "🔤  Tokenización":
    st.markdown('<div class="section-title">Tokenización: Del Texto a los IDs</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <b>¿Qué es un token?</b> La unidad mínima que procesa un LLM. Puede ser un carácter, 
    una subpalabra, una palabra o un símbolo especial. El vocabulario de GPT-4 tiene ~100k tokens; 
    LLaMA 3 tiene ~128k. <b>Los modelos nunca ven texto crudo: solo ven IDs enteros.</b>
    </div>
    """, unsafe_allow_html=True)

    input_text = st.text_area(
        "Texto de entrada",
        value=SAMPLE_TEXTS["mixed"],
        height=120,
        help="Escribe cualquier texto. Prueba con código, números, emojis, palabras raras."
    )

    tab_bpe, tab_bert, tab_nltk, tab_compare = st.tabs(
        ["🔧 BPE / GPT-2 (tiktoken)", "🤗 BERT (WordPiece)", "📚 NLTK (word/sent)", "📊 Comparativa"]
    )

    with tab_bpe:
        col_explain, col_result = st.columns([1, 1])
        with col_explain:
            st.markdown("""
            <div class="section-title" style="font-size:1rem;">Byte-Pair Encoding (BPE)</div>
            <div class="formula-box">
            Algoritmo BPE:
            1. Inicializa vocabulario con caracteres únicos
            2. Cuenta frecuencia de cada par adyacente
            3. Fusiona el par más frecuente → nuevo token
            4. Repite hasta vocabulario de tamaño K
            
            GPT-2/3/4 usan BPE con:
            • Vocab: ~50,256 tokens (GPT-2) / ~100,277 (GPT-4)
            • Encoding: cl100k_base (GPT-3.5+/GPT-4)
            • Nunca rompe palabras si caben enteras
            
            Eficiencia:
            1 token ≈ 0.75 palabras (EN)
            1 token ≈ 0.50 palabras (ES/FR/DE)
            </div>
            """, unsafe_allow_html=True)
            
        with col_result:
            if TIKTOKEN_AVAILABLE:
                tokens_bpe = tokenize_with_tiktoken(input_text)
                n_tokens = len(tokens_bpe)
                n_chars = len(input_text)
                n_words = len(input_text.split())
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Tokens", n_tokens)
                c2.metric("Palabras", n_words)
                c3.metric("Ratio tok/word", f"{n_tokens/max(n_words,1):.2f}")
                
                st.markdown("**Visualización de tokens BPE:**")
                st.markdown(render_token_chips(tokens_bpe[:80]), unsafe_allow_html=True)
                if len(tokens_bpe) > 80:
                    st.caption(f"... mostrando primeros 80 de {n_tokens} tokens")
                
                # Token length distribution
                token_lengths = [len(t) for t in tokens_bpe]
                fig_dist = px.histogram(
                    x=token_lengths, nbins=20,
                    title="Distribución de longitud de tokens",
                    labels={"x": "Caracteres por token", "y": "Frecuencia"},
                    color_discrete_sequence=["#f0883e"]
                )
                fig_dist.update_layout(
                    paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                    font=dict(color="#e6edf3"), title_font=dict(size=12), height=200
                )
                fig_dist.update_xaxes(gridcolor="#30363d")
                fig_dist.update_yaxes(gridcolor="#30363d")
                st.plotly_chart(fig_dist, use_container_width=True)
            else:
                st.warning("tiktoken no disponible. Instala: `pip install tiktoken`")

    with tab_bert:
        col_explain, col_result = st.columns([1, 1])
        with col_explain:
            st.markdown("""
            <div class="section-title" style="font-size:1rem;">WordPiece (BERT)</div>
            <div class="formula-box">
            Algoritmo WordPiece:
            
            Objetivo: maximizar la log-probabilidad
            de los datos de entrenamiento dado el vocab.
            
            Fusiona pares que maximizan:
            score(a,b) = freq(ab) / (freq(a) × freq(b))
            
            Diferencia vs BPE:
            • BPE: fusiona por frecuencia absoluta
            • WordPiece: fusiona por ganancia relativa
            
            Prefijo "##": token es continuación
            Ejemplo:
            "transformación" →
            ['trans', '##form', '##aci', '##ón']
            
            Vocab BERT-multilingual: 119,547 tokens
            </div>
            """, unsafe_allow_html=True)
            
        with col_result:
            bert_tok = load_bert_tokenizer()
            if bert_tok:
                try:
                    encoding = bert_tok(input_text, return_tensors=None, truncation=True, max_length=512)
                    tokens_bert = bert_tok.convert_ids_to_tokens(encoding["input_ids"])
                    ids_bert = encoding["input_ids"]
                    
                    c1, c2 = st.columns(2)
                    c1.metric("Tokens BERT", len(tokens_bert))
                    c2.metric("Incluye [CLS]/[SEP]", "Sí")
                    
                    st.markdown("**Tokens WordPiece:**")
                    st.markdown(render_token_chips(tokens_bert[:80]), unsafe_allow_html=True)
                    
                    # Show token IDs table
                    with st.expander("Ver IDs de tokens"):
                        df_toks = pd.DataFrame({
                            "Posición": range(len(tokens_bert[:30])),
                            "Token": tokens_bert[:30],
                            "ID": ids_bert[:30],
                            "¿Subpalabra?": ["Sí" if t.startswith("##") else "No" for t in tokens_bert[:30]]
                        })
                        st.dataframe(df_toks, use_container_width=True)
                except Exception as e:
                    st.error(f"Error tokenizando: {e}")
            else:
                st.info("Cargando tokenizador BERT-multilingual... (requiere descarga inicial)")

    with tab_nltk:
        if NLTK_AVAILABLE:
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Word Tokenization (NLTK)**")
                try:
                    word_tokens = word_tokenize(input_text, language='spanish')
                    st.markdown(render_token_chips(word_tokens[:60]), unsafe_allow_html=True)
                    st.metric("Tokens de palabras", len(word_tokens))
                except Exception:
                    word_tokens = input_text.split()
                    st.markdown(render_token_chips(word_tokens[:60]), unsafe_allow_html=True)
                
                st.markdown("**Sentence Tokenization**")
                try:
                    sent_tokens = sent_tokenize(input_text, language='spanish')
                    for i, sent in enumerate(sent_tokens):
                        st.markdown(f"""
                        <div style="background:#161b22;border:1px solid #30363d;border-radius:4px;
                                    padding:0.4rem 0.7rem;margin:0.3rem 0;font-size:0.82rem;">
                            <span style="color:#8b949e;font-family:'JetBrains Mono',monospace;font-size:0.7rem;">S{i+1}</span>
                            {sent}
                        </div>""", unsafe_allow_html=True)
                    st.metric("Oraciones detectadas", len(sent_tokens))
                except Exception as e:
                    st.error(str(e))
                    
            with col_b:
                st.markdown("**N-gramas del texto**")
                n_gram_n = st.slider("N para n-grama", 1, 5, 2)
                try:
                    word_tokens_clean = [w.lower() for w in word_tokenize(input_text) 
                                        if w.isalpha() and len(w) > 1]
                    grams = list(ngrams(word_tokens_clean, n_gram_n))
                    gram_freq = Counter(grams).most_common(15)
                    
                    if gram_freq:
                        df_ngrams = pd.DataFrame(
                            [((" ".join(g)), c) for g, c in gram_freq],
                            columns=["N-grama", "Frecuencia"]
                        )
                        fig_ng = px.bar(
                            df_ngrams, x="Frecuencia", y="N-grama",
                            orientation="h",
                            title=f"Top 15 {n_gram_n}-gramas",
                            color_discrete_sequence=["#58a6ff"]
                        )
                        fig_ng.update_layout(
                            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                            font=dict(color="#e6edf3"), height=350,
                            title_font=dict(size=12), yaxis=dict(autorange="reversed")
                        )
                        fig_ng.update_xaxes(gridcolor="#30363d")
                        st.plotly_chart(fig_ng, use_container_width=True)
                except Exception as e:
                    st.error(f"Error generando n-gramas: {e}")
        else:
            st.warning("NLTK no disponible. Instala con: `pip install nltk`")

    with tab_compare:
        st.markdown('<div class="section-title" style="font-size:1rem;">Comparativa de Tokenizadores</div>', unsafe_allow_html=True)
        
        results = {}
        
        # BPE
        if TIKTOKEN_AVAILABLE:
            toks = tokenize_with_tiktoken(input_text)
            results["BPE (GPT/cl100k)"] = {
                "tokens": len(toks),
                "example": " | ".join(toks[:10]) + "...",
                "vocab_size": "~100,277",
                "algo": "Byte-Pair Encoding",
                "used_by": "GPT-3.5, GPT-4, LLaMA 3"
            }
        
        # BERT WordPiece
        bt = load_bert_tokenizer()
        if bt:
            try:
                enc = bt(input_text, truncation=True, max_length=512)
                toks_b = bt.convert_ids_to_tokens(enc["input_ids"])
                results["WordPiece (BERT-multilingual)"] = {
                    "tokens": len(toks_b),
                    "example": " | ".join(toks_b[:10]) + "...",
                    "vocab_size": "119,547",
                    "algo": "WordPiece (max likelihood)",
                    "used_by": "BERT, DistilBERT, RoBERTa"
                }
            except Exception:
                pass
        
        # GPT-2
        gpt2_tok = load_gpt2_tokenizer()
        if gpt2_tok:
            try:
                enc2 = gpt2_tok(input_text, truncation=True, max_length=512)
                toks_g = gpt2_tok.convert_ids_to_tokens(enc2["input_ids"])
                results["BPE (GPT-2 original)"] = {
                    "tokens": len(toks_g),
                    "example": " | ".join(str(t) for t in toks_g[:10]) + "...",
                    "vocab_size": "50,257",
                    "algo": "BPE (byte-level)",
                    "used_by": "GPT-2, original OPT"
                }
            except Exception:
                pass
        
        # NLTK
        if NLTK_AVAILABLE:
            try:
                wt = word_tokenize(input_text)
                results["NLTK word_tokenize"] = {
                    "tokens": len(wt),
                    "example": " | ".join(wt[:10]) + "...",
                    "vocab_size": "Vocabulario abierto",
                    "algo": "Punkt (basado en reglas)",
                    "used_by": "Investigación, NLP clásico"
                }
            except Exception:
                pass

        # Simple whitespace
        ws_toks = input_text.split()
        results["Whitespace split (baseline)"] = {
            "tokens": len(ws_toks),
            "example": " | ".join(ws_toks[:10]) + "...",
            "vocab_size": "Vocabulario abierto",
            "algo": "Separación por espacios",
            "used_by": "Baseline, bag-of-words"
        }
        
        if results:
            df_compare = pd.DataFrame(results).T.reset_index()
            df_compare.columns = ["Tokenizador", "# Tokens", "Ejemplo (primeros 10)", "Vocab", "Algoritmo", "Usado por"]
            st.dataframe(df_compare, use_container_width=True)
            
            # Bar chart
            fig_tok = px.bar(
                df_compare, x="Tokenizador", y="# Tokens",
                title="Número de tokens según tokenizador",
                color="# Tokens",
                color_continuous_scale=["#58a6ff", "#f0883e", "#f85149"]
            )
            fig_tok.update_layout(
                paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                font=dict(color="#e6edf3"), height=300, showlegend=False
            )
            fig_tok.update_xaxes(gridcolor="#30363d", tickangle=-30)
            fig_tok.update_yaxes(gridcolor="#30363d")
            st.plotly_chart(fig_tok, use_container_width=True)
        
        st.markdown("""
        <div class="warn-box">
        <b>⚠️ Observación clave:</b> El mismo texto produce cantidades de tokens muy distintas según el algoritmo. 
        Esto afecta directamente el <b>costo</b> de las APIs (cobran por token), 
        los <b>límites de contexto</b> y la <b>eficiencia de la memoria</b> en inferencia.
        El español genera ~30–50% más tokens que el inglés equivalente en modelos entrenados principalmente en inglés.
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# MODULE 2: EMBEDDINGS
# ════════════════════════════════════════════════════════════
elif module == "📐  Embeddings & Similitud":
    st.markdown('<div class="section-title">Embeddings y Similitud Semántica</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    Los <b>embeddings</b> son representaciones vectoriales densas en ℝᵈ donde la distancia geométrica 
    captura proximidad semántica. "banco financiero" y "institución crediticia" deben tener vectores 
    cercanos. Este es el salto fundamental de BoW/TF-IDF a representaciones distribuidas.
    </div>
    """, unsafe_allow_html=True)
    
    tab_tfidf, tab_sbert, tab_viz = st.tabs(["📊 TF-IDF", "🤗 Sentence-BERT", "🗺️ Visualización 2D"])
    
    with tab_tfidf:
        st.markdown('<div class="section-title" style="font-size:1rem;">TF-IDF: Representación Dispersa</div>', unsafe_allow_html=True)
        
        col_params, col_results = st.columns([1, 2])
        with col_params:
            st.markdown("**Documentos de prueba:**")
            default_docs = [
                "La inteligencia artificial transforma la educación universitaria.",
                "Los modelos de lenguaje aprenden patrones en texto masivo.",
                "El fútbol colombiano tiene talento mundial en sus jugadores.",
                "BERT y GPT son arquitecturas Transformer para NLP.",
                "La economía colombiana creció un 3% en el último trimestre.",
                "Deep learning revolucionó el reconocimiento de imágenes y texto.",
            ]
            docs_input = st.text_area(
                "Un documento por línea:",
                value="\n".join(default_docs),
                height=180,
                help="Cada línea es un documento separado"
            )
            documents = [d.strip() for d in docs_input.split("\n") if d.strip()]
            
            tfidf_max_feat = st.slider("Max features", 10, 100, 30)
            tfidf_ngram_max = st.radio("N-gramas", [1, 2], index=1, horizontal=True)
            
        with col_results:
            if SKLEARN_AVAILABLE and len(documents) >= 2:
                try:
                    vec = TfidfVectorizer(
                        max_features=tfidf_max_feat,
                        ngram_range=(1, tfidf_ngram_max),
                        min_df=1
                    )
                    X = vec.fit_transform(documents)
                    feature_names = vec.get_feature_names_out()
                    
                    # TF-IDF heatmap
                    X_dense = X.toarray()
                    doc_labels = [f"Doc {i+1}: {d[:25]}..." for i, d in enumerate(documents)]
                    
                    # Top features per doc
                    top_features_idx = np.argsort(X_dense.sum(axis=0))[-min(20, len(feature_names)):]
                    X_top = X_dense[:, top_features_idx]
                    feat_top = feature_names[top_features_idx]
                    
                    fig_heat = px.imshow(
                        X_top,
                        x=feat_top,
                        y=[f"D{i+1}" for i in range(len(documents))],
                        title="Matriz TF-IDF (top términos)",
                        color_continuous_scale="Blues",
                        aspect="auto"
                    )
                    fig_heat.update_layout(
                        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                        font=dict(color="#e6edf3", size=10),
                        height=250, title_font=dict(size=12)
                    )
                    st.plotly_chart(fig_heat, use_container_width=True)
                    
                    # Cosine similarity between documents
                    sim_matrix = cosine_similarity(X)
                    fig_sim = px.imshow(
                        sim_matrix,
                        x=[f"D{i+1}" for i in range(len(documents))],
                        y=[f"D{i+1}" for i in range(len(documents))],
                        title="Similitud Coseno entre documentos (TF-IDF)",
                        color_continuous_scale="RdYlGn",
                        zmin=0, zmax=1, text_auto=".2f"
                    )
                    fig_sim.update_layout(
                        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                        font=dict(color="#e6edf3", size=10),
                        height=280, title_font=dict(size=12)
                    )
                    st.plotly_chart(fig_sim, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.warning("Necesitas al menos 2 documentos y sklearn instalado.")

    with tab_sbert:
        st.markdown('<div class="section-title" style="font-size:1rem;">Sentence-BERT: Embeddings Contextuales</div>', unsafe_allow_html=True)
        
        sbert_model_name = st.selectbox(
            "Modelo de embedding:",
            [
                "paraphrase-multilingual-MiniLM-L12-v2",
                "all-MiniLM-L6-v2",
                "all-mpnet-base-v2",
            ],
            help="MiniLM-L12-v2: multilingüe, 117M params, 384 dims. all-mpnet: inglés, 109M params, 768 dims."
        )
        
        default_sents = [
            "El banco aprobó el crédito hipotecario.",
            "La institución financiera otorgó el préstamo.",
            "Me senté en el banco del parque a leer.",
            "La inteligencia artificial cambia el mundo.",
            "El partido de fútbol terminó en empate.",
            "Los modelos de lenguaje comprenden el texto.",
        ]
        
        sents_input = st.text_area(
            "Frases (una por línea):",
            value="\n".join(default_sents),
            height=160,
            help="Prueba con paráfrasis, sinónimos, textos en diferentes idiomas"
        )
        sentences = [s.strip() for s in sents_input.split("\n") if s.strip()]
        
        if st.button("⚡ Calcular Embeddings", key="sbert_btn") and len(sentences) >= 2:
            sbert = load_sbert_model(sbert_model_name)
            if sbert:
                with st.spinner("Calculando embeddings..."):
                    t0 = time.perf_counter()
                    embeddings = sbert.encode(sentences, show_progress_bar=False)
                    elapsed = time.perf_counter() - t0
                
                st.success(f"✅ {len(sentences)} embeddings de {embeddings.shape[1]} dimensiones en {elapsed:.3f}s")
                
                # Cosine similarity matrix
                sim = cosine_sim_matrix(embeddings)
                
                fig_csim = px.imshow(
                    sim,
                    x=[f"S{i+1}" for i in range(len(sentences))],
                    y=[f"S{i+1}" for i in range(len(sentences))],
                    title="Similitud Coseno — Sentence-BERT",
                    color_continuous_scale="RdYlGn",
                    zmin=-1, zmax=1, text_auto=".3f",
                    hover_data={"Frases": [s[:40] for s in sentences]}
                )
                fig_csim.update_layout(
                    paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                    font=dict(color="#e6edf3"), height=350, title_font=dict(size=13)
                )
                st.plotly_chart(fig_csim, use_container_width=True)
                
                # Legend
                for i, s in enumerate(sentences):
                    st.markdown(f"""
                    <span style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;
                                 color:#f0883e;background:#1c2333;padding:2px 6px;
                                 border-radius:3px;">S{i+1}</span>
                    <span style="font-size:0.85rem;"> {s}</span><br>
                    """, unsafe_allow_html=True)
                
                # Ranking: most similar pairs
                pairs = []
                for i in range(len(sentences)):
                    for j in range(i+1, len(sentences)):
                        pairs.append({
                            "Frase A": f"S{i+1}: {sentences[i][:45]}...",
                            "Frase B": f"S{j+1}: {sentences[j][:45]}...",
                            "Similitud": round(float(sim[i,j]), 4)
                        })
                df_pairs = pd.DataFrame(pairs).sort_values("Similitud", ascending=False)
                st.markdown("**Ranking de pares por similitud:**")
                st.dataframe(df_pairs, use_container_width=True, height=220)
                
                st.markdown("""
                <div class="warn-box">
                <b>💡 Observa:</b> S1 y S2 ("banco crédito" y "préstamo") deberían tener alta similitud.
                S1 y S3 ("banco parque") deberían tener baja similitud a pesar de usar la misma palabra "banco".
                Esto demuestra que los embeddings contextuales resuelven la polisemia.
                </div>
                """, unsafe_allow_html=True)

    with tab_viz:
        st.markdown('<div class="section-title" style="font-size:1rem;">Proyección 2D de Embeddings (PCA / t-SNE conceptual)</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        Los embeddings viven en espacios de alta dimensión (384, 768, 4096 dims). Para visualizarlos, 
        proyectamos a 2D usando SVD (similar a PCA). La separación en clusters indica que el modelo 
        captura distintas áreas semánticas.
        </div>
        """, unsafe_allow_html=True)
        
        thematic_sentences = {
            "IA/NLP": [
                "Los transformers revolucionaron el procesamiento de lenguaje.",
                "BERT usa atención bidireccional para embeddings contextuales.",
                "GPT-4 genera texto coherente con 1.8 billones de parámetros.",
            ],
            "Economía": [
                "La inflación colombiana bajó al 5.2% en diciembre.",
                "El Banco de la República mantuvo tasas de interés.",
                "El PIB creció impulsado por las exportaciones de café.",
            ],
            "Deportes": [
                "Colombia clasificó al mundial con una victoria contundente.",
                "El equipo de fútbol entrenó en la altitud de Bogotá.",
                "James Rodríguez marcó un golazo en el último partido.",
            ],
        }
        
        all_sents = []
        labels = []
        colors_list = []
        color_map = {"IA/NLP": "#f0883e", "Economía": "#58a6ff", "Deportes": "#3fb950"}
        
        for cat, sents in thematic_sentences.items():
            all_sents.extend(sents)
            labels.extend([cat] * len(sents))
        
        if st.button("🗺️ Visualizar espacio semántico", key="viz_btn"):
            sbert = load_sbert_model()
            if sbert and SKLEARN_AVAILABLE:
                with st.spinner("Calculando embeddings y proyección..."):
                    embs = sbert.encode(all_sents)
                    svd = TruncatedSVD(n_components=2, random_state=42)
                    coords = svd.fit_transform(embs)
                
                df_viz = pd.DataFrame({
                    "x": coords[:, 0],
                    "y": coords[:, 1],
                    "Categoría": labels,
                    "Texto": [s[:50] + "..." for s in all_sents]
                })
                
                fig_viz = px.scatter(
                    df_viz, x="x", y="y",
                    color="Categoría",
                    text="Texto",
                    title="Proyección 2D de embeddings por categoría semántica",
                    color_discrete_map=color_map,
                    hover_data={"Texto": True, "x": False, "y": False}
                )
                fig_viz.update_traces(
                    textposition="top center",
                    textfont=dict(size=8),
                    marker=dict(size=12, opacity=0.85)
                )
                fig_viz.update_layout(
                    paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                    font=dict(color="#e6edf3"), height=450, title_font=dict(size=13)
                )
                fig_viz.update_xaxes(gridcolor="#30363d")
                fig_viz.update_yaxes(gridcolor="#30363d")
                st.plotly_chart(fig_viz, use_container_width=True)
                
                st.markdown("""
                <div class="success-box">
                ✅ Observa cómo las frases de la misma categoría se agrupan en el espacio 2D.
                Los embeddings capturan la semántica sin ninguna supervisión explícita de categorías.
                Esta capacidad es la base del RAG, la búsqueda semántica y la clasificación zero-shot.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("Necesitas sentence-transformers y sklearn instalados.")


# ════════════════════════════════════════════════════════════
# MODULE 3: NLP CLÁSICO
# ════════════════════════════════════════════════════════════
elif module == "🏷️  NLP Clásico (POS, NER, Sentimientos)":
    st.markdown('<div class="section-title">NLP Clásico: POS, NER, Sentimientos y Estadísticas</div>', unsafe_allow_html=True)
    
    text_nlp = st.text_area(
        "Texto para analizar:",
        value=SAMPLE_TEXTS["es"],
        height=140
    )
    
    if not NLTK_AVAILABLE:
        st.error("NLTK no disponible. Instala con: `pip install nltk`")
        st.stop()
    
    tab_pos, tab_ner, tab_sent, tab_stats = st.tabs(
        ["🏷️ POS Tagging", "👤 NER", "😊 Sentimientos", "📈 Estadísticas de Texto"]
    )
    
    with tab_pos:
        st.markdown("""
        <div class="info-box">
        <b>Part-of-Speech Tagging</b> asigna una categoría gramatical a cada token. 
        Es fundamental para análisis sintáctico, extracción de información y comprensión semántica.
        </div>
        <div class="formula-box">
        Etiquetas POS (Penn Treebank):
        NN=Noun  VB=Verb  JJ=Adjective  RB=Adverb  
        DT=Determiner  IN=Preposition  CC=Conjunction
        PRP=Pronoun  NNP=Proper Noun  CD=Cardinal number
        </div>
        """, unsafe_allow_html=True)
        
        try:
            tokens_pos = word_tokenize(text_nlp)
            pos_tags = pos_tag(tokens_pos)
            
            # Color coding per POS
            pos_colors = {
                'NN': '#58a6ff', 'NNS': '#58a6ff', 'NNP': '#bc8cff', 'NNPS': '#bc8cff',
                'VB': '#3fb950', 'VBD': '#3fb950', 'VBG': '#39d353', 'VBN': '#39d353',
                'JJ': '#f0883e', 'JJR': '#f0883e', 'JJS': '#f0883e',
                'RB': '#ffa657', 'RBR': '#ffa657',
                'DT': '#8b949e', 'IN': '#8b949e', 'CC': '#8b949e',
                'PRP': '#d2a8ff', 'PRP$': '#d2a8ff',
                'CD': '#ff7b72',
            }
            
            chips = []
            for word, tag in pos_tags[:60]:
                color = pos_colors.get(tag, "#6e7681")
                chips.append(
                    f'<span class="token-chip" style="background:{color}22;border-color:{color};color:{color}" '
                    f'title="{tag}">{word}<sub style="font-size:0.6em;">{tag}</sub></span>'
                )
            st.markdown(f'<div class="token-container">{"".join(chips)}</div>', unsafe_allow_html=True)
            
            # POS frequency chart
            pos_freq = Counter([tag for _, tag in pos_tags]).most_common(12)
            df_pos = pd.DataFrame(pos_freq, columns=["POS", "Frecuencia"])
            fig_pos = px.bar(
                df_pos, x="POS", y="Frecuencia",
                title="Distribución de categorías gramaticales",
                color="Frecuencia",
                color_continuous_scale=["#161b22", "#58a6ff", "#f0883e"]
            )
            fig_pos.update_layout(
                paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                font=dict(color="#e6edf3"), height=280, showlegend=False
            )
            fig_pos.update_xaxes(gridcolor="#30363d")
            fig_pos.update_yaxes(gridcolor="#30363d")
            st.plotly_chart(fig_pos, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error en POS tagging: {e}")

    with tab_ner:
        st.markdown("""
        <div class="info-box">
        <b>Named Entity Recognition (NER)</b> identifica y clasifica entidades nombradas: 
        personas (PER), organizaciones (ORG), lugares (LOC), fechas, valores monetarios, etc.
        Fundamental para extracción de información y construcción de knowledge graphs.
        </div>
        """, unsafe_allow_html=True)
        
        try:
            tokens_ner = word_tokenize(text_nlp)
            pos_tags_ner = pos_tag(tokens_ner)
            tree = ne_chunk(pos_tags_ner)
            
            entities = []
            for subtree in tree:
                if hasattr(subtree, 'label'):
                    entity_text = " ".join([w for w, t in subtree.leaves()])
                    entities.append({"Entidad": entity_text, "Tipo": subtree.label()})
            
            if entities:
                ner_colors = {
                    "PERSON": "#bc8cff", "ORGANIZATION": "#58a6ff",
                    "GPE": "#3fb950", "FACILITY": "#f0883e",
                    "LOCATION": "#39d353", "GSP": "#ffa657"
                }
                
                st.markdown("**Entidades detectadas:**")
                chips_ner = []
                for e in entities:
                    color = ner_colors.get(e["Tipo"], "#8b949e")
                    chips_ner.append(
                        f'<span class="token-chip" style="background:{color}33;border-color:{color};color:{color};font-size:0.85rem;">'
                        f'{e["Entidad"]} <sub>{e["Tipo"]}</sub></span>'
                    )
                st.markdown(f'<div class="token-container">{"".join(chips_ner)}</div>', unsafe_allow_html=True)
                
                df_ner = pd.DataFrame(entities)
                st.dataframe(df_ner, use_container_width=True, height=200)
                
                ner_type_freq = Counter([e["Tipo"] for e in entities])
                fig_ner = px.pie(
                    values=list(ner_type_freq.values()),
                    names=list(ner_type_freq.keys()),
                    title="Distribución de tipos de entidades",
                    color_discrete_sequence=list(ner_colors.values())
                )
                fig_ner.update_layout(
                    paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                    font=dict(color="#e6edf3"), height=280
                )
                st.plotly_chart(fig_ner, use_container_width=True)
            else:
                st.info("No se detectaron entidades nombradas con NLTK. Prueba con texto en inglés o usa spaCy para mejor soporte en español.")
                
            st.markdown("""
            <div class="warn-box">
            <b>💡 Nota pedagógica:</b> NLTK NER funciona mejor en inglés. Para español, usar 
            <b>spaCy + es_core_news_lg</b> o <b>BERT fine-tuned para NER en español</b> 
            (e.g. PlanTL-GOB-ES/roberta-base-bne-capitel-ner).
            </div>
            """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Error en NER: {e}")

    with tab_sent:
        st.markdown("""
        <div class="info-box">
        <b>Análisis de Sentimientos</b>: VADER (Valence Aware Dictionary and sEntiment Reasoner) 
        es un lexicón con reglas que funciona bien en inglés para redes sociales. 
        Retorna scores: Positivo, Negativo, Neutro y Compound (síntesis -1 a +1).
        </div>
        """, unsafe_allow_html=True)
        
        test_sentences = [
            "I love this product, it is absolutely amazing!",
            "This is the worst experience I have ever had.",
            "The weather today is okay, nothing special.",
            "Artificial intelligence is transforming education in incredible ways.",
            "I'm not sure if I like this feature or not.",
            "Despite some issues, the overall experience was positive.",
        ]
        
        custom_sent = st.text_area(
            "Frases para analizar (una por línea, inglés funciona mejor):",
            value="\n".join(test_sentences),
            height=150
        )
        
        sent_list = [s.strip() for s in custom_sent.split("\n") if s.strip()]
        
        try:
            sia = SentimentIntensityAnalyzer()
            results_sent = []
            for sentence in sent_list:
                scores = sia.polarity_scores(sentence)
                sentiment = "Positivo" if scores["compound"] >= 0.05 else \
                           "Negativo" if scores["compound"] <= -0.05 else "Neutro"
                results_sent.append({
                    "Texto": sentence[:60] + ("..." if len(sentence) > 60 else ""),
                    "Positivo": round(scores["pos"], 3),
                    "Negativo": round(scores["neg"], 3),
                    "Neutro": round(scores["neu"], 3),
                    "Compound": round(scores["compound"], 3),
                    "Sentimiento": sentiment
                })
            
            df_sent = pd.DataFrame(results_sent)
            st.dataframe(df_sent, use_container_width=True, height=250)
            
            # Sentiment bar chart
            fig_sent = go.Figure()
            fig_sent.add_trace(go.Bar(
                name="Positivo", x=df_sent["Texto"].str[:30], y=df_sent["Positivo"],
                marker_color="#3fb950"
            ))
            fig_sent.add_trace(go.Bar(
                name="Negativo", x=df_sent["Texto"].str[:30], y=df_sent["Negativo"],
                marker_color="#f85149"
            ))
            fig_sent.add_trace(go.Scatter(
                name="Compound", x=df_sent["Texto"].str[:30], y=df_sent["Compound"],
                mode="lines+markers", marker=dict(color="#f0883e", size=8),
                line=dict(color="#f0883e", width=2), yaxis="y2"
            ))
            fig_sent.update_layout(
                title="Análisis de Sentimientos VADER",
                barmode="group",
                paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                font=dict(color="#e6edf3"), height=350,
                xaxis=dict(tickangle=-30, gridcolor="#30363d"),
                yaxis=dict(title="Score 0–1", gridcolor="#30363d"),
                yaxis2=dict(title="Compound -1/+1", overlaying="y", side="right",
                           gridcolor="#30363d"),
                legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1)
            )
            st.plotly_chart(fig_sent, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error en análisis de sentimientos: {e}")

    with tab_stats:
        st.markdown('<div class="section-title" style="font-size:1rem;">Estadísticas Lingüísticas del Texto</div>', unsafe_allow_html=True)
        
        try:
            tokens_all = word_tokenize(text_nlp)
            words_only = [w.lower() for w in tokens_all if w.isalpha()]
            
            try:
                stop_words_es = set(stopwords.words('spanish'))
                stop_words_en = set(stopwords.words('english'))
                stop_all = stop_words_es | stop_words_en
            except Exception:
                stop_all = set()
            
            content_words = [w for w in words_only if w not in stop_all]
            
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            col_s1.metric("Total tokens", len(tokens_all))
            col_s2.metric("Palabras únicas", len(set(words_only)))
            col_s3.metric("Riqueza léxica", f"{len(set(words_only))/max(len(words_only),1):.3f}")
            col_s4.metric("Palabras de contenido", len(content_words))
            
            # Word frequency
            freq = Counter(content_words).most_common(20)
            df_freq = pd.DataFrame(freq, columns=["Palabra", "Frecuencia"])
            
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                fig_freq = px.bar(
                    df_freq.head(15), x="Frecuencia", y="Palabra",
                    orientation="h", title="Top 15 palabras de contenido",
                    color_discrete_sequence=["#58a6ff"]
                )
                fig_freq.update_layout(
                    paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                    font=dict(color="#e6edf3"), height=350,
                    yaxis=dict(autorange="reversed"), title_font=dict(size=12)
                )
                fig_freq.update_xaxes(gridcolor="#30363d")
                st.plotly_chart(fig_freq, use_container_width=True)
            
            with col_f2:
                if WORDCLOUD_AVAILABLE and content_words:
                    try:
                        text_for_wc = " ".join(content_words)
                        wc = WordCloud(
                            width=600, height=350,
                            background_color="#0d1117",
                            colormap="Blues",
                            max_words=60
                        ).generate(text_for_wc)
                        fig_wc, ax = plt.subplots(figsize=(6, 3.5))
                        ax.imshow(wc, interpolation='bilinear')
                        ax.axis('off')
                        fig_wc.patch.set_facecolor('#0d1117')
                        st.pyplot(fig_wc, use_container_width=True)
                        plt.close(fig_wc)
                    except Exception:
                        st.dataframe(df_freq, use_container_width=True)
                else:
                    st.dataframe(df_freq, use_container_width=True)
            
            # Character-level stats
            char_freq = Counter(text_nlp.lower())
            letters = {k: v for k, v in char_freq.items() if k.isalpha()}
            df_chars = pd.DataFrame(
                sorted(letters.items(), key=lambda x: -x[1])[:15],
                columns=["Carácter", "Frecuencia"]
            )
            fig_chars = px.bar(
                df_chars, x="Carácter", y="Frecuencia",
                title="Distribución de caracteres (Zipf's Law)",
                color_discrete_sequence=["#bc8cff"]
            )
            fig_chars.update_layout(
                paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                font=dict(color="#e6edf3"), height=250, title_font=dict(size=12)
            )
            fig_chars.update_xaxes(gridcolor="#30363d")
            fig_chars.update_yaxes(gridcolor="#30363d")
            st.plotly_chart(fig_chars, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error en estadísticas: {e}")


# ════════════════════════════════════════════════════════════
# MODULE 4: LLM LAB — PARÁMETROS
# ════════════════════════════════════════════════════════════
elif module == "⚡  LLM Lab — Parámetros":
    st.markdown('<div class="section-title">LLM Lab: Explorando Parámetros de Inferencia</div>', unsafe_allow_html=True)
    
    if not api_key:
        st.markdown("""
        <div class="warn-box">
        ⚠️ <b>Se requiere una Groq API Key</b> para este módulo. 
        Obtén una gratis en <a href="https://console.groq.com" style="color:#58a6ff;">console.groq.com</a> 
        e ingrésala en la barra lateral.
        </div>
        """, unsafe_allow_html=True)
        st.stop()
    
    client = get_groq_client(api_key)
    
    col_config, col_lab = st.columns([1, 2])
    
    with col_config:
        st.markdown("### ⚙️ Configuración de Parámetros")
        
        with st.expander("🌡️ Temperature", expanded=True):
            temperature = st.slider(
                "temperature", 0.0, 2.0, 0.7, 0.05,
                help="Controla aleatoriedad. 0=determinístico, 1=original, 2=caótico"
            )
            st.markdown(explain_parameter("temperature"), unsafe_allow_html=False)
        
        with st.expander("🎯 Top-P (Nucleus Sampling)"):
            top_p = st.slider(
                "top_p", 0.01, 1.0, 1.0, 0.01,
                help="Masa de probabilidad acumulada. 0.9 = núcleo del 90% más probable"
            )
            st.markdown(explain_parameter("top_p"))
        
        with st.expander("📏 Max Tokens"):
            max_tokens = st.slider(
                "max_tokens", 10, 4096, 512, 10,
                help="Límite duro de tokens en la respuesta"
            )
            st.markdown(explain_parameter("max_tokens"))
        
        with st.expander("🔁 Frequency & Presence Penalty (concepto — no soportado por Groq)"):
            st.markdown("""
            <div class="warn-box" style="font-size:0.8rem;">
            ⚠️ <b>Nota pedagógica:</b> <code>frequency_penalty</code> y <code>presence_penalty</code>
            son parámetros de la <b>API de OpenAI</b>. La API de Groq <b>no los soporta</b>
            (los rechaza con error 400/401). Los sliders son solo para aprender la teoría;
            los valores <b>NO se envían</b> en la llamada real a Groq.
            </div>
            """, unsafe_allow_html=True)
            freq_penalty = st.slider(
                "frequency_penalty (conceptual, no enviado)", -2.0, 2.0, 0.0, 0.1,
                help="SOLO EDUCATIVO — No enviado a Groq. Penaliza tokens según frecuencia"
            )
            pres_penalty = st.slider(
                "presence_penalty (conceptual, no enviado)", -2.0, 2.0, 0.0, 0.1,
                help="SOLO EDUCATIVO — No enviado a Groq. Penaliza presencia binaria"
            )
            st.markdown(explain_parameter("frequency_penalty"))
        
        with st.expander("🌱 Seed & Stop Sequences"):
            use_seed = st.checkbox("Usar seed fijo (reproducibilidad)", value=False)
            seed_val = st.number_input("Seed", 0, 999999, 42, disabled=not use_seed)
            
            stop_raw = st.text_input(
                "Stop sequences (separadas por coma):",
                placeholder='###, <end>, \\n\\n',
                help="El modelo para al encontrar cualquiera de estas cadenas"
            )
            stop_seqs = [s.strip().replace("\\n", "\n") for s in stop_raw.split(",") if s.strip()] or None
            st.markdown(explain_parameter("stop"))
        
        with st.expander("💬 System Prompt"):
            system_prompt = st.text_area(
                "System Prompt:",
                value="Eres un asistente experto en ciencia de datos y NLP. "
                      "Responde de forma precisa, concisa y pedagógica en español. "
                      "Cuando sea relevante, incluye ejemplos de código Python.",
                height=120,
                help="Instrucciones globales de comportamiento para el modelo"
            )
        
        # Summary box
        st.markdown(f"""
        <div style="background:#0d1117;border:1px solid #30363d;border-radius:6px;
                    padding:0.8rem;margin-top:0.5rem;font-family:'JetBrains Mono',monospace;
                    font-size:0.72rem;color:#8b949e;">
        <b style="color:#f0883e;">Configuración actual (enviada a Groq):</b><br>
        model: {selected_model}<br>
        temperature: {temperature}<br>
        top_p: {top_p}<br>
        max_tokens: {max_tokens}<br>
        seed: {seed_val if use_seed else "None"}<br>
        stop: {stop_seqs}<br>
        <span style="color:#856404;">freq_penalty/pres_penalty: conceptual (no enviado)</span>
        </div>
        """, unsafe_allow_html=True)

    with col_lab:
        st.markdown("### 💬 Prueba de Generación")
        
        # Preset queries
        preset_queries = {
            "Custom": "",
            "Explicar temperatura en LLMs": 
                "Explica con una analogía simple qué hace el parámetro temperature en un LLM y por qué values extremos son problemáticos.",
            "Comparar RNN vs Transformer":
                "Compara RNN/LSTM con la arquitectura Transformer en 5 puntos clave. ¿Por qué los Transformers ganaron?",
            "Código: calcular TF-IDF en Python":
                "Escribe un ejemplo de código Python usando sklearn para calcular TF-IDF de 5 documentos y encontrar los 3 términos más relevantes de cada uno.",
            "Generar historia creativa (probar temp alta)":
                "Escribe un párrafo de ciencia ficción donde una IA llamada EAFIT-GPT descubre que puede sentir emociones. Sé muy creativo.",
            "Pregunta factual (probar temp=0)":
                "¿Cuántos parámetros tiene BERT-base? ¿Y GPT-3? ¿Y LLaMA 3.1 70B? Sé preciso.",
            "Test de stop sequences":
                "Enumera 10 aplicaciones de NLP en empresas colombianas. Numera cada una así: 1. 2. 3. ...",
        }
        
        preset = st.selectbox("Queries de ejemplo:", list(preset_queries.keys()))
        
        user_query = st.text_area(
            "Tu query:",
            value=preset_queries[preset],
            height=120,
            placeholder="Escribe tu pregunta o instrucción al modelo...",
            key="llm_query"
        )
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
        run_btn = col_btn1.button("⚡ Generar", key="run_gen", use_container_width=True)
        run_x3 = col_btn2.button("🔄 Generar ×3", key="run_x3", use_container_width=True,
                                  help="Genera 3 respuestas con los mismos parámetros para ver variabilidad")
        
        if run_btn and user_query.strip():
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ]
            
            with st.spinner(f"⚡ Generando con {selected_model}..."):
                result = call_groq(
                    client=client,
                    model=selected_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    # frequency_penalty / presence_penalty NOT sent (Groq doesn't support them)
                    stop=stop_seqs,
                    seed=seed_val if use_seed else None,
                )
            
            if result["success"]:
                # Metrics
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("⏱️ Latencia", f"{result['latency']:.2f}s")
                c2.metric("⚡ Tokens/s", f"{result['tokens_per_sec']:.0f}")
                c3.metric("📤 Input tokens", result["usage"]["prompt_tokens"])
                c4.metric("📥 Output tokens", result["usage"]["completion_tokens"])
                
                finish_color = "#3fb950" if result["finish_reason"] == "stop" else "#f0883e"
                st.markdown(f"""
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;
                            color:{finish_color};margin-bottom:0.5rem;">
                finish_reason: {result['finish_reason']}
                {"✅ Terminó naturalmente" if result['finish_reason']=='stop' else "⚠️ Truncado por max_tokens"}
                </div>""", unsafe_allow_html=True)
                
                st.markdown(f'<div class="llm-response">{result["content"]}</div>', unsafe_allow_html=True)
                
                # Token visualization
                with st.expander("🔍 Ver tokens del output (BPE)"):
                    output_tokens = tokenize_with_tiktoken(result["content"])
                    st.markdown(render_token_chips(output_tokens[:100]), unsafe_allow_html=True)
                    st.caption(f"Total: {len(output_tokens)} tokens visualizados")
                    
            else:
                st.error(f"Error: {result.get('error', 'Unknown error')}")
        
        elif run_x3 and user_query.strip():
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ]
            
            responses = []
            with st.spinner("Generando 3 respuestas para comparar variabilidad..."):
                for i in range(3):
                    r = call_groq(
                        client=client,
                        model=selected_model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        stop=stop_seqs,
                    )
                    responses.append(r)
                    time.sleep(0.3)
            
            st.markdown(f"""
            <div class="info-box">
            <b>Variabilidad con temperature={temperature}</b><br>
            Con temperature=0, las 3 respuestas deberían ser idénticas o casi idénticas.<br>
            Con temperature alta (>0.8), espera diferencias sustanciales.
            </div>
            """, unsafe_allow_html=True)
            
            for i, r in enumerate(responses):
                if r["success"]:
                    with st.expander(f"Respuesta {i+1} | {r['latency']:.2f}s | {r['usage']['completion_tokens']} tokens"):
                        st.markdown(f'<div class="llm-response" style="max-height:200px;">{r["content"]}</div>', unsafe_allow_html=True)
            
            # Similarity between responses
            if SBERT_AVAILABLE and all(r["success"] for r in responses):
                sbert = load_sbert_model()
                if sbert:
                    texts_to_compare = [r["content"] for r in responses]
                    embs = sbert.encode(texts_to_compare)
                    sim = cosine_sim_matrix(embs)
                    st.markdown(f"""
                    <div class="metric-card" style="text-align:left;">
                    <b>Similitud coseno entre respuestas:</b><br>
                    R1 vs R2: <b>{sim[0,1]:.3f}</b> | 
                    R1 vs R3: <b>{sim[0,2]:.3f}</b> | 
                    R2 vs R3: <b>{sim[1,2]:.3f}</b><br>
                    <span style="color:#8b949e;font-size:0.8rem;">
                    Rango: 0=completamente diferente, 1=idéntico semánticamente
                    </span>
                    </div>
                    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# MODULE 5: COMPARADOR DE MODELOS
# ════════════════════════════════════════════════════════════
elif module == "⚖️  Comparador de Modelos":
    st.markdown('<div class="section-title">Comparador de Modelos: Misma Query, Múltiples LLMs</div>', unsafe_allow_html=True)
    
    if not api_key:
        st.markdown('<div class="warn-box">⚠️ Se requiere Groq API Key.</div>', unsafe_allow_html=True)
        st.stop()
    
    client = get_groq_client(api_key)
    
    st.markdown("""
    <div class="info-box">
    Envía exactamente la misma query a múltiples modelos y compara: calidad de respuesta, latencia, 
    tokens por segundo, uso de tokens y costo estimado. 
    Esto permite entender el <b>trade-off calidad vs velocidad vs costo</b>.
    </div>
    """, unsafe_allow_html=True)
    
    # Models to compare
    models_to_compare = st.multiselect(
        "Selecciona modelos a comparar (máximo 4):",
        options=list(GROQ_MODELS.keys()),
        default=["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "mixtral-8x7b-32768"],
        format_func=lambda m: f"{GROQ_MODELS[m]['family']} ({GROQ_MODELS[m]['params']})",
        max_selections=4
    )
    
    # Shared parameters
    col_p1, col_p2, col_p3 = st.columns(3)
    cmp_temperature = col_p1.slider("temperature", 0.0, 2.0, 0.7, 0.1, key="cmp_t")
    cmp_max_tokens = col_p2.slider("max_tokens", 50, 2048, 400, 50, key="cmp_mt")
    cmp_top_p = col_p3.slider("top_p", 0.1, 1.0, 1.0, 0.1, key="cmp_tp")
    
    cmp_system = st.text_input(
        "System prompt (compartido):",
        value="Eres un asistente experto en IA y ciencia de datos. Responde en español, de forma clara y concisa.",
        key="cmp_sys"
    )
    
    preset_compare = {
        "Custom": "",
        "Razonamiento lógico":
            "Un tren sale de Medellín a las 8:00 AM a 120 km/h. Otro sale de Bogotá (420 km de distancia) a las 9:00 AM a 90 km/h. ¿A qué hora y dónde se encuentran? Muestra el razonamiento paso a paso.",
        "Creatividad":
            "Escribe un haiku sobre la arquitectura Transformer. Luego explica qué metáfora elegiste.",
        "Código Python":
            "Escribe una función Python que calcule la similitud coseno entre dos vectores numpy sin usar sklearn. Incluye docstring y ejemplo de uso.",
        "Conocimiento técnico":
            "Explica la diferencia entre los algoritmos de tokenización BPE, WordPiece y SentencePiece en exactamente 3 puntos cada uno.",
        "Instrucción larga":
            "Actúa como profesor de la Maestría en Ciencia de Datos de EAFIT. Diseña una evaluación de 5 preguntas (con respuestas) sobre la arquitectura Transformer para estudiantes de posgrado.",
    }
    
    preset_cmp = st.selectbox("Query de ejemplo:", list(preset_compare.keys()), key="cmp_preset")
    cmp_query = st.text_area(
        "Query para todos los modelos:",
        value=preset_compare[preset_cmp],
        height=100,
        key="cmp_query"
    )
    
    if st.button("⚡ Comparar modelos", key="run_compare", use_container_width=True) and cmp_query.strip():
        if not models_to_compare:
            st.warning("Selecciona al menos un modelo.")
            st.stop()
        
        results_compare = {}
        progress = st.progress(0, text="Iniciando comparación...")
        
        for i, model in enumerate(models_to_compare):
            progress.progress((i / len(models_to_compare)), text=f"Consultando {GROQ_MODELS[model]['family']}...")
            messages = [
                {"role": "system", "content": cmp_system},
                {"role": "user", "content": cmp_query}
            ]
            result = call_groq(
                client=client,
                model=model,
                messages=messages,
                temperature=cmp_temperature,
                max_tokens=cmp_max_tokens,
                top_p=cmp_top_p,
            )
            results_compare[model] = result
            time.sleep(0.2)
        
        progress.progress(1.0, text="✅ Comparación completada")
        
        # Separate successful from failed
        successful = {m: r for m, r in results_compare.items() if r.get("success")}
        failed     = {m: r for m, r in results_compare.items() if not r.get("success")}

        # ── Show errors for failed models ──
        if failed:
            st.markdown("### ⚠️ Errores")
            for model, result in failed.items():
                err_msg = result.get("error", "Error desconocido")
                st.markdown(f"""
                <div class="warn-box">
                <b>❌ {GROQ_MODELS[model]['family']} ({GROQ_MODELS[model]['params']})</b><br>
                <code style="font-size:0.8rem;">{err_msg}</code><br>
                <span style="font-size:0.78rem;color:#8b949e;">
                💡 Verifica que la API Key sea válida y que el modelo esté disponible en tu cuenta Groq.
                </span>
                </div>
                """, unsafe_allow_html=True)

        if not successful:
            st.error("Ningún modelo respondió correctamente. Verifica tu API Key en la barra lateral.")
            st.stop()

        # ── Summary metrics (only successful) ──
        st.markdown("### 📊 Métricas Comparativas")
        n_successful = len(successful)
        metric_cols = st.columns(max(1, n_successful))
        
        for col, (model, result) in zip(metric_cols, successful.items()):
            with col:
                model_color = GROQ_MODELS[model]["color"]
                st.markdown(f"""
                <div class="metric-card" style="border-color:{model_color};">
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:{model_color};">
                {GROQ_MODELS[model]['family']}
                </div>
                <div class="metric-value" style="color:{model_color};font-size:1.3rem;">
                {result['latency']:.2f}s
                </div>
                <div class="metric-label">Latencia</div>
                <div style="margin-top:0.4rem;font-size:0.8rem;color:#8b949e;">
                ⚡ {result['tokens_per_sec']:.0f} tok/s<br>
                📤 {result['usage']['prompt_tokens']} prompt<br>
                📥 {result['usage']['completion_tokens']} output<br>
                🏁 {result.get('finish_reason', 'n/a')}
                </div>
                </div>
                """, unsafe_allow_html=True)
        
        # ── Speed comparison chart ──
        if n_successful >= 2:
            fig_speed = go.Figure()
            model_labels = [
                f"{GROQ_MODELS[m]['family']}\n({GROQ_MODELS[m]['params']})"
                for m in successful.keys()
            ]
            fig_speed.add_trace(go.Bar(
                name="Latencia (s)",
                x=model_labels,
                y=[r["latency"] for r in successful.values()],
                marker_color=[GROQ_MODELS[m]["color"] for m in successful.keys()],
                opacity=0.85, yaxis="y"
            ))
            fig_speed.add_trace(go.Scatter(
                name="Tokens/segundo",
                x=model_labels,
                y=[r["tokens_per_sec"] for r in successful.values()],
                mode="markers+lines",
                marker=dict(size=10, color="#ffffff"),
                line=dict(color="#ffffff", dash="dot"),
                yaxis="y2"
            ))
            fig_speed.update_layout(
                title="Latencia vs Velocidad de Generación",
                paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                font=dict(color="#e6edf3"),
                yaxis=dict(title="Latencia (s)", gridcolor="#30363d"),
                yaxis2=dict(title="Tokens/s", overlaying="y", side="right"),
                height=320,
                legend=dict(bgcolor="#161b22", bordercolor="#30363d")
            )
            st.plotly_chart(fig_speed, use_container_width=True)
        
        # ── Responses side by side ──
        st.markdown("### 💬 Respuestas Comparadas")
        resp_cols = st.columns(max(1, n_successful))
        for col, (model, result) in zip(resp_cols, successful.items()):
            with col:
                model_color = GROQ_MODELS[model]["color"]
                st.markdown(f"""
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;
                            color:{model_color};margin-bottom:0.3rem;font-weight:600;">
                {GROQ_MODELS[model]['family']} ({GROQ_MODELS[model]['params']})
                </div>
                """, unsafe_allow_html=True)
                content_display = result.get("content", "") or "*(sin respuesta)*"
                st.markdown(
                    f'<div class="llm-response" style="font-size:0.82rem;">{content_display}</div>',
                    unsafe_allow_html=True
                )
        
        # ── Token usage table ──
        df_tokens = pd.DataFrame([{
            "Modelo": GROQ_MODELS[m]["family"],
            "Prompt tokens": r["usage"]["prompt_tokens"],
            "Output tokens": r["usage"]["completion_tokens"],
            "Total tokens": r["usage"]["total_tokens"],
            "Tokens/s": round(r["tokens_per_sec"], 1),
            "Latencia (s)": round(r["latency"], 3),
        } for m, r in successful.items()])
        
        st.markdown("### 📋 Tabla de Uso de Tokens")
        st.dataframe(df_tokens, use_container_width=True)


# ════════════════════════════════════════════════════════════
# MODULE 6: BENCHMARK DE VELOCIDAD
# ════════════════════════════════════════════════════════════
elif module == "📊  Benchmark de Velocidad":
    st.markdown('<div class="section-title">Benchmark de Latencia y Throughput</div>', unsafe_allow_html=True)
    
    if not api_key:
        st.markdown('<div class="warn-box">⚠️ Se requiere Groq API Key.</div>', unsafe_allow_html=True)
        st.stop()
    
    client = get_groq_client(api_key)
    
    st.markdown("""
    <div class="info-box">
    Ejecuta múltiples llamadas al mismo modelo para medir la <b>distribución estadística de latencia</b> 
    (no solo el promedio). En sistemas de producción, importan el percentil P95 y P99, no solo la media.
    </div>
    """, unsafe_allow_html=True)
    
    col_bm1, col_bm2 = st.columns(2)
    bm_model = col_bm1.selectbox(
        "Modelo a benchmarkear:",
        list(GROQ_MODELS.keys()),
        format_func=lambda m: f"{GROQ_MODELS[m]['family']} ({GROQ_MODELS[m]['params']})"
    )
    n_runs = col_bm2.slider("Número de llamadas:", 3, 15, 5)
    
    bm_prompt = st.text_area(
        "Prompt del benchmark:",
        value="Explica en exactamente 3 oraciones qué es el mecanismo de atención en los Transformers.",
        height=80
    )
    bm_max_tokens = st.slider("max_tokens para benchmark:", 50, 500, 150, 25)
    
    if st.button("🏁 Iniciar Benchmark", key="bm_run", use_container_width=True):
        latencies = []
        tok_per_sec_list = []
        output_tokens_list = []
        
        progress_bm = st.progress(0, text="Ejecutando benchmark...")
        messages_bm = [{"role": "user", "content": bm_prompt}]
        
        for i in range(n_runs):
            progress_bm.progress((i + 1) / n_runs, text=f"Llamada {i+1}/{n_runs}...")
            r = call_groq(
                client=client,
                model=bm_model,
                messages=messages_bm,
                temperature=0.0,
                max_tokens=bm_max_tokens,
            )
            if r["success"]:
                latencies.append(r["latency"])
                tok_per_sec_list.append(r["tokens_per_sec"])
                output_tokens_list.append(r["usage"]["completion_tokens"])
            time.sleep(0.1)
        
        progress_bm.progress(1.0, text="✅ Benchmark completado")
        
        if latencies:
            # Stats
            col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
            col_s1.metric("Media", f"{np.mean(latencies):.3f}s")
            col_s2.metric("Mediana", f"{np.median(latencies):.3f}s")
            col_s3.metric("P95", f"{np.percentile(latencies, 95):.3f}s")
            col_s4.metric("Std Dev", f"{np.std(latencies):.3f}s")
            col_s5.metric("Promedio tok/s", f"{np.mean(tok_per_sec_list):.0f}")
            
            # Latency distribution
            fig_bm = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Distribución de Latencia", "Tokens/segundo por llamada")
            )
            
            fig_bm.add_trace(
                go.Histogram(x=latencies, nbinsx=min(n_runs, 10),
                            marker_color="#f0883e", opacity=0.85, name="Latencia"),
                row=1, col=1
            )
            fig_bm.add_vline(
                x=np.mean(latencies), line_dash="dash", line_color="#58a6ff",
                annotation_text=f"Media: {np.mean(latencies):.3f}s",
                row=1, col=1
            )
            fig_bm.add_vline(
                x=np.percentile(latencies, 95), line_dash="dot", line_color="#f85149",
                annotation_text=f"P95: {np.percentile(latencies, 95):.3f}s",
                row=1, col=1
            )
            
            fig_bm.add_trace(
                go.Scatter(
                    x=list(range(1, len(tok_per_sec_list)+1)),
                    y=tok_per_sec_list,
                    mode="lines+markers",
                    marker=dict(color="#3fb950", size=8),
                    line=dict(color="#3fb950"),
                    name="Tokens/s"
                ),
                row=1, col=2
            )
            fig_bm.add_hline(
                y=np.mean(tok_per_sec_list), line_dash="dash", line_color="#58a6ff",
                annotation_text=f"Media: {np.mean(tok_per_sec_list):.0f} tok/s",
                row=1, col=2
            )
            
            fig_bm.update_layout(
                paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                font=dict(color="#e6edf3"), height=350,
                showlegend=False
            )
            for axis in ['xaxis', 'yaxis', 'xaxis2', 'yaxis2']:
                fig_bm.update_layout(**{f"{axis}_gridcolor": "#30363d"})
            
            st.plotly_chart(fig_bm, use_container_width=True)
            
            # Data table
            df_bm = pd.DataFrame({
                "Llamada": range(1, len(latencies)+1),
                "Latencia (s)": [round(l, 4) for l in latencies],
                "Tokens/s": [round(t, 1) for t in tok_per_sec_list],
                "Output tokens": output_tokens_list,
            })
            st.dataframe(df_bm, use_container_width=True)
            
            st.markdown(f"""
            <div class="info-box">
            <b>Análisis del Benchmark — {GROQ_MODELS[bm_model]['family']}:</b><br>
            • Coeficiente de variación: {(np.std(latencies)/np.mean(latencies)*100):.1f}% 
            (menor=más estable)<br>
            • Overhead P95 vs media: +{((np.percentile(latencies,95)/np.mean(latencies)-1)*100):.1f}%<br>
            • Para diseño de sistemas: usar P95 ({np.percentile(latencies,95):.3f}s) 
            como SLA conservador, no la media ({np.mean(latencies):.3f}s).
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# MODULE 7: ATTENTION VISUALIZER
# ════════════════════════════════════════════════════════════
elif module == "🎯  Attention Visualizer":
    st.markdown('<div class="section-title">Attention Visualizer: Entendiendo Qué Atiende el Modelo</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <b>Atención conceptual:</b> Este módulo calcula un proxy de la atención usando similitud TF-IDF 
    entre tokens y el query. Para atención real de un Transformer, se necesita acceso a los pesos 
    internos del modelo (posible con HuggingFace + BertViz). Aquí visualizamos la <b>intuición</b> 
    del mecanismo.
    </div>
    
    <div class="formula-box">
    Fórmula real de Self-Attention:
    
    Attention(Q,K,V) = softmax(QKᵀ / √dₖ) · V
    
    Para el token i, el score de atención hacia el token j es:
    α(i,j) = softmax( qᵢ · kⱼ / √dₖ )
    
    Donde qᵢ, kⱼ son proyecciones aprendidas del embedding del token i, j.
    La visualización abajo aproxima este proceso con similitud de embeddings.
    </div>
    """, unsafe_allow_html=True)
    
    attn_text = st.text_input(
        "Oración para visualizar atención:",
        value="La directora de la empresa aprobó el nuevo presupuesto porque consideró que era viable.",
        help="Frases con pronombres, correferencias y relaciones semánticas son más interesantes"
    )
    
    if SKLEARN_AVAILABLE and attn_text.strip():
        tokens_attn = attn_text.split()
        n_toks = len(tokens_attn)
        
        if n_toks >= 3:
            # Compute proxy attention matrix using TF-IDF cosine similarity
            # Create character n-gram features for each token
            if SKLEARN_AVAILABLE:
                try:
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    vec_attn = TfidfVectorizer(analyzer='char_wb', ngram_range=(2,4), min_df=1)
                    
                    # Context windows as pseudo-documents
                    context_docs = []
                    for i in range(n_toks):
                        left = max(0, i-3)
                        right = min(n_toks, i+4)
                        context_docs.append(" ".join(tokens_attn[left:right]))
                    
                    X_attn = vec_attn.fit_transform(context_docs)
                    sim_attn = cosine_similarity(X_attn).astype(float)
                    
                    # Normalize rows (softmax-like)
                    sim_attn = np.exp(sim_attn * 3)
                    row_sums = sim_attn.sum(axis=1, keepdims=True)
                    sim_attn = sim_attn / row_sums
                    
                    # Attention heatmap
                    fig_attn = px.imshow(
                        sim_attn,
                        x=tokens_attn,
                        y=tokens_attn,
                        title="Matriz de Atención Aproximada (proxy TF-IDF)",
                        color_continuous_scale="Oranges",
                        text_auto=".2f",
                        aspect="auto"
                    )
                    fig_attn.update_layout(
                        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                        font=dict(color="#e6edf3", size=11),
                        height=max(300, n_toks * 25 + 100),
                        xaxis=dict(tickangle=-45),
                        coloraxis_showscale=True
                    )
                    st.plotly_chart(fig_attn, use_container_width=True)
                    
                    # Attention per query token
                    st.markdown("### 🔍 Selecciona un token para ver qué atiende")
                    query_token_idx = st.select_slider(
                        "Token de consulta (Query):",
                        options=list(range(n_toks)),
                        format_func=lambda i: f"[{i}] '{tokens_attn[i]}'",
                        value=min(3, n_toks-1)
                    )
                    
                    attn_row = sim_attn[query_token_idx]
                    
                    col_viz1, col_viz2 = st.columns([2, 1])
                    with col_viz1:
                        fig_row = px.bar(
                            x=tokens_attn,
                            y=attn_row,
                            title=f"Distribución de atención del token '{tokens_attn[query_token_idx]}'",
                            color=attn_row,
                            color_continuous_scale=["#161b22", "#f0883e", "#ffa657"],
                        )
                        fig_row.update_layout(
                            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                            font=dict(color="#e6edf3"), height=280,
                            xaxis=dict(tickangle=-30), showlegend=False
                        )
                        fig_row.update_xaxes(gridcolor="#30363d")
                        fig_row.update_yaxes(gridcolor="#30363d")
                        st.plotly_chart(fig_row, use_container_width=True)
                    
                    with col_viz2:
                        st.markdown(f"**Token foco:** `{tokens_attn[query_token_idx]}`")
                        st.markdown("**Top 5 tokens atendidos:**")
                        top5_idx = np.argsort(attn_row)[::-1][:5]
                        for rank, idx in enumerate(top5_idx):
                            color = token_color(rank, 5)
                            bar_width = int(attn_row[idx] * 100)
                            st.markdown(f"""
                            <div style="margin:0.3rem 0;">
                            <span style="font-family:'JetBrains Mono',monospace;
                                        font-size:0.8rem;color:{color};">
                            {rank+1}. '{tokens_attn[idx]}'
                            </span>
                            <div style="background:{color}33;height:6px;border-radius:3px;
                                        width:{bar_width}%;margin-top:2px;"></div>
                            <span style="font-size:0.7rem;color:#8b949e;">{attn_row[idx]:.3f}</span>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Color-coded sentence
                    st.markdown("**Oración con intensidad de atención coloreada:**")
                    sentence_html = ""
                    for i, (tok, val) in enumerate(zip(tokens_attn, attn_row)):
                        alpha = 0.2 + val * 0.8
                        r_val = int(240 * val)
                        g_val = int(136 * (1 - val))
                        bg = f"rgba({r_val},{g_val},62,{alpha:.2f})"
                        sentence_html += (
                            f'<span class="attn-cell" style="background:{bg};" '
                            f'title="Atención: {val:.3f}">{tok}</span> '
                        )
                    st.markdown(f"<div>{sentence_html}</div>", unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error calculando atención: {e}")
        else:
            st.warning("La oración debe tener al menos 3 palabras.")
    
    st.markdown("""
    <div class="warn-box">
    <b>⚠️ Nota importante:</b> Esta visualización es una <b>aproximación pedagógica</b> basada en 
    similitud de n-gramas, no la atención real del Transformer. La atención real usa vectores Q, K, V 
    aprendidos de millones de parámetros y captura relaciones mucho más complejas. 
    Para atención real, usa: <code>pip install bertviz</code> con modelos HuggingFace.
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# MODULE 8: PLAYGROUND LIBRE
# ════════════════════════════════════════════════════════════
elif module == "🧪  Playground Libre":
    st.markdown('<div class="section-title">Playground Libre: Experimenta sin Restricciones</div>', unsafe_allow_html=True)
    
    if not api_key:
        st.markdown('<div class="warn-box">⚠️ Se requiere Groq API Key.</div>', unsafe_allow_html=True)
        st.stop()
    
    client = get_groq_client(api_key)
    
    st.markdown("""
    <div class="info-box">
    Modo de experimentación libre. Diseña tu propio sistema multi-turn, prueba técnicas de 
    Prompt Engineering avanzadas (Chain of Thought, Few-Shot, ReAct) y explora el límite de los modelos.
    </div>
    """, unsafe_allow_html=True)
    
    # Multi-turn conversation
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "pg_system" not in st.session_state:
        st.session_state.pg_system = "Eres un asistente de investigación para la Maestría en Ciencia de Datos de EAFIT."
    
    tab_chat, tab_prompts, tab_cheatsheet = st.tabs(
        ["💬 Chat Multi-Turn", "🎨 Prompt Engineering", "📋 Cheatsheet de Parámetros"]
    )
    
    with tab_chat:
        col_pg1, col_pg2 = st.columns([3, 1])
        
        with col_pg2:
            st.markdown("**Configuración:**")
            pg_model = st.selectbox(
                "Modelo:",
                list(GROQ_MODELS.keys()),
                format_func=lambda m: f"{GROQ_MODELS[m]['family']}",
                key="pg_model"
            )
            pg_temp = st.slider("temperature", 0.0, 2.0, 0.7, 0.05, key="pg_temp")
            pg_max = st.slider("max_tokens", 100, 4096, 1024, 100, key="pg_max")
            pg_top_p = st.slider("top_p", 0.1, 1.0, 0.95, 0.05, key="pg_tp")
            
            st.markdown("**System Prompt:**")
            st.session_state.pg_system = st.text_area(
                "system",
                value=st.session_state.pg_system,
                height=120, label_visibility="collapsed",
                key="pg_sys_input"
            )
            
            if st.button("🗑️ Limpiar conversación", key="pg_clear"):
                st.session_state.conversation_history = []
                st.rerun()
            
            st.markdown(f"""
            <div style="font-size:0.72rem;font-family:'JetBrains Mono',monospace;color:#8b949e;
                        margin-top:0.5rem;">
            Turnos: {len(st.session_state.conversation_history)//2}<br>
            Tokens acum: ~{count_tokens_tiktoken(' '.join([m['content'] for m in st.session_state.conversation_history]))}<br>
            Ctx limit: {GROQ_MODELS[pg_model]['context']:,}
            </div>
            """, unsafe_allow_html=True)
        
        with col_pg1:
            # Display conversation
            for msg in st.session_state.conversation_history:
                if msg["role"] == "user":
                    st.markdown(f"""
                    <div style="background:#1c2333;border:1px solid #30363d;border-radius:8px;
                                padding:0.8rem 1rem;margin:0.4rem 0;border-left:3px solid #58a6ff;">
                    <span style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                                 color:#58a6ff;">👤 Usuario</span><br>
                    <span style="font-size:0.9rem;">{msg['content']}</span>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background:#0d1117;border:1px solid #30363d;border-radius:8px;
                                padding:0.8rem 1rem;margin:0.4rem 0;border-left:3px solid #f0883e;">
                    <span style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                                 color:#f0883e;">🤖 {GROQ_MODELS[pg_model]['family']}</span><br>
                    <span style="font-size:0.9rem;">{msg['content']}</span>
                    </div>""", unsafe_allow_html=True)
            
            pg_user_input = st.text_area(
                "Tu mensaje:", height=80,
                placeholder="Escribe tu mensaje... (multi-turn habilitado)",
                key="pg_input"
            )
            
            if st.button("📤 Enviar", key="pg_send", use_container_width=True) and pg_user_input.strip():
                st.session_state.conversation_history.append({
                    "role": "user", "content": pg_user_input
                })
                
                messages_pg = [{"role": "system", "content": st.session_state.pg_system}]
                messages_pg.extend(st.session_state.conversation_history)
                
                with st.spinner("Generando respuesta..."):
                    r = call_groq(
                        client=client,
                        model=pg_model,
                        messages=messages_pg,
                        temperature=pg_temp,
                        max_tokens=pg_max,
                        top_p=pg_top_p,
                    )
                
                if r["success"]:
                    st.session_state.conversation_history.append({
                        "role": "assistant", "content": r["content"]
                    })
                    st.rerun()
                else:
                    st.error(f"Error: {r.get('error', 'Unknown')}")
    
    with tab_prompts:
        st.markdown('<div class="section-title" style="font-size:1rem;">Técnicas de Prompt Engineering</div>', unsafe_allow_html=True)
        
        techniques = {
            "Zero-Shot": {
                "desc": "Sin ejemplos. Solo instrucción directa.",
                "prompt": """Clasifica el sentimiento de este texto en Positivo/Negativo/Neutro:

Texto: "La nueva versión de LLaMA supera mis expectativas en razonamiento matemático."
Sentimiento:""",
                "when": "Cuando el modelo ya conoce bien la tarea. Más simple, menos tokens."
            },
            "Few-Shot": {
                "desc": "2–5 ejemplos para guiar el formato y la tarea.",
                "prompt": """Clasifica el sentimiento. Responde SOLO con la etiqueta.

Texto: "El modelo tardó demasiado y la respuesta fue incorrecta." → Negativo
Texto: "La tokenización funciona perfectamente para mi caso de uso." → Positivo
Texto: "Los resultados son aceptables pero hay margen de mejora." → Neutro
Texto: "Esta arquitectura Transformer es increíblemente eficiente." → """,
                "when": "Cuando el formato de salida debe ser exacto o el modelo tiende a divagar."
            },
            "Chain of Thought": {
                "desc": "Forzar razonamiento paso a paso antes de responder.",
                "prompt": """Resuelve el siguiente problema paso a paso, mostrando cada razonamiento:

Un modelo LLM procesa 500 tokens por segundo.
Tengo un documento de 50,000 palabras.
Asumiendo 1 token ≈ 0.75 palabras, ¿cuántos segundos tardará en procesar el documento completo?

Razonamiento:""",
                "when": "Problemas matemáticos, lógicos o que requieren múltiples pasos. Aumenta precisión ~40%."
            },
            "Role Prompting": {
                "desc": "Asignar un rol o persona específica al modelo.",
                "prompt": """Eres el Prof. Andrej Karpathy, experto mundial en arquitecturas de redes neuronales y ex-Director de IA de Tesla. 
Tienes un estilo didáctico, usas analogías físicas y siempre das ejemplos de código Python.

Explica el mecanismo de Multi-Head Attention a un estudiante de primer año de maestría.""",
                "when": "Cuando necesitas un estilo específico, expertise técnico, o perspectiva particular."
            },
            "Structured Output": {
                "desc": "Forzar output en formato estructurado (JSON, XML, Markdown).",
                "prompt": """Analiza el siguiente modelo de lenguaje y responde ÚNICAMENTE en JSON válido con esta estructura exacta, sin texto adicional:
{
  "nombre": string,
  "arquitectura": string,
  "parametros": string,
  "contexto_tokens": number,
  "fortalezas": [string, string, string],
  "limitaciones": [string, string],
  "casos_uso_ideales": [string, string, string],
  "score_general": number entre 0 y 10
}

Modelo a analizar: LLaMA 3.1 70B""",
                "when": "Integración con código, APIs, parseo automático de respuestas."
            },
            "ReAct (Reason + Act)": {
                "desc": "Entrelaza razonamiento y acciones con herramientas.",
                "prompt": """Responde usando el formato: Pensamiento → Acción → Observación → Respuesta Final.

Pregunta: ¿Cuántos tokens tiene el prompt "La inteligencia artificial transforma el mundo" en BPE?

Pensamiento: Necesito calcular los tokens BPE. Según la regla 1 token ≈ 0.75 palabras en inglés, para español usaré 0.5.
Acción: Contar palabras: "La inteligencia artificial transforma el mundo" = 6 palabras. 6 / 0.5 = ~12 tokens estimados.
Observación: El texto tiene palabras de longitud media, algunas subpalabras como "inteligencia" pueden tokenizarse como 2-3 tokens.
Respuesta Final:""",
                "when": "Agentes que usan herramientas, razonamiento con acciones externas, cálculos complejos."
            },
        }
        
        for tech_name, tech_info in techniques.items():
            with st.expander(f"**{tech_name}** — {tech_info['desc']}"):
                col_t1, col_t2 = st.columns([3, 1])
                with col_t1:
                    edited_prompt = st.text_area(
                        "Prompt:", value=tech_info["prompt"],
                        height=150, key=f"prompt_{tech_name}"
                    )
                with col_t2:
                    st.markdown(f"""
                    <div class="info-box" style="font-size:0.8rem;">
                    <b>¿Cuándo usar?</b><br>{tech_info['when']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button(f"▶️ Ejecutar", key=f"run_{tech_name}") and api_key:
                        with st.spinner("Generando..."):
                            r = call_groq(
                                client=client,
                                model=selected_model,
                                messages=[{"role": "user", "content": edited_prompt}],
                                temperature=0.7,
                                max_tokens=500,
                            )
                        if r["success"]:
                            st.markdown(f"""
                            <div class="llm-response" style="font-size:0.82rem;max-height:200px;">
                            {r['content']}
                            </div>
                            <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;
                                        color:#8b949e;margin-top:0.3rem;">
                            {r['latency']:.2f}s · {r['usage']['completion_tokens']} tokens
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.error(r.get("error", "Error"))
    
    with tab_cheatsheet:
        st.markdown('<div class="section-title" style="font-size:1rem;">Cheatsheet Completo de Parámetros LLM</div>', unsafe_allow_html=True)
        
        params_data = [
            {
                "Parámetro": "temperature",
                "Tipo": "float [0.0, 2.0]",
                "Default": "1.0",
                "Efecto": "Escala logits antes del softmax. T→0: greedy. T→∞: uniforme.",
                "Producción": "0.0–0.3 (hechos/código), 0.7–1.0 (general), 1.0–1.5 (creatividad)",
                "Interacción": "Se combina con top_p/top_k. No usar ambos muy restrictivos."
            },
            {
                "Parámetro": "top_p",
                "Tipo": "float (0, 1]",
                "Default": "1.0",
                "Efecto": "Nucleus sampling. Mantiene tokens hasta acumular P de probabilidad.",
                "Producción": "0.9–0.95 para uso general. 1.0 = deshabilitado.",
                "Interacción": "Con temperature. OpenAI recomienda no modificar ambos a la vez."
            },
            {
                "Parámetro": "top_k",
                "Tipo": "int [0, vocab_size]",
                "Default": "0 (off)",
                "Efecto": "Solo muestra del top-K tokens más probables. 0 = deshabilitado.",
                "Producción": "50–100 para uso general. 1 = greedy.",
                "Interacción": "Alternativa a top_p. Se puede usar en combinación."
            },
            {
                "Parámetro": "max_tokens",
                "Tipo": "int [1, context_limit]",
                "Default": "Varía por modelo",
                "Efecto": "Límite duro de output. finish_reason='length' si se alcanza.",
                "Producción": "Estimar según tarea. Chat: 256–1024. Código: 1024–4096.",
                "Interacción": "Afecta costo (output tokens más caros). No afecta calidad."
            },
            {
                "Parámetro": "frequency_penalty",
                "Tipo": "float [-2.0, 2.0]",
                "Default": "0.0",
                "Efecto": "logit(t) -= penalty × count(t). Penaliza según frecuencia acumulada.",
                "Producción": "0.0–0.5 para evitar repetición. Evitar >1.0 (vocabulario raro).",
                "Interacción": "Se suma con presence_penalty. Cuidado con valores negativos."
            },
            {
                "Parámetro": "presence_penalty",
                "Tipo": "float [-2.0, 2.0]",
                "Default": "0.0",
                "Efecto": "logit(t) -= penalty × (1 if appeared else 0). Penaliza presencia binaria.",
                "Producción": "0.0–0.6 para diversidad temática.",
                "Interacción": "Distinto de frequency_penalty: es binario, no acumulativo."
            },
            {
                "Parámetro": "seed",
                "Tipo": "int [0, 2^32-1]",
                "Default": "None (aleatorio)",
                "Efecto": "Semilla RNG. Mismo seed + params → output determinístico (aprox.).",
                "Producción": "Usar en experimentos, tests, A/B testing de prompts.",
                "Interacción": "No garantiza reproducibilidad perfecta por paralelismo GPU."
            },
            {
                "Parámetro": "stop",
                "Tipo": "list[str] o str",
                "Default": "None",
                "Efecto": "Para generación al encontrar cualquiera de estos strings.",
                "Producción": '["\\n\\n", "###"] para controlar formato. No incluido en output.',
                "Interacción": "Alternativa a max_tokens para control de longitud."
            },
            {
                "Parámetro": "stream",
                "Tipo": "bool",
                "Default": "False",
                "Efecto": "Transmite tokens en tiempo real (Server-Sent Events).",
                "Producción": "True para UX interactiva (ver tokens llegar). Misma calidad final.",
                "Interacción": "No compatible con algunas métricas de uso (token count)."
            },
        ]
        
        df_params = pd.DataFrame(params_data)
        st.dataframe(df_params, use_container_width=True, height=360)
        
        # Visual guide: temperature effect
        st.markdown("### 🌡️ Efecto Visual de la Temperatura")
        
        # Simulate logits
        np.random.seed(42)
        logits = np.array([3.5, 2.1, 1.8, 0.9, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02])
        vocab_sample = ["gato", "perro", "casa", "árbol", "sol", "luna", "río", "mar", "cielo", "aire"]
        
        temps_to_show = [0.1, 0.5, 1.0, 1.5, 2.0]
        
        fig_temp_effect = go.Figure()
        for T in temps_to_show:
            scaled = logits / T
            probs = np.exp(scaled - scaled.max())
            probs = probs / probs.sum()
            fig_temp_effect.add_trace(go.Bar(
                name=f"T={T}",
                x=vocab_sample,
                y=probs,
                opacity=0.8,
            ))
        
        fig_temp_effect.update_layout(
            title="Distribución de probabilidades con distintas temperaturas",
            barmode="group",
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            font=dict(color="#e6edf3"),
            height=350,
            legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1),
            xaxis=dict(gridcolor="#30363d"),
            yaxis=dict(title="Probabilidad", gridcolor="#30363d"),
        )
        st.plotly_chart(fig_temp_effect, use_container_width=True)
        
        st.markdown("""
        <div class="info-box">
        <b>Observa:</b> Con T=0.1 (azul), casi toda la probabilidad se concentra en "gato". 
        Con T=2.0 (morado), la distribución se aplana y tokens menos probables como "cielo" 
        tienen una chance real. Esto explica por qué temperaturas altas generan texto más "sorprendente" 
        pero potencialmente incoherente.
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div style="text-align:center;font-family:'JetBrains Mono',monospace;font-size:0.68rem;
            color:#30363d;padding:0.5rem 0;">
    EAFIT Universidad · Maestría en Ciencia de Datos · 
    NLP & LLM Interactive Lab · 
    Prof. Jorge Iván Padilla-Buriticá · 
    <a href="https://www.linkedin.com/in/jipadilla" style="color:#30363d;">linkedin/jipadilla</a>
</div>
""", unsafe_allow_html=True)
