# ============================================================
# EAFIT — Maestría en Ciencia de Datos
# NLP & LLM Interactive Lab
# Prof. Jorge Iván Padilla-Buriticá | linkedin.com/in/jipadilla
# ============================================================
# Módulos cubiertos:
#   0. Inicio & Teoría
#   1. Tokenización       (BPE/tiktoken, GPT-2, BERT WordPiece, IDs)
#   2. Embeddings         (TF-IDF, embeddings densos, similitud coseno, proyección 2D)
#   3. Chunking para RAG  (Estructural, Recursivo, Semántico, Agéntico)
#   4. NLP Clásico        (POS, NER, Sentimientos)
#   5. LLM Lab            (temperature, top_p, max_tokens, stop, seed)
#   6. Comparador         (misma query, varios modelos Groq)
#   7. Benchmark          (latencia, tokens/s, percentiles)
#   8. Attention Viz      (heat-map conceptual de atención)
#   9. Playground Libre   (chat multi-turn + prompt engineering)
#
# ── SIN PyTorch / torchvision / transformers / sentence-transformers ──
# Estas librerías no se usan a propósito. `transformers` registra
# internamente cientos de submódulos (incluidos modelos de visión como
# ViTMatte/ViTPose/YOLOS), y el *file watcher* de Streamlit los recorre
# a todos al arrancar, disparando `ModuleNotFoundError: No module named
# 'torchvision'` aunque la app nunca los use. La única forma robusta de
# eliminar ese error es no depender de esas librerías. Aquí se reemplazan
# por alternativas 100% CPU, sin PyTorch:
#   • tiktoken   → tokenización BPE (GPT-2 original y cl100k moderno)
#   • tokenizers → tokenización WordPiece de BERT (Rust puro)
#   • fastembed  → embeddings densos vía ONNX Runtime
#
# Todos los modelos de generación (LLM Lab, Comparador, Benchmark,
# Playground) se sirven vía Groq API — el catálogo se verificó en
# console.groq.com/docs/models. Ese catálogo cambia con frecuencia:
# revísalo antes de una clase en vivo.
# ============================================================

import os
import re
import time
from collections import Counter

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
except Exception:
    GROQ_AVAILABLE = False

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except Exception:
    TIKTOKEN_AVAILABLE = False

try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.util import ngrams
    from nltk.corpus import stopwords
    from nltk import pos_tag, ne_chunk
    from nltk.sentiment import SentimentIntensityAnalyzer
    for pkg in ['punkt', 'averaged_perceptron_tagger', 'maxent_ne_chunker',
                'words', 'stopwords', 'vader_lexicon', 'punkt_tab',
                'averaged_perceptron_tagger_eng', 'maxent_ne_chunker_tab']:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass
    NLTK_AVAILABLE = True
except Exception:
    NLTK_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.decomposition import TruncatedSVD
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    from tokenizers import Tokenizer as HFTokenizer  # HuggingFace `tokenizers`, sin PyTorch
    TOKENIZERS_AVAILABLE = True
except Exception:
    TOKENIZERS_AVAILABLE = False

try:
    from fastembed import TextEmbedding  # embeddings vía ONNX Runtime, sin PyTorch
    FASTEMBED_AVAILABLE = True
except Exception:
    FASTEMBED_AVAILABLE = False

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
# CUSTOM CSS — Dark Academic / Research Lab, sobrio y elegante
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
    --purple:     #bc8cff;
    --teal:       #39d353;
}
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background-color: var(--bg); color: var(--text); }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
/* No ocultar todo el <header>: ahí vive el botón que abre/cierra la
   barra lateral. Lo hacemos transparente y forzamos que el control de
   colapso sea siempre visible y con buen contraste sobre fondo oscuro. */
header[data-testid="stHeader"] { background: transparent; box-shadow: none; visibility: visible; }
button[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"] {
    visibility: visible !important;
    display: flex !important;
    opacity: 1 !important;
}
button[data-testid="collapsedControl"] svg,
[data-testid="stSidebarCollapsedControl"] svg {
    fill: var(--accent) !important;
    color: var(--accent) !important;
}
.block-container { padding-top: 1rem; max-width: 100%; }
.lab-header {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #1a1f2e 100%);
    border-bottom: 2px solid var(--accent);
    padding: 1.2rem 2rem; margin: -1rem -1rem 1.5rem -1rem;
    display: flex; align-items: center; gap: 1rem; flex-wrap: wrap;
}
.lab-header h1 { font-family: 'Crimson Pro', serif; font-size: 1.8rem; font-weight: 600; color: var(--text); margin: 0; letter-spacing: -0.02em; }
.lab-header .subtitle { font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: var(--accent); margin-top: 0.15rem; }
.badge { background: var(--accent); color: #000; font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; font-weight: 600; padding: 0.2rem 0.6rem; border-radius: 20px; white-space: nowrap; }
.section-title { font-family: 'Crimson Pro', serif; font-size: 1.5rem; font-weight: 600; color: var(--text); border-left: 4px solid var(--accent); padding-left: 0.8rem; margin: 1.5rem 0 1rem 0; }
.info-box { background: var(--surface2); border: 1px solid var(--accent2); border-left: 4px solid var(--accent2); border-radius: 6px; padding: 0.8rem 1rem; margin: 0.5rem 0; font-size: 0.88rem; }
.warn-box { background: #2d1f00; border: 1px solid var(--accent); border-left: 4px solid var(--accent); border-radius: 6px; padding: 0.8rem 1rem; margin: 0.5rem 0; font-size: 0.88rem; }
.success-box { background: #0d2818; border: 1px solid var(--accent3); border-left: 4px solid var(--accent3); border-radius: 6px; padding: 0.8rem 1rem; margin: 0.5rem 0; font-size: 0.88rem; }
.formula-box { background: #0d1117; border: 1px solid var(--border); border-left: 4px solid var(--purple); border-radius: 6px; padding: 0.8rem 1.2rem; margin: 0.5rem 0; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: var(--purple); white-space: pre-wrap; }
.metric-card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 1rem; text-align: center; transition: border-color 0.2s; }
.metric-card:hover { border-color: var(--accent); }
.metric-value { font-family: 'JetBrains Mono', monospace; font-size: 1.8rem; font-weight: 600; color: var(--accent); }
.metric-label { font-size: 0.75rem; color: var(--text-muted); margin-top: 0.2rem; }
.token-container { display: flex; flex-wrap: wrap; gap: 4px; margin: 0.5rem 0; }
.token-chip { font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; padding: 2px 8px; border-radius: 4px; border: 1px solid; }
.id-chip { font-family: 'JetBrains Mono', monospace; font-size: 0.62rem; color: #6e7681; display:block; text-align:center; }
section[data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--border); }
section[data-testid="stSidebar"] .stSelectbox label, section[data-testid="stSidebar"] .stSlider label, section[data-testid="stSidebar"] .stTextInput label { color: var(--text-muted); font-size: 0.82rem; font-family: 'JetBrains Mono', monospace; }
.stTextArea textarea { background: var(--surface2) !important; color: var(--text) !important; border-color: var(--border) !important; font-family: 'DM Sans', sans-serif !important; }
.stButton > button { background: var(--accent) !important; color: #000 !important; border: none !important; font-weight: 600 !important; border-radius: 6px !important; font-family: 'DM Sans', sans-serif !important; }
.stButton > button:hover { background: #e07830 !important; transform: translateY(-1px); }
div[data-testid="stExpander"] { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 8px !important; }
.stTabs [data-baseweb="tab"] { font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: var(--text-muted); }
.stTabs [aria-selected="true"] { color: var(--accent) !important; border-bottom-color: var(--accent) !important; }
.stDataFrame { border: 1px solid var(--border) !important; border-radius: 6px; }
.llm-response { background: var(--surface2); border: 1px solid var(--border); border-radius: 8px; padding: 1.2rem; font-size: 0.92rem; line-height: 1.7; white-space: pre-wrap; max-height: 500px; overflow-y: auto; }
.attn-cell { display: inline-block; padding: 2px 6px; border-radius: 3px; font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; margin: 2px; }
.chunk-box { background: var(--surface); border: 1px solid var(--border); border-left: 4px solid var(--teal); border-radius: 6px; padding: 0.7rem 1rem; margin: 0.4rem 0; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# CATÁLOGO DE MODELOS — GROQ (verificado en console.groq.com/docs/models)
# ════════════════════════════════════════════════════════════
GROQ_MODELS = {
    "openai/gpt-oss-120b": {
        "family": "GPT-OSS 120B", "params": "120B (MoE)", "context": 131_072,
        "type": "Decoder-only · Mixture of Experts · Reasoning", "license": "Apache 2.0",
        "strengths": "Razonamiento profundo, instrucciones complejas, reasoning_effort configurable",
        "color": "#f0883e"
    },
    "openai/gpt-oss-20b": {
        "family": "GPT-OSS 20B", "params": "20B (MoE)", "context": 131_072,
        "type": "Decoder-only · Mixture of Experts · Reasoning", "license": "Apache 2.0",
        "strengths": "Versión ligera de GPT-OSS. Buen balance velocidad/calidad para el aula",
        "color": "#58a6ff"
    },
    "openai/gpt-oss-safeguard-20b": {
        "family": "GPT-OSS Safeguard 20B", "params": "20B", "context": 131_072,
        "type": "Clasificador de seguridad (policy-aware)", "license": "Apache 2.0",
        "strengths": "Clasifica contenido contra una política dada en el prompt. Guardrails/moderación",
        "color": "#f85149"
    },
    "qwen/qwen3-32b": {
        "family": "Qwen3 32B", "params": "32B", "context": 131_072,
        "type": "Decoder-only · Reasoning conmutable", "license": "Apache 2.0",
        "strengths": "Reasoning on/off vía reasoning_effort=none|default. Fuerte en matemáticas",
        "color": "#bc8cff"
    },
    "llama-3.3-70b-versatile": {
        "family": "LLaMA 3.3", "params": "70B", "context": 131_072,
        "type": "Decoder-only", "license": "Meta Community",
        "strengths": "Razonamiento general, multilingüe, buen todoterreno para comparaciones",
        "color": "#3fb950"
    },
    "llama-3.1-8b-instant": {
        "family": "LLaMA 3.1", "params": "8B", "context": 131_072,
        "type": "Decoder-only", "license": "Meta Community",
        "strengths": "Velocidad máxima, bajo costo. Ideal para benchmarks de latencia",
        "color": "#39d353"
    },
    "deepseek-r1-distill-llama-70b": {
        "family": "DeepSeek R1 Distill", "params": "70B", "context": 131_072,
        "type": "Decoder-only · Reasoning (CoT destilado en LLaMA 70B)", "license": "MIT",
        "strengths": "Razonamiento matemático/lógico de DeepSeek-R1, destilado en una base LLaMA",
        "color": "#ffa657"
    },
}

# ════════════════════════════════════════════════════════════
# CATÁLOGO DE MODELOS DE EMBEDDING (vía fastembed / ONNX, sin PyTorch)
# ════════════════════════════════════════════════════════════
EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": {
        "hf_id": "sentence-transformers/all-MiniLM-L6-v2",
        "dims": 384, "context": 256, "size": "~90 MB",
        "note": "Clásico, ligero y rápido. Ideal para prototipos. Entrenado sobre todo en "
                "inglés (funciona en español con algo de pérdida de calidad).",
    },
    "bge-m3": {
        "hf_id": "BAAI/bge-m3",
        "dims": 1024, "context": 8192, "size": "~2.2 GB",
        "note": "BAAI. Muy fuerte en corpus multilingüe/técnico en español, contexto de 8192 tokens.",
    },
    "nomic-embed-text": {
        "hf_id": "nomic-ai/nomic-embed-text-v1.5",
        "dims": 768, "context": 8192, "size": "~550 MB",
        "note": "Nomic AI. Popular para RAG con Groq en la comunidad open-source. Contexto largo.",
    },
}

# Textos de ejemplo
SAMPLE_TEXTS = {
    "es": """La inteligencia artificial está transformando profundamente la educación superior en Colombia.
Las universidades como EAFIT están adoptando modelos de lenguaje grande para personalizar el aprendizaje
y automatizar la evaluación formativa. Sin embargo, los docentes señalan que el pensamiento crítico
y la creatividad humana siguen siendo irreemplazables. El Ministerio de Educación analiza marcos
regulatorios para garantizar el uso ético de estas tecnologías en el aula.""",
    "mixed": """El Transformer architecture introduced by Vaswani et al. (2017) propone que
la atención es todo lo que necesitas. Esta arquitectura utiliza self-attention con matrices Q, K, V
para calcular: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V.
Los LLMs modernos como GPT-OSS-120B tienen decenas de miles de millones de parámetros y contextos de 131k tokens.""",
    "rag_doc": """# Política de Datos EAFIT

## 1. Introducción
La Universidad EAFIT recolecta datos académicos con fines de mejora continua.

## 2. Alcance
Esta política aplica a estudiantes, profesores y personal administrativo de todos los programas.

### 2.1 Datos de estudiantes
Se recolectan calificaciones, asistencia y uso de plataformas virtuales.

### 2.2 Datos de investigación
Los grupos de investigación gestionan sus propios repositorios bajo supervisión de la Vicerrectoría.

## 3. Seguridad
Todos los datos se almacenan cifrados y se auditan trimestralmente. El acceso no autorizado
está prohibido y será sancionado según el reglamento estudiantil vigente."""
}

# ════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="⚙️ Cargando modelo de embeddings (ONNX, sin PyTorch)…")
def load_embedding_model(model_key: str = "all-MiniLM-L6-v2"):
    if not FASTEMBED_AVAILABLE:
        return None
    info = EMBEDDING_MODELS.get(model_key)
    if info is None:
        return None
    try:
        return TextEmbedding(model_name=info["hf_id"])
    except Exception as e:
        st.warning(f"No se pudo cargar {model_key}: {e}")
        return None

def embed_texts(model, texts: list) -> np.ndarray:
    """fastembed .embed() devuelve un generador de arrays numpy; lo materializamos en matriz."""
    return np.array(list(model.embed(texts)))

@st.cache_resource(show_spinner="⚙️ Cargando tokenizador BERT (WordPiece, sin PyTorch)…")
def load_bert_tokenizer(model_name: str = "bert-base-multilingual-cased"):
    if not TOKENIZERS_AVAILABLE:
        return None
    try:
        return HFTokenizer.from_pretrained(model_name)
    except Exception:
        return None

def tokenize_with_tiktoken(text: str, encoding_name: str = "cl100k_base"):
    """Devuelve (tokens, ids) usando una codificación BPE de tiktoken.

    encoding_name="cl100k_base" -> BPE de referencia de GPT-3.5/4 (~100,277 vocab)
    encoding_name="gpt2"        -> BPE ORIGINAL de GPT-2 (2019, 50,257 vocab), sin PyTorch.
    """
    if not TIKTOKEN_AVAILABLE:
        toks = text.split()
        return toks, list(range(len(toks)))
    try:
        enc = tiktoken.get_encoding(encoding_name)
        token_ids = enc.encode(text)
        tokens = [enc.decode([t]) for t in token_ids]
        return tokens, token_ids
    except Exception:
        toks = text.split()
        return toks, list(range(len(toks)))

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
    stop=None,
    seed=None,
    reasoning_effort=None,
) -> dict:
    """
    Wrapper de la API de Groq con los parámetros que realmente soporta hoy.
    Groq NO soporta frequency_penalty / presence_penalty (son de la API de OpenAI);
    nunca se envían. reasoning_effort solo se envía a modelos con razonamiento
    (gpt-oss-*, qwen3), y se ignora en los demás para no provocar un error 400.
    """
    if client is None:
        return {"success": False, "error": "Cliente Groq no inicializado. Verifica tu API Key.", "latency": 0,
                "content": "", "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}

    params = {
        "model": model, "messages": messages,
        "temperature": float(temperature), "max_tokens": int(max_tokens), "top_p": float(top_p),
    }
    if stop:
        params["stop"] = stop
    if seed is not None:
        params["seed"] = int(seed)
    reasoning_capable = ("gpt-oss" in model) or ("qwen3" in model)
    if reasoning_effort and reasoning_capable:
        params["reasoning_effort"] = reasoning_effort

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
        return {"success": True, "content": content, "usage": usage, "latency": latency,
                "tokens_per_sec": tokens_per_sec, "finish_reason": finish_reason, "model": model, "params": params}
    except Exception as e:
        return {"success": False, "error": str(e), "latency": time.perf_counter() - t0,
                "content": "", "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}

def cosine_sim_matrix(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    normed = embeddings / norms
    return normed @ normed.T

def token_color(idx: int, total: int) -> str:
    colors = ["#f0883e", "#58a6ff", "#3fb950", "#bc8cff", "#ffa657",
              "#79c0ff", "#7ee787", "#d2a8ff", "#ff7b72", "#39d353"]
    return colors[idx % len(colors)]

def render_token_chips(tokens: list, ids: list = None) -> str:
    """Renderiza tokens como chips de color, con el ID numérico debajo cuando se provee."""
    chips = []
    for i, tok in enumerate(tokens):
        color = token_color(i, len(tokens))
        display = repr(tok).strip("'") if tok.strip() == "" else tok
        id_html = f'<span class="id-chip">{ids[i]}</span>' if ids is not None and i < len(ids) else ""
        chip = (
            f'<span style="display:inline-flex;flex-direction:column;align-items:center;margin:1px;">'
            f'<span class="token-chip" style="background:{color}22;border-color:{color};color:{color}" '
            f'title="Token {i}: {repr(tok)}">{display}</span>{id_html}</span>'
        )
        chips.append(chip)
    return f'<div class="token-container">{"".join(chips)}</div>'

def explain_parameter(param: str) -> str:
    explanations = {
        "temperature": """
**🌡️ Temperature** — Controla la *aleatoriedad* de la distribución de probabilidades.

- **Fórmula:** `P(wᵢ) = softmax(logits / T)ᵢ`
- **T → 0:** Determinístico (greedy). Respuestas repetibles y conservadoras.
- **T = 1:** Distribución original del modelo sin modificar.
- **T > 1:** Distribución más plana. Mayor diversidad, mayor riesgo de incoherencia.
- **Rango práctico:** 0.0–2.0. Código/datos: 0.0–0.3. Creatividad: 0.7–1.2.
""",
        "top_p": """
**🎯 Top-P (Nucleus Sampling)** — Muestrea solo del conjunto mínimo de tokens cuyas probabilidades acumuladas alcanzan P.

- **top_p = 1.0:** Considera todos los tokens (sin filtro).
- **top_p = 0.9:** Considera el 90% más probable de la masa de probabilidad.
- **Interacción con temperature:** se aplica después de escalar los logits.
""",
        "max_tokens": """
**📏 Max Tokens** — Límite duro del número máximo de tokens en la respuesta.

- No afecta la *calidad*, solo la *longitud*.
- **finish_reason = "length"**: truncado. **finish_reason = "stop"**: terminó naturalmente.
- Los tokens de output suelen costar más que los de input.
""",
        "reasoning_effort": """
**🧠 Reasoning Effort** — Solo en modelos con razonamiento interno (GPT-OSS, Qwen3).

- `none`: sin tokens de razonamiento (más rápido).
- `low` / `medium` / `high` (GPT-OSS) o `default` (Qwen3): más pasos internos antes de responder.
- Mayor esfuerzo → más latencia y tokens, pero mejor precisión en tareas lógicas/matemáticas.
""",
        "seed": """
**🌱 Seed** — Semilla del generador de números aleatorios para reproducibilidad.

- Mismo seed + mismos parámetros → output (aprox.) idéntico.
- No todos los backends garantizan reproducibilidad perfecta (paralelismo de GPU/LPU).
""",
        "stop": """
**🛑 Stop Sequences** — Strings que detienen la generación al aparecer.

- Ejemplos: `["\\n\\n", "###", "Usuario:"]`
- El string de parada NO se incluye en el output final.
""",
    }
    return explanations.get(param, f"Parámetro: **{param}**")

# ── Chunking helpers (implementaciones ligeras, sin dependencia de LangChain) ──

def chunk_fixed_recursive(text: str, chunk_size: int = 300, overlap: int = 50) -> list:
    """Splitter recursivo: intenta cortar en \\n\\n, luego \\n, luego '. ', luego espacio."""
    seps = ["\n\n", "\n", ". ", " "]

    def _split(t, seps):
        if len(t) <= chunk_size:
            return [t]
        sep = seps[0] if seps else ""
        parts = t.split(sep) if sep else list(t)
        chunks, current = [], ""
        for p in parts:
            piece = (p + sep) if sep else p
            if len(current) + len(piece) <= chunk_size:
                current += piece
            else:
                if current:
                    chunks.append(current)
                if len(piece) > chunk_size and len(seps) > 1:
                    chunks.extend(_split(piece, seps[1:]))
                    current = ""
                else:
                    current = piece
        if current:
            chunks.append(current)
        return chunks

    raw_chunks = [c.strip() for c in _split(text, seps) if c.strip()]
    if overlap <= 0 or len(raw_chunks) < 2:
        return raw_chunks
    overlapped = []
    for i, c in enumerate(raw_chunks):
        if i == 0:
            overlapped.append(c)
        else:
            tail = raw_chunks[i-1][-overlap:]
            overlapped.append((tail + " " + c).strip())
    return overlapped

def chunk_structural_markdown(text: str) -> list:
    """Corta por encabezados Markdown (#, ##, ###), preservando la jerarquía."""
    lines = text.split("\n")
    chunks, current_lines, current_headers = [], [], {}
    header_re = re.compile(r"^(#{1,6})\s+(.*)")

    def flush():
        body = "\n".join(current_lines).strip()
        if body:
            chunks.append({"headers": dict(current_headers), "text": body})

    for line in lines:
        m = header_re.match(line)
        if m:
            flush()
            current_lines = []
            level = len(m.group(1))
            title = m.group(2).strip()
            for k in [k for k in current_headers if k >= level]:
                del current_headers[k]
            current_headers[level] = title
        else:
            current_lines.append(line)
    flush()
    return chunks

def chunk_semantic(sentences: list, embeddings: np.ndarray, threshold: float = 0.55) -> list:
    """Agrupa oraciones consecutivas mientras la similitud coseno con la anterior sea alta."""
    if len(sentences) == 0:
        return []
    sim = cosine_sim_matrix(embeddings)
    groups, current = [], [0]
    for i in range(1, len(sentences)):
        if sim[i, i-1] >= threshold:
            current.append(i)
        else:
            groups.append(current)
            current = [i]
    groups.append(current)
    chunks = []
    for g in groups:
        chunks.append({
            "text": " ".join(sentences[i] for i in g),
            "n_sentences": len(g),
            "avg_boundary_sim": float(np.mean([sim[g[k], g[k-1]] for k in range(1, len(g))])) if len(g) > 1 else 1.0,
        })
    return chunks

PROPOSITIONAL_PROMPT = """Descompón el siguiente texto en proposiciones atómicas: oraciones \
independientes, cada una con un único hecho verificable, sin pronombres ambiguos (reemplázalos \
por el sustantivo que refieren). Devuelve SOLO una lista numerada, una proposición por línea, \
sin texto adicional.

Texto:
\"\"\"{text}\"\"\"

Proposiciones:"""

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

    st.markdown('<p style="font-family:\'JetBrains Mono\',monospace;font-size:0.75rem;color:#8b949e;">🔑 GROQ API KEY</p>', unsafe_allow_html=True)
    api_key_env = os.environ.get("GROQ_API_KEY", "")
    api_key_input = st.text_input(
        "Groq API Key", value=api_key_env, type="password", placeholder="gsk_...",
        label_visibility="collapsed", help="Obtén tu clave gratis en console.groq.com"
    )
    api_key = api_key_input or api_key_env

    if api_key:
        st.markdown('<div class="success-box" style="font-size:0.75rem;">✅ API Key detectada</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="warn-box" style="font-size:0.75rem;">⚠️ Sin API Key — los módulos que llaman a Groq quedan deshabilitados. Tokenización, Embeddings, Chunking (estructural/recursivo/semántico) y NLP Clásico funcionan igual sin key.</div>', unsafe_allow_html=True)

    st.markdown('<hr style="border-color:#30363d;margin:0.8rem 0;">', unsafe_allow_html=True)
    st.markdown('<p style="font-family:\'JetBrains Mono\',monospace;font-size:0.75rem;color:#8b949e;">📍 MÓDULO ACTIVO</p>', unsafe_allow_html=True)
    module = st.selectbox(
        "Módulo",
        options=[
            "🏠  Inicio & Teoría",
            "🔤  Tokenización",
            "📐  Embeddings & Similitud",
            "🧩  Chunking para RAG",
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
    st.markdown('<p style="font-family:\'JetBrains Mono\',monospace;font-size:0.75rem;color:#8b949e;">🤖 MODELO GROQ ACTIVO</p>', unsafe_allow_html=True)
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
    st.caption("⚠️ Catálogo verificado en 2026; puede cambiar. Confirma en [console.groq.com/docs/models](https://console.groq.com/docs/models).")

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
    <span class="badge" style="background:#58a6ff;">Sin PyTorch</span>
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
        st.markdown('<div class="metric-card"><div class="metric-value">10</div><div class="metric-label">Módulos interactivos</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#58a6ff;">{len(GROQ_MODELS)}</div><div class="metric-label">Modelos Groq (gratis)</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#3fb950;">{len(EMBEDDING_MODELS)}</div><div class="metric-label">Modelos de embedding</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><div class="metric-value" style="color:#bc8cff;">0</div><div class="metric-label">Dependencias de PyTorch</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    tab_theory, tab_models, tab_embed, tab_chunk = st.tabs(
        ["📖 Teoría NLP", "🤖 Modelos Groq", "📐 Embeddings", "🧩 Chunking"]
    )

    with tab_theory:
        col_left, col_right = st.columns([1, 1])
        with col_left:
            st.markdown("""
            <div class="section-title" style="font-size:1.1rem;">Pipeline NLP Completo</div>
            <div class="info-box"><b>1. Tokenización</b><br>Texto crudo → unidades mínimas. Algoritmos: BPE, WordPiece, SentencePiece.</div>
            <div class="info-box"><b>2. Normalización</b><br>Minúsculas, acentos, URLs, stopwords.</div>
            <div class="info-box"><b>3. Representación vectorial</b><br>BoW → TF-IDF → Embeddings densos contextuales.</div>
            <div class="info-box"><b>4. Chunking</b><br>Dividir documentos largos en fragmentos indexables para RAG.</div>
            <div class="info-box"><b>5. Modelos de secuencia</b><br>n-gramas → RNN/LSTM → Transformer.</div>
            <div class="info-box"><b>6. Tareas downstream</b><br>Clasificación, NER, Sentiment, QA, RAG, Generación.</div>
            """, unsafe_allow_html=True)
        with col_right:
            st.markdown('<div class="section-title" style="font-size:1.1rem;">Ecuaciones Clave</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="formula-box">TF-IDF(t,d) = tf(t,d) x log(N / df(t))

tf(t,d)  = f(t,d) / sum_k f(k,d)
idf(t)   = log(N / |{d: t in d}|)</div>
            <div class="formula-box">Similitud Coseno:
cos(theta) = (u . v) / (||u|| ||v||)  en [-1, 1]

cos = 1  -> vectores idénticos en dirección
cos = 0  -> vectores ortogonales (sin relación)
cos = -1 -> vectores opuestos</div>
            <div class="formula-box">Softmax con temperatura T:
P(w_i) = exp(logit_i / T) / sum_j exp(logit_j / T)

T->0: determinístico (greedy)
T=1:  distribución original
T->inf: uniforme (caótico)</div>
            <div class="formula-box">Scaled Dot-Product Attention:
A(Q,K,V) = softmax(QK^T / sqrt(d_k)) . V</div>
            """, unsafe_allow_html=True)

    with tab_models:
        st.markdown('<div class="section-title" style="font-size:1.1rem;">Modelos Disponibles en Groq (gratis)</div>', unsafe_allow_html=True)
        rows = []
        for m, info in GROQ_MODELS.items():
            rows.append({"Model ID (Groq)": m, "Familia": info["family"], "Parámetros": info["params"],
                         "Contexto (tokens)": f"{info['context']:,}", "Tipo": info["type"],
                         "Licencia": info["license"], "Fortalezas": info["strengths"]})
        st.dataframe(pd.DataFrame(rows), width="stretch", height=280)
        st.markdown("""
        <div class="info-box">
        <b>💡 ¿Por qué Groq?</b> Usa hardware <b>LPU (Language Processing Unit)</b> especializado,
        con velocidades de inferencia 10–100x superiores a GPUs convencionales:
        típicamente 200–800+ tokens/segundo, sin tarjeta de crédito.
        </div>
        <div class="warn-box">
        <b>⚠️ El catálogo cambia:</b> Groq retira y agrega modelos con frecuencia (por ejemplo,
        <code>mixtral-8x7b-32768</code> y <code>gemma2-9b-it</code> ya no están disponibles).
        Verifica siempre <a href="https://console.groq.com/docs/models" style="color:#58a6ff;">console.groq.com/docs/models</a>
        antes de una clase en vivo.
        </div>
        """, unsafe_allow_html=True)

    with tab_embed:
        st.markdown('<div class="section-title" style="font-size:1.1rem;">Modelos de Embedding</div>', unsafe_allow_html=True)
        rows_e = [{"Modelo": k, "HF id": v["hf_id"], "Dimensiones": v["dims"], "Contexto (tokens)": v["context"],
                   "Tamaño aprox.": v["size"], "Nota": v["note"]} for k, v in EMBEDDING_MODELS.items()]
        st.dataframe(pd.DataFrame(rows_e), width="stretch", height=200)
        st.markdown("""
        <div class="info-box">
        Estos tres modelos corren localmente vía <code>fastembed</code> (ONNX Runtime), <b>sin
        PyTorch</b> — Groq no ofrece un endpoint propio de embeddings, así que la práctica estándar
        en la comunidad es: <b>embeddings locales</b> + <b>generación vía Groq</b>.
        </div>
        """, unsafe_allow_html=True)

    with tab_chunk:
        st.markdown('<div class="section-title" style="font-size:1.1rem;">Estrategias de Chunking para RAG</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box"><b>Estructural</b> (Markdown headers) — corta siguiendo la jerarquía real
        del documento (#, ##, ###). Preserva contexto de sección.</div>
        <div class="info-box"><b>Recursivo / Fixed-size</b> — corta por tamaño de caracteres, respetando
        párrafos y oraciones antes de cortar a la fuerza. Con overlap para no perder contexto.</div>
        <div class="info-box"><b>Semántico</b> — agrupa oraciones consecutivas mientras su similitud
        coseno (embeddings) se mantenga alta; corta cuando el tema cambia.</div>
        <div class="info-box"><b>Proposicional / Agéntico</b> — usa un LLM (Groq) para reescribir el
        texto en proposiciones atómicas autocontenidas antes de indexar. Más costoso, más preciso.</div>
        """, unsafe_allow_html=True)
        st.markdown("Pruébalas en el módulo **🧩 Chunking para RAG** de la barra lateral.")

# ════════════════════════════════════════════════════════════
# MODULE 1: TOKENIZACIÓN
# ════════════════════════════════════════════════════════════
elif module == "🔤  Tokenización":
    st.markdown('<div class="section-title">Tokenización: Del Texto a los IDs</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <b>¿Qué es un token?</b> La unidad mínima que procesa un LLM. Puede ser un carácter,
    una subpalabra, una palabra o un símbolo especial. <b>Los modelos nunca ven texto crudo:
    solo ven IDs enteros</b> — un índice dentro del vocabulario del tokenizador.
    </div>
    """, unsafe_allow_html=True)

    input_text = st.text_area("Texto de entrada", value=SAMPLE_TEXTS["mixed"], height=120,
                               help="Escribe cualquier texto. Prueba con código, números, palabras raras.")

    tab_bpe, tab_gpt2, tab_bert, tab_compare = st.tabs(
        ["🔧 BPE (cl100k, ref. GPT-3.5/4)", "🐣 GPT-2 (BPE original, 2019)", "🤗 BERT (WordPiece)", "📊 Comparativa"]
    )

    with tab_bpe:
        col_explain, col_result = st.columns([1, 1])
        with col_explain:
            st.markdown("""
            <div class="section-title" style="font-size:1rem;">Byte-Pair Encoding (BPE)</div>
            <div class="formula-box">Algoritmo BPE:
1. Vocabulario inicial = caracteres únicos
2. Cuenta frecuencia de cada par adyacente
3. Fusiona el par más frecuente -> nuevo token
4. Repite hasta vocabulario de tamaño K

cl100k_base: ~100,277 tokens de vocabulario
Referencia histórica de la familia BPE moderna
(no es el tokenizador exacto de los modelos que
sirve hoy Groq, que usan sus propios BPE internos,
pero sirve como referencia comparativa universal)</div>
            """, unsafe_allow_html=True)
        with col_result:
            if TIKTOKEN_AVAILABLE:
                tokens_bpe, ids_bpe = tokenize_with_tiktoken(input_text, "cl100k_base")
                n_tokens, n_words = len(tokens_bpe), len(input_text.split())
                c1, c2, c3 = st.columns(3)
                c1.metric("Tokens", n_tokens)
                c2.metric("Palabras", n_words)
                c3.metric("Ratio tok/word", f"{n_tokens/max(n_words,1):.2f}")
                st.markdown("**Tokens + Token IDs:**")
                st.markdown(render_token_chips(tokens_bpe[:60], ids_bpe[:60]), unsafe_allow_html=True)
                if len(tokens_bpe) > 60:
                    st.caption(f"... mostrando primeros 60 de {n_tokens} tokens")
                token_lengths = [len(t) for t in tokens_bpe]
                fig_dist = px.histogram(x=token_lengths, nbins=20, title="Distribución de longitud de tokens",
                                         labels={"x": "Caracteres por token", "y": "Frecuencia"},
                                         color_discrete_sequence=["#f0883e"])
                fig_dist.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                                        font=dict(color="#e6edf3"), title_font=dict(size=12), height=200)
                fig_dist.update_xaxes(gridcolor="#30363d"); fig_dist.update_yaxes(gridcolor="#30363d")
                st.plotly_chart(fig_dist, width="stretch")
            else:
                st.warning("tiktoken no disponible. Instala: `pip install tiktoken`")

    with tab_gpt2:
        col_explain, col_result = st.columns([1, 1])
        with col_explain:
            st.markdown("""
            <div class="section-title" style="font-size:1rem;">GPT-2 (BPE byte-level, 2019)</div>
            <div class="formula-box">Vocabulario: 50,257 tokens
Byte-level BPE: opera sobre bytes UTF-8,
nunca produce un token "desconocido" (UNK)

Este es el tokenizador ORIGINAL de GPT-2 (OpenAI,
2019), el ancestro directo de todos los BPE
modernos. Se obtiene aquí vía tiktoken, sin
ninguna dependencia de PyTorch/transformers.

Compáralo con cl100k_base: mismo algoritmo,
vocabulario ~2x más pequeño -> más tokens
para el mismo texto.</div>
            """, unsafe_allow_html=True)
        with col_result:
            if TIKTOKEN_AVAILABLE:
                tokens_g, ids_g = tokenize_with_tiktoken(input_text, "gpt2")
                toks_g_clean = [t.replace("Ġ", "·").replace("\n", "⏎") for t in tokens_g]
                c1, c2 = st.columns(2)
                c1.metric("Tokens GPT-2", len(tokens_g))
                c2.metric("Vocab size", "50,257")
                st.caption("`·` representa un espacio antes del token (carácter Ġ interno de GPT-2, byte-level BPE)")
                st.markdown(render_token_chips(toks_g_clean[:60], ids_g[:60]), unsafe_allow_html=True)
                if len(tokens_g) > 60:
                    st.caption(f"... mostrando primeros 60 de {len(tokens_g)} tokens")
            else:
                st.warning("tiktoken no disponible. Instala: `pip install tiktoken`")

    with tab_bert:
        col_explain, col_result = st.columns([1, 1])
        with col_explain:
            st.markdown("""
            <div class="section-title" style="font-size:1rem;">WordPiece (BERT)</div>
            <div class="formula-box">Fusiona pares que maximizan:
score(a,b) = freq(ab) / (freq(a) x freq(b))

Diferencia vs BPE: BPE fusiona por frecuencia
absoluta; WordPiece por ganancia relativa.

Prefijo "##" = continuación de la palabra anterior
"transformación" -> ['trans','##form','##aci','##ón']

Vocab BERT-multilingual: 119,547 tokens
(vía la librería `tokenizers`, sin PyTorch)</div>
            """, unsafe_allow_html=True)
        with col_result:
            bert_tok = load_bert_tokenizer()
            if bert_tok:
                try:
                    encoding = bert_tok.encode(input_text)
                    ids_bert, tokens_bert = encoding.ids, encoding.tokens
                    c1, c2 = st.columns(2)
                    c1.metric("Tokens BERT", len(tokens_bert))
                    c2.metric("Incluye [CLS]/[SEP]", "Sí" if tokens_bert and tokens_bert[0] == "[CLS]" else "Depende del modelo")
                    st.markdown("**Tokens + IDs WordPiece:**")
                    st.markdown(render_token_chips(tokens_bert[:60], ids_bert[:60]), unsafe_allow_html=True)
                    with st.expander("Ver tabla completa de IDs"):
                        df_toks = pd.DataFrame({
                            "Posición": range(len(tokens_bert[:40])), "Token": tokens_bert[:40],
                            "ID": ids_bert[:40],
                            "¿Subpalabra?": ["Sí" if t.startswith("##") else "No" for t in tokens_bert[:40]]
                        })
                        st.dataframe(df_toks, width="stretch")
                except Exception as e:
                    st.error(f"Error tokenizando: {e}")
            else:
                st.info("Cargando tokenizador BERT-multilingual... (descarga inicial ligera del tokenizer.json, sin PyTorch)")

    with tab_compare:
        st.markdown('<div class="section-title" style="font-size:1rem;">Comparativa de Tokenizadores</div>', unsafe_allow_html=True)
        results = {}
        if TIKTOKEN_AVAILABLE:
            toks, _ = tokenize_with_tiktoken(input_text, "cl100k_base")
            results["BPE cl100k (ref. GPT-3.5/4)"] = {"tokens": len(toks), "vocab_size": "~100,277",
                "algo": "Byte-Pair Encoding", "era": "2023", "ejemplo": " | ".join(toks[:8]) + "..."}
            toks_g, _ = tokenize_with_tiktoken(input_text, "gpt2")
            results["BPE (GPT-2 original)"] = {"tokens": len(toks_g), "vocab_size": "50,257",
                "algo": "BPE byte-level", "era": "2019", "ejemplo": " | ".join(str(t) for t in toks_g[:8]) + "..."}
        bt = load_bert_tokenizer()
        if bt:
            try:
                enc = bt.encode(input_text)
                toks_b = enc.tokens
                results["WordPiece (BERT-multilingual)"] = {"tokens": len(toks_b), "vocab_size": "119,547",
                    "algo": "WordPiece", "era": "2018", "ejemplo": " | ".join(toks_b[:8]) + "..."}
            except Exception:
                pass
        results["Whitespace split (baseline)"] = {"tokens": len(input_text.split()), "vocab_size": "Abierto",
            "algo": "Separación por espacios", "era": "—", "ejemplo": " | ".join(input_text.split()[:8]) + "..."}

        df_compare = pd.DataFrame(results).T.reset_index()
        df_compare.columns = ["Tokenizador", "# Tokens", "Vocab", "Algoritmo", "Época", "Ejemplo (primeros 8)"]
        st.dataframe(df_compare, width="stretch")

        fig_tok = px.bar(df_compare, x="Tokenizador", y="# Tokens", title="Número de tokens según tokenizador",
                          color="# Tokens", color_continuous_scale=["#58a6ff", "#f0883e", "#f85149"])
        fig_tok.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22", font=dict(color="#e6edf3"),
                               height=300, showlegend=False)
        fig_tok.update_xaxes(gridcolor="#30363d", tickangle=-20); fig_tok.update_yaxes(gridcolor="#30363d")
        st.plotly_chart(fig_tok, width="stretch")

        st.markdown("""
        <div class="warn-box">
        <b>⚠️ Observación clave:</b> El mismo texto produce cantidades de tokens muy distintas según
        el algoritmo y su vocabulario. Esto afecta directamente el <b>costo</b> de las APIs (cobran
        por token), los <b>límites de contexto</b> y la <b>eficiencia de memoria</b> en inferencia.
        El español suele generar ~30–50% más tokens que el inglés equivalente en modelos entrenados
        mayormente en inglés.
        </div>
        """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# MODULE 2: EMBEDDINGS
# ════════════════════════════════════════════════════════════
elif module == "📐  Embeddings & Similitud":
    st.markdown('<div class="section-title">Embeddings y Similitud Semántica</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    Los <b>embeddings</b> son vectores densos en ℝᵈ donde la distancia geométrica captura
    proximidad semántica. Este es el salto de BoW/TF-IDF (dispersos, sin significado) a
    representaciones distribuidas (densas, contextuales).
    </div>
    """, unsafe_allow_html=True)

    tab_bow, tab_tfidf, tab_words, tab_dense, tab_viz = st.tabs(
        ["🧺 Bag of Words", "📊 TF-IDF (disperso)", "🔍 Palabras cortas/similares",
         "🤗 Embeddings densos + similitud coseno", "🗺️ Visualización 2D"]
    )

    with tab_bow:
        st.markdown('<div class="section-title" style="font-size:1rem;">Bag of Words (BoW): Conteo Puro</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box"><b>Bag of Words</b> representa cada documento como un vector de
        <i>conteos de palabras</i>, ignorando orden y gramática — la representación vectorial más
        simple de todas. A diferencia de TF-IDF, aquí una palabra muy frecuente en <b>todos</b> los
        documentos (poco informativa) pesa exactamente igual que una palabra rara y distintiva.</div>
        <div class="formula-box">BoW(t,d) = conteo de veces que el término t aparece en el documento d
(vector disperso de tamaño |vocabulario|, sin ponderar por importancia)</div>
        """, unsafe_allow_html=True)
        col_bow1, col_bow2 = st.columns([1, 2])
        with col_bow1:
            default_bow_docs = [
                "El gato duerme en el sofá todo el día.",
                "El perro juega en el parque con el gato.",
                "Los modelos de lenguaje procesan texto y generan texto.",
            ]
            bow_docs_input = st.text_area("Un documento por línea:", value="\n".join(default_bow_docs), height=140, key="bow_docs")
            bow_documents = [d.strip() for d in bow_docs_input.split("\n") if d.strip()]
            bow_binary = st.checkbox("Binario (presencia/ausencia, no conteo)", value=False, key="bow_binary")
            bow_lowercase = st.checkbox("Minúsculas", value=True, key="bow_lower")
        with col_bow2:
            if SKLEARN_AVAILABLE and len(bow_documents) >= 1:
                try:
                    from sklearn.feature_extraction.text import CountVectorizer
                    cv = CountVectorizer(lowercase=bow_lowercase, binary=bow_binary)
                    Xc = cv.fit_transform(bow_documents)
                    vocab = cv.get_feature_names_out()
                    Xc_dense = Xc.toarray()
                    df_bow = pd.DataFrame(Xc_dense, columns=vocab, index=[f"D{i+1}" for i in range(len(bow_documents))])
                    st.markdown(f"**Vocabulario:** {len(vocab)} términos únicos")
                    st.dataframe(df_bow, width="stretch", height=180)

                    freq_total = Xc_dense.sum(axis=0)
                    top_n = min(15, len(vocab))
                    top_order = np.argsort(freq_total)[::-1][:top_n]
                    fig_bow = px.bar(x=vocab[top_order], y=freq_total[top_order],
                                      title="Frecuencia total de términos (corpus completo)",
                                      labels={"x": "término", "y": "frecuencia"})
                    fig_bow.update_traces(marker_color="#f0883e")
                    fig_bow.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                                           font=dict(color="#e6edf3"), height=280, xaxis=dict(tickangle=-40))
                    st.plotly_chart(fig_bow, width="stretch")

                    if len(bow_documents) >= 2:
                        sim_bow = cosine_similarity(Xc)
                        fig_bowsim = px.imshow(sim_bow, x=[f"D{i+1}" for i in range(len(bow_documents))],
                                                y=[f"D{i+1}" for i in range(len(bow_documents))],
                                                title="Similitud coseno entre documentos (BoW)",
                                                color_continuous_scale="RdYlGn", zmin=0, zmax=1, text_auto=".2f")
                        fig_bowsim.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                                                  font=dict(color="#e6edf3", size=10), height=260)
                        st.plotly_chart(fig_bowsim, width="stretch")
                    st.markdown("""
                    <div class="warn-box"><b>💡 Limitación clave de BoW:</b> "el gato" y "gato el"
                    producen el <b>mismo</b> vector — se pierde el orden. Y palabras muy comunes
                    (como "el", "de", "en") dominan los conteos sin aportar significado; por eso
                    TF-IDF (siguiente pestaña) pondera por rareza, y los embeddings (más adelante)
                    capturan significado real.</div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error calculando Bag of Words: {e}")
            else:
                st.warning("Escribe al menos un documento y asegúrate de tener scikit-learn instalado.")

    with tab_tfidf:
        st.markdown('<div class="section-title" style="font-size:1rem;">TF-IDF: Representación Dispersa</div>', unsafe_allow_html=True)
        col_params, col_results = st.columns([1, 2])
        with col_params:
            default_docs = [
                "La inteligencia artificial transforma la educación universitaria.",
                "Los modelos de lenguaje aprenden patrones en texto masivo.",
                "El fútbol colombiano tiene talento mundial en sus jugadores.",
                "BERT y GPT son arquitecturas Transformer para NLP.",
                "La economía colombiana creció un 3% en el último trimestre.",
                "Deep learning revolucionó el reconocimiento de imágenes y texto.",
            ]
            docs_input = st.text_area("Un documento por línea:", value="\n".join(default_docs), height=180)
            documents = [d.strip() for d in docs_input.split("\n") if d.strip()]
            tfidf_max_feat = st.slider("Max features", 10, 100, 30)
            tfidf_ngram_max = st.radio("N-gramas", [1, 2], index=1, horizontal=True)
        with col_results:
            if SKLEARN_AVAILABLE and len(documents) >= 2:
                try:
                    vec = TfidfVectorizer(max_features=tfidf_max_feat, ngram_range=(1, tfidf_ngram_max), min_df=1)
                    X = vec.fit_transform(documents)
                    feature_names = vec.get_feature_names_out()
                    X_dense = X.toarray()
                    top_idx = np.argsort(X_dense.sum(axis=0))[-min(20, len(feature_names)):]
                    fig_heat = px.imshow(X_dense[:, top_idx], x=feature_names[top_idx],
                                          y=[f"D{i+1}" for i in range(len(documents))],
                                          title="Matriz TF-IDF (top términos)", color_continuous_scale="Blues", aspect="auto")
                    fig_heat.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                                            font=dict(color="#e6edf3", size=10), height=250, title_font=dict(size=12))
                    st.plotly_chart(fig_heat, width="stretch")

                    sim_matrix = cosine_similarity(X)
                    fig_sim = px.imshow(sim_matrix, x=[f"D{i+1}" for i in range(len(documents))],
                                         y=[f"D{i+1}" for i in range(len(documents))],
                                         title="Similitud Coseno entre documentos (TF-IDF)",
                                         color_continuous_scale="RdYlGn", zmin=0, zmax=1, text_auto=".2f")
                    fig_sim.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                                           font=dict(color="#e6edf3", size=10), height=280, title_font=dict(size=12))
                    st.plotly_chart(fig_sim, width="stretch")
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.warning("Necesitas al menos 2 documentos y scikit-learn instalado.")

    with tab_words:
        st.markdown('<div class="section-title" style="font-size:1rem;">Distancia Coseno entre Palabras Cortas / Similares</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">Compara palabras sueltas (no oraciones) para ver qué tan cerca están
        en el espacio vectorial. Útil para explorar sinónimos, variaciones morfológicas
        ("gato" vs "gata" vs "gatos") o pares no relacionados. Puedes elegir dos motores:</div>
        <ul style="font-size:0.85rem;color:#8b949e;">
        <li><b>Char n-gramas (TF-IDF)</b>: no requiere descargar ningún modelo, funciona sin
        internet. Detecta similitud <i>de forma/ortografía</i> (útil para variantes morfológicas),
        no de significado.</li>
        <li><b>Embeddings densos</b>: usa uno de los modelos de la pestaña anterior, detecta
        similitud <i>de significado</i> aunque las palabras se escriban distinto.</li>
        </ul>
        """, unsafe_allow_html=True)

        word_engine = st.radio("Motor de comparación:", ["Char n-gramas (TF-IDF, sin descargas)", "Embeddings densos"],
                                horizontal=True, key="word_engine")
        default_words = "gato\ngata\ngatos\nperro\nautomóvil\ncarro\nplátano"
        words_input = st.text_area("Una palabra (o palabra corta) por línea:", value=default_words, height=150, key="words_input")
        word_list = [w.strip() for w in words_input.split("\n") if w.strip()]

        if len(word_list) >= 2:
            try:
                if word_engine.startswith("Char"):
                    if not SKLEARN_AVAILABLE:
                        st.warning("Necesitas scikit-learn instalado.")
                        word_sim = None
                    else:
                        wv = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), min_df=1)
                        Xw = wv.fit_transform(word_list)
                        word_sim = cosine_similarity(Xw)
                else:
                    word_embed_choice = st.selectbox(
                        "Modelo de embedding:", list(EMBEDDING_MODELS.keys()),
                        format_func=lambda k: f"{k} ({EMBEDDING_MODELS[k]['dims']}d)", key="word_embed_model"
                    )
                    word_sim = None
                    if st.button("⚡ Calcular", key="word_emb_btn"):
                        emb_model_w = load_embedding_model(word_embed_choice)
                        if emb_model_w:
                            with st.spinner("Calculando embeddings de las palabras..."):
                                embs_w = embed_texts(emb_model_w, word_list)
                            word_sim = cosine_sim_matrix(embs_w)
                        else:
                            st.error(f"No se pudo cargar {word_embed_choice}.")

                if word_sim is not None:
                    fig_wsim = px.imshow(word_sim, x=word_list, y=word_list,
                                          title=f"Similitud coseno entre palabras — {word_engine}",
                                          color_continuous_scale="RdYlGn", zmin=0 if word_engine.startswith("Char") else -1,
                                          zmax=1, text_auto=".2f", aspect="auto")
                    fig_wsim.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                                            font=dict(color="#e6edf3", size=11), height=max(300, len(word_list) * 35 + 100))
                    st.plotly_chart(fig_wsim, width="stretch")

                    pairs_w = []
                    for i in range(len(word_list)):
                        for j in range(i + 1, len(word_list)):
                            pairs_w.append({
                                "Palabra A": word_list[i], "Palabra B": word_list[j],
                                "Similitud coseno": round(float(word_sim[i, j]), 4),
                                "Distancia coseno (1 - sim)": round(1.0 - float(word_sim[i, j]), 4),
                            })
                    df_pairs_w = pd.DataFrame(pairs_w).sort_values("Similitud coseno", ascending=False)
                    st.markdown("**Ranking de pares (más similares primero):**")
                    st.dataframe(df_pairs_w, width="stretch", height=min(300, 40 + 35 * len(df_pairs_w)))
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Escribe al menos 2 palabras (una por línea).")

    with tab_dense:
        st.markdown('<div class="section-title" style="font-size:1rem;">Embeddings Contextuales Densos</div>', unsafe_allow_html=True)
        embed_choice = st.selectbox(
            "Modelo de embedding:", list(EMBEDDING_MODELS.keys()),
            format_func=lambda k: f"{k} ({EMBEDDING_MODELS[k]['dims']}d, {EMBEDDING_MODELS[k]['size']})"
        )
        emb_info = EMBEDDING_MODELS[embed_choice]
        st.markdown(f"""
        <div class="info-box">
        <b>{embed_choice}</b> — {emb_info['note']}<br>
        📦 <code>{emb_info['hf_id']}</code> · 📐 {emb_info['dims']} dimensiones ·
        📚 contexto {emb_info['context']} tokens · 💾 {emb_info['size']}
        </div>
        """, unsafe_allow_html=True)

        default_sents = [
            "El banco aprobó el crédito hipotecario.",
            "La institución financiera otorgó el préstamo.",
            "Me senté en el banco del parque a leer.",
            "La inteligencia artificial cambia el mundo.",
            "El partido de fútbol terminó en empate.",
            "Los modelos de lenguaje comprenden el texto.",
        ]
        sents_input = st.text_area("Frases (una por línea):", value="\n".join(default_sents), height=160)
        sentences = [s.strip() for s in sents_input.split("\n") if s.strip()]

        if st.button("⚡ Calcular Embeddings", key="emb_btn") and len(sentences) >= 2:
            emb_model = load_embedding_model(embed_choice)
            if emb_model:
                with st.spinner("Calculando embeddings (ONNX Runtime, sin PyTorch)..."):
                    t0 = time.perf_counter()
                    embeddings = embed_texts(emb_model, sentences)
                    elapsed = time.perf_counter() - t0

                st.success(f"✅ {len(sentences)} embeddings de {embeddings.shape[1]} dimensiones en {elapsed:.3f}s")

                with st.expander("🔍 Ver el vector crudo (primeras 20 dimensiones) de la Frase 1"):
                    df_vec = pd.DataFrame({"Dimensión (índice)": range(20), "Valor": embeddings[0][:20].round(4)})
                    st.dataframe(df_vec, width="stretch", height=200)
                    st.caption("A diferencia de un token ID (entero, categórico), cada componente de un embedding "
                               "es un número real continuo — no representa una palabra por sí sola, solo cobra "
                               "sentido en conjunto con las demás dimensiones.")

                sim = cosine_sim_matrix(embeddings)
                fig_csim = px.imshow(sim, x=[f"S{i+1}" for i in range(len(sentences))],
                                      y=[f"S{i+1}" for i in range(len(sentences))],
                                      title=f"Similitud Coseno — {embed_choice}", color_continuous_scale="RdYlGn",
                                      zmin=-1, zmax=1, text_auto=".3f")
                fig_csim.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                                        font=dict(color="#e6edf3"), height=350, title_font=dict(size=13))
                st.plotly_chart(fig_csim, width="stretch")

                for i, s in enumerate(sentences):
                    st.markdown(f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.75rem;color:#f0883e;background:#1c2333;padding:2px 6px;border-radius:3px;">S{i+1}</span> <span style="font-size:0.85rem;"> {s}</span><br>', unsafe_allow_html=True)

                pairs = []
                for i in range(len(sentences)):
                    for j in range(i+1, len(sentences)):
                        pairs.append({"Frase A": f"S{i+1}: {sentences[i][:45]}...",
                                       "Frase B": f"S{j+1}: {sentences[j][:45]}...",
                                       "Similitud": round(float(sim[i, j]), 4)})
                df_pairs = pd.DataFrame(pairs).sort_values("Similitud", ascending=False)
                st.markdown("**Ranking de pares por similitud coseno:**")
                st.dataframe(df_pairs, width="stretch", height=220)

                st.markdown("""
                <div class="warn-box">
                <b>💡 Observa:</b> S1 y S2 ("banco crédito" y "préstamo") deberían tener alta similitud.
                S1 y S3 ("banco parque") deberían tener baja similitud a pesar de compartir la palabra
                "banco". Esto demuestra que los embeddings contextuales resuelven la polisemia — algo
                que TF-IDF nunca podría lograr.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error(f"No se pudo cargar {embed_choice}. Revisa que `fastembed` esté instalado (`pip install fastembed`).")

    with tab_viz:
        st.markdown('<div class="section-title" style="font-size:1rem;">Proyección 2D de Embeddings (SVD)</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
        Los embeddings viven en espacios de alta dimensión (384–1024 dims aquí). Proyectamos a 2D
        con SVD (similar a PCA). La separación en clusters indica que el modelo captura distintas
        áreas semánticas sin supervisión explícita.
        </div>
        """, unsafe_allow_html=True)
        thematic_sentences = {
            "IA/NLP": ["Los transformers revolucionaron el procesamiento de lenguaje.",
                       "BERT usa atención bidireccional para embeddings contextuales.",
                       "Los LLMs generan texto coherente con miles de millones de parámetros."],
            "Economía": ["La inflación colombiana bajó al 5.2% en diciembre.",
                         "El Banco de la República mantuvo tasas de interés.",
                         "El PIB creció impulsado por las exportaciones de café."],
            "Deportes": ["Colombia clasificó al mundial con una victoria contundente.",
                        "El equipo de fútbol entrenó en la altitud de Bogotá.",
                        "El delantero marcó un golazo en el último partido."],
        }
        all_sents, labels = [], []
        color_map = {"IA/NLP": "#f0883e", "Economía": "#58a6ff", "Deportes": "#3fb950"}
        for cat, sents in thematic_sentences.items():
            all_sents.extend(sents); labels.extend([cat] * len(sents))

        viz_model = st.selectbox("Modelo para la proyección:", list(EMBEDDING_MODELS.keys()), key="viz_model")
        if st.button("🗺️ Visualizar espacio semántico", key="viz_btn"):
            emb_model = load_embedding_model(viz_model)
            if emb_model and SKLEARN_AVAILABLE:
                with st.spinner("Calculando embeddings y proyección..."):
                    embs = embed_texts(emb_model, all_sents)
                    svd = TruncatedSVD(n_components=2, random_state=42)
                    coords = svd.fit_transform(embs)
                df_viz = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "Categoría": labels,
                                        "Texto": [s[:50] + "..." for s in all_sents]})
                fig_viz = px.scatter(df_viz, x="x", y="y", color="Categoría", text="Texto",
                                      title="Proyección 2D de embeddings por categoría semántica",
                                      color_discrete_map=color_map)
                fig_viz.update_traces(textposition="top center", textfont=dict(size=8), marker=dict(size=12, opacity=0.85))
                fig_viz.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                                       font=dict(color="#e6edf3"), height=450, title_font=dict(size=13))
                fig_viz.update_xaxes(gridcolor="#30363d"); fig_viz.update_yaxes(gridcolor="#30363d")
                st.plotly_chart(fig_viz, width="stretch")
                st.markdown("""
                <div class="success-box">
                ✅ Las frases de la misma categoría se agrupan en el espacio 2D sin ninguna
                supervisión de categorías. Esta capacidad es la base del RAG, la búsqueda semántica
                y la clasificación zero-shot.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("Necesitas `fastembed` y `scikit-learn` instalados.")

# ════════════════════════════════════════════════════════════
# MODULE 3: CHUNKING PARA RAG
# ════════════════════════════════════════════════════════════
elif module == "🧩  Chunking para RAG":
    st.markdown('<div class="section-title">Estrategias de Chunking para RAG</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    Antes de indexar un documento en un vector store, hay que dividirlo en <b>chunks</b>. La
    estrategia de corte determina la calidad del retrieval: chunks mal cortados rompen el
    contexto; chunks demasiado grandes diluyen la relevancia. Prueba las 4 estrategias con
    el mismo documento.
    </div>
    """, unsafe_allow_html=True)

    tab_struct, tab_fixed, tab_sem, tab_agent = st.tabs(
        ["🏗️ Estructural (Markdown)", "✂️ Recursivo / Fixed-size", "🧠 Semántico (coseno)", "🤖 Proposicional / Agéntico"]
    )

    with tab_struct:
        st.markdown("""
        <div class="formula-box">Corta por encabezados Markdown (#, ##, ###...)
Preserva la jerarquía como metadata de cada chunk.</div>
        """, unsafe_allow_html=True)
        doc_md = st.text_area("Documento Markdown:", value=SAMPLE_TEXTS["rag_doc"], height=220, key="md_doc")
        chunks_struct = chunk_structural_markdown(doc_md)
        st.markdown(f"**{len(chunks_struct)} chunks generados:**")
        for i, c in enumerate(chunks_struct):
            path = " › ".join(v for k, v in sorted(c["headers"].items()))
            st.markdown(f"""
            <div class="chunk-box">
            <span style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#39d353;">
            Chunk {i+1} — {path if path else "(sin encabezado)"}</span><br>
            {c['text'][:300]}{'...' if len(c['text']) > 300 else ''}
            </div>""", unsafe_allow_html=True)

    with tab_fixed:
        st.markdown("""
        <div class="formula-box">Corta por tamaño fijo de caracteres, intentando
respetar párrafos (\\n\\n) y oraciones antes de
cortar a la fuerza. Con overlap para no perder
contexto en los bordes entre chunks.</div>
        """, unsafe_allow_html=True)
        doc_fixed = st.text_area("Documento:", value=SAMPLE_TEXTS["rag_doc"], height=180, key="fixed_doc")
        c1, c2 = st.columns(2)
        chunk_size = c1.slider("chunk_size (caracteres)", 50, 800, 200, 10)
        chunk_overlap = c2.slider("overlap (caracteres)", 0, 200, 30, 5)
        chunks_fixed = chunk_fixed_recursive(doc_fixed, chunk_size, chunk_overlap)
        st.markdown(f"**{len(chunks_fixed)} chunks generados** (tamaño objetivo: {chunk_size} chars):")
        lengths = [len(c) for c in chunks_fixed]
        for i, c in enumerate(chunks_fixed):
            st.markdown(f"""
            <div class="chunk-box">
            <span style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#39d353;">
            Chunk {i+1} — {len(c)} caracteres</span><br>{c}
            </div>""", unsafe_allow_html=True)
        if lengths:
            fig_len = px.bar(x=[f"C{i+1}" for i in range(len(lengths))], y=lengths,
                              title="Longitud de cada chunk (caracteres)", color_discrete_sequence=["#58a6ff"])
            fig_len.add_hline(y=chunk_size, line_dash="dash", line_color="#f0883e", annotation_text="chunk_size objetivo")
            fig_len.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22", font=dict(color="#e6edf3"), height=250)
            st.plotly_chart(fig_len, width="stretch")

    with tab_sem:
        st.markdown("""
        <div class="formula-box">1. Divide el texto en oraciones
2. Calcula el embedding de cada oración
3. Agrupa oraciones consecutivas mientras la
   similitud coseno con la anterior >= threshold
4. Corta cuando el tema cambia (similitud cae)</div>
        """, unsafe_allow_html=True)
        if not (FASTEMBED_AVAILABLE and NLTK_AVAILABLE):
            st.warning("Requiere `fastembed` y NLTK instalados.")
        else:
            doc_sem = st.text_area("Documento:", value=SAMPLE_TEXTS["rag_doc"].replace("#", "").replace("\n\n", " "),
                                    height=150, key="sem_doc")
            sem_threshold = st.slider("Umbral de similitud coseno (threshold)", 0.1, 0.95, 0.55, 0.05)
            sem_model = st.selectbox("Modelo de embedding:", list(EMBEDDING_MODELS.keys()), key="sem_model")
            if st.button("🧠 Chunkear semánticamente", key="run_sem_chunk"):
                try:
                    sentences = [s.strip() for s in sent_tokenize(doc_sem) if s.strip()]
                except Exception:
                    sentences = [s.strip() for s in doc_sem.split(".") if s.strip()]
                emb_model = load_embedding_model(sem_model)
                if emb_model and len(sentences) >= 2:
                    with st.spinner("Calculando embeddings por oración..."):
                        embs = embed_texts(emb_model, sentences)
                    chunks_sem = chunk_semantic(sentences, embs, sem_threshold)
                    st.markdown(f"**{len(chunks_sem)} chunks semánticos** (de {len(sentences)} oraciones):")
                    for i, c in enumerate(chunks_sem):
                        st.markdown(f"""
                        <div class="chunk-box">
                        <span style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#bc8cff;">
                        Chunk {i+1} — {c['n_sentences']} oraciones · similitud interna media: {c['avg_boundary_sim']:.3f}</span><br>
                        {c['text']}
                        </div>""", unsafe_allow_html=True)
                else:
                    st.warning("Se necesitan al menos 2 oraciones y un modelo de embedding cargado.")

    with tab_agent:
        st.markdown("""
        <div class="formula-box">Un LLM (Groq) reescribe el texto en proposiciones
atómicas: hechos independientes, sin pronombres
ambiguos. Mayor costo (llamada a LLM), mayor
precisión de retrieval.</div>
        """, unsafe_allow_html=True)
        if not api_key:
            st.markdown('<div class="warn-box">⚠️ Este método requiere Groq API Key (usa un LLM para chunkear).</div>', unsafe_allow_html=True)
        else:
            doc_agent = st.text_area("Documento (pasaje corto, funciona mejor):",
                                      value="La Dra. Ana Gómez lidera el grupo de investigación en IA de EAFIT. "
                                            "Ella publicó 12 artículos en 2025 y su equipo ganó una beca Minciencias "
                                            "que financiará el proyecto durante dos años.",
                                      height=100, key="agent_doc")
            agent_model = st.selectbox("Modelo Groq para chunking:", list(GROQ_MODELS.keys()),
                                        format_func=lambda m: GROQ_MODELS[m]['family'], key="agent_model")
            if st.button("🤖 Generar proposiciones", key="run_agent_chunk"):
                client = get_groq_client(api_key)
                prompt = PROPOSITIONAL_PROMPT.format(text=doc_agent)
                with st.spinner("El LLM está descomponiendo el texto..."):
                    r = call_groq(client=client, model=agent_model,
                                  messages=[{"role": "user", "content": prompt}],
                                  temperature=0.0, max_tokens=400)
                if r["success"]:
                    props = [p.strip() for p in r["content"].split("\n") if p.strip()]
                    st.markdown(f"**{len(props)} proposiciones atómicas generadas:**")
                    for p in props:
                        clean = re.sub(r"^\d+[\.\)]\s*", "", p)
                        st.markdown(f'<div class="chunk-box">{clean}</div>', unsafe_allow_html=True)
                    st.caption(f"⏱️ {r['latency']:.2f}s · {r['usage']['completion_tokens']} tokens de salida")
                else:
                    st.error(r.get("error", "Error"))

    st.markdown("""
    <div class="warn-box">
    <b>💡 ¿Cuál elegir?</b> Estructural para documentos ya bien formateados (manuales, políticas).
    Recursivo/fixed-size como baseline robusto y barato. Semántico cuando el documento no tiene
    estructura clara pero sí cambios temáticos. Proposicional/Agéntico para máxima precisión en
    dominios donde el costo por llamada a LLM se justifica (legal, médico, investigación).
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# MODULE 4: NLP CLÁSICO
# ════════════════════════════════════════════════════════════
elif module == "🏷️  NLP Clásico (POS, NER, Sentimientos)":
    st.markdown('<div class="section-title">NLP Clásico: POS, NER, Sentimientos y Estadísticas</div>', unsafe_allow_html=True)
    text_nlp = st.text_area("Texto para analizar:", value=SAMPLE_TEXTS["es"], height=140)

    if not NLTK_AVAILABLE:
        st.error("NLTK no disponible. Instala con: `pip install nltk`")
        st.stop()

    tab_pos, tab_ner, tab_sent, tab_stats = st.tabs(["🏷️ POS Tagging", "👤 NER", "😊 Sentimientos", "📈 Estadísticas"])

    with tab_pos:
        st.markdown("""
        <div class="info-box"><b>Part-of-Speech Tagging</b> asigna una categoría gramatical a cada token.</div>
        <div class="formula-box">Penn Treebank tags:
NN=Noun VB=Verb JJ=Adjective RB=Adverb DT=Determiner
IN=Preposition CC=Conjunction PRP=Pronoun NNP=Proper Noun CD=Cardinal</div>
        """, unsafe_allow_html=True)
        try:
            pos_tags = pos_tag(word_tokenize(text_nlp))
            pos_colors = {'NN': '#58a6ff', 'NNS': '#58a6ff', 'NNP': '#bc8cff', 'NNPS': '#bc8cff',
                          'VB': '#3fb950', 'VBD': '#3fb950', 'VBG': '#39d353', 'VBN': '#39d353',
                          'JJ': '#f0883e', 'JJR': '#f0883e', 'JJS': '#f0883e', 'RB': '#ffa657', 'RBR': '#ffa657',
                          'DT': '#8b949e', 'IN': '#8b949e', 'CC': '#8b949e', 'PRP': '#d2a8ff', 'PRP$': '#d2a8ff', 'CD': '#ff7b72'}
            chips = []
            for word, tag in pos_tags[:60]:
                color = pos_colors.get(tag, "#6e7681")
                chips.append(f'<span class="token-chip" style="background:{color}22;border-color:{color};color:{color}" title="{tag}">{word}<sub style="font-size:0.6em;">{tag}</sub></span>')
            st.markdown(f'<div class="token-container">{"".join(chips)}</div>', unsafe_allow_html=True)
            pos_freq = Counter([tag for _, tag in pos_tags]).most_common(12)
            df_pos = pd.DataFrame(pos_freq, columns=["POS", "Frecuencia"])
            fig_pos = px.bar(df_pos, x="POS", y="Frecuencia", title="Distribución de categorías gramaticales",
                              color="Frecuencia", color_continuous_scale=["#161b22", "#58a6ff", "#f0883e"])
            fig_pos.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22", font=dict(color="#e6edf3"), height=280, showlegend=False)
            st.plotly_chart(fig_pos, width="stretch")
        except Exception as e:
            st.error(f"Error en POS tagging: {e}")

    with tab_ner:
        st.markdown("""
        <div class="info-box"><b>NER</b> identifica personas (PER), organizaciones (ORG), lugares (LOC), etc.</div>
        """, unsafe_allow_html=True)
        try:
            tree = ne_chunk(pos_tag(word_tokenize(text_nlp)))
            entities = [{"Entidad": " ".join(w for w, t in s.leaves()), "Tipo": s.label()} for s in tree if hasattr(s, 'label')]
            if entities:
                ner_colors = {"PERSON": "#bc8cff", "ORGANIZATION": "#58a6ff", "GPE": "#3fb950",
                              "FACILITY": "#f0883e", "LOCATION": "#39d353", "GSP": "#ffa657"}
                chips_ner = [f'<span class="token-chip" style="background:{ner_colors.get(e["Tipo"],"#8b949e")}33;border-color:{ner_colors.get(e["Tipo"],"#8b949e")};color:{ner_colors.get(e["Tipo"],"#8b949e")};font-size:0.85rem;">{e["Entidad"]} <sub>{e["Tipo"]}</sub></span>' for e in entities]
                st.markdown(f'<div class="token-container">{"".join(chips_ner)}</div>', unsafe_allow_html=True)
                st.dataframe(pd.DataFrame(entities), width="stretch", height=200)
            else:
                st.info("No se detectaron entidades con NLTK. Funciona mejor en inglés — para español, considera spaCy o un BERT fine-tuned para NER.")
            st.markdown('<div class="warn-box"><b>💡 Nota:</b> NLTK NER funciona mejor en inglés. Para español, la opción estándar es spaCy (es_core_news_lg) o un modelo BERT afinado para NER en español.</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error en NER: {e}")

    with tab_sent:
        st.markdown('<div class="info-box"><b>VADER</b> es un lexicón basado en reglas, funciona mejor en inglés. Retorna Positivo/Negativo/Neutro/Compound.</div>', unsafe_allow_html=True)
        test_sentences = ["I love this product, it is absolutely amazing!",
                          "This is the worst experience I have ever had.",
                          "The weather today is okay, nothing special.",
                          "Artificial intelligence is transforming education in incredible ways.",
                          "I'm not sure if I like this feature or not.",
                          "Despite some issues, the overall experience was positive."]
        custom_sent = st.text_area("Frases (una por línea, inglés funciona mejor):", value="\n".join(test_sentences), height=150)
        sent_list = [s.strip() for s in custom_sent.split("\n") if s.strip()]
        try:
            sia = SentimentIntensityAnalyzer()
            results_sent = []
            for sentence in sent_list:
                scores = sia.polarity_scores(sentence)
                sentiment = "Positivo" if scores["compound"] >= 0.05 else "Negativo" if scores["compound"] <= -0.05 else "Neutro"
                results_sent.append({"Texto": sentence[:60], "Positivo": round(scores["pos"], 3),
                                     "Negativo": round(scores["neg"], 3), "Neutro": round(scores["neu"], 3),
                                     "Compound": round(scores["compound"], 3), "Sentimiento": sentiment})
            df_sent = pd.DataFrame(results_sent)
            st.dataframe(df_sent, width="stretch", height=250)
            fig_sent = px.bar(df_sent, x="Texto", y="Compound", color="Sentimiento",
                               color_discrete_map={"Positivo": "#3fb950", "Negativo": "#f85149", "Neutro": "#8b949e"},
                               title="Score Compound (VADER) por frase")
            fig_sent.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22", font=dict(color="#e6edf3"), height=320, xaxis=dict(tickangle=-20))
            st.plotly_chart(fig_sent, width="stretch")
        except Exception as e:
            st.error(f"Error en análisis de sentimientos: {e}")

    with tab_stats:
        st.markdown('<div class="section-title" style="font-size:1rem;">Estadísticas Lingüísticas y N-gramas</div>', unsafe_allow_html=True)
        try:
            tokens_all = word_tokenize(text_nlp)
            words_only = [w.lower() for w in tokens_all if w.isalpha()]
            try:
                stop_all = set(stopwords.words('spanish')) | set(stopwords.words('english'))
            except Exception:
                stop_all = set()
            content_words = [w for w in words_only if w not in stop_all]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total tokens", len(tokens_all))
            c2.metric("Palabras únicas", len(set(words_only)))
            c3.metric("Riqueza léxica", f"{len(set(words_only))/max(len(words_only),1):.3f}")
            c4.metric("Palabras de contenido", len(content_words))

            col_f1, col_f2 = st.columns(2)
            with col_f1:
                freq = Counter(content_words).most_common(15)
                df_freq = pd.DataFrame(freq, columns=["Palabra", "Frecuencia"])
                fig_freq = px.bar(df_freq, x="Frecuencia", y="Palabra", orientation="h", title="Top 15 palabras de contenido",
                                   color_discrete_sequence=["#58a6ff"])
                fig_freq.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22", font=dict(color="#e6edf3"),
                                        height=380, yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_freq, width="stretch")
            with col_f2:
                n_gram_n = st.slider("N para n-grama", 1, 5, 2)
                try:
                    word_tokens_clean = [w.lower() for w in tokens_all if w.isalpha() and len(w) > 1]
                    grams = list(ngrams(word_tokens_clean, n_gram_n))
                    gram_freq = Counter(grams).most_common(15)
                    if gram_freq:
                        df_ngrams = pd.DataFrame([(" ".join(g), c) for g, c in gram_freq], columns=["N-grama", "Frecuencia"])
                        fig_ng = px.bar(df_ngrams, x="Frecuencia", y="N-grama", orientation="h",
                                         title=f"Top 15 {n_gram_n}-gramas", color_discrete_sequence=["#bc8cff"])
                        fig_ng.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22", font=dict(color="#e6edf3"),
                                              height=380, yaxis=dict(autorange="reversed"))
                        st.plotly_chart(fig_ng, width="stretch")
                except Exception as e:
                    st.error(f"Error generando n-gramas: {e}")
        except Exception as e:
            st.error(f"Error en estadísticas: {e}")

# ════════════════════════════════════════════════════════════
# MODULE 5: LLM LAB — PARÁMETROS
# ════════════════════════════════════════════════════════════
elif module == "⚡  LLM Lab — Parámetros":
    st.markdown('<div class="section-title">LLM Lab: Explorando Parámetros de Inferencia</div>', unsafe_allow_html=True)
    if not api_key:
        st.markdown('<div class="warn-box">⚠️ Se requiere una Groq API Key. Obtén una gratis en <a href="https://console.groq.com" style="color:#58a6ff;">console.groq.com</a>.</div>', unsafe_allow_html=True)
        st.stop()
    client = get_groq_client(api_key)
    col_config, col_lab = st.columns([1, 2])

    with col_config:
        st.markdown("### ⚙️ Configuración de Parámetros")
        with st.expander("🌡️ Temperature", expanded=True):
            temperature = st.slider("temperature", 0.0, 2.0, 0.7, 0.05)
            st.markdown(explain_parameter("temperature"))
        with st.expander("🎯 Top-P"):
            top_p = st.slider("top_p", 0.01, 1.0, 1.0, 0.01)
            st.markdown(explain_parameter("top_p"))
        with st.expander("📏 Max Tokens"):
            max_tokens = st.slider("max_tokens", 10, 4096, 512, 10)
            st.markdown(explain_parameter("max_tokens"))
        reasoning_capable = ("gpt-oss" in selected_model) or ("qwen3" in selected_model)
        with st.expander(f"🧠 Reasoning Effort {'' if reasoning_capable else '(no aplica a este modelo)'}"):
            if reasoning_capable:
                if "qwen3" in selected_model:
                    reasoning_effort = st.radio("reasoning_effort", ["none", "default"], index=1, horizontal=True)
                else:
                    reasoning_effort = st.radio("reasoning_effort", ["low", "medium", "high"], index=1, horizontal=True)
            else:
                reasoning_effort = None
                st.caption("Solo disponible en GPT-OSS y Qwen3.")
            st.markdown(explain_parameter("reasoning_effort"))
        with st.expander("🌱 Seed & Stop Sequences"):
            use_seed = st.checkbox("Usar seed fijo (reproducibilidad)", value=False)
            seed_val = st.number_input("Seed", 0, 999999, 42, disabled=not use_seed)
            stop_raw = st.text_input("Stop sequences (separadas por coma):", placeholder='###, <end>')
            stop_seqs = [s.strip().replace("\\n", "\n") for s in stop_raw.split(",") if s.strip()] or None
            st.markdown(explain_parameter("stop"))
        with st.expander("💬 System Prompt"):
            system_prompt = st.text_area("System Prompt:",
                value="Eres un asistente experto en ciencia de datos y NLP. Responde de forma precisa, "
                      "concisa y pedagógica en español.", height=100)
        st.markdown(f"""
        <div style="background:#0d1117;border:1px solid #30363d;border-radius:6px;padding:0.8rem;margin-top:0.5rem;font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:#8b949e;">
        <b style="color:#f0883e;">Config enviada a Groq:</b><br>
        model: {selected_model}<br>temperature: {temperature}<br>top_p: {top_p}<br>
        max_tokens: {max_tokens}<br>reasoning_effort: {reasoning_effort}<br>
        seed: {seed_val if use_seed else "None"}<br>stop: {stop_seqs}
        </div>
        """, unsafe_allow_html=True)

    with col_lab:
        st.markdown("### 💬 Prueba de Generación")
        preset_queries = {
            "Custom": "",
            "Explicar temperatura en LLMs": "Explica con una analogía simple qué hace el parámetro temperature en un LLM.",
            "Comparar RNN vs Transformer": "Compara RNN/LSTM con la arquitectura Transformer en 5 puntos clave.",
            "Código: TF-IDF en Python": "Escribe un ejemplo de código Python usando sklearn para calcular TF-IDF de 5 documentos.",
            "Razonamiento matemático": "Un tren sale a 120 km/h y otro a 90 km/h uno hacia el otro, separados 420 km. ¿Cuándo se encuentran? Razona paso a paso.",
        }
        preset = st.selectbox("Queries de ejemplo:", list(preset_queries.keys()))
        user_query = st.text_area("Tu query:", value=preset_queries[preset], height=120, key="llm_query")

        if st.button("⚡ Generar", key="run_gen", width="stretch") and user_query.strip():
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_query}]
            with st.spinner(f"⚡ Generando con {selected_model}..."):
                result = call_groq(client=client, model=selected_model, messages=messages, temperature=temperature,
                                    max_tokens=max_tokens, top_p=top_p, stop=stop_seqs,
                                    seed=seed_val if use_seed else None, reasoning_effort=reasoning_effort)
            if result["success"]:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("⏱️ Latencia", f"{result['latency']:.2f}s")
                c2.metric("⚡ Tokens/s", f"{result['tokens_per_sec']:.0f}")
                c3.metric("📤 Input tokens", result["usage"]["prompt_tokens"])
                c4.metric("📥 Output tokens", result["usage"]["completion_tokens"])
                finish_color = "#3fb950" if result["finish_reason"] == "stop" else "#f0883e"
                st.markdown(f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.72rem;color:{finish_color};">finish_reason: {result["finish_reason"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="llm-response">{result["content"]}</div>', unsafe_allow_html=True)
                with st.expander("🔍 Ver tokens + IDs del output (cl100k BPE)"):
                    out_toks, out_ids = tokenize_with_tiktoken(result["content"], "cl100k_base")
                    st.markdown(render_token_chips(out_toks[:100], out_ids[:100]), unsafe_allow_html=True)
            else:
                st.error(f"Error: {result.get('error', 'Unknown error')}")


# ════════════════════════════════════════════════════════════
# MODULE 6: COMPARADOR DE MODELOS
# ════════════════════════════════════════════════════════════
elif module == "⚖️  Comparador de Modelos":
    st.markdown('<div class="section-title">Comparador: Misma Query, Múltiples LLMs Groq</div>', unsafe_allow_html=True)
    if not api_key:
        st.markdown('<div class="warn-box">⚠️ Se requiere Groq API Key.</div>', unsafe_allow_html=True)
        st.stop()
    client = get_groq_client(api_key)

    models_to_compare = st.multiselect(
        "Selecciona modelos a comparar (máximo 4):", options=list(GROQ_MODELS.keys()),
        default=["llama-3.1-8b-instant", "openai/gpt-oss-20b", "qwen/qwen3-32b"],
        format_func=lambda m: f"{GROQ_MODELS[m]['family']} ({GROQ_MODELS[m]['params']})", max_selections=4
    )
    col_p1, col_p2, col_p3 = st.columns(3)
    cmp_temperature = col_p1.slider("temperature", 0.0, 2.0, 0.7, 0.1, key="cmp_t")
    cmp_max_tokens = col_p2.slider("max_tokens", 50, 2048, 400, 50, key="cmp_mt")
    cmp_top_p = col_p3.slider("top_p", 0.1, 1.0, 1.0, 0.1, key="cmp_tp")
    cmp_system = st.text_input("System prompt (compartido):",
        value="Eres un asistente experto en IA y ciencia de datos. Responde en español, claro y conciso.")
    cmp_query = st.text_area("Query para todos los modelos:",
        value="Explica la diferencia entre BPE, WordPiece y SentencePiece en 3 puntos cada uno.", height=90)

    if st.button("⚡ Comparar modelos", key="run_compare", width="stretch") and cmp_query.strip():
        if not models_to_compare:
            st.warning("Selecciona al menos un modelo."); st.stop()
        results_compare = {}
        progress = st.progress(0, text="Iniciando comparación...")
        for i, model in enumerate(models_to_compare):
            progress.progress(i / len(models_to_compare), text=f"Consultando {GROQ_MODELS[model]['family']}...")
            r = call_groq(client=client, model=model,
                          messages=[{"role": "system", "content": cmp_system}, {"role": "user", "content": cmp_query}],
                          temperature=cmp_temperature, max_tokens=cmp_max_tokens, top_p=cmp_top_p)
            results_compare[model] = r
            time.sleep(0.2)
        progress.progress(1.0, text="✅ Comparación completada")

        successful = {m: r for m, r in results_compare.items() if r.get("success")}
        failed = {m: r for m, r in results_compare.items() if not r.get("success")}
        if failed:
            st.markdown("### ⚠️ Errores")
            for model, result in failed.items():
                st.markdown(f'<div class="warn-box"><b>❌ {GROQ_MODELS[model]["family"]}</b><br><code style="font-size:0.8rem;">{result.get("error","")}</code></div>', unsafe_allow_html=True)
        if not successful:
            st.error("Ningún modelo respondió correctamente."); st.stop()

        st.markdown("### 📊 Métricas Comparativas")
        metric_cols = st.columns(max(1, len(successful)))
        for col, (model, result) in zip(metric_cols, successful.items()):
            with col:
                mc = GROQ_MODELS[model]["color"]
                st.markdown(f"""
                <div class="metric-card" style="border-color:{mc};">
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:{mc};">{GROQ_MODELS[model]['family']}</div>
                <div class="metric-value" style="color:{mc};font-size:1.3rem;">{result['latency']:.2f}s</div>
                <div class="metric-label">Latencia</div>
                <div style="margin-top:0.4rem;font-size:0.8rem;color:#8b949e;">
                ⚡ {result['tokens_per_sec']:.0f} tok/s<br>📤 {result['usage']['prompt_tokens']} prompt<br>
                📥 {result['usage']['completion_tokens']} output</div></div>
                """, unsafe_allow_html=True)

        if len(successful) >= 2:
            fig_speed = go.Figure()
            labels = [GROQ_MODELS[m]['family'] for m in successful]
            fig_speed.add_trace(go.Bar(name="Latencia (s)", x=labels, y=[r["latency"] for r in successful.values()],
                                        marker_color=[GROQ_MODELS[m]["color"] for m in successful], opacity=0.85))
            fig_speed.add_trace(go.Scatter(name="Tokens/s", x=labels, y=[r["tokens_per_sec"] for r in successful.values()],
                                            mode="markers+lines", marker=dict(size=10, color="#ffffff"),
                                            line=dict(color="#ffffff", dash="dot"), yaxis="y2"))
            fig_speed.update_layout(title="Latencia vs Velocidad de Generación", paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                                     font=dict(color="#e6edf3"), yaxis=dict(title="Latencia (s)"),
                                     yaxis2=dict(title="Tokens/s", overlaying="y", side="right"), height=320)
            st.plotly_chart(fig_speed, width="stretch")

        st.markdown("### 💬 Respuestas Comparadas")
        resp_cols = st.columns(max(1, len(successful)))
        for col, (model, result) in zip(resp_cols, successful.items()):
            with col:
                mc = GROQ_MODELS[model]["color"]
                st.markdown(f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.72rem;color:{mc};font-weight:600;">{GROQ_MODELS[model]["family"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="llm-response" style="font-size:0.82rem;">{result.get("content","") or "*(sin respuesta)*"}</div>', unsafe_allow_html=True)

        df_tokens = pd.DataFrame([{"Modelo": GROQ_MODELS[m]["family"], "Prompt tokens": r["usage"]["prompt_tokens"],
                                    "Output tokens": r["usage"]["completion_tokens"], "Tokens/s": round(r["tokens_per_sec"], 1),
                                    "Latencia (s)": round(r["latency"], 3)} for m, r in successful.items()])
        st.markdown("### 📋 Tabla de Uso de Tokens")
        st.dataframe(df_tokens, width="stretch")


# ════════════════════════════════════════════════════════════
# MODULE 7: BENCHMARK DE VELOCIDAD
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
    (no solo el promedio). En producción, importan el percentil P95/P99, no solo la media.
    </div>
    """, unsafe_allow_html=True)

    col_bm1, col_bm2 = st.columns(2)
    bm_model = col_bm1.selectbox("Modelo a benchmarkear:", list(GROQ_MODELS.keys()),
                                  format_func=lambda m: f"{GROQ_MODELS[m]['family']} ({GROQ_MODELS[m]['params']})")
    n_runs = col_bm2.slider("Número de llamadas:", 3, 15, 5)
    bm_prompt = st.text_area("Prompt del benchmark:",
        value="Explica en exactamente 3 oraciones qué es el mecanismo de atención en los Transformers.", height=80)
    bm_max_tokens = st.slider("max_tokens para benchmark:", 50, 500, 150, 25)

    if st.button("🏁 Iniciar Benchmark", key="bm_run", width="stretch"):
        latencies, tok_per_sec_list, output_tokens_list = [], [], []
        progress_bm = st.progress(0, text="Ejecutando benchmark...")
        messages_bm = [{"role": "user", "content": bm_prompt}]
        for i in range(n_runs):
            progress_bm.progress((i + 1) / n_runs, text=f"Llamada {i+1}/{n_runs}...")
            r = call_groq(client=client, model=bm_model, messages=messages_bm, temperature=0.0, max_tokens=bm_max_tokens)
            if r["success"]:
                latencies.append(r["latency"]); tok_per_sec_list.append(r["tokens_per_sec"]); output_tokens_list.append(r["usage"]["completion_tokens"])
            time.sleep(0.1)
        progress_bm.progress(1.0, text="✅ Benchmark completado")

        if latencies:
            col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
            col_s1.metric("Media", f"{np.mean(latencies):.3f}s")
            col_s2.metric("Mediana", f"{np.median(latencies):.3f}s")
            col_s3.metric("P95", f"{np.percentile(latencies, 95):.3f}s")
            col_s4.metric("Std Dev", f"{np.std(latencies):.3f}s")
            col_s5.metric("Promedio tok/s", f"{np.mean(tok_per_sec_list):.0f}")

            fig_bm = make_subplots(rows=1, cols=2, subplot_titles=("Distribución de Latencia", "Tokens/segundo por llamada"))
            fig_bm.add_trace(go.Histogram(x=latencies, nbinsx=min(n_runs, 10), marker_color="#f0883e", opacity=0.85, name="Latencia"), row=1, col=1)
            fig_bm.add_vline(x=np.mean(latencies), line_dash="dash", line_color="#58a6ff", annotation_text=f"Media: {np.mean(latencies):.3f}s", row=1, col=1)
            fig_bm.add_vline(x=np.percentile(latencies, 95), line_dash="dot", line_color="#f85149", annotation_text=f"P95: {np.percentile(latencies,95):.3f}s", row=1, col=1)
            fig_bm.add_trace(go.Scatter(x=list(range(1, len(tok_per_sec_list)+1)), y=tok_per_sec_list, mode="lines+markers",
                                         marker=dict(color="#3fb950", size=8), line=dict(color="#3fb950"), name="Tokens/s"), row=1, col=2)
            fig_bm.add_hline(y=np.mean(tok_per_sec_list), line_dash="dash", line_color="#58a6ff",
                              annotation_text=f"Media: {np.mean(tok_per_sec_list):.0f} tok/s", row=1, col=2)
            fig_bm.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22", font=dict(color="#e6edf3"), height=350, showlegend=False)
            for axis in ['xaxis', 'yaxis', 'xaxis2', 'yaxis2']:
                fig_bm.update_layout(**{f"{axis}_gridcolor": "#30363d"})
            st.plotly_chart(fig_bm, width="stretch")

            df_bm = pd.DataFrame({"Llamada": range(1, len(latencies)+1), "Latencia (s)": [round(l, 4) for l in latencies],
                                   "Tokens/s": [round(t, 1) for t in tok_per_sec_list], "Output tokens": output_tokens_list})
            st.dataframe(df_bm, width="stretch")

            st.markdown(f"""
            <div class="info-box">
            <b>Análisis del Benchmark — {GROQ_MODELS[bm_model]['family']}:</b><br>
            • Coeficiente de variación: {(np.std(latencies)/np.mean(latencies)*100):.1f}% (menor = más estable)<br>
            • Overhead P95 vs media: +{((np.percentile(latencies,95)/np.mean(latencies)-1)*100):.1f}%<br>
            • Para diseño de sistemas: usa P95 ({np.percentile(latencies,95):.3f}s) como SLA conservador, no la media.
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# MODULE 8: ATTENTION VISUALIZER
# ════════════════════════════════════════════════════════════
elif module == "🎯  Attention Visualizer":
    st.markdown('<div class="section-title">Attention Visualizer: Entendiendo Qué Atiende el Modelo</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box"><b>Proxy pedagógico:</b> este módulo aproxima la atención con similitud
    coseno de n-gramas de caracteres, no son los pesos reales de un Transformer.</div>
    <div class="formula-box">Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) . V
Para el token i, el score de atención hacia el token j:
alpha(i,j) = softmax( q_i . k_j / sqrt(d_k) )</div>
    """, unsafe_allow_html=True)

    attn_text = st.text_input("Oración para visualizar atención:",
        value="La directora de la empresa aprobó el nuevo presupuesto porque consideró que era viable.")

    if SKLEARN_AVAILABLE and attn_text.strip():
        tokens_attn = attn_text.split()
        n_toks = len(tokens_attn)
        if n_toks >= 3:
            try:
                vec_attn = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=1)
                context_docs = [" ".join(tokens_attn[max(0, i-3):min(n_toks, i+4)]) for i in range(n_toks)]
                X_attn = vec_attn.fit_transform(context_docs)
                sim_attn = cosine_similarity(X_attn).astype(float)
                sim_attn = np.exp(sim_attn * 3)
                sim_attn = sim_attn / sim_attn.sum(axis=1, keepdims=True)

                fig_attn = px.imshow(sim_attn, x=tokens_attn, y=tokens_attn, title="Matriz de Atención Aproximada (similitud coseno)",
                                      color_continuous_scale="Oranges", text_auto=".2f", aspect="auto")
                fig_attn.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22", font=dict(color="#e6edf3", size=11),
                                        height=max(300, n_toks * 25 + 100), xaxis=dict(tickangle=-45))
                st.plotly_chart(fig_attn, width="stretch")

                query_token_idx = st.select_slider("Token de consulta (Query):", options=list(range(n_toks)),
                                                     format_func=lambda i: f"[{i}] '{tokens_attn[i]}'", value=min(3, n_toks-1))
                attn_row = sim_attn[query_token_idx]
                col_viz1, col_viz2 = st.columns([2, 1])
                with col_viz1:
                    fig_row = px.bar(x=tokens_attn, y=attn_row, title=f"Distribución de atención de '{tokens_attn[query_token_idx]}'",
                                      color=attn_row, color_continuous_scale=["#161b22", "#f0883e", "#ffa657"])
                    fig_row.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22", font=dict(color="#e6edf3"),
                                           height=280, xaxis=dict(tickangle=-30), showlegend=False)
                    st.plotly_chart(fig_row, width="stretch")
                with col_viz2:
                    st.markdown("**Top 5 tokens atendidos:**")
                    for rank, idx in enumerate(np.argsort(attn_row)[::-1][:5]):
                        color = token_color(rank, 5)
                        st.markdown(f"""
                        <div style="margin:0.3rem 0;">
                        <span style="font-family:'JetBrains Mono',monospace;font-size:0.8rem;color:{color};">{rank+1}. '{tokens_attn[idx]}'</span>
                        <div style="background:{color}33;height:6px;border-radius:3px;width:{int(attn_row[idx]*100)}%;margin-top:2px;"></div>
                        <span style="font-size:0.7rem;color:#8b949e;">{attn_row[idx]:.3f}</span></div>
                        """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error calculando atención: {e}")
        else:
            st.warning("La oración debe tener al menos 3 palabras.")


# ════════════════════════════════════════════════════════════
# MODULE 9: PLAYGROUND LIBRE
# ════════════════════════════════════════════════════════════
elif module == "🧪  Playground Libre":
    st.markdown('<div class="section-title">Playground Libre</div>', unsafe_allow_html=True)
    if not api_key:
        st.markdown('<div class="warn-box">⚠️ Se requiere Groq API Key.</div>', unsafe_allow_html=True)
        st.stop()
    client = get_groq_client(api_key)

    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "pg_system" not in st.session_state:
        st.session_state.pg_system = "Eres un asistente de investigación para la Maestría en Ciencia de Datos de EAFIT."

    tab_chat, tab_prompts = st.tabs(["💬 Chat Multi-Turn", "🎨 Prompt Engineering"])

    with tab_chat:
        col_pg1, col_pg2 = st.columns([3, 1])
        with col_pg2:
            pg_model = st.selectbox("Modelo:", list(GROQ_MODELS.keys()), format_func=lambda m: GROQ_MODELS[m]['family'], key="pg_model")
            pg_temp = st.slider("temperature", 0.0, 2.0, 0.7, 0.05, key="pg_temp")
            pg_max = st.slider("max_tokens", 100, 4096, 1024, 100, key="pg_max")
            st.session_state.pg_system = st.text_area("System Prompt:", value=st.session_state.pg_system, height=100, key="pg_sys_input")
            if st.button("🗑️ Limpiar conversación", key="pg_clear"):
                st.session_state.conversation_history = []
                st.rerun()
        with col_pg1:
            for msg in st.session_state.conversation_history:
                role_color = "#58a6ff" if msg["role"] == "user" else "#f0883e"
                role_label = "👤 Usuario" if msg["role"] == "user" else f"🤖 {GROQ_MODELS[pg_model]['family']}"
                st.markdown(f"""
                <div style="background:#1c2333;border:1px solid #30363d;border-radius:8px;padding:0.8rem 1rem;margin:0.4rem 0;border-left:3px solid {role_color};">
                <span style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:{role_color};">{role_label}</span><br>
                <span style="font-size:0.9rem;">{msg['content']}</span></div>""", unsafe_allow_html=True)

            pg_user_input = st.text_area("Tu mensaje:", height=80, key="pg_input")
            if st.button("📤 Enviar", key="pg_send", width="stretch") and pg_user_input.strip():
                st.session_state.conversation_history.append({"role": "user", "content": pg_user_input})
                messages_pg = [{"role": "system", "content": st.session_state.pg_system}] + st.session_state.conversation_history
                with st.spinner("Generando respuesta..."):
                    r = call_groq(client=client, model=pg_model, messages=messages_pg, temperature=pg_temp, max_tokens=pg_max)
                if r["success"]:
                    st.session_state.conversation_history.append({"role": "assistant", "content": r["content"]})
                    st.rerun()
                else:
                    st.error(f"Error: {r.get('error', 'Unknown')}")

    with tab_prompts:
        techniques = {
            "Zero-Shot": {"desc": "Sin ejemplos, instrucción directa.",
                "prompt": 'Clasifica el sentimiento: "La nueva versión supera mis expectativas." Sentimiento:',
                "when": "Cuando el modelo ya conoce bien la tarea."},
            "Few-Shot": {"desc": "2-5 ejemplos guían el formato.",
                "prompt": 'Texto: "Tardó demasiado" -> Negativo\nTexto: "Funciona perfecto" -> Positivo\nTexto: "Es aceptable" -> Neutro\nTexto: "Es increíblemente eficiente" ->',
                "when": "Cuando el formato de salida debe ser exacto."},
            "Chain of Thought": {"desc": "Fuerza razonamiento paso a paso.",
                "prompt": "Un modelo procesa 500 tokens/s. Documento de 50,000 palabras, 1 token≈0.75 palabras. ¿Cuántos segundos tardará? Razona paso a paso.",
                "when": "Problemas matemáticos o de múltiples pasos."},
            "Role Prompting": {"desc": "Asigna un rol específico al modelo.",
                "prompt": "Eres un profesor experto en Transformers con estilo didáctico. Explica Multi-Head Attention a un estudiante de maestría.",
                "when": "Cuando necesitas un estilo o expertise particular."},
            "Structured Output": {"desc": "Fuerza salida en JSON/XML.",
                "prompt": 'Responde SOLO en JSON: {"nombre": str, "parametros": str, "fortalezas": [str]}. Modelo: GPT-OSS-120B',
                "when": "Integración con código, parseo automático."},
        }
        for tech_name, tech_info in techniques.items():
            with st.expander(f"**{tech_name}** — {tech_info['desc']}"):
                col_t1, col_t2 = st.columns([3, 1])
                with col_t1:
                    edited_prompt = st.text_area("Prompt:", value=tech_info["prompt"], height=120, key=f"prompt_{tech_name}")
                with col_t2:
                    st.markdown(f'<div class="info-box" style="font-size:0.8rem;"><b>¿Cuándo usar?</b><br>{tech_info["when"]}</div>', unsafe_allow_html=True)
                    if st.button("▶️ Ejecutar", key=f"run_{tech_name}"):
                        with st.spinner("Generando..."):
                            r = call_groq(client=client, model=selected_model,
                                         messages=[{"role": "user", "content": edited_prompt}], temperature=0.7, max_tokens=400)
                        if r["success"]:
                            st.markdown(f'<div class="llm-response" style="font-size:0.82rem;max-height:200px;">{r["content"]}</div>', unsafe_allow_html=True)
                            st.caption(f"{r['latency']:.2f}s · {r['usage']['completion_tokens']} tokens")
                        else:
                            st.error(r.get("error", "Error"))


# ════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div style="text-align:center;font-family:'JetBrains Mono',monospace;font-size:0.68rem;color:#30363d;padding:0.5rem 0;">
    EAFIT Universidad · Maestría en Ciencia de Datos · NLP & LLM Interactive Lab ·
    Prof. Jorge Iván Padilla-Buriticá ·
    <a href="https://www.linkedin.com/in/jipadilla" style="color:#30363d;">linkedin/jipadilla</a>
</div>
""", unsafe_allow_html=True)
