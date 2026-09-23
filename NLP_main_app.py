# ============================================================
# EAFIT — Maestría en Ciencia de Datos
# NLP & LLM Interactive Lab  (v2 — catálogo corregido, 2026)
# Prof. Jorge Iván Padilla-Buriticá | linkedin.com/in/jipadilla
# ============================================================
# Módulos:
#   0. Inicio & Teoría
#   1. Tokenización            (BPE/tiktoken, GPT-2, BERT WordPiece, IDs)
#   2. Embeddings & Similitud  (TF-IDF, Sentence-Transformers, coseno, 2D)
#   3. Modelos Clásicos vs Modernos (GPT-2 local  vs  Groq LLM)
#   4. Chunking para RAG       (Estructural, Recursivo, Semántico, Agéntico)
#   5. NLP Clásico             (POS, NER, Sentimientos)
#   6. LLM Lab — Parámetros    (temperature, top_p, max_tokens, stop, seed)
#   7. Comparador de Modelos   (misma query, varios LLMs Groq)
#   8. Attention Visualizer    (proxy pedagógico de self-attention)
#   9. Playground Libre        (chat multi-turn + prompt engineering)
#
# NOTA IMPORTANTE SOBRE EL CATÁLOGO DE MODELOS:
# Groq NO aloja modelos clásicos (GPT-2, BERT, RoBERTa) ni GPT-3.5 para
# inferencia — solo sirve modelos "open-weight" modernos optimizados para
# su hardware LPU. Por eso este laboratorio separa dos mundos:
#   • Modelos MODERNOS → se consultan vía Groq API (requieren API key)
#   • Modelos CLÁSICOS → solo se usan para TOKENIZACIÓN y EMBEDDINGS,
#     con librerías ligeras sin PyTorch (tiktoken, tokenizers, fastembed).
#     No se ejecuta generación real de GPT-2/BERT en este laboratorio.
# El catálogo de Groq cambia con frecuencia: verifica siempre
# https://console.groq.com/docs/models antes de una clase en vivo.
#
# NOTA TÉCNICA SOBRE DEPENDENCIAS (importante si despliegas en Streamlit
# Community Cloud): esta app NO usa `torch`, `torchvision` ni
# `transformers`/`sentence-transformers` a propósito. Esas librerías
# registran cientos de submódulos internos (incluidos modelos de visión
# como ViTMatte/ViTPose/YOLOS) y el *file watcher* de Streamlit los
# recorre todos al arrancar, disparando `ModuleNotFoundError: No module
# named 'torchvision'` en los logs aunque la app nunca los use — un bug
# de interacción Streamlit+HuggingFace bien documentado y molesto de
# silenciar por completo. La solución robusta es no depender de esas
# librerías: aquí se reemplazan por alternativas 100% CPU, sin PyTorch:
#   • tiktoken   → tokenización BPE (GPT-2 y familia GPT-3.5/4)
#   • tokenizers → tokenización WordPiece (BERT), sin PyTorch
#   • fastembed  → embeddings densos vía ONNX Runtime, sin PyTorch
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

# ── Optional imports with graceful fallback ──────────────────
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except Exception:
    GROQ_AVAILABLE = False

try:
    import anthropic as anthropic_sdk
    ANTHROPIC_AVAILABLE = True
except Exception:
    ANTHROPIC_AVAILABLE = False

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except Exception:
    TIKTOKEN_AVAILABLE = False

try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
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
/* IMPORTANTE: no ocultar todo el <header> — ahí vive el botón que abre/cierra
   la barra lateral. Solo lo hacemos transparente y forzamos que el control
   de colapso sea siempre visible y con buen contraste sobre el fondo oscuro. */
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
.danger-box { background: #2d0d0d; border: 1px solid var(--danger); border-left: 4px solid var(--danger); border-radius: 6px; padding: 0.8rem 1rem; margin: 0.5rem 0; font-size: 0.88rem; }
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
# CATÁLOGO DE MODELOS — GROQ (vigente; verificado en console.groq.com/docs/models)
# ════════════════════════════════════════════════════════════
GROQ_MODELS = {
    "openai/gpt-oss-120b": {
        "family": "GPT-OSS 120B", "params": "120B (MoE)", "context": 131_072,
        "type": "Decoder-only · Mixture of Experts · Reasoning", "license": "Apache 2.0",
        "strengths": "Razonamiento profundo, instrucciones complejas, reasoning_effort configurable (low/medium/high)",
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
        "strengths": "Clasifica contenido contra una política dada en el prompt. Guardrails / moderación",
        "color": "#f85149"
    },
    "qwen/qwen3-32b": {
        "family": "Qwen3 32B", "params": "32B", "context": 131_072,
        "type": "Decoder-only · Reasoning conmutable", "license": "Apache 2.0",
        "strengths": "Alibaba Cloud. Reasoning on/off vía reasoning_effort=none|default. Fuerte en matemáticas",
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
        "strengths": "Razonamiento matemático/lógico de DeepSeek-R1, destilado en una base LLaMA. "
                     "Gratis en el free tier de Groq (límites diarios más bajos que los demás modelos).",
        "color": "#ffa657"
    },
}

# ════════════════════════════════════════════════════════════
# CATÁLOGO OPCIONAL — ANTHROPIC (Claude)
# NO es gratuito: Anthropic no ofrece un tier gratuito de API como
# Groq o Google AI Studio (solo créditos de prueba limitados al
# crear la cuenta). Se incluye como comparación opcional, de pago,
# requiere la propia API key de Anthropic del usuario.
# ════════════════════════════════════════════════════════════
ANTHROPIC_MODELS = {
    "claude-haiku-4-5-20251001": {
        "family": "Claude Haiku 4.5", "params": "No divulgado", "context": 200_000,
        "type": "Decoder-only (propietario)", "license": "Propietaria — de pago",
        "strengths": "El modelo Claude más económico disponible; útil como referencia de un LLM "
                     "propietario frente a los open-weight servidos por Groq.",
        "color": "#d2a8ff"
    },
}

# Vista combinada solo para metadatos de presentación (color, familia, params) —
# las llamadas reales siguen yendo por call_groq() o call_anthropic() según el proveedor.
MODEL_META = {**GROQ_MODELS, **ANTHROPIC_MODELS}

# ════════════════════════════════════════════════════════════
# CATÁLOGO DE MODELOS CLÁSICOS — cargados LOCALMENTE (HuggingFace)
# Estos NO pasan por Groq. Sirven para contrastar "antes vs después"
# de la era de los LLMs a escala.
# ════════════════════════════════════════════════════════════
CLASSIC_MODELS = {
    "gpt2": {
        "label": "GPT-2 (small)", "year": 2019, "params": "124M",
        "note": "OpenAI, 2019. El ancestro directo de la familia GPT. Decoder-only, "
                "sin RLHF ni instruction-tuning: completa texto, no sigue instrucciones.",
        "task": "causal_lm"
    },
    "distilgpt2": {
        "label": "DistilGPT-2", "year": 2019, "params": "82M",
        "note": "Versión destilada de GPT-2 (~40% más pequeña, ~60% más rápida, ~97% del rendimiento).",
        "task": "causal_lm"
    },
    "bert-base-uncased": {
        "label": "BERT base (uncased)", "year": 2018, "params": "110M",
        "note": "Google, 2018. Encoder-only, bidireccional. No genera texto: se usa para "
                "embeddings, clasificación, NER, MLM.",
        "task": "encoder"
    },
    "bert-base-multilingual-cased": {
        "label": "BERT multilingual", "year": 2018, "params": "178M",
        "note": "Mismo BERT, entrenado en 104 idiomas incl. español. WordPiece de 119,547 tokens.",
        "task": "encoder"
    },
    "roberta-base": {
        "label": "RoBERTa base", "year": 2019, "params": "125M",
        "note": "Facebook AI, 2019. BERT re-entrenado más tiempo, sin NSP, con BPE en vez de WordPiece.",
        "task": "encoder"
    },
}

# ════════════════════════════════════════════════════════════
# CATÁLOGO DE MODELOS DE EMBEDDING
# ════════════════════════════════════════════════════════════
EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": {
        "hf_id": "sentence-transformers/all-MiniLM-L6-v2",
        "dims": 384, "context": 256, "size": "~90 MB",
        "note": "Clásico, extremadamente ligero y rápido. Ideal para prototipos donde la "
                "latencia es la prioridad. Entrenado principalmente en inglés (funciona en "
                "español pero con pérdida de calidad).",
    },
    "bge-m3": {
        "hf_id": "BAAI/bge-m3",
        "dims": 1024, "context": 8192, "size": "~2.2 GB",
        "note": "BAAI. Uno de los más potentes para corpus multilingüe/técnico en español. "
                "Contexto de 8192 tokens. Descarga pesada: úsalo si tu conexión lo permite.",
    },
    "nomic-embed-text": {
        "hf_id": "nomic-ai/nomic-embed-text-v1.5",
        "dims": 768, "context": 8192, "size": "~550 MB",
        "note": "Nomic AI. El más popular para RAG con Groq en la comunidad open-source. "
                "Contexto largo de 8192 tokens.",
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
    """Load a fastembed (ONNX Runtime) text-embedding model from the EMBEDDING_MODELS catalog."""
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
    """fastembed .embed() returns a generator of numpy arrays; materialize into a matrix."""
    return np.array(list(model.embed(texts)))

@st.cache_resource(show_spinner="⚙️ Cargando tokenizador BERT (WordPiece, sin PyTorch)…")
def load_bert_tokenizer(model_name: str = "bert-base-multilingual-cased"):
    if not TOKENIZERS_AVAILABLE:
        return None
    try:
        return HFTokenizer.from_pretrained(model_name)
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

def tokenize_with_tiktoken(text: str, encoding_name: str = "cl100k_base"):
    """Returns (tokens, ids) using a tiktoken BPE encoding.

    encoding_name="cl100k_base" -> BPE de referencia de GPT-3.5/4 (~100,277 vocab)
    encoding_name="gpt2"        -> BPE ORIGINAL de GPT-2 (2019, 50,257 vocab),
                                    sin ninguna dependencia de PyTorch/transformers.
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
    Wrapper around Groq's chat.completions API with the parameters Groq actually
    supports today. Groq does NOT support OpenAI-only params like frequency_penalty
    or presence_penalty — sending them raises a 400 error, so they are never sent.
    reasoning_effort is only forwarded for reasoning-capable models (gpt-oss-*, qwen3-32b);
    it is silently dropped for the others to avoid a 400 error on non-reasoning models.
    """
    if client is None:
        return {"success": False, "error": "Cliente Groq no inicializado. Verifica tu API Key.", "latency": 0,
                "content": "", "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}

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
        return {
            "success": True, "content": content, "usage": usage, "latency": latency,
            "tokens_per_sec": tokens_per_sec, "finish_reason": finish_reason,
            "model": model, "params": params,
        }
    except Exception as e:
        return {
            "success": False, "error": str(e), "latency": time.perf_counter() - t0,
            "content": "", "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

def get_anthropic_client(api_key: str):
    """Anthropic (Claude) — opcional, de pago. No hay tier gratuito de API."""
    if not ANTHROPIC_AVAILABLE or not api_key:
        return None
    try:
        return anthropic_sdk.Anthropic(api_key=api_key)
    except Exception:
        return None

def call_anthropic(client, model: str, messages: list, system: str = "",
                    temperature: float = 0.7, max_tokens: int = 1024) -> dict:
    """
    Wrapper de la API de Claude, con la misma forma de retorno que call_groq()
    para que ambos proveedores se puedan comparar con el mismo código de UI.
    `messages` debe traer solo turnos user/assistant; el system prompt va aparte.
    """
    if client is None:
        return {"success": False, "error": "Cliente Anthropic no inicializado. Verifica tu API Key.",
                "latency": 0, "content": "", "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
    t0 = time.perf_counter()
    try:
        response = client.messages.create(
            model=model, max_tokens=int(max_tokens), temperature=float(temperature),
            system=system or "", messages=messages,
        )
        latency = time.perf_counter() - t0
        content = "".join(block.text for block in response.content if getattr(block, "type", "") == "text")
        usage = {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
        }
        tokens_per_sec = usage["completion_tokens"] / latency if latency > 0 else 0
        return {"success": True, "content": content, "usage": usage, "latency": latency,
                "tokens_per_sec": tokens_per_sec, "finish_reason": response.stop_reason, "model": model}
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
    """Render tokens as colored HTML chips, with the numeric token ID underneath when given."""
    chips = []
    for i, tok in enumerate(tokens):
        color = token_color(i, len(tokens))
        display = repr(tok).strip("'") if tok.strip() == "" else tok
        id_html = ""
        if ids is not None and i < len(ids):
            id_html = f'<span class="id-chip">{ids[i]}</span>'
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
- Mayor esfuerzo → más latencia y tokens consumidos, pero mejor precisión en tareas lógicas/matemáticas.
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
    """Recursive-character-like splitter: intenta cortar en \\n\\n, luego \\n, luego espacio."""
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
    """Split by Markdown headers (#, ##, ###), keeping header hierarchy as metadata."""
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
            keys_to_drop = [k for k in current_headers if k >= level]
            for k in keys_to_drop:
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
        s = sim[i, i-1]
        if s >= threshold:
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
        st.markdown('<div class="warn-box" style="font-size:0.75rem;">⚠️ Sin API Key — módulos que llaman a Groq quedan deshabilitados. Los módulos locales (tokenización, embeddings, modelos clásicos) funcionan igual.</div>', unsafe_allow_html=True)

    st.markdown('<hr style="border-color:#30363d;margin:0.8rem 0;">', unsafe_allow_html=True)
    st.markdown('<p style="font-family:\'JetBrains Mono\',monospace;font-size:0.75rem;color:#8b949e;">🟣 ANTHROPIC API KEY (opcional, de pago)</p>', unsafe_allow_html=True)
    anthropic_key_env = os.environ.get("ANTHROPIC_API_KEY", "")
    anthropic_key_input = st.text_input(
        "Anthropic API Key", value=anthropic_key_env, type="password", placeholder="sk-ant-...",
        label_visibility="collapsed", help="A diferencia de Groq, Anthropic NO tiene tier gratuito de API."
    )
    anthropic_key = anthropic_key_input or anthropic_key_env
    if anthropic_key:
        st.markdown('<div class="success-box" style="font-size:0.72rem;">✅ Claude habilitado como comparación opcional</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-box" style="font-size:0.72rem;">ℹ️ Opcional. Anthropic no ofrece API gratuita (solo créditos de prueba al crear cuenta) — a diferencia de Groq, aquí sí se cobra por token desde el primer uso.</div>', unsafe_allow_html=True)

    st.markdown('<hr style="border-color:#30363d;margin:0.8rem 0;">', unsafe_allow_html=True)
    st.markdown('<p style="font-family:\'JetBrains Mono\',monospace;font-size:0.75rem;color:#8b949e;">📍 MÓDULO ACTIVO</p>', unsafe_allow_html=True)
    module = st.selectbox(
        "Módulo",
        options=[
            "🏠  Inicio & Teoría",
            "🔤  Tokenización",
            "📐  Embeddings & Similitud",
            "🕰️  Modelos Clásicos vs Modernos",
            "🧩  Chunking para RAG",
            "🏷️  NLP Clásico (POS, NER, Sentimientos)",
            "⚡  LLM Lab — Parámetros",
            "⚖️  Comparador de Modelos",
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
    <span class="badge" style="background:#58a6ff;">HuggingFace local</span>
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
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#58a6ff;">{len(GROQ_MODELS)}</div><div class="metric-label">Modelos Groq (modernos)</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#3fb950;">{len(CLASSIC_MODELS)}</div><div class="metric-label">Modelos clásicos locales</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#bc8cff;">{len(EMBEDDING_MODELS)}</div><div class="metric-label">Modelos de embedding</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    tab_theory, tab_models, tab_classic, tab_embed, tab_chunk = st.tabs(
        ["📖 Teoría NLP", "🤖 Modelos Groq", "🕰️ Modelos Clásicos", "📐 Embeddings", "🧩 Chunking"]
    )

    with tab_theory:
        col_left, col_right = st.columns([1, 1])
        with col_left:
            st.markdown("""
            <div class="section-title" style="font-size:1.1rem;">Pipeline NLP Completo</div>
            <div class="info-box"><b>1. Tokenización</b><br>Texto crudo → unidades mínimas. Algoritmos: BPE, WordPiece, SentencePiece.</div>
            <div class="info-box"><b>2. Normalización</b><br>Minúsculas, acentos, URLs, stopwords.</div>
            <div class="info-box"><b>3. Representación vectorial</b><br>BoW → TF-IDF → Word2Vec → Embeddings contextuales (BERT, GPT).</div>
            <div class="info-box"><b>4. Modelos de secuencia</b><br>n-gramas → HMM → RNN/LSTM → Transformer.</div>
            <div class="info-box"><b>5. Tareas downstream</b><br>Clasificación, NER, Sentiment, QA, RAG, Generación.</div>
            """, unsafe_allow_html=True)
        with col_right:
            st.markdown('<div class="section-title" style="font-size:1.1rem;">Ecuaciones Clave</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="formula-box">TF-IDF(t,d) = tf(t,d) x log(N / df(t))

tf(t,d)  = f(t,d) / sum_k f(k,d)
idf(t)   = log(N / |{d: t in d}|)</div>
            <div class="formula-box">Similitud Coseno:
cos(theta) = (u . v) / (||u|| ||v||)  en [-1, 1]</div>
            <div class="formula-box">Softmax con temperatura T:
P(w_i) = exp(logit_i / T) / sum_j exp(logit_j / T)

T->0: determinístico (greedy)
T=1:  distribución original
T->inf: uniforme (caótico)</div>
            <div class="formula-box">Scaled Dot-Product Attention:
A(Q,K,V) = softmax(QK^T / sqrt(d_k)) . V</div>
            """, unsafe_allow_html=True)

    with tab_models:
        st.markdown('<div class="section-title" style="font-size:1.1rem;">Modelos Modernos Disponibles en Groq (2026)</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="warn-box">
        <b>⚠️ Qué cambió respecto a catálogos antiguos:</b> Groq retiró modelos como
        <code>mixtral-8x7b-32768</code>, <code>llama-3.1-70b-versatile</code> y <code>gemma2-9b-it</code>
        de su catálogo estándar. El catálogo vigente prioriza la familia <b>GPT-OSS</b> (OpenAI, open-weight)
        y <b>Qwen3</b>, junto a LLaMA 3.3/3.1 para uso general. Verifica siempre
        <a href="https://console.groq.com/docs/models" style="color:#58a6ff;">console.groq.com/docs/models</a>.
        </div>
        """, unsafe_allow_html=True)
        rows = []
        for m, info in GROQ_MODELS.items():
            rows.append({"Model ID (Groq)": m, "Familia": info["family"], "Parámetros": info["params"],
                         "Contexto (tokens)": f"{info['context']:,}", "Tipo": info["type"],
                         "Licencia": info["license"], "Fortalezas": info["strengths"]})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=260)
        st.markdown("""
        <div class="info-box">
        <b>💡 ¿Por qué Groq?</b> Usa hardware <b>LPU (Language Processing Unit)</b> especializado,
        con velocidades de inferencia 10–100x superiores a GPUs convencionales:
        típicamente 200–800+ tokens/segundo. Además de LLaMA/GPT-OSS/Qwen3, Groq también sirve
        <b>DeepSeek R1 Distill</b> (incluido en el catálogo de arriba) totalmente gratis, con
        límites diarios más ajustados que los demás modelos.
        </div>
        <div class="warn-box">
        <b>🟣 ¿Y Claude (Anthropic)?</b> A diferencia de Groq, Google AI Studio o Cloudflare
        Workers AI, <b>Anthropic no ofrece un tier gratuito de API</b> — solo créditos de prueba
        limitados al crear la cuenta, luego se cobra por token desde el primer uso. Por eso Claude
        no está en el catálogo gratuito de arriba: se añadió como opción <b>opcional y de pago</b>
        (clave de Anthropic propia) únicamente en el módulo <b>Comparador de Modelos</b>, para
        contrastar un LLM propietario cerrado contra los open-weight servidos por Groq.
        </div>
        """, unsafe_allow_html=True)

    with tab_classic:
        st.markdown('<div class="section-title" style="font-size:1.1rem;">Modelos Clásicos (tokenización en vivo, sin PyTorch)</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
        Groq no aloja GPT-2, BERT ni RoBERTa: solo sirve LLMs modernos "instruct". Para comparar la
        era pre-LLM con la era actual sin arrastrar PyTorch/`transformers` completo (que en
        Streamlit Cloud dispara errores de <code>torchvision</code> por sus módulos de visión no
        usados), este laboratorio reproduce la <b>tokenización real</b> de GPT-2 y BERT con
        librerías ligeras (<code>tiktoken</code>, <code>tokenizers</code>) y usa una comparación
        <b>ilustrativa</b> —claramente marcada como tal— para la generación de texto.
        </div>
        """, unsafe_allow_html=True)
        rows_c = [{"Modelo": k, "Nombre": v["label"], "Año": v["year"], "Parámetros": v["params"],
                   "Tarea": "Generación (decoder)" if v["task"] == "causal_lm" else "Codificación (encoder)",
                   "Nota": v["note"]} for k, v in CLASSIC_MODELS.items()]
        st.dataframe(pd.DataFrame(rows_c), use_container_width=True, height=240)
        st.markdown("""
        <div class="warn-box">
        <b>🎓 Valor pedagógico:</b> GPT-2 (124M parámetros) genera texto notablemente menos
        coherente que GPT-OSS-120B, no por mala suerte, sino porque tiene ~1000x menos parámetros,
        sin RLHF ni instruction-tuning. Este contraste es el corazón del módulo
        <b>"Modelos Clásicos vs Modernos"</b>.
        </div>
        """, unsafe_allow_html=True)

    with tab_embed:
        st.markdown('<div class="section-title" style="font-size:1.1rem;">Modelos de Embedding</div>', unsafe_allow_html=True)
        rows_e = [{"Modelo": k, "HF id": v["hf_id"], "Dimensiones": v["dims"], "Contexto (tokens)": v["context"],
                   "Tamaño aprox.": v["size"], "Nota": v["note"]} for k, v in EMBEDDING_MODELS.items()]
        st.dataframe(pd.DataFrame(rows_e), use_container_width=True, height=200)
        st.markdown("""
        <div class="info-box">
        Estos tres modelos se usan localmente (vía <code>fastembed</code> / ONNX Runtime, sin PyTorch), no por Groq API
        — Groq no expone hoy un endpoint de embeddings propio para estos modelos, así que la práctica
        estándar en la comunidad es: <b>embeddings locales / HF Inference</b> + <b>generación vía Groq</b>.
        </div>
        """, unsafe_allow_html=True)

    with tab_chunk:
        st.markdown('<div class="section-title" style="font-size:1.1rem;">Estrategias de Chunking para RAG</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box"><b>Estructural</b> (Markdown/HTML headers) — corta siguiendo la jerarquía real
        del documento (#, ##, ###). Preserva contexto de sección, ideal para manuales y políticas.</div>
        <div class="info-box"><b>Recursivo / Fixed-size</b> — corta por tamaño de caracteres, intentando
        respetar párrafos y oraciones antes de cortar a la fuerza. Equivalente simplificado a
        <code>RecursiveCharacterTextSplitter</code> de LangChain.</div>
        <div class="info-box"><b>Semántico</b> — agrupa oraciones consecutivas mientras su similitud
        coseno (embeddings) se mantenga alta; corta cuando el tema cambia. Equivalente conceptual a
        <code>SemanticChunker</code>.</div>
        <div class="info-box"><b>Proposicional / Agéntico</b> — usa un LLM (aquí, Groq) para reescribir
        el texto en proposiciones atómicas autocontenidas, resolviendo pronombres y ambigüedades antes
        de indexar. Más costoso, mayor precisión en retrieval.</div>
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
        ["🔧 BPE (cl100k, familia GPT-3.5/4)", "🐣 GPT-2 (BPE original)", "🤗 BERT (WordPiece)", "📊 Comparativa"]
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
Usado por: GPT-3.5, GPT-4 (referencia histórica de
la familia BPE moderna; no es el tokenizador de
los modelos servidos hoy por Groq, que usan sus
propios tokenizadores BPE/SentencePiece internos)</div>
            """, unsafe_allow_html=True)
        with col_result:
            if TIKTOKEN_AVAILABLE:
                tokens_bpe, ids_bpe = tokenize_with_tiktoken(input_text)
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
                st.plotly_chart(fig_dist, use_container_width=True)
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

Este es el tokenizador ORIGINAL de GPT-2,
el ancestro directo de todos los BPE modernos.
Compáralo con cl100k_base: mismo algoritmo,
vocabulario ~2x más pequeño -> más tokens
para el mismo texto.</div>
            """, unsafe_allow_html=True)
        with col_result:
            tokens_g, ids_g = tokenize_with_tiktoken(input_text, encoding_name="gpt2")
            toks_g_clean = [t.replace("Ġ", "·").replace("\n", "⏎") for t in tokens_g]  # Ġ = espacio precedente
            c1, c2 = st.columns(2)
            c1.metric("Tokens GPT-2", len(tokens_g))
            c2.metric("Vocab size", "50,257")
            st.caption("`·` representa un espacio antes del token (carácter Ġ interno de GPT-2, byte-level BPE)")
            st.markdown(render_token_chips(toks_g_clean[:60], ids_g[:60]), unsafe_allow_html=True)
            if len(tokens_g) > 60:
                st.caption(f"... mostrando primeros 60 de {len(tokens_g)} tokens")

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

Vocab BERT-multilingual: 119,547 tokens</div>
            """, unsafe_allow_html=True)
        with col_result:
            bert_tok = load_bert_tokenizer()
            if bert_tok:
                try:
                    encoding = bert_tok.encode(input_text)
                    ids_bert = encoding.ids
                    tokens_bert = encoding.tokens
                    c1, c2 = st.columns(2)
                    c1.metric("Tokens BERT", len(tokens_bert))
                    c2.metric("Incluye [CLS]/[SEP]", "Sí" if tokens_bert and tokens_bert[0] in ("[CLS]", "<s>") else "Depende del modelo")
                    st.markdown("**Tokens + IDs WordPiece:**")
                    st.markdown(render_token_chips(tokens_bert[:60], ids_bert[:60]), unsafe_allow_html=True)
                    with st.expander("Ver tabla completa de IDs"):
                        df_toks = pd.DataFrame({
                            "Posición": range(len(tokens_bert[:40])), "Token": tokens_bert[:40],
                            "ID": ids_bert[:40],
                            "¿Subpalabra?": ["Sí" if t.startswith("##") else "No" for t in tokens_bert[:40]]
                        })
                        st.dataframe(df_toks, use_container_width=True)
                except Exception as e:
                    st.error(f"Error tokenizando: {e}")
            else:
                st.info("Cargando tokenizador BERT-multilingual... (requiere descarga inicial del tokenizer.json, ~1-2 MB, sin PyTorch)")

    with tab_compare:
        st.markdown('<div class="section-title" style="font-size:1rem;">Comparativa de Tokenizadores</div>', unsafe_allow_html=True)
        results = {}
        if TIKTOKEN_AVAILABLE:
            toks, ids_ = tokenize_with_tiktoken(input_text)
            results["BPE cl100k (ref. GPT-3.5/4)"] = {"tokens": len(toks), "vocab_size": "~100,277",
                "algo": "Byte-Pair Encoding", "era": "2023", "ejemplo": " | ".join(toks[:8]) + "..."}
            toks_g, _ = tokenize_with_tiktoken(input_text, encoding_name="gpt2")
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
        st.dataframe(df_compare, use_container_width=True)

        fig_tok = px.bar(df_compare, x="Tokenizador", y="# Tokens", title="Número de tokens según tokenizador",
                          color="# Tokens", color_continuous_scale=["#58a6ff", "#f0883e", "#f85149"])
        fig_tok.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22", font=dict(color="#e6edf3"),
                               height=300, showlegend=False)
        fig_tok.update_xaxes(gridcolor="#30363d", tickangle=-20); fig_tok.update_yaxes(gridcolor="#30363d")
        st.plotly_chart(fig_tok, use_container_width=True)

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

    tab_tfidf, tab_sbert, tab_viz = st.tabs(["📊 TF-IDF (disperso)", "🤗 Embeddings densos + IDs", "🗺️ Visualización 2D"])

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
                    st.plotly_chart(fig_heat, use_container_width=True)

                    sim_matrix = cosine_similarity(X)
                    fig_sim = px.imshow(sim_matrix, x=[f"D{i+1}" for i in range(len(documents))],
                                         y=[f"D{i+1}" for i in range(len(documents))],
                                         title="Similitud Coseno entre documentos (TF-IDF)",
                                         color_continuous_scale="RdYlGn", zmin=0, zmax=1, text_auto=".2f")
                    fig_sim.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                                           font=dict(color="#e6edf3", size=10), height=280, title_font=dict(size=12))
                    st.plotly_chart(fig_sim, use_container_width=True)
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.warning("Necesitas al menos 2 documentos y sklearn instalado.")

    with tab_sbert:
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

        if st.button("⚡ Calcular Embeddings", key="sbert_btn") and len(sentences) >= 2:
            emb_model = load_embedding_model(embed_choice)
            if emb_model:
                with st.spinner("Calculando embeddings (ONNX Runtime)..."):
                    t0 = time.perf_counter()
                    embeddings = embed_texts(emb_model, sentences)
                    elapsed = time.perf_counter() - t0

                st.success(f"✅ {len(sentences)} embeddings de {embeddings.shape[1]} dimensiones en {elapsed:.3f}s")

                # Show a slice of the raw vector + its "IDs" (dimension index) for one sentence
                with st.expander("🔍 Ver el vector crudo (primeras 20 dimensiones) de la Frase 1"):
                    df_vec = pd.DataFrame({"Dimensión (índice)": range(20), "Valor": embeddings[0][:20].round(4)})
                    st.dataframe(df_vec, use_container_width=True, height=200)
                    st.caption("A diferencia de un token ID (entero, categórico), cada componente de un embedding "
                               "es un número real continuo — no representa una palabra por sí solo, solo cobra "
                               "sentido en conjunto con las demás dimensiones.")

                sim = cosine_sim_matrix(embeddings)
                fig_csim = px.imshow(sim, x=[f"S{i+1}" for i in range(len(sentences))],
                                      y=[f"S{i+1}" for i in range(len(sentences))],
                                      title=f"Similitud Coseno — {embed_choice}", color_continuous_scale="RdYlGn",
                                      zmin=-1, zmax=1, text_auto=".3f")
                fig_csim.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                                        font=dict(color="#e6edf3"), height=350, title_font=dict(size=13))
                st.plotly_chart(fig_csim, use_container_width=True)

                for i, s in enumerate(sentences):
                    st.markdown(f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.75rem;color:#f0883e;background:#1c2333;padding:2px 6px;border-radius:3px;">S{i+1}</span> <span style="font-size:0.85rem;"> {s}</span><br>', unsafe_allow_html=True)

                pairs = []
                for i in range(len(sentences)):
                    for j in range(i+1, len(sentences)):
                        pairs.append({"Frase A": f"S{i+1}: {sentences[i][:45]}...",
                                       "Frase B": f"S{j+1}: {sentences[j][:45]}...",
                                       "Similitud": round(float(sim[i, j]), 4)})
                df_pairs = pd.DataFrame(pairs).sort_values("Similitud", ascending=False)
                st.markdown("**Ranking de pares por similitud:**")
                st.dataframe(df_pairs, use_container_width=True, height=220)

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
                st.plotly_chart(fig_viz, use_container_width=True)
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
# MODULE 3: MODELOS CLÁSICOS vs MODERNOS
# ════════════════════════════════════════════════════════════
elif module == "🕰️  Modelos Clásicos vs Modernos":
    st.markdown('<div class="section-title">De GPT-2 (2019) a GPT-OSS (2025): dos eras de LLMs</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    Este módulo compara la era <b>pre-instruction-tuning</b> (GPT-2, BERT — 2018/2019) con la era
    actual de LLMs (GPT-OSS, Qwen3, LLaMA — vía Groq). La comparación de <b>generación</b> es
    ilustrativa (no ejecuta pesos de GPT-2 en vivo — ver nota técnica abajo); la comparación de
    <b>tokenización</b> sí es 100% en vivo con las librerías reales de cada época.
    </div>
    """, unsafe_allow_html=True)

    rows_c = [{"Modelo": k, "Nombre": v["label"], "Año": v["year"], "Parámetros": v["params"],
               "Tarea nativa": "Generación autoregresiva (decoder)" if v["task"] == "causal_lm" else "Codificación bidireccional (encoder)",
               "Nota": v["note"]} for k, v in CLASSIC_MODELS.items()]
    st.dataframe(pd.DataFrame(rows_c), use_container_width=True, height=220)

    st.markdown('<div class="section-title" style="font-size:1.1rem;">🔤 Tokenización en vivo: GPT-2 (2019) vs BPE moderno</div>', unsafe_allow_html=True)
    prompt_cm = st.text_area(
        "Texto a tokenizar con ambos algoritmos:",
        value="Artificial intelligence will change education because it personalizes learning at scale.",
        height=70
    )
    if TIKTOKEN_AVAILABLE and prompt_cm.strip():
        toks_old, ids_old = tokenize_with_tiktoken(prompt_cm, encoding_name="gpt2")
        toks_new, ids_new = tokenize_with_tiktoken(prompt_cm, encoding_name="cl100k_base")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown(f"**GPT-2 BPE (2019, vocab 50,257)** — {len(toks_old)} tokens")
            st.markdown(render_token_chips([t.replace('Ġ', '·') for t in toks_old[:40]], ids_old[:40]), unsafe_allow_html=True)
        with col_t2:
            st.markdown(f"**cl100k BPE (ref. GPT-3.5/4, vocab ~100,277)** — {len(toks_new)} tokens")
            st.markdown(render_token_chips(toks_new[:40], ids_new[:40]), unsafe_allow_html=True)
        st.caption("Mismo texto, mismo algoritmo (BPE) — el vocabulario más grande y mejor entrenado de la "
                   "era moderna produce, casi siempre, menos tokens para el mismo texto.")

    st.markdown('<div class="section-title" style="font-size:1.1rem;">💬 Generación: ilustrativo (GPT-2) vs en vivo (Groq)</div>', unsafe_allow_html=True)
    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        st.markdown("### 🐣 GPT-2 (124M, 2019) — ilustrativo")
        st.markdown("""
        <div class="warn-box" style="font-size:0.8rem;">
        ⚠️ No se ejecuta inferencia real aquí: correr pesos de GPT-2 requiere PyTorch, y
        <code>transformers</code> arrastra cientos de submódulos de visión que rompen el
        <i>file watcher</i> de Streamlit Cloud incluso sin usarlos (ver nota al final de la página).
        El texto de abajo es una <b>reconstrucción representativa</b> del estilo típico de GPT-2
        con este tipo de prompt: continúa la frase de forma plausible pero divaga, pierde el hilo
        o se repite tras pocas oraciones — no sigue instrucciones porque nunca fue entrenado para eso.
        </div>
        """, unsafe_allow_html=True)
        illustrative_gpt2 = (
            prompt_cm.strip() + " of the most important thing in the world. It is also a good way "
            "to get a job in the future, and it is also a good way to get a job in the future. "
            "The future of education is not the same as the future of education."
        )
        st.markdown(f'<div class="llm-response" style="font-size:0.85rem;">{illustrative_gpt2}</div>', unsafe_allow_html=True)
        st.caption("↑ Ilustrativo — nota la repetición y la falta de una respuesta real a la instrucción implícita.")
    with col_cfg2:
        st.markdown("### 🚀 Modelo moderno (Groq) — en vivo")
        modern_choice = st.selectbox(
            "Modelo moderno:", list(GROQ_MODELS.keys()),
            format_func=lambda m: f"{GROQ_MODELS[m]['family']} ({GROQ_MODELS[m]['params']})", key="modern_choice"
        )
        modern_max = st.slider("max_tokens", 50, 500, 150, key="modern_mt")
        modern_temp = st.slider("temperature", 0.0, 2.0, 0.7, 0.1, key="modern_temp")
        if st.button("⚡ Generar con el modelo moderno", key="run_modern", use_container_width=True) and prompt_cm.strip():
            if not api_key:
                st.markdown('<div class="warn-box">⚠️ Se requiere Groq API Key en la barra lateral.</div>', unsafe_allow_html=True)
            else:
                client = get_groq_client(api_key)
                with st.spinner("Generando vía Groq..."):
                    r = call_groq(client=client, model=modern_choice,
                                  messages=[{"role": "user", "content": prompt_cm}],
                                  temperature=modern_temp, max_tokens=modern_max)
                if r["success"]:
                    st.markdown(f'<div class="llm-response" style="font-size:0.85rem;">{r["content"]}</div>', unsafe_allow_html=True)
                    st.caption(f"⏱️ {r['latency']:.2f}s · {r['tokens_per_sec']:.0f} tok/s · instruction-tuned + RLHF")
                else:
                    st.error(r.get("error", "Error"))

    st.markdown("""
    <div class="success-box">
    🎓 <b>Qué observar:</b> GPT-2 tiende a divagar, repetirse o perder el hilo tras pocas
    oraciones — está prediciendo el token más probable sin ningún concepto de "responder a una
    instrucción". El modelo moderno entiende la intención, mantiene coherencia larga y sigue un
    formato conversacional. La diferencia no es solo de tamaño: es de <i>paradigma de
    entrenamiento</i> (pretraining puro vs. pretraining + SFT + RLHF).
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-title" style="font-size:1.1rem;">Encoders clásicos: BERT / RoBERTa (Masked Language Modeling)</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    BERT y RoBERTa son <b>encoder-only</b>: no completan texto de forma autoregresiva. Su tarea
    nativa de preentrenamiento es <b>Masked Language Modeling (MLM)</b> — predecir una palabra
    oculta usando contexto de AMBOS lados a la vez (por eso son "bidireccionales").
    </div>
    <div class="formula-box">P(w_mask | contexto_izq, contexto_der) = softmax(W · h_mask + b)

h_mask = representación final de la posición enmascarada,
que ya incorporó información de TODA la oración (self-attention
sin máscara causal, a diferencia de un decoder como GPT-2/GPT-OSS)</div>
    """, unsafe_allow_html=True)
    mlm_text = st.text_input("Frase con un hueco a completar mentalmente:",
                              value="La capital de Colombia es ___.")
    st.markdown("""
    <div class="chunk-box">
    🧠 <b>Ejercicio guiado</b> (sin inferencia en vivo): si fueras BERT, ¿qué palabras tendrían
    probabilidad alta para el hueco de arriba? Piensa en: (1) el tipo gramatical esperado
    (sustantivo propio), (2) el conocimiento de mundo necesario (capitales de países),
    (3) qué tan única es la respuesta dado el contexto. Este es exactamente el tipo de señal
    que MLM usa para aprender representaciones — sin ninguna etiqueta humana, solo prediciendo
    palabras ocultas en texto masivo.
    </div>
    """, unsafe_allow_html=True)
    st.caption("Para inferencia real de MLM (BERT/RoBERTa) fuera de este lab: `pip install transformers torch` "
               "en un entorno separado y usa `pipeline('fill-mask', model=...)`.")


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
        ["🏗️ Estructural (Markdown)", "✂️ Recursivo / Fixed-size", "🧠 Semántico", "🤖 Proposicional / Agéntico"]
    )

    with tab_struct:
        st.markdown("""
        <div class="formula-box">Corta por encabezados Markdown (#, ##, ###...)
Preserva la jerarquía como metadata de cada chunk.
Equivalente a: MarkdownHeaderTextSplitter</div>
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
contexto en los bordes.
Equivalente simplificado a: RecursiveCharacterTextSplitter</div>
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
            st.plotly_chart(fig_len, use_container_width=True)

    with tab_sem:
        st.markdown("""
        <div class="formula-box">1. Divide el texto en oraciones
2. Calcula el embedding de cada oración
3. Agrupa oraciones consecutivas mientras la
   similitud coseno con la anterior >= threshold
4. Corta cuando el tema cambia (similitud cae)
Equivalente conceptual a: SemanticChunker</div>
        """, unsafe_allow_html=True)
        if not (FASTEMBED_AVAILABLE and NLTK_AVAILABLE):
            st.warning("Requiere `fastembed` y NLTK instalados.")
        else:
            doc_sem = st.text_area("Documento:", value=SAMPLE_TEXTS["rag_doc"].replace("#", "").replace("\n\n", " "),
                                    height=150, key="sem_doc")
            sem_threshold = st.slider("Umbral de similitud (threshold)", 0.1, 0.95, 0.55, 0.05)
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
        <div class="formula-box">Un LLM reescribe el texto en proposiciones
atómicas: hechos independientes, sin pronombres
ambiguos. Mayor costo (llamada a LLM por chunk),
mayor precisión de retrieval.
Equivalente a: chunking "Propositional" / "Agentic"
con ChatGroq + PromptTemplate</div>
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
# MODULE 5: NLP CLÁSICO
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
            st.plotly_chart(fig_pos, use_container_width=True)
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
                st.dataframe(pd.DataFrame(entities), use_container_width=True, height=200)
            else:
                st.info("No se detectaron entidades con NLTK. Prueba en inglés o usa un modelo BERT-NER en español desde el módulo Playground.")
            st.markdown('<div class="warn-box"><b>💡 Nota:</b> NLTK NER funciona mejor en inglés. Para español, usar spaCy (es_core_news_lg) o un BERT fine-tuned para NER en español.</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error en NER: {e}")

    with tab_sent:
        st.markdown('<div class="info-box"><b>VADER</b> es un lexicón basado en reglas, funciona mejor en inglés. Retorna Positivo/Negativo/Neutro/Compound.</div>', unsafe_allow_html=True)
        test_sentences = ["I love this product, it is absolutely amazing!",
                          "This is the worst experience I have ever had.",
                          "The weather today is okay, nothing special.",
                          "Artificial intelligence is transforming education in incredible ways."]
        custom_sent = st.text_area("Frases (una por línea, inglés funciona mejor):", value="\n".join(test_sentences), height=130)
        sent_list = [s.strip() for s in custom_sent.split("\n") if s.strip()]
        try:
            sia = SentimentIntensityAnalyzer()
            results_sent = []
            for sentence in sent_list:
                scores = sia.polarity_scores(sentence)
                sentiment = "Positivo" if scores["compound"] >= 0.05 else "Negativo" if scores["compound"] <= -0.05 else "Neutro"
                results_sent.append({"Texto": sentence[:60], "Positivo": round(scores["pos"], 3),
                                     "Negativo": round(scores["neg"], 3), "Compound": round(scores["compound"], 3), "Sentimiento": sentiment})
            df_sent = pd.DataFrame(results_sent)
            st.dataframe(df_sent, use_container_width=True, height=220)
            fig_sent = px.bar(df_sent, x="Texto", y="Compound", color="Sentimiento",
                               color_discrete_map={"Positivo": "#3fb950", "Negativo": "#f85149", "Neutro": "#8b949e"},
                               title="Score Compound por frase")
            fig_sent.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22", font=dict(color="#e6edf3"), height=300, xaxis=dict(tickangle=-20))
            st.plotly_chart(fig_sent, use_container_width=True)
        except Exception as e:
            st.error(f"Error en análisis de sentimientos: {e}")

    with tab_stats:
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
            freq = Counter(content_words).most_common(15)
            df_freq = pd.DataFrame(freq, columns=["Palabra", "Frecuencia"])
            fig_freq = px.bar(df_freq, x="Frecuencia", y="Palabra", orientation="h", title="Top 15 palabras de contenido",
                               color_discrete_sequence=["#58a6ff"])
            fig_freq.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22", font=dict(color="#e6edf3"), height=380, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_freq, use_container_width=True)
        except Exception as e:
            st.error(f"Error en estadísticas: {e}")

# ════════════════════════════════════════════════════════════
# MODULE 6: LLM LAB — PARÁMETROS
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

        if st.button("⚡ Generar", key="run_gen", use_container_width=True) and user_query.strip():
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
                    out_toks, out_ids = tokenize_with_tiktoken(result["content"])
                    st.markdown(render_token_chips(out_toks[:100], out_ids[:100]), unsafe_allow_html=True)
            else:
                st.error(f"Error: {result.get('error', 'Unknown error')}")


# ════════════════════════════════════════════════════════════
# MODULE 7: COMPARADOR DE MODELOS
# ════════════════════════════════════════════════════════════
elif module == "⚖️  Comparador de Modelos":
    st.markdown('<div class="section-title">Comparador: Misma Query, Múltiples LLMs Groq</div>', unsafe_allow_html=True)
    if not api_key:
        st.markdown('<div class="warn-box">⚠️ Se requiere Groq API Key.</div>', unsafe_allow_html=True)
        st.stop()
    client = get_groq_client(api_key)

    models_to_compare = st.multiselect(
        "Selecciona modelos Groq a comparar (máximo 4, todos gratis):", options=list(GROQ_MODELS.keys()),
        default=["llama-3.1-8b-instant", "openai/gpt-oss-20b", "qwen/qwen3-32b"],
        format_func=lambda m: f"{MODEL_META[m]['family']} ({MODEL_META[m]['params']})", max_selections=4
    )

    include_claude = st.checkbox(
        "🟣 Incluir Claude (Anthropic) en la comparación — opcional, de pago, requiere tu propia API key",
        value=False,
        disabled=not anthropic_key,
        help="Ingresa tu Anthropic API Key en la barra lateral para habilitar esta opción."
    )
    if include_claude and not anthropic_key:
        st.caption("Ingresa una Anthropic API Key en la barra lateral para habilitarlo.")
    claude_model_for_cmp = None
    if include_claude and anthropic_key:
        claude_model_for_cmp = st.selectbox(
            "Modelo Claude:", list(ANTHROPIC_MODELS.keys()),
            format_func=lambda m: f"{ANTHROPIC_MODELS[m]['family']} — de pago", key="cmp_claude_model"
        )
    col_p1, col_p2, col_p3 = st.columns(3)
    cmp_temperature = col_p1.slider("temperature", 0.0, 2.0, 0.7, 0.1, key="cmp_t")
    cmp_max_tokens = col_p2.slider("max_tokens", 50, 2048, 400, 50, key="cmp_mt")
    cmp_top_p = col_p3.slider("top_p", 0.1, 1.0, 1.0, 0.1, key="cmp_tp")
    cmp_system = st.text_input("System prompt (compartido):",
        value="Eres un asistente experto en IA y ciencia de datos. Responde en español, claro y conciso.")
    cmp_query = st.text_area("Query para todos los modelos:",
        value="Explica la diferencia entre BPE, WordPiece y SentencePiece en 3 puntos cada uno.", height=90)

    if st.button("⚡ Comparar modelos", key="run_compare", use_container_width=True) and cmp_query.strip():
        if not models_to_compare and not (include_claude and claude_model_for_cmp):
            st.warning("Selecciona al menos un modelo."); st.stop()
        results_compare = {}
        total_calls = len(models_to_compare) + (1 if (include_claude and claude_model_for_cmp) else 0)
        progress = st.progress(0, text="Iniciando comparación...")
        for i, model in enumerate(models_to_compare):
            progress.progress(i / total_calls, text=f"Consultando {MODEL_META[model]['family']}...")
            r = call_groq(client=client, model=model,
                          messages=[{"role": "system", "content": cmp_system}, {"role": "user", "content": cmp_query}],
                          temperature=cmp_temperature, max_tokens=cmp_max_tokens, top_p=cmp_top_p)
            results_compare[model] = r
            time.sleep(0.2)
        if include_claude and claude_model_for_cmp:
            progress.progress(len(models_to_compare) / total_calls, text=f"Consultando {ANTHROPIC_MODELS[claude_model_for_cmp]['family']} (de pago)...")
            claude_client = get_anthropic_client(anthropic_key)
            r_claude = call_anthropic(client=claude_client, model=claude_model_for_cmp,
                                       messages=[{"role": "user", "content": cmp_query}], system=cmp_system,
                                       temperature=cmp_temperature, max_tokens=cmp_max_tokens)
            results_compare[claude_model_for_cmp] = r_claude
        progress.progress(1.0, text="✅ Comparación completada")

        successful = {m: r for m, r in results_compare.items() if r.get("success")}
        failed = {m: r for m, r in results_compare.items() if not r.get("success")}
        if failed:
            st.markdown("### ⚠️ Errores")
            for model, result in failed.items():
                st.markdown(f'<div class="warn-box"><b>❌ {MODEL_META[model]["family"]}</b><br><code style="font-size:0.8rem;">{result.get("error","")}</code></div>', unsafe_allow_html=True)
        if not successful:
            st.error("Ningún modelo respondió correctamente."); st.stop()

        st.markdown("### 📊 Métricas Comparativas")
        metric_cols = st.columns(max(1, len(successful)))
        for col, (model, result) in zip(metric_cols, successful.items()):
            with col:
                mc = MODEL_META[model]["color"]
                st.markdown(f"""
                <div class="metric-card" style="border-color:{mc};">
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:{mc};">{MODEL_META[model]['family']}</div>
                <div class="metric-value" style="color:{mc};font-size:1.3rem;">{result['latency']:.2f}s</div>
                <div class="metric-label">Latencia</div>
                <div style="margin-top:0.4rem;font-size:0.8rem;color:#8b949e;">
                ⚡ {result['tokens_per_sec']:.0f} tok/s<br>📤 {result['usage']['prompt_tokens']} prompt<br>
                📥 {result['usage']['completion_tokens']} output</div></div>
                """, unsafe_allow_html=True)

        if len(successful) >= 2:
            fig_speed = go.Figure()
            labels = [MODEL_META[m]['family'] for m in successful]
            fig_speed.add_trace(go.Bar(name="Latencia (s)", x=labels, y=[r["latency"] for r in successful.values()],
                                        marker_color=[MODEL_META[m]["color"] for m in successful], opacity=0.85))
            fig_speed.add_trace(go.Scatter(name="Tokens/s", x=labels, y=[r["tokens_per_sec"] for r in successful.values()],
                                            mode="markers+lines", marker=dict(size=10, color="#ffffff"),
                                            line=dict(color="#ffffff", dash="dot"), yaxis="y2"))
            fig_speed.update_layout(title="Latencia vs Velocidad de Generación", paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                                     font=dict(color="#e6edf3"), yaxis=dict(title="Latencia (s)"),
                                     yaxis2=dict(title="Tokens/s", overlaying="y", side="right"), height=320)
            st.plotly_chart(fig_speed, use_container_width=True)

        st.markdown("### 💬 Respuestas Comparadas")
        resp_cols = st.columns(max(1, len(successful)))
        for col, (model, result) in zip(resp_cols, successful.items()):
            with col:
                mc = MODEL_META[model]["color"]
                st.markdown(f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.72rem;color:{mc};font-weight:600;">{MODEL_META[model]["family"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="llm-response" style="font-size:0.82rem;">{result.get("content","") or "*(sin respuesta)*"}</div>', unsafe_allow_html=True)

        df_tokens = pd.DataFrame([{"Modelo": MODEL_META[m]["family"], "Prompt tokens": r["usage"]["prompt_tokens"],
                                    "Output tokens": r["usage"]["completion_tokens"], "Tokens/s": round(r["tokens_per_sec"], 1),
                                    "Latencia (s)": round(r["latency"], 3)} for m, r in successful.items()])
        st.markdown("### 📋 Tabla de Uso de Tokens")
        st.dataframe(df_tokens, use_container_width=True)


# ════════════════════════════════════════════════════════════
# MODULE 8: ATTENTION VISUALIZER
# ════════════════════════════════════════════════════════════
elif module == "🎯  Attention Visualizer":
    st.markdown('<div class="section-title">Attention Visualizer: Entendiendo Qué Atiende el Modelo</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box"><b>Proxy pedagógico:</b> este módulo aproxima la atención con similitud
    de n-gramas de caracteres, no son los pesos reales de un Transformer (para eso: <code>pip install bertviz</code>).</div>
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

                fig_attn = px.imshow(sim_attn, x=tokens_attn, y=tokens_attn, title="Matriz de Atención Aproximada",
                                      color_continuous_scale="Oranges", text_auto=".2f", aspect="auto")
                fig_attn.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22", font=dict(color="#e6edf3", size=11),
                                        height=max(300, n_toks * 25 + 100), xaxis=dict(tickangle=-45))
                st.plotly_chart(fig_attn, use_container_width=True)

                query_token_idx = st.select_slider("Token de consulta (Query):", options=list(range(n_toks)),
                                                     format_func=lambda i: f"[{i}] '{tokens_attn[i]}'", value=min(3, n_toks-1))
                attn_row = sim_attn[query_token_idx]
                col_viz1, col_viz2 = st.columns([2, 1])
                with col_viz1:
                    fig_row = px.bar(x=tokens_attn, y=attn_row, title=f"Distribución de atención de '{tokens_attn[query_token_idx]}'",
                                      color=attn_row, color_continuous_scale=["#161b22", "#f0883e", "#ffa657"])
                    fig_row.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22", font=dict(color="#e6edf3"),
                                           height=280, xaxis=dict(tickangle=-30), showlegend=False)
                    st.plotly_chart(fig_row, use_container_width=True)
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
            if st.button("📤 Enviar", key="pg_send", use_container_width=True) and pg_user_input.strip():
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
