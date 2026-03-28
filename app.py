"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         DebateLM — Version 2                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

RAG Stack:
  • Parent-Child Chunking  (small retrieval chunks → large context windows)
  • Hybrid Search          (BM25 lexical + ChromaDB semantic)
  • Cross-Encoder Re-Rank  (ms-marco-MiniLM for result quality)
  • HyDE                   (Hypothetical Document Embeddings for better recall)
  • MMR Diversity          (avoid redundant context)

Search Grounding:
  • Serper (Google Search API) for real-time web evidence

AI Backbone:
  • Streaming responses with st.write_stream
  • Per-user isolated storage (IP-hashed)
"""

import streamlit as st
import os, json, datetime, uuid, hashlib, shutil, time, re
from typing import Optional
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_community.retrievers import EnsembleRetriever
import requests

load_dotenv()

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")
SERPER_API_KEY  = os.getenv("SERPER_API_KEY", "")

AVAILABLE_MODELS = [
    "gemini-3-flash-preview",
    "gemini-3.1-pro-preview",
    "gemini-3.1-flash-lite-preview",
]
DEFAULT_MODEL = AVAILABLE_MODELS[0]
MAX_DEBATES   = 50

PARENT_CHUNK_SIZE    = 2000
PARENT_CHUNK_OVERLAP = 200
CHILD_CHUNK_SIZE     = 400
CHILD_CHUNK_OVERLAP  = 50
TOP_K_RETRIEVAL      = 20
TOP_K_FINAL          = 8

# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  — dark, judicial theme
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="DebateLM",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d0f14;
    color: #e8e2d5;
}
.stApp { background: #0d0f14; }

/* ── Hero ── */
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3.8rem;
    font-weight: 900;
    background: linear-gradient(135deg, #c9a96e 0%, #f0d9a8 50%, #c9a96e 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    letter-spacing: -0.02em;
    line-height: 1.1;
}
.hero-sub {
    text-align: center;
    color: #7a7060;
    font-size: 1.05rem;
    font-weight: 300;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 0.3rem;
}
.hero-divider {
    border: none;
    border-top: 1px solid #2a2820;
    margin: 1.5rem 0 2rem 0;
}

/* ── Agent cards ── */
.agent-card {
    border: 1px solid #2a2820;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    background: linear-gradient(145deg, #13151c, #1a1c24);
    margin-bottom: 0.6rem;
}

/* ── Debate bubble ── */
.debate-bubble {
    border-left: 3px solid #c9a96e;
    background: #13151c;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    font-size: 0.95rem;
    line-height: 1.7;
}
.debate-bubble.judge {
    border-left-color: #5ba4a4;
    background: #0f1a1a;
}
.debate-bubble .speaker-tag {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #c9a96e;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.4rem;
}
.debate-bubble.judge .speaker-tag { color: #5ba4a4; }

/* ── Round badge ── */
.round-badge {
    text-align: center;
    font-family: 'Playfair Display', serif;
    font-size: 1rem;
    color: #4a4540;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin: 1.5rem 0 0.5rem 0;
    border-top: 1px solid #2a2820;
    border-bottom: 1px solid #2a2820;
    padding: 0.4rem 0;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0a0c10 !important;
    border-right: 1px solid #1e2028;
}
section[data-testid="stSidebar"] .stButton > button {
    background: #1a1c24;
    border: 1px solid #2e2c28;
    color: #c9a96e;
    border-radius: 8px;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.82rem;
    letter-spacing: 0.05em;
    transition: all 0.2s;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: #c9a96e;
    color: #0d0f14;
    border-color: #c9a96e;
}

/* ── Primary button ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #c9a96e, #a07840);
    border: none;
    color: #0d0f14;
    font-weight: 500;
    border-radius: 8px;
    font-family: 'DM Sans', sans-serif;
    letter-spacing: 0.05em;
    padding: 0.6rem 2rem;
    transition: opacity 0.2s;
}
.stButton > button[kind="primary"]:hover { opacity: 0.85; }

/* ── Evidence pills ── */
.evidence-pill {
    display: inline-block;
    background: #1a2a1a;
    border: 1px solid #2a4a2a;
    color: #6abf6a;
    border-radius: 20px;
    padding: 0.15rem 0.7rem;
    font-size: 0.75rem;
    font-family: 'DM Mono', monospace;
    margin: 0.1rem 0.2rem;
}
.evidence-pill.web {
    background: #1a1a2a;
    border-color: #3a3a6a;
    color: #8888ff;
}

/* ── Source box ── */
.source-box {
    background: #0f1115;
    border: 1px solid #1e2028;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-size: 0.82rem;
    font-family: 'DM Mono', monospace;
    color: #7a7a8a;
    margin-top: 0.5rem;
    line-height: 1.6;
}

/* ── Status strip ── */
.status-strip {
    background: #0f1115;
    border: 1px solid #1e2028;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    font-size: 0.82rem;
    color: #c9a96e;
    margin: 0.4rem 0;
    font-family: 'DM Mono', monospace;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# USER ISOLATION
# ═══════════════════════════════════════════════════════════════════════════
def get_user_id() -> str:
    try:
        ip = st.context.headers.get("X-Forwarded-For", "")
        if ip:
            return hashlib.sha256(ip.split(",")[0].strip().encode()).hexdigest()[:32]
    except Exception:
        pass
    if "uid" in st.query_params:
        return st.query_params["uid"]
    uid = str(uuid.uuid4()).replace("-", "")[:32]
    st.query_params["uid"] = uid
    return uid

if "session_id" not in st.session_state:
    st.session_state.session_id = get_user_id()

USER_DIR      = os.path.join("user_data", st.session_state.session_id)
AUTOSAVE_FILE = os.path.join(USER_DIR, "autosave.json")
HISTORY_FILE  = os.path.join(USER_DIR, "debate_history_db.json")
PERSIST_DIR   = os.path.join(USER_DIR, "chroma_db")
DOCSTORE_FILE = os.path.join(USER_DIR, "docstore.json")
TEMP_DIR      = os.path.join(USER_DIR, "temp_docs")
os.makedirs(USER_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# PERSISTENCE HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def auto_save():
    data = {k: st.session_state[k] for k in st.session_state
            if k.startswith("inst_") or k == "debate_topic"}
    with open(AUTOSAVE_FILE, "w") as f:
        json.dump(data, f)

def load_past_debates():
    if not os.path.exists(HISTORY_FILE):
        return []
    with open(HISTORY_FILE, "r") as f:
        records = json.load(f)
    for i, r in enumerate(records):
        r.setdefault("id", f"legacy_{i}")
        r.setdefault("research_brief", "")
        r.setdefault("web_evidence", [])
        r.setdefault("post_chat", [])
        r.setdefault("judge_model", DEFAULT_MODEL)
    return records

def save_new_debate(topic, history, verdict, research_brief, web_evidence, judge_model):
    record = {
        "id":             str(datetime.datetime.now().timestamp()),
        "date":           datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "topic":          topic,
        "research_brief": research_brief,
        "web_evidence":   web_evidence,
        "history":        history,
        "verdict":        verdict,
        "post_chat":      [],
        "judge_model":    judge_model,
    }
    st.session_state.past_debates.insert(0, record)
    with open(HISTORY_FILE, "w") as f:
        json.dump(st.session_state.past_debates, f, indent=2)
    return record

# ── Docstore (parent chunks) ────────────────────────────────────────────
def load_docstore() -> dict:
    if os.path.exists(DOCSTORE_FILE):
        with open(DOCSTORE_FILE, "r") as f:
            return json.load(f)
    return {}

def save_docstore(ds: dict):
    with open(DOCSTORE_FILE, "w") as f:
        json.dump(ds, f)

# ═══════════════════════════════════════════════════════════════════════════
# INIT SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════
if "loaded_autosave" not in st.session_state:
    if os.path.exists(AUTOSAVE_FILE):
        with open(AUTOSAVE_FILE, "r") as f:
            for k, v in json.load(f).items():
                st.session_state[k] = v
    st.session_state.loaded_autosave = True

if "past_debates"  not in st.session_state:
    st.session_state.past_debates = load_past_debates()
if "current_view"  not in st.session_state:
    st.session_state.current_view = "new"
if "selected_history" not in st.session_state:
    st.session_state.selected_history = None
if "docstore" not in st.session_state:
    st.session_state.docstore = load_docstore()

# ── Re-load vectorstore if it already exists on disk ────────────────────
if "vectorstore" not in st.session_state:
    if GEMINI_API_KEY and os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        try:
            emb = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=GEMINI_API_KEY,
            )
            st.session_state.vectorstore = Chroma(
                persist_directory=PERSIST_DIR,
                embedding_function=emb,
            )
        except Exception:
            st.session_state.vectorstore = None
    else:
        st.session_state.vectorstore = None

if "bm25_retriever" not in st.session_state:
    st.session_state.bm25_retriever = None

# ═══════════════════════════════════════════════════════════════════════════
# ── INDUSTRY RAG PIPELINE ──────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def process_documents(uploaded_files, is_append: bool = False):
    """
    Parent-Child Chunking Strategy:
    1. Split into PARENT chunks (2 000 tokens) → saved to docstore for LLM context
    2. Split PARENT chunks into CHILD chunks (400 tokens) → embedded in ChromaDB
    3. Each child chunk carries parent_id metadata for lookup
    """
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=PARENT_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    raw_docs: list[Document] = []
    for uf in uploaded_files:
        path = os.path.join(TEMP_DIR, uf.name)
        with open(path, "wb") as f:
            f.write(uf.getbuffer())
        if uf.name.lower().endswith(".pdf"):
            raw_docs.extend(PyPDFLoader(path).load())
        else:
            raw_docs.extend(TextLoader(path).load())

    parent_docs = parent_splitter.split_documents(raw_docs)
    child_docs: list[Document] = []
    docstore: dict = st.session_state.docstore if is_append else {}

    for parent in parent_docs:
        parent_id = str(uuid.uuid4())
        docstore[parent_id] = {
            "content":  parent.page_content,
            "metadata": parent.metadata,
        }
        children = child_splitter.split_documents([parent])
        for child in children:
            child.metadata["parent_id"] = parent_id
            child.metadata["source"]    = parent.metadata.get("source", "unknown")
        child_docs.extend(children)

    st.session_state.docstore = docstore
    save_docstore(docstore)

    emb = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GEMINI_API_KEY,
    )

    if is_append and st.session_state.vectorstore is not None:
        st.session_state.vectorstore.add_documents(child_docs)
        vs = st.session_state.vectorstore
    else:
        vs = Chroma.from_documents(
            documents=child_docs,
            embedding=emb,
            persist_directory=PERSIST_DIR,
        )

    # ── Build / rebuild BM25 retriever ──────────────────────────────────
    all_children = vs.get()["documents"]
    bm25_docs = [Document(page_content=txt) for txt in all_children]
    st.session_state.bm25_retriever = BM25Retriever.from_documents(bm25_docs)
    st.session_state.bm25_retriever.k = TOP_K_RETRIEVAL

    return vs

def _get_parent_context(child_docs: list[Document]) -> str:
    """Swap child chunks for their full parent chunks."""
    seen = set()
    parents = []
    for doc in child_docs:
        pid = doc.metadata.get("parent_id")
        if pid and pid not in seen:
            seen.add(pid)
            entry = st.session_state.docstore.get(pid)
            if entry:
                src = entry["metadata"].get("source", "?")
                parents.append(f"[Source: {src}]\n{entry['content']}")
            else:
                parents.append(doc.page_content)
        elif not pid:
            parents.append(doc.page_content)
    return "\n\n---\n\n".join(parents)

def hyde_query(topic: str, llm: ChatGoogleGenerativeAI) -> str:
    """Hypothetical Document Embedding — generates a fake 'ideal document' for better retrieval."""
    prompt = (
        f"Write a 3-sentence excerpt from an authoritative academic paper or expert report "
        f"that directly addresses: '{topic}'. Be factual and specific."
    )
    resp = llm.invoke(prompt)
    return parse_response(resp)

def gemini_rerank(query: str, candidates: list[Document], llm: ChatGoogleGenerativeAI) -> list[Document]:
    if not candidates:
        return candidates
    numbered = "\n".join(
        f"{i+1}. {doc.page_content[:300]}" for i, doc in enumerate(candidates)
    )
    prompt = (
        f"You are a relevance ranker. Given the query below, rank the following passages "
        f"by relevance (most relevant first). Reply ONLY with the numbers in order, comma-separated. "
        f"Example: 3,1,5,2,4\n\nQuery: {query}\n\nPassages:\n{numbered}"
    )
    try:
        resp = llm.invoke(prompt)
        raw = parse_response(resp).strip()
        indices = [int(x.strip()) - 1 for x in raw.split(",") if x.strip().isdigit()]
        reranked = [candidates[i] for i in indices if 0 <= i < len(candidates)]
        seen = set(id(d) for d in reranked)
        for d in candidates:
            if id(d) not in seen:
                reranked.append(d)
        return reranked[:TOP_K_FINAL]
    except Exception:
        return candidates[:TOP_K_FINAL]


def hybrid_retrieve(query: str, llm: ChatGoogleGenerativeAI) -> list[Document]:
    vs = st.session_state.vectorstore
    bm25 = st.session_state.bm25_retriever
    if vs is None:
        return []

    try:
        hyp_doc = hyde_query(query, llm)
        retrieval_query = f"{query}\n\n{hyp_doc}"
    except Exception:
        retrieval_query = query

    semantic_retriever = vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K_RETRIEVAL, "fetch_k": TOP_K_RETRIEVAL * 3, "lambda_mult": 0.6},
    )

    if bm25 is not None:
        ensemble = EnsembleRetriever(
            retrievers=[bm25, semantic_retriever],
            weights=[0.35, 0.65],
        )
        candidates = ensemble.invoke(retrieval_query)
    else:
        candidates = semantic_retriever.invoke(retrieval_query)

    return gemini_rerank(query, candidates, llm)

# ═══════════════════════════════════════════════════════════════════════════
# WEB SEARCH (Serper)
# ═══════════════════════════════════════════════════════════════════════════
def serper_search(query: str, num_results: int = 6) -> list[dict]:
    """Real-time Google Search via Serper API."""
    if not SERPER_API_KEY:
        return []
    try:
        resp = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
            json={"q": query, "num": num_results, "hl": "en"},
            timeout=10,
        )
        data = resp.json()
        results = []
        for item in data.get("organic", [])[:num_results]:
            results.append({
                "title":   item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "link":    item.get("link", ""),
            })
        return results
    except Exception:
        return []

def format_web_evidence(results: list[dict]) -> str:
    if not results:
        return ""
    lines = ["### 🌐 Live Web Evidence\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. **{r['title']}**\n   {r['snippet']}\n   Source: {r['link']}\n")
    return "\n".join(lines)

# ═══════════════════════════════════════════════════════════════════════════
# CORE AI FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════
def parse_response(response) -> str:
    content = response.content
    if isinstance(content, list):
        return "\n".join(
            block.get("text", "") for block in content
            if isinstance(block, dict) and "text" in block
        )
    return str(content)

def get_llm(model: str, streaming: bool = False) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=GEMINI_API_KEY,
        streaming=streaming,
        temperature=0.7,
    )

def generate_persona(topic: str, agent_idx: int, model: str) -> str:
    llm = get_llm(model)
    prompt = (
        f"Topic: '{topic}'\n"
        f"Create a 2-sentence system instruction for Debate Agent {agent_idx + 1}. "
        f"Give them a very specific professional background (e.g. 'former World Bank economist turned climate activist'), "
        f"a strong bias, and a distinct rhetorical style. Be creative and specific."
    )
    return parse_response(llm.invoke(prompt))

def generate_research_brief(topic: str, judge_model: str) -> tuple[str, list[dict]]:
    """
    Produces the Research Brief by combining:
    - Parent-child RAG retrieval (local PDFs)
    - Live Serper web search results
    """
    llm = get_llm(judge_model)
    rag_context = ""
    web_results: list[dict] = []

    # ── Local knowledge base ─────────────────────────────────────────────
    if st.session_state.vectorstore is not None:
        child_docs = hybrid_retrieve(topic, llm)
        rag_context = _get_parent_context(child_docs)

    # ── Live web grounding ───────────────────────────────────────────────
    if SERPER_API_KEY:
        web_results = serper_search(f"{topic} expert analysis research 2025", num_results=6)

    web_text = format_web_evidence(web_results)
    has_rag  = bool(rag_context.strip())
    has_web  = bool(web_text.strip())

    if not has_rag and not has_web:
        return "No external documents or web evidence provided. Agents will rely on general knowledge.", []

    context_block = ""
    if has_rag:
        context_block += f"\n\n=== KNOWLEDGE BASE (Uploaded Documents) ===\n{rag_context}"
    if has_web:
        context_block += f"\n\n=== LIVE WEB EVIDENCE ===\n{web_text}"

    system = (
        "You are the Chief Research Analyst. Your job is to synthesize all available evidence "
        "into a concise, structured Research Brief for AI debaters. "
        "Always cite sources inline using [Source: filename] for documents and [Web: URL] for web results. "
        "Identify key facts, contested claims, and open questions."
    )
    user = (
        f"Debate Topic: '{topic}'\n\n"
        f"EVIDENCE:{context_block}\n\n"
        "Write a structured Research Brief (400–600 words) with sections: "
        "KEY FACTS, CONTESTED CLAIMS, EXPERT PERSPECTIVES, OPEN QUESTIONS. "
        "Cite every claim."
    )
    resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return parse_response(resp), web_results

def run_agent_turn(
    agent_idx: int,
    agent_data: dict,
    topic: str,
    research_brief: str,
    web_evidence: list[dict],
    debate_history: list[str],
    round_num: int,
) -> str:
    llm = get_llm(agent_data["model"])

    web_context = format_web_evidence(web_evidence) if web_evidence else ""
    citation_rule = (
        "Cite specific sources from the Research Brief using [Source: ...] or [Web: URL] notation. "
        "Every empirical claim MUST have a citation."
        if research_brief and research_brief != "No external documents or web evidence provided. Agents will rely on general knowledge."
        else "Base your arguments on first principles and general knowledge. Do NOT fabricate citations."
    )

    system = (
        f"IDENTITY: {agent_data['instruction']}\n\n"
        f"RESEARCH BRIEF (ground truth):\n{research_brief}\n\n"
        f"LIVE WEB EVIDENCE:\n{web_context}\n\n"
        "DEBATE RULES:\n"
        f"- {citation_rule}\n"
        "- Make 2–3 sharp, well-structured arguments per turn.\n"
        "- Directly rebut the PREVIOUS speaker's weakest claim.\n"
        "- Be intellectually aggressive but factually rigorous.\n"
        "- End with a memorable closing line this round.\n"
        "- Keep response to 200–350 words.\n"
        "- Format: clear paragraphs, no bullet lists."
    )

    if not debate_history:
        user_msg = f"Round {round_num} — Opening statement. Topic: '{topic}'"
    else:
        history_text = "\n\n".join(debate_history[-6:])  # last 3 exchanges for context
        user_msg = (
            f"Round {round_num} — Debate so far:\n\n{history_text}\n\n"
            f"Now make your strongest argument on: '{topic}'"
        )

    resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user_msg)])
    return parse_response(resp)

def run_judge(topic: str, research_brief: str, debate_history: list[str], judge_model: str) -> str:
    llm = get_llm(judge_model)
    transcript = "\n\n".join(debate_history)
    system = (
        "You are the Final Synthesizer — a neutral, wise adjudicator. "
        "Your verdict must be evidence-based, fair, and intellectually honest. "
        "Cite specific arguments and sources. Speak directly to the reader."
    )
    user = (
        f"TOPIC: '{topic}'\n\n"
        f"RESEARCH BRIEF:\n{research_brief}\n\n"
        f"FULL TRANSCRIPT:\n{transcript}\n\n"
        "Deliver your FINAL VERDICT with these sections:\n"
        "1. STRONGEST ARGUMENTS (per side)\n"
        "2. WEAKEST ARGUMENTS (what failed)\n"
        "3. EVIDENCE QUALITY (who cited best)\n"
        "4. SYNTHESIS — the truth as best you can determine it\n"
        "5. WHAT REMAINS CONTESTED\n\n"
        "Be bold, precise, and cite sources throughout. 400–600 words."
    )
    resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return parse_response(resp)

# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        "<div style='font-family:Playfair Display,serif;font-size:1.3rem;"
        "color:#c9a96e;text-align:center;padding:0.5rem 0 1rem 0;'>⚖️ DebateLM</div>",
        unsafe_allow_html=True,
    )

    st.button(
        "➕ New Debate",
        type="primary",
        use_container_width=True,
        on_click=lambda: st.session_state.update(current_view="new"),
    )

    st.markdown("---")
    st.markdown("#### ⚙️ Debate Settings")
    num_agents = st.slider("Debaters", 2, 5, 2)
    num_rounds = st.slider("Rounds", 1, 5, 2)
    judge_model = st.selectbox("Judge Model", AVAILABLE_MODELS, index=0)

    st.markdown("---")
    st.markdown("#### 📚 Knowledge Base")

    doc_count = 0
    if st.session_state.vectorstore:
        try:
            doc_count = st.session_state.vectorstore._collection.count()
        except Exception:
            pass
        st.success(f"✅ {doc_count} chunks indexed")

        with st.expander("Add more documents"):
            more_files = st.file_uploader(
                "Upload PDFs / TXT",
                type=["pdf", "txt"],
                accept_multiple_files=True,
                key="append_files",
            )
            if st.button("Add to KB ➕", use_container_width=True):
                if more_files:
                    with st.spinner("Indexing…"):
                        st.session_state.vectorstore = process_documents(more_files, is_append=True)
                    st.rerun()
    else:
        uploaded_files = st.file_uploader(
            "Upload PDFs / TXT",
            type=["pdf", "txt"],
            accept_multiple_files=True,
        )
        if st.button("Build Knowledge Base 🧠", use_container_width=True, type="primary"):
            if uploaded_files:
                with st.spinner("Building RAG pipeline… (parent-child chunking + BM25 + embeddings)"):
                    st.session_state.vectorstore = process_documents(uploaded_files)
                st.success("Knowledge base ready!")
                st.rerun()

    if st.session_state.vectorstore and st.button("🗑️ Clear Knowledge Base", use_container_width=True):
        if os.path.exists(PERSIST_DIR):
            shutil.rmtree(PERSIST_DIR)
        st.session_state.vectorstore   = None
        st.session_state.bm25_retriever = None
        st.session_state.docstore       = {}
        save_docstore({})
        st.rerun()

    # API key status
    st.markdown("---")
    st.markdown("#### 🔑 API Status")
    col1, col2 = st.columns(2)
    col1.markdown(
        f"<div style='font-size:0.75rem;color:{'#6abf6a' if GEMINI_API_KEY else '#bf6a6a'}'>"
        f"{'✅' if GEMINI_API_KEY else '❌'} Gemini</div>",
        unsafe_allow_html=True,
    )
    col2.markdown(
        f"<div style='font-size:0.75rem;color:{'#6abf6a' if SERPER_API_KEY else '#7a7070'}'>"
        f"{'✅' if SERPER_API_KEY else '⚪'} Serper</div>",
        unsafe_allow_html=True,
    )

    # Quota
    debates_used = len(st.session_state.past_debates)
    st.markdown("---")
    st.markdown(f"#### 📊 Usage  `{debates_used}/{MAX_DEBATES}`")
    st.progress(min(debates_used / MAX_DEBATES, 1.0))

    # History list
    st.markdown("---")
    st.markdown("#### 🗄️ Past Debates")
    if not st.session_state.past_debates:
        st.caption("No debates yet.")
    for rec in st.session_state.past_debates:
        short = rec["topic"][:28] + "…" if len(rec["topic"]) > 28 else rec["topic"]
        if st.button(f"🗓 {short}", key=f"h_{rec['id']}", use_container_width=True):
            st.session_state.current_view    = "history"
            st.session_state.selected_history = rec
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
# HERO HEADER
# ═══════════════════════════════════════════════════════════════════════════
st.markdown(
    "<div class='hero-title'>DebateLM</div>"
    "<div class='hero-sub'>NotebookLM-Level Retrieval · Multi-Agent AI Debate · Live Web Grounding</div>"
    "<hr class='hero-divider'>",
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════════
# VIEW: NEW DEBATE
# ═══════════════════════════════════════════════════════════════════════════
if st.session_state.current_view == "new":
    can_debate = debates_used < MAX_DEBATES and bool(GEMINI_API_KEY)

    st.subheader("1 · Configure Debaters")
    agents_config = []
    cols = st.columns(min(num_agents, 3))
    for i in range(num_agents):
        col = cols[i % len(cols)]
        with col:
            with st.container(border=True):
                st.markdown(f"**Agent {i + 1}**")
                model_sel = st.selectbox("Model", AVAILABLE_MODELS, key=f"model_{i}", index=0)
                mode = st.radio("Persona", ["AI Generated", "Manual"], key=f"mode_{i}", horizontal=True)
                if mode == "Manual":
                    if f"inst_{i}" not in st.session_state:
                        st.session_state[f"inst_{i}"] = "You are a rigorous empiricist. Demand evidence for every claim."
                    instruction = st.text_area("System prompt", key=f"inst_{i}", height=80, on_change=auto_save)
                else:
                    instruction = "__AI_GENERATED__"
                    st.caption("✨ Persona auto-generated from topic")
                agents_config.append({"id": i, "mode": mode, "instruction": instruction, "model": model_sel})

    st.divider()
    st.subheader("2 · Enter the Debate Topic")
    if "debate_topic" not in st.session_state:
        st.session_state.debate_topic = ""
    topic = st.text_area(
        "Topic",
        key="debate_topic",
        placeholder="e.g. Should central banks adopt CBDC? What are the economic trade-offs of UBI? Is AGI an existential risk?",
        height=90,
        on_change=auto_save,
        label_visibility="collapsed",
    )

    # Web search toggle
    use_web = st.toggle("🌐 Enable live web search (Serper)", value=bool(SERPER_API_KEY), disabled=not SERPER_API_KEY,
                        help="Requires SERPER_API_KEY in .env")

    if not GEMINI_API_KEY:
        st.error("❌ GEMINI_API_KEY missing from .env")
    if debates_used >= MAX_DEBATES:
        st.warning(f"🚨 Debate quota reached ({MAX_DEBATES}). Clear history to continue.")

    launch = st.button("⚖️ Launch Debate", type="primary", use_container_width=True, disabled=not can_debate)

    if launch:
        if not topic.strip():
            st.error("Please enter a debate topic first.")
            st.stop()

        # ── Step 1: Research Brief ─────────────────────────────────────
        with st.status("🕵️ Chief Research Analyst is building the brief…", expanded=True) as status:
            st.write("Running hybrid RAG retrieval (BM25 + semantic + HyDE + re-rank)…")
            with st.spinner():
                research_brief, web_evidence = generate_research_brief(
                    topic,
                    judge_model,
                )
            status.update(label="✅ Research brief ready", state="complete")

        with st.expander("📄 Research Brief (click to expand)", expanded=False):
            st.markdown(research_brief)
            if web_evidence:
                st.markdown("---")
                st.markdown("**🌐 Web sources consulted:**")
                for r in web_evidence:
                    st.markdown(f"- [{r['title']}]({r['link']})")

        # ── Step 2: Finalize personas ──────────────────────────────────
        with st.spinner("🎭 Generating agent personas…"):
            active_agents = []
            for ag in agents_config:
                if ag["mode"] == "AI Generated":
                    instr = generate_persona(topic, ag["id"], ag["model"])
                else:
                    instr = ag["instruction"]
                active_agents.append({"instruction": instr, "model": ag["model"]})

        # ── Step 3: Debate rounds ──────────────────────────────────────
        debate_history: list[str] = []

        AGENT_COLORS = ["#c9a96e", "#7ab8c9", "#c97ab8", "#7ac97a", "#c97a7a"]

        for r in range(num_rounds):
            st.markdown(
                f"<div class='round-badge'>— Round {r + 1} of {num_rounds} —</div>",
                unsafe_allow_html=True,
            )
            for i, ag in enumerate(active_agents):
                color = AGENT_COLORS[i % len(AGENT_COLORS)]
                with st.spinner(f"Agent {i+1} ({ag['model']}) is formulating…"):
                    argument = run_agent_turn(
                        agent_idx=i,
                        agent_data=ag,
                        topic=topic,
                        research_brief=research_brief,
                        web_evidence=web_evidence if use_web else [],
                        debate_history=debate_history,
                        round_num=r + 1,
                    )
                st.markdown(
                    f"<div class='debate-bubble'>"
                    f"<div class='speaker-tag' style='color:{color}'>Agent {i+1} · {ag['model'].split('/')[0]}</div>"
                    f"{argument.replace(chr(10), '<br>')}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                debate_history.append(f"Agent {i+1}: {argument}")

        # ── Step 4: Judge ──────────────────────────────────────────────
        st.markdown("<div class='round-badge'>— Final Verdict —</div>", unsafe_allow_html=True)
        with st.spinner(f"⚖️ Judge ({judge_model}) is deliberating…"):
            verdict = run_judge(topic, research_brief, debate_history, judge_model)

        st.markdown(
            f"<div class='debate-bubble judge'>"
            f"<div class='speaker-tag'>⚖️ Final Synthesizer · {judge_model}</div>"
            f"{verdict.replace(chr(10), '<br>')}"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── Save & navigate ────────────────────────────────────────────
        new_rec = save_new_debate(
            topic, debate_history, verdict, research_brief,
            web_evidence if use_web else [], judge_model,
        )
        st.session_state.current_view     = "history"
        st.session_state.selected_history = new_rec
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
# VIEW: HISTORY / POST-DEBATE CHAT
# ═══════════════════════════════════════════════════════════════════════════
elif st.session_state.current_view == "history":
    past = st.session_state.selected_history
    if not past:
        st.session_state.current_view = "new"
        st.rerun()

    st.button("← Back to Setup", on_click=lambda: st.session_state.update(current_view="new"))

    st.markdown(f"<div style='color:#7a7060;font-size:0.8rem;'>📅 {past['date']}</div>", unsafe_allow_html=True)
    st.markdown(f"## {past['topic']}")

    tab1, tab2, tab3 = st.tabs(["📜 Transcript", "📄 Research Brief", "💬 Follow-up Chat"])

    with tab1:
        AGENT_COLORS = ["#c9a96e", "#7ab8c9", "#c97ab8", "#7ac97a", "#c97a7a"]
        for msg in past["history"]:
            if ": " in msg:
                speaker, text = msg.split(": ", 1)
                idx = int(re.search(r"\d+", speaker).group()) - 1 if re.search(r"\d+", speaker) else 0
                color = AGENT_COLORS[idx % len(AGENT_COLORS)]
                st.markdown(
                    f"<div class='debate-bubble'>"
                    f"<div class='speaker-tag' style='color:{color}'>{speaker}</div>"
                    f"{text.replace(chr(10), '<br>')}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("<div class='round-badge'>— Final Verdict —</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='debate-bubble judge'>"
            f"<div class='speaker-tag'>⚖️ Final Synthesizer</div>"
            f"{past['verdict'].replace(chr(10), '<br>')}"
            f"</div>",
            unsafe_allow_html=True,
        )

    with tab2:
        brief = past.get("research_brief", "No brief available.")
        st.markdown(brief)
        web_ev = past.get("web_evidence", [])
        if web_ev:
            st.markdown("---\n**🌐 Web sources consulted:**")
            for r in web_ev:
                st.markdown(f"- [{r['title']}]({r['link']})")

    with tab3:
        st.caption(
            "Ask the Synthesizer anything about this debate — "
            "it remembers the full transcript, research brief, and verdict."
        )
        if "post_chat" not in past:
            past["post_chat"] = []

        for cm in past["post_chat"]:
            with st.chat_message(cm["role"]):
                st.markdown(cm["content"])

        if q := st.chat_input("Ask a follow-up question…"):
            past["post_chat"].append({"role": "user", "content": q})
            with st.chat_message("user"):
                st.markdown(q)

            chat_llm = get_llm(past.get("judge_model", DEFAULT_MODEL))
            sys_ctx = (
                f"You are the Final Synthesizer. You concluded a debate on: '{past['topic']}'.\n\n"
                f"RESEARCH BRIEF:\n{past.get('research_brief', 'N/A')}\n\n"
                f"YOUR VERDICT:\n{past['verdict']}\n\n"
                "Answer follow-up questions with precision. Cite sources from the brief where relevant."
            )
            messages = [SystemMessage(content=sys_ctx)]
            for cm in past["post_chat"]:
                if cm["role"] == "user":
                    messages.append(HumanMessage(content=cm["content"]))
                else:
                    messages.append(AIMessage(content=cm["content"]))

            with st.chat_message("assistant"):
                with st.spinner("Synthesizer is thinking…"):
                    resp = chat_llm.invoke(messages)
                    ans  = parse_response(resp)
                    st.markdown(ans)

            past["post_chat"].append({"role": "assistant", "content": ans})

            for i, rec in enumerate(st.session_state.past_debates):
                if rec["id"] == past["id"]:
                    st.session_state.past_debates[i] = past
                    break
            with open(HISTORY_FILE, "w") as f:
                json.dump(st.session_state.past_debates, f, indent=2)
