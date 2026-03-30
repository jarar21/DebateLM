import streamlit as st
import os, json, datetime, uuid, hashlib, shutil, re
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
import requests

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")

AVAILABLE_MODELS =[
    "gemini-3-flash-preview",
    "gemini-3.1-pro-preview",
    "gemini-3.1-flash-lite-preview",
]
DEFAULT_MODEL        = AVAILABLE_MODELS[0]
MAX_DEBATES          = 50
PARENT_CHUNK_SIZE    = 2000
PARENT_CHUNK_OVERLAP = 200
CHILD_CHUNK_SIZE     = 400
CHILD_CHUNK_OVERLAP  = 50
TOP_K_RETRIEVAL      = 20
TOP_K_FINAL          = 8

st.set_page_config(page_title="DebateLM", page_icon="⚖", layout="wide", initial_sidebar_state="expanded")

# Cleaned-up CSS that uses native Streamlit CSS variables 
# var(--primary-color), var(--background-color), var(--secondary-background-color), var(--text-color)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; }
footer, .stDeployButton { display: none !important; }
.block-container { padding: 3rem 2rem 4rem 2rem !important; max-width: 1400px !important; }

/* Custom Layout Classes inheriting Streamlit Native Colors */
.masthead { display:flex; align-items:flex-end; justify-content:space-between; padding:2.2rem 0 1.4rem 0; border-bottom:1px solid var(--secondary-background-color); margin-bottom:2rem; }
.masthead-wordmark { font-family:'Inter', sans-serif; font-size:2.4rem; font-weight:800; color:var(--text-color); letter-spacing:-0.03em; line-height:1; }
.masthead-tagline { font-family:'Inter', sans-serif; font-size:0.85rem; color:var(--text-color); opacity:0.7; font-weight: 500; margin-top: 0.5rem; }
.masthead-right { display:flex; gap:2.5rem; align-items:flex-end; padding-bottom:0.1rem; }
.masthead-stat-val { font-family:'Inter', sans-serif; font-size:1.4rem; font-weight:700; color:var(--text-color); line-height:1; text-align:right; }
.masthead-stat-lbl { font-family:'Inter', sans-serif; font-size:0.75rem; color:var(--text-color); opacity:0.7; font-weight: 500; margin-top:0.3rem; text-align:right; text-transform:uppercase; letter-spacing:0.05em; }

.section-hd { display:flex; align-items:center; gap:1rem; margin:0.5rem 0 1.2rem 0; }
.section-hd-num { font-family:'Inter', sans-serif; font-size:2rem; font-weight:800; color:var(--secondary-background-color); line-height:1; min-width:2rem; filter: brightness(0.8); }
.section-hd-title { font-family:'Inter', sans-serif; font-size:0.85rem; font-weight:700; letter-spacing:0.05em; text-transform:uppercase; color:var(--text-color); }
.section-hd-desc { font-family:'Inter', sans-serif; font-size:0.8rem; color:var(--text-color); opacity:0.7; font-weight: 500; margin-top:0.15rem; }

.agent-card { border:1px solid var(--secondary-background-color); background:var(--secondary-background-color); padding:0.8rem 1rem; border-radius: 4px; position:relative; margin-bottom:0.5rem; }
.agent-card::before { content:''; position:absolute; top:0; left:0; width:4px; border-radius: 4px 0 0 4px; height:100%; }
.agent-id { font-family:'Inter', sans-serif; font-size:0.75rem; font-weight: 700; letter-spacing:0.05em; text-transform:uppercase; }

.round-divider { display:flex; align-items:center; gap:1.2rem; margin:2.5rem 0 1.5rem 0; }
.round-divider-line { flex:1; height:1px; background:var(--secondary-background-color); }
.round-divider-label { font-family:'Inter', sans-serif; font-size:0.75rem; font-weight: 600; color:var(--text-color); opacity:0.7; text-transform:uppercase; letter-spacing:0.1em; white-space:nowrap; text-align: center; }

.arg-card { border:1px solid var(--secondary-background-color); background:var(--background-color); border-radius: 6px; margin-bottom:1.2rem; position:relative; overflow:hidden; }
.arg-card-accent { position:absolute; top:0; left:0; width:4px; height:100%; }
.arg-card-header { display:flex; align-items:center; justify-content:space-between; padding:0.75rem 1.2rem; border-bottom:1px solid var(--secondary-background-color); background:var(--secondary-background-color); }
.arg-card-speaker { font-family:'Inter', sans-serif; font-size:0.75rem; font-weight: 700; letter-spacing:0.05em; text-transform:uppercase; }
.arg-card-model { font-family:'Inter', sans-serif; font-size:0.7rem; color:var(--text-color); opacity:0.7; font-weight: 500; }
.arg-card-body { padding:1.2rem 1.4rem; font-family:'Inter', sans-serif; font-size:0.95rem; line-height:1.6; color:var(--text-color); word-wrap: break-word; }
.arg-card-query { padding:0.6rem 1.4rem; font-family:'Inter', sans-serif; font-size:0.75rem; color:var(--text-color); opacity:0.7; border-top:1px dashed var(--secondary-background-color); background:var(--secondary-background-color); word-wrap: break-word; }

.verdict-card { border:1px solid var(--primary-color); border-radius: 6px; background:var(--secondary-background-color); margin-top:1.5rem; position:relative; overflow:hidden; }
.verdict-header { display:flex; align-items:center; justify-content:space-between; padding:0.85rem 1.4rem; border-bottom:1px solid var(--primary-color); background:var(--background-color); }
.verdict-title { font-family:'Inter', sans-serif; font-size:0.75rem; font-weight: 700; letter-spacing:0.05em; text-transform:uppercase; color:var(--primary-color); }
.verdict-model { font-family:'Inter', sans-serif; font-size:0.7rem; font-weight: 500; color:var(--primary-color); opacity: 0.8; }
.verdict-body { padding:1.5rem; font-family:'Inter', sans-serif; font-size:0.95rem; line-height:1.6; color:var(--text-color); word-wrap: break-word; }

.sb-logo { font-family:'Inter', sans-serif; font-size:1.2rem; font-weight:800; color:var(--text-color); letter-spacing:-0.02em; padding:1.4rem 1.2rem 0.2rem 1.2rem; border-bottom:none; }
.sb-logo-sub { font-family:'Inter', sans-serif; font-size:0.7rem; font-weight: 500; color:var(--text-color); opacity:0.7; padding:0.2rem 1.2rem 0.8rem 1.2rem; border-bottom:1px solid var(--secondary-background-color); }
.sb-section { padding:1rem 1.2rem; border-bottom:1px solid var(--secondary-background-color); }
.sb-label { font-family:'Inter', sans-serif; font-size:0.75rem; font-weight: 700; text-transform:uppercase; color:var(--text-color); opacity:0.7; margin-bottom:0.8rem; letter-spacing:0.02em; }

.quota-row { padding:0.8rem 1.2rem; border-bottom:1px solid var(--secondary-background-color); }
.quota-label { display:flex; justify-content:space-between; font-family:'Inter', sans-serif; font-size:0.75rem; font-weight: 600; color:var(--text-color); opacity:0.7; margin-bottom:0.5rem; }
.quota-track { height:4px; background:var(--secondary-background-color); filter: brightness(0.9); border-radius: 2px; overflow: hidden; }
.quota-fill { height:100%; background:var(--primary-color); }

.api-status { display:flex; gap:0.5rem; padding:0.7rem 1.2rem; border-bottom:1px solid var(--secondary-background-color); flex-wrap: wrap; }
.api-pill { display:flex; align-items:center; gap:0.4rem; font-family:'Inter', sans-serif; font-size:0.7rem; font-weight: 600; padding:0.25rem 0.6rem; border:1px solid var(--secondary-background-color); border-radius: 4px; }
.api-pill.on { border-color:#059669; color:#059669; background: rgba(16, 185, 129, 0.05); }
.api-pill.off { border-color:var(--secondary-background-color); color:var(--text-color); opacity:0.5; }
.api-dot { width:6px; height:6px; border-radius:50%; }
.api-pill.on .api-dot { background:#059669; }
.api-pill.off .api-dot { background:var(--text-color); opacity:0.5; }

.record-meta { display:flex; gap:1rem; align-items:baseline; margin-bottom:0.8rem; padding-bottom:1rem; border-bottom:1px solid var(--secondary-background-color); }
.record-date { font-family:'Inter', sans-serif; font-size:0.8rem; font-weight: 500; color:var(--text-color); opacity:0.7; }
.record-topic { font-family:'Inter', sans-serif; font-size:1.6rem; font-weight:800; color:var(--text-color); line-height:1.3; margin-bottom:1rem; letter-spacing:-0.02em; word-wrap: break-word; }

/* === RESPONSIVE MEDIA QUERIES === */
@media screen and (max-width: 768px) {
    .block-container { padding: 1.5rem 1rem 3rem 1rem !important; }
    .masthead { flex-direction: column; align-items: flex-start; padding: 1.5rem 0 1rem 0; gap: 1.5rem; }
    .masthead-wordmark { font-size: 2rem; }
    .masthead-right { width: 100%; justify-content: space-between; align-items: flex-start; gap: 1rem; }
    .masthead-stat-val { text-align: left; font-size: 1.2rem; }
    .masthead-stat-lbl { text-align: left; font-size: 0.7rem; }
    .section-hd { flex-direction: column; align-items: flex-start; gap: 0.5rem; }
    .section-hd-num { font-size: 1.5rem; min-width: auto; }
    .arg-card-header, .verdict-header { flex-direction: column; align-items: flex-start; gap: 0.3rem; }
    .arg-card-body, .verdict-body { padding: 1rem; font-size: 0.9rem; }
    .record-meta { flex-direction: column; gap: 0.4rem; }
    .record-topic { font-size: 1.3rem; }
    .round-divider { gap: 0.8rem; margin: 2rem 0 1rem 0; }
    .round-divider-label { font-size: 0.7rem; }
}
</style>
""", unsafe_allow_html=True)


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

def auto_save():
    data = {k: st.session_state[k] for k in st.session_state if k.startswith("inst_") or k == "debate_topic"}
    with open(AUTOSAVE_FILE, "w") as f: json.dump(data, f)

def load_past_debates():
    if not os.path.exists(HISTORY_FILE): return[]
    with open(HISTORY_FILE, "r") as f: records = json.load(f)
    for i, r in enumerate(records):
        r.setdefault("id", f"legacy_{i}")
        r.setdefault("research_logs",[])
        r.setdefault("post_chat",[]) 
        r.setdefault("judge_model", DEFAULT_MODEL)
    return records

def save_new_debate(topic, history, verdict, research_logs, judge_model):
    record = {"id": str(datetime.datetime.now().timestamp()), "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
              "topic": topic, "research_logs": research_logs,
              "history": history, "verdict": verdict, "post_chat":[], "judge_model": judge_model}
    st.session_state.past_debates.insert(0, record)
    with open(HISTORY_FILE, "w") as f: json.dump(st.session_state.past_debates, f, indent=2)
    return record

def load_docstore() -> dict:
    if os.path.exists(DOCSTORE_FILE):
        with open(DOCSTORE_FILE, "r") as f: return json.load(f)
    return {}

def save_docstore(ds: dict):
    with open(DOCSTORE_FILE, "w") as f: json.dump(ds, f)

if "loaded_autosave" not in st.session_state:
    if os.path.exists(AUTOSAVE_FILE):
        with open(AUTOSAVE_FILE, "r") as f:
            for k, v in json.load(f).items(): st.session_state[k] = v
    st.session_state.loaded_autosave = True
if "past_debates"     not in st.session_state: st.session_state.past_debates = load_past_debates()
if "current_view"     not in st.session_state: st.session_state.current_view = "new"
if "selected_history" not in st.session_state: st.session_state.selected_history = None
if "docstore"         not in st.session_state: st.session_state.docstore = load_docstore()

if "vectorstore" not in st.session_state:
    if GEMINI_API_KEY and os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        try:
            emb = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=GEMINI_API_KEY)
            st.session_state.vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=emb)
        except Exception:
            st.session_state.vectorstore = None
    else:
        st.session_state.vectorstore = None
if "bm25_retriever" not in st.session_state: st.session_state.bm25_retriever = None


def process_documents(uploaded_files, is_append=False):
    p_split = RecursiveCharacterTextSplitter(chunk_size=PARENT_CHUNK_SIZE, chunk_overlap=PARENT_CHUNK_OVERLAP, separators=["\n\n","\n",". "," ",""])
    c_split = RecursiveCharacterTextSplitter(chunk_size=CHILD_CHUNK_SIZE,  chunk_overlap=CHILD_CHUNK_OVERLAP,  separators=["\n\n","\n",". "," ",""])
    raw_docs: list[Document] =[]
    for uf in uploaded_files:
        path = os.path.join(TEMP_DIR, uf.name)
        with open(path, "wb") as f: f.write(uf.getbuffer())
        raw_docs.extend(PyPDFLoader(path).load() if uf.name.lower().endswith(".pdf") else TextLoader(path).load())
    parent_docs = p_split.split_documents(raw_docs)
    child_docs: list[Document] =[]
    docstore: dict = st.session_state.docstore if is_append else {}
    for parent in parent_docs:
        pid = str(uuid.uuid4())
        docstore[pid] = {"content": parent.page_content, "metadata": parent.metadata}
        children = c_split.split_documents([parent])
        for child in children:
            child.metadata["parent_id"] = pid
            child.metadata["source"]    = parent.metadata.get("source", "unknown")
        child_docs.extend(children)
    st.session_state.docstore = docstore
    save_docstore(docstore)
    emb = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=GEMINI_API_KEY)
    if is_append and st.session_state.vectorstore is not None:
        st.session_state.vectorstore.add_documents(child_docs)
        vs = st.session_state.vectorstore
    else:
        vs = Chroma.from_documents(documents=child_docs, embedding=emb, persist_directory=PERSIST_DIR)
    all_children = vs.get()["documents"]
    bm25_docs =[Document(page_content=txt) for txt in all_children]
    st.session_state.bm25_retriever = BM25Retriever.from_documents(bm25_docs)
    st.session_state.bm25_retriever.k = TOP_K_RETRIEVAL
    return vs

def _get_parent_context(child_docs):
    if not child_docs: return ""
    seen, parents = set(),[]
    for doc in child_docs:
        pid = doc.metadata.get("parent_id")
        if pid and pid not in seen:
            seen.add(pid)
            entry = st.session_state.docstore.get(pid)
            if entry:
                parents.append(f"[Source: {entry['metadata'].get('source','?')}]\n{entry['content']}")
            else:
                parents.append(doc.page_content)
        elif not pid:
            parents.append(doc.page_content)
    return "\n\n---\n\n".join(parents)

def parse_response(response) -> str:
    c = response.content
    if isinstance(c, list):
        return "\n".join(b.get("text","") for b in c if isinstance(b,dict) and "text" in b)
    return str(c)

def get_llm(model: str):
    return ChatGoogleGenerativeAI(model=model, google_api_key=GEMINI_API_KEY, temperature=0.7)

def hyde_query(topic, llm) -> str:
    return parse_response(llm.invoke(f"Write a 3-sentence excerpt from an authoritative paper directly addressing: '{topic}'. Be factual and specific."))

def gemini_rerank(query, candidates, llm):
    if not candidates: return candidates
    numbered = "\n".join(f"{i+1}. {doc.page_content[:300]}" for i, doc in enumerate(candidates))
    try:
        raw = parse_response(llm.invoke(f"Rank by relevance to query. Reply ONLY with numbers comma-separated.\nQuery: {query}\n\nPassages:\n{numbered}")).strip()
        indices =[int(x.strip())-1 for x in raw.split(",") if x.strip().isdigit()]
        reranked = [candidates[i] for i in indices if 0 <= i < len(candidates)]
        seen = set(id(d) for d in reranked)
        for d in candidates:
            if id(d) not in seen: reranked.append(d)
        return reranked[:TOP_K_FINAL]
    except Exception:
        return candidates[:TOP_K_FINAL]

def rrf_merge(ranked_lists, k=60):
    scores, doc_map = {}, {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked):
            key = doc.page_content[:150]
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            doc_map[key] = doc
    return[doc_map[k] for k in sorted(scores, key=lambda x: scores[x], reverse=True)]

def hybrid_retrieve(query, llm):
    vs = st.session_state.vectorstore
    bm25 = st.session_state.bm25_retriever
    if vs is None: return[]
    try:
        retrieval_query = f"{query}\n\n{hyde_query(query, llm)}"
    except Exception:
        retrieval_query = query
    semantic = vs.as_retriever(search_type="mmr", search_kwargs={"k": TOP_K_RETRIEVAL, "fetch_k": TOP_K_RETRIEVAL*3, "lambda_mult": 0.6}).invoke(retrieval_query)
    if bm25 is not None:
        bm25.k = TOP_K_RETRIEVAL
        candidates = rrf_merge([bm25.invoke(query), semantic])
    else:
        candidates = semantic
    return gemini_rerank(query, candidates, llm)

def serper_search(query, num_results=4):
    if not SERPER_API_KEY: return[]
    try:
        resp = requests.post("https://google.serper.dev/search", headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}, json={"q": query, "num": num_results, "hl": "en"}, timeout=10)
        return[{"title": r.get("title",""), "snippet": r.get("snippet",""), "link": r.get("link","")} for r in resp.json().get("organic",[])[:num_results]]
    except Exception:
        return[]

def format_web_evidence(results):
    if not results: return ""
    lines =[]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r['title']}\n   {r['snippet']}\n   [{r['link']}]\n")
    return "\n".join(lines)

def generate_persona(topic, agent_idx, model):
    llm = get_llm(model)
    return parse_response(llm.invoke(f"Topic: '{topic}'\nCreate a 2-sentence system instruction for Debate Agent {agent_idx+1}. Give them a very specific professional background, a strong bias, and a distinct rhetorical style."))

def run_agent_turn(agent_idx, agent_data, topic, query, rag_context, web_evidence, debate_history, round_num):
    llm = get_llm(agent_data["model"])
    
    context_block = ""
    if rag_context: context_block += f"\nINTERNAL KNOWLEDGE BASE:\n{rag_context}\n"
    if web_evidence: context_block += f"\nLIVE WEB EVIDENCE:\n{format_web_evidence(web_evidence)}\n"
    
    no_kb = not context_block.strip()
    citation_rule = "Do NOT fabricate citations. Base arguments on logic and first principles." if no_kb else "Cite empirical claims using[Source: ...] or [Web: URL]."
    
    system = (f"IDENTITY: {agent_data['instruction']}\n\n"
              f"You just researched: '{query}'. Based on this, you found:\n{context_block if context_block else 'No external documents or web evidence provided. Rely on general knowledge.'}\n\n"
              f"DEBATE RULES:\n- {citation_rule}\n- Make 2-3 sharp arguments per turn.\n"
              "- Directly rebut the previous speaker's weakest claim if this isn't the opening.\n"
              "- Be intellectually aggressive but factually rigorous.\n- End with a memorable closing line.\n- 200-350 words. Clear paragraphs, no bullet lists.")
    
    if not debate_history:
        user_msg = f"Round {round_num} — Opening statement. Topic: '{topic}'"
    else:
        user_msg = f"Round {round_num} — Debate so far:\n\n{chr(10).join(debate_history[-6:])}\n\nMake your strongest argument on: '{topic}'"
        
    return parse_response(llm.invoke([SystemMessage(content=system), HumanMessage(content=user_msg)]))

def run_judge(topic, debate_history, judge_model):
    llm = get_llm(judge_model)
    return parse_response(llm.invoke([
        SystemMessage(content="You are the Final Synthesizer — a neutral, wise adjudicator. Evidence-based, fair, intellectually honest. Evaluate the debate purely on the transcripts and citations provided."),
        HumanMessage(content=f"TOPIC: '{topic}'\n\nFULL TRANSCRIPT:\n{chr(10).join(debate_history)}\n\n"
                     "Deliver your FINAL VERDICT with:\n1. STRONGEST ARGUMENTS (per side)\n2. WEAKEST ARGUMENTS\n3. EVIDENCE QUALITY\n4. SYNTHESIS\n5. WHAT REMAINS CONTESTED\n\nBold, precise. 400-600 words.")
    ]))


AGENT_COLORS =["#6366F1", "#38BDF8", "#A855F7", "#10B981", "#F43F5E"]
AGENT_NAMES  =["AGENT I", "AGENT II", "AGENT III", "AGENT IV", "AGENT V"]

def _esc(s): return s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace("\n","<br>")

def render_argument(speaker, model_name, text, color, query=None):
    query_html = f'<div class="arg-card-query">🔍 Research query: <i>{_esc(query)}</i></div>' if query else ""
    st.markdown(f"""<div class="arg-card">
        <div class="arg-card-accent" style="background:{color}"></div>
        <div class="arg-card-header">
            <span class="arg-card-speaker" style="color:{color}">{speaker}</span>
            <span class="arg-card-model">{model_name}</span>
        </div>
        <div class="arg-card-body">{_esc(text)}</div>
        {query_html}
    </div>""", unsafe_allow_html=True)

def render_verdict(text, model_name):
    st.markdown(f"""<div class="verdict-card">
        <div class="verdict-header">
            <span class="verdict-title">⚖ Final Verdict</span>
            <span class="verdict-model">{model_name}</span>
        </div>
        <div class="verdict-body">{_esc(text)}</div>
    </div>""", unsafe_allow_html=True)

def render_round_divider(r, total):
    st.markdown(f"""<div class="round-divider">
        <div class="round-divider-line"></div>
        <div class="round-divider-label">Round {r} of {total}</div>
        <div class="round-divider-line"></div>
    </div>""", unsafe_allow_html=True)

def render_section_header(num, title, desc=""):
    desc_html = f"<div class='section-hd-desc'>{desc}</div>" if desc else ""
    st.markdown(f"""<div class="section-hd">
        <div class="section-hd-num">{num}</div>
        <div><div class="section-hd-title">{title}</div>{desc_html}</div>
    </div>""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown('<div class="sb-logo">⚖ DebateLM</div><div class="sb-logo-sub">Intelligence Debate System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-section">', unsafe_allow_html=True)
    st.button("+ New Debate", type="primary", use_container_width=True, on_click=lambda: st.session_state.update(current_view="new"))
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-section"><div class="sb-label">Configuration</div>', unsafe_allow_html=True)
    num_agents  = st.slider("Debaters", 2, 5, 2)
    num_rounds  = st.slider("Rounds", 1, 5, 2)
    judge_model = st.selectbox("Judge Model", AVAILABLE_MODELS, index=0)
    st.markdown('</div>', unsafe_allow_html=True)

    doc_count = 0
    if st.session_state.vectorstore:
        try: doc_count = st.session_state.vectorstore._collection.count()
        except Exception: pass
    st.markdown('<div class="sb-section"><div class="sb-label">Knowledge Base</div>', unsafe_allow_html=True)
    if st.session_state.vectorstore:
        st.markdown(f'<div style="font-family:\'Inter\',sans-serif;font-size:0.75rem;font-weight:600;color:#059669;margin-bottom:0.7rem;">● {doc_count} chunks indexed</div>', unsafe_allow_html=True)
        with st.expander("Add documents"):
            more_files = st.file_uploader("PDFs / TXT", type=["pdf","txt"], accept_multiple_files=True, key="append_files", label_visibility="collapsed")
            if st.button("Add to KB", use_container_width=True):
                if more_files:
                    with st.spinner("Indexing…"):
                        st.session_state.vectorstore = process_documents(more_files, is_append=True)
                    st.rerun()
        if st.button("Clear KB", use_container_width=True):
            # 1. Safely delete the Chroma collection data first
            if st.session_state.vectorstore is not None:
                try:
                    st.session_state.vectorstore.delete_collection()
                except Exception:
                    pass
            
            # 2. Remove references so the connection can close
            st.session_state.vectorstore = None
            st.session_state.bm25_retriever = None
            
            # 3. Force garbage collection to release Windows file locks
            import gc
            gc.collect()
            
            # 4. Safely delete the directory (ignoring locked file errors)
            if os.path.exists(PERSIST_DIR): 
                shutil.rmtree(PERSIST_DIR, ignore_errors=True)
                
            # 5. Reset document store and rerun
            st.session_state.docstore = {}
            save_docstore({})
            st.rerun()
    else:
        uploaded_files = st.file_uploader("PDFs / TXT", type=["pdf","txt"], accept_multiple_files=True, label_visibility="collapsed")
        if st.button("Build Knowledge Base", use_container_width=True, type="primary"):
            if uploaded_files:
                with st.spinner("Processing…"):
                    st.session_state.vectorstore = process_documents(uploaded_files)
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    debates_used = len(st.session_state.past_debates)
    pct = int(min(debates_used / MAX_DEBATES, 1.0) * 100)
    g_on, s_on = bool(GEMINI_API_KEY), bool(SERPER_API_KEY)
    st.markdown(f"""
    <div class="quota-row">
        <div class="quota-label"><span>Session Usage</span><span>{debates_used}/{MAX_DEBATES}</span></div>
        <div class="quota-track"><div class="quota-fill" style="width:{pct}%"></div></div>
    </div>
    <div class="api-status">
        <div class="api-pill {'on' if g_on else 'off'}"><div class="api-dot"></div>GEMINI</div>
        <div class="api-pill {'on' if s_on else 'off'}"><div class="api-dot"></div>SERPER</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-section"><div class="sb-label">Archive</div>', unsafe_allow_html=True)
    if not st.session_state.past_debates:
        st.markdown('<div style="font-family:\'Inter\',sans-serif;font-weight:500;font-size:0.75rem;color:var(--text-color);opacity:0.7;padding:0.3rem 0;">No debates recorded.</div>', unsafe_allow_html=True)
    for rec in st.session_state.past_debates:
        short = rec["topic"][:34] + "…" if len(rec["topic"]) > 34 else rec["topic"]
        if st.button(short, key=f"h_{rec['id']}", use_container_width=True):
            st.session_state.current_view = "history"; st.session_state.selected_history = rec; st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


kb_status = f"{doc_count} chunks" if doc_count else "No KB"
web_status = "Live" if bool(SERPER_API_KEY) else "Offline"
st.markdown(f"""
<div class="masthead">
    <div>
        <div class="masthead-wordmark">DebateLM</div>
        <div class="masthead-tagline">Multi-Agent Intelligence Debate · Agent-Specific RAG</div>
    </div>
    <div class="masthead-right">
        <div>
            <div class="masthead-stat-val">{debates_used}</div>
            <div class="masthead-stat-lbl">Debates Run</div>
        </div>
        <div>
            <div class="masthead-stat-val">{kb_status}</div>
            <div class="masthead-stat-lbl">Knowledge Base</div>
        </div>
        <div>
            <div class="masthead-stat-val">{web_status}</div>
            <div class="masthead-stat-lbl">Web Search</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


if st.session_state.current_view == "new":
    can_debate = debates_used < MAX_DEBATES and bool(GEMINI_API_KEY)

    render_section_header("01", "CONFIGURE DEBATERS", "Assign models and personas to each agent")
    agents_config =[]
    cols = st.columns(min(num_agents, 3), gap="small")
    for i in range(num_agents):
        col = cols[i % min(num_agents, 3)]
        with col:
            color = AGENT_COLORS[i]
            st.markdown(f'<div class="agent-card" style="border-left:4px solid {color}"><div class="agent-id" style="color:{color}">{AGENT_NAMES[i]}</div></div>', unsafe_allow_html=True)
            model_sel = st.selectbox("Model", AVAILABLE_MODELS, key=f"model_{i}", index=0, label_visibility="collapsed")
            mode = st.radio("Persona",["AI Generated", "Manual"], key=f"mode_{i}", horizontal=True)
            if mode == "Manual":
                if f"inst_{i}" not in st.session_state:
                    st.session_state[f"inst_{i}"] = "You are a rigorous empiricist. Demand evidence for every claim."
                instruction = st.text_area("System prompt", key=f"inst_{i}", height=72, on_change=auto_save, label_visibility="collapsed", placeholder="Write this agent's persona and bias…")
            else:
                instruction = "__AI_GENERATED__"
                st.caption("Auto-generated from topic")
            agents_config.append({"id": i, "mode": mode, "instruction": instruction, "model": model_sel})

    st.markdown("<br>", unsafe_allow_html=True)
    render_section_header("02", "THE MOTION", "State the proposition to be debated")
    if "debate_topic" not in st.session_state: st.session_state.debate_topic = ""
    topic = st.text_area("Motion", key="debate_topic",
        placeholder="State the motion — e.g. 'This house believes that AGI will be net-positive for humanity within 20 years.'",
        height=100, on_change=auto_save, label_visibility="collapsed")

    use_web = st.toggle("Enable live web search grounding via Serper", value=bool(SERPER_API_KEY), disabled=not SERPER_API_KEY)
    if not GEMINI_API_KEY: st.error("GEMINI_API_KEY not found.")
    if debates_used >= MAX_DEBATES: st.warning(f"Session quota reached ({MAX_DEBATES}).")

    st.markdown("<br>", unsafe_allow_html=True)
    launch = st.button("Launch Debate", type="primary", disabled=not can_debate)

    if launch:
        if not topic.strip(): st.error("Please enter a topic."); st.stop()

        with st.spinner("Generating agent personas..."):
            active_agents =[]
            for ag in agents_config:
                instr = generate_persona(topic, ag["id"], ag["model"]) if ag["mode"] == "AI Generated" else ag["instruction"]
                active_agents.append({"instruction": instr, "model": ag["model"]})

        debate_history: list[str] =[]
        agent_research_logs =[]

        for r in range(num_rounds):
            render_round_divider(r + 1, num_rounds)
            for i, ag in enumerate(active_agents):
                agent_llm = get_llm(ag["model"])
                with st.spinner(f"{AGENT_NAMES[i]} researching & formulating..."):
                    search_query = topic
                    
                    # Knowledge Base is retrieved in every round
                    rag_docs = hybrid_retrieve(search_query, agent_llm) if st.session_state.vectorstore else[]
                    rag_context = _get_parent_context(rag_docs)
                    
                    # Web search ONLY in the beginning (Round 1)
                    if r == 0:
                        web_results = serper_search(search_query, 4) if use_web else[]
                    else:
                        web_results =[]
                    
                    # Agent formulates argument based on their personal search
                    argument = run_agent_turn(i, ag, topic, search_query, rag_context, web_results, debate_history, r + 1)
                    
                    # Store logs for transparency
                    agent_research_logs.append({
                        "round": r + 1,
                        "agent": AGENT_NAMES[i],
                        "query": search_query,
                        "web": web_results,
                        "rag_found": bool(rag_context)
                    })
                    
                render_argument(AGENT_NAMES[i], ag["model"], argument, AGENT_COLORS[i % len(AGENT_COLORS)], search_query)
                debate_history.append(f"{AGENT_NAMES[i]}: {argument}")

        st.markdown('<div class="round-divider"><div class="round-divider-line"></div><div class="round-divider-label">Deliberation</div><div class="round-divider-line"></div></div>', unsafe_allow_html=True)
        with st.spinner(f"Final Synthesizer ({judge_model}) deliberating..."):
            verdict = run_judge(topic, debate_history, judge_model)
        render_verdict(verdict, judge_model)

        new_rec = save_new_debate(topic, debate_history, verdict, agent_research_logs, judge_model)
        st.session_state.current_view = "history"; st.session_state.selected_history = new_rec; st.rerun()


elif st.session_state.current_view == "history":
    past = st.session_state.selected_history
    if not past: st.session_state.current_view = "new"; st.rerun()

    if st.button("← Return to Setup"):
        st.session_state.current_view = "new"; st.rerun()

    st.markdown(f"""
    <div class="record-meta">
        <div class="record-date">{past['date']}</div>
        <div class="record-date">{past.get('judge_model','')}</div>
    </div>
    <div class="record-topic">{past['topic']}</div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Transcript", "Research Logs", "Follow-up"])

    with tab1:
        # Determine if we have stored queries inside our legacy or current history
        logs = past.get("research_logs",[])
        log_idx = 0
        for msg in past["history"]:
            if ": " not in msg: continue
            speaker, text = msg.split(": ", 1)
            roman = {"I": 0, "II": 1, "III": 2, "IV": 3, "V": 4}
            m = re.search(r"(V?I{0,3}|I{1,3}V?)$", speaker.strip())
            idx = roman.get(m.group().strip(), 0) if m else 0
            
            # Match logs to speaker
            query = None
            if log_idx < len(logs) and logs[log_idx].get("agent") == speaker:
                query = logs[log_idx].get("query")
                log_idx += 1
                
            render_argument(speaker, "", text, AGENT_COLORS[idx % len(AGENT_COLORS)], query)
        st.markdown('<div class="round-divider"><div class="round-divider-line"></div><div class="round-divider-label">Final Verdict</div><div class="round-divider-line"></div></div>', unsafe_allow_html=True)
        render_verdict(past["verdict"], past.get("judge_model",""))

    with tab2:
        logs = past.get("research_logs",[])
        if not logs:
            if "research_brief" in past and past["research_brief"]:
                st.caption("Legacy debate: Displaying centralized Research Brief.")
                st.markdown(f'<div class="brief-card">{past["research_brief"]}</div>', unsafe_allow_html=True)
            else:
                st.caption("No research logs recorded for this debate.")
        else:
            for log in logs:
                st.markdown(f"**{log['agent']}** (Round {log['round']})")
                st.markdown(f"`🔍 {log['query']}`")
                
                has_sources = False
                if log.get("rag_found"):
                    st.markdown("- 📄 *Found relevant context in Knowledge Base*")
                    has_sources = True
                
                web_sources = log.get("web",[])
                if web_sources:
                    has_sources = True
                    for w in web_sources:
                        st.markdown(f"- 🌐 [{w['title']}]({w['link']})")
                        
                if not has_sources:
                    st.caption("No external sources retrieved for this query.")
                    
                st.divider()

    with tab3:
        st.caption("The Synthesizer retains full memory of this debate.")
        if "post_chat" not in past: past["post_chat"] =[]
        for cm in past["post_chat"]:
            with st.chat_message(cm["role"]): st.markdown(cm["content"])

        if q := st.chat_input("Ask a follow-up question…"):
            past["post_chat"].append({"role": "user", "content": q})
            with st.chat_message("user"): st.markdown(q)
            chat_llm = get_llm(past.get("judge_model", DEFAULT_MODEL))
            msgs =[SystemMessage(content=f"You are the Final Synthesizer. Debate: '{past['topic']}'.\nVERDICT:\n{past['verdict']}\nAnswer precisely. Cite arguments used by the agents.")]
            for cm in past["post_chat"]:
                msgs.append(HumanMessage(content=cm["content"]) if cm["role"]=="user" else AIMessage(content=cm["content"]))
            with st.chat_message("assistant"):
                with st.spinner("Synthesizer deliberating…"):
                    ans = parse_response(chat_llm.invoke(msgs))
                    st.markdown(ans)
            past["post_chat"].append({"role": "assistant", "content": ans})
            for i, rec in enumerate(st.session_state.past_debates):
                if rec["id"] == past["id"]:
                    st.session_state.past_debates[i] = past; break
            with open(HISTORY_FILE, "w") as f:
                json.dump(st.session_state.past_debates, f, indent=2)