import streamlit as st
import os, json, datetime, uuid, hashlib, time, re, tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from supabase import create_client, Client
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import requests

load_dotenv()

# --- CLOUD SECRETS & API KEYS ---
def get_secret(key):
    return st.secrets[key] if key in st.secrets else os.getenv(key, "")

GEMINI_API_KEY      = get_secret("GEMINI_API_KEY")
SERPER_API_KEY      = get_secret("SERPER_API_KEY")
SUPABASE_URL        = get_secret("SUPABASE_URL")
SUPABASE_KEY        = get_secret("SUPABASE_KEY")
PINECONE_API_KEY    = get_secret("PINECONE_API_KEY")
PINECONE_INDEX_NAME = get_secret("PINECONE_INDEX_NAME")

AVAILABLE_MODELS = [
    "gemini-3-flash-preview",
    "gemini-3.1-pro-preview",
    "gemini-3.1-flash-lite-preview",
]
DEFAULT_MODEL        = AVAILABLE_MODELS[0]
MAX_DEBATES          = 5
TOP_K_RETRIEVAL      = 15
TOP_K_FINAL          = 8

st.set_page_config(page_title="DebateLM", page_icon="⚖", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
*, *::before, *::after { box-sizing: border-box; }
footer, .stDeployButton { display: none !important; }
.block-container { padding: 3rem 2rem 4rem 2rem !important; max-width: 1400px !important; }
.masthead { display:flex; align-items:flex-end; justify-content:space-between; padding:2.2rem 0 1.4rem 0; border-bottom:1px solid var(--secondary-background-color); margin-bottom:2rem; }
.masthead-wordmark { font-family:'Inter', sans-serif; font-size:2.4rem; font-weight:800; color:var(--text-color); letter-spacing:-0.03em; line-height:1; }
.masthead-tagline { font-family:'Inter', sans-serif; font-size:0.85rem; color:var(--text-color); opacity:0.7; font-weight: 500; margin-top: 0.5rem; }
.section-hd { display:flex; align-items:center; gap:1rem; margin:0.5rem 0 1.2rem 0; }
.section-hd-num { font-family:'Inter', sans-serif; font-size:2rem; font-weight:800; color:var(--secondary-background-color); line-height:1; min-width:2rem; filter: brightness(0.8); }
.section-hd-title { font-family:'Inter', sans-serif; font-size:0.85rem; font-weight:700; letter-spacing:0.05em; text-transform:uppercase; color:var(--text-color); }
.agent-card { border:1px solid var(--secondary-background-color); background:var(--secondary-background-color); padding:0.8rem 1rem; border-radius: 4px; position:relative; margin-bottom:0.5rem; }
.agent-card::before { content:''; position:absolute; top:0; left:0; width:4px; border-radius: 4px 0 0 4px; height:100%; }
.agent-id { font-family:'Inter', sans-serif; font-size:0.75rem; font-weight: 700; letter-spacing:0.05em; text-transform:uppercase; }
.round-divider { display:flex; align-items:center; gap:1.2rem; margin:2.5rem 0 1.5rem 0; }
.round-divider-line { flex:1; height:1px; background:var(--secondary-background-color); }
.round-divider-label { font-family:'Inter', sans-serif; font-size:0.75rem; font-weight: 600; color:var(--text-color); opacity:0.7; text-transform:uppercase; letter-spacing:0.1em; white-space:nowrap; text-align: center; }
.sb-logo { font-family:'Inter', sans-serif; font-size:1.2rem; font-weight:800; color:var(--text-color); letter-spacing:-0.02em; padding:1.4rem 1.2rem 0.2rem 1.2rem; border-bottom:none; }
.sb-logo-sub { font-family:'Inter', sans-serif; font-size:0.7rem; font-weight: 500; color:var(--text-color); opacity:0.7; padding:0.2rem 1.2rem 0.8rem 1.2rem; border-bottom:1px solid var(--secondary-background-color); }
.sb-section { padding:1rem 1.2rem; border-bottom:1px solid var(--secondary-background-color); }
.sb-label { font-family:'Inter', sans-serif; font-size:0.75rem; font-weight: 700; text-transform:uppercase; color:var(--text-color); opacity:0.7; margin-bottom:0.8rem; letter-spacing:0.02em; }
.quota-row { padding:0.8rem 1.2rem; border-bottom:1px solid var(--secondary-background-color); }
.quota-label { display:flex; justify-content:space-between; font-family:'Inter', sans-serif; font-size:0.75rem; font-weight: 600; color:var(--text-color); opacity:0.7; margin-bottom:0.5rem; }
.quota-track { height:4px; background:var(--secondary-background-color); filter: brightness(0.9); border-radius: 2px; overflow: hidden; }
.quota-fill { height:100%; background:var(--primary-color); }
.record-meta { display:flex; gap:1rem; align-items:baseline; margin-bottom:0.8rem; padding-bottom:1rem; border-bottom:1px solid var(--secondary-background-color); }
.record-date { font-family:'Inter', sans-serif; font-size:0.8rem; font-weight: 500; color:var(--text-color); opacity:0.7; }
</style>
""", unsafe_allow_html=True)

# --- 1. GUEST ID SYSTEM ---
def get_guest_id() -> str:
    if "uid" in st.query_params:
        uid = st.query_params["uid"]
        st.session_state.guest_id = uid
        return uid
    if "guest_id" in st.session_state:
        return st.session_state.guest_id
    uid = "guest_" + str(uuid.uuid4()).replace("-", "")[:16]
    st.session_state.guest_id = uid
    return uid

guest_id = get_guest_id()

# --- 2. SUPABASE (WITH CLOUD CACHE) ---
@st.cache_resource
def init_supabase() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase = init_supabase()

def is_valid_supabase_guest(gid: str) -> bool:
    if not gid: return False
    return bool(re.match(r"^guest_[a-z0-9]{20,}$", gid))

# 🔥 NEW: Merged function to get guest quota AND preferences efficiently
def get_guest_session_data(gid):
    if not is_valid_supabase_guest(gid): 
        return {"debates_run": 0, "preferences": {}}
    try:
        res = supabase.table("guest_sessions").select("debates_run, preferences").eq("guest_id", gid).execute()
        if res.data:
            return {
                "debates_run": res.data[0].get("debates_run", 0),
                "preferences": res.data[0].get("preferences") or {}
            }
        else:
            supabase.table("guest_sessions").upsert({"guest_id": gid, "preferences": {}}).execute()
            return {"debates_run": 0, "preferences": {}}
    except Exception:
        return {"debates_run": 0, "preferences": {}}

# 🔥 NEW: Save preferences instantly to cloud
def save_preferences(gid, prefs_dict):
    if is_valid_supabase_guest(gid):
        try:
            supabase.table("guest_sessions").update({"preferences": prefs_dict}).eq("guest_id", gid).execute()
        except Exception:
            pass

def increment_quota(gid):
    if is_valid_supabase_guest(gid):
        try:
            current = get_guest_session_data(gid)["debates_run"]
            supabase.table("guest_sessions").update({"debates_run": current + 1, "last_active": "now()"}).eq("guest_id", gid).execute()
        except Exception:
            pass

def load_past_debates(gid):
    if not is_valid_supabase_guest(gid): return []
    try:
        res = supabase.table("debates").select("*").eq("guest_id", gid).order("created_at", desc=True).execute()
        debates = []
        for row in res.data:
            d = row["history"]
            d["id"] = row["debate_id"]
            d["verdict"] = row.get("verdict", d.get("verdict", "")) 
            debates.append(d)
        return debates
    except Exception:
        return []

# 🔥 FIX: Removed the duplicate increment_quota call
def save_new_debate(topic, history, verdict, research_logs, judge_model, gid):
    debate_id = str(uuid.uuid4())
    record = {
        "id": debate_id, "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "topic": topic, "research_logs": research_logs, "history": history, 
        "verdict": verdict, "post_chat": [], "judge_model": judge_model
    }
    if is_valid_supabase_guest(gid):
        try:
            supabase.table("debates").insert({
                "debate_id": debate_id, "guest_id": gid, "topic": topic,
                "verdict": verdict, "history": record
            }).execute()
            increment_quota(gid) # ✅ Only called ONCE here
        except Exception as e: 
            print("DB Error:", e)
            
    st.session_state.past_debates.insert(0, record)
    return record

def update_debate_chat(debate_id, record):
    if is_valid_supabase_guest(guest_id):
        try:
            supabase.table("debates").update({"history": record, "verdict": record.get("verdict", "")}).eq("debate_id", debate_id).execute()
        except Exception: pass

# --- INITIALIZE STATE ---
guest_data = get_guest_session_data(guest_id)
if "user_prefs" not in st.session_state:
    st.session_state.user_prefs = guest_data["preferences"]
debates_used = guest_data["debates_run"]

if "past_debates" not in st.session_state: st.session_state.past_debates = load_past_debates(guest_id)
if "current_view" not in st.session_state: st.session_state.current_view = "new"
if "selected_history" not in st.session_state: st.session_state.selected_history = None
if "uploaded_file_names" not in st.session_state: st.session_state.uploaded_file_names = []

# Helper to safely load selected model indexes from Cache
def get_model_idx(model_name):
    return AVAILABLE_MODELS.index(model_name) if model_name in AVAILABLE_MODELS else 0

# --- 3. PINECONE INTEGRATION ---
@st.cache_resource
def init_pinecone(): return Pinecone(api_key=PINECONE_API_KEY)
pc = init_pinecone()

def get_vectorstore():
    emb = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=GEMINI_API_KEY, output_dimensionality=768)
    return PineconeVectorStore(index_name=PINECONE_INDEX_NAME, embedding=emb, pinecone_api_key=PINECONE_API_KEY)

def process_documents(uploaded_files):
    if not uploaded_files: return 0
    p_split = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=400)
    c_split = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    progress_bar = st.progress(0.0, text="Reading files...")
    raw_docs = []
    for uf in uploaded_files:
        suffix = ".pdf" if uf.name.lower().endswith(".pdf") else ".txt"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uf.getbuffer())
            tmp_path = tmp_file.name
        
        try:
            file_docs = PyMuPDFLoader(tmp_path).load() if suffix == ".pdf" else TextLoader(tmp_path).load()
            for d in file_docs: d.metadata["file_name"] = uf.name
            raw_docs.extend(file_docs)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            
    parent_docs = p_split.split_documents(raw_docs)
    child_docs = []
    for parent in parent_docs:
        children = c_split.split_documents([parent])
        for child in children:
            child.metadata.update({"guest_id": guest_id, "file_name": parent.metadata.get("file_name", "unknown"), "parent_text": parent.page_content, "source": parent.metadata.get("source", "unknown")})
            if "page" in child.metadata: del child.metadata["page"]
        child_docs.extend(children)
        
    total_chunks = len(child_docs)
    if total_chunks == 0:
        progress_bar.empty()
        return 0
        
    vs = get_vectorstore()
    BATCH_SIZE = 150 
    total_batches = (total_chunks + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in range(0, total_chunks, BATCH_SIZE):
        batch = child_docs[i : i + BATCH_SIZE]
        progress_bar.progress(0.2 + (0.8 * ((i // BATCH_SIZE + 1) / total_batches)), text=f"Uploading batch {(i // BATCH_SIZE) + 1} of {total_batches}...")
        try:
            vs.add_documents(batch)
        except Exception:
            progress_bar.progress(0.2 + (0.8 * ((i // BATCH_SIZE + 1) / total_batches)), text=f"⚠️ Network hiccup. Retrying batch {(i // BATCH_SIZE) + 1} in 2s...")
            time.sleep(2)
            vs.add_documents(batch)
            
    progress_bar.empty()
    return total_chunks

def delete_documents(file_names):
    index = pc.Index(PINECONE_INDEX_NAME)
    for fname in file_names:
        try: index.delete(filter={"file_name": {"$eq": fname}, "guest_id": {"$eq": guest_id}})
        except Exception: pass

# --- CORE LLM LOGIC ---
def parse_response(response):
    c = response.content
    return "\n".join(b.get("text","") for b in c if isinstance(b,dict) and "text" in b) if isinstance(c, list) else str(c)

def get_llm(model: str):
    return ChatGoogleGenerativeAI(model=model, google_api_key=GEMINI_API_KEY, temperature=0.7)

def generate_agent_queries(topic, agents_config, llm):
    try:
        prompt = f"Topic: '{topic}'\nGenerate a short Google search query (under 8 words) for each agent persona to find evidence supporting their specific view.\n\n"
        for i, ag in enumerate(agents_config):
            prompt += f"Agent {i} Persona: {ag['instruction']}\n"
        prompt += "\nReturn ONLY a JSON list of strings, e.g. [\"query1\", \"query2\"]."
        raw = parse_response(llm.invoke([SystemMessage(content="You are a research planner."), HumanMessage(content=prompt)]))
        raw_json = re.search(r'\[.*\]', raw.replace('\n', ''))
        if raw_json:
            queries = json.loads(raw_json.group())
            if len(queries) == len(agents_config): return queries
    except Exception as e: print("Query Gen Error:", e)
    return [topic] * len(agents_config)

def gemini_rerank(query, candidates, llm, persona=None):
    if not candidates: return candidates
    numbered = "\n".join(f"{i+1}. {doc.page_content[:300]}" for i, enumerate in enumerate(candidates))
    rank_prompt = f"Rank by relevance to the query AND this specific persona: '{persona}'. Reply ONLY with comma-separated numbers.\nQuery: {query}\n\n{numbered}" if persona else f"Rank by relevance. Reply ONLY with comma-separated numbers.\nQuery: {query}\n\n{numbered}"
    try:
        raw = parse_response(llm.invoke(rank_prompt)).strip()
        indices =[int(x.strip())-1 for x in raw.split(",") if x.strip().isdigit()]
        reranked = [candidates[i] for i in indices if 0 <= i < len(candidates)]
        seen = set(id(d) for d in reranked)
        for d in candidates:
            if id(d) not in seen: reranked.append(d)
        return reranked[:TOP_K_FINAL]
    except Exception: return candidates[:TOP_K_FINAL]

def retrieve_from_pinecone(query, llm, persona=None):
    try:
        results = get_vectorstore().similarity_search(query, k=TOP_K_RETRIEVAL, filter={"guest_id": guest_id})
        reranked = gemini_rerank(query, results, llm, persona)
        parents, context_blocks = set(), []
        for doc in reranked:
            pt = doc.metadata.get("parent_text", doc.page_content)
            if pt not in parents:
                parents.add(pt)
                context_blocks.append(f"[Source: {doc.metadata.get('source','?')}]\n{pt}")
        return "\n\n---\n\n".join(context_blocks)
    except Exception: return ""

def serper_search(query, num_results=4):
    if not SERPER_API_KEY: return[]
    try:
        resp = requests.post("https://google.serper.dev/search", headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}, json={"q": query, "num": num_results, "hl": "en"}, timeout=10)
        return[{"title": r.get("title",""), "snippet": r.get("snippet",""), "link": r.get("link","")} for r in resp.json().get("organic",[])[:num_results]]
    except Exception: return[]

def format_web_evidence(results):
    return "\n".join(f"{i}. {r['title']}\n   {r['snippet']}\n   [{r['link']}]\n" for i, r in enumerate(results, 1))

def run_agent_turn(agent_idx, agent_data, topic, query, rag_context, web_evidence, debate_history, round_num):
    llm = get_llm(agent_data["model"])
    context_block = ""
    if rag_context: context_block += f"\nINTERNAL KB:\n{rag_context}\n"
    if web_evidence: context_block += f"\nLIVE WEB:\n{format_web_evidence(web_evidence)}\n"
    
    citation_rule = "Do NOT fabricate citations." if not context_block.strip() else "Cite strictly using [Source: Title] or [Web: Domain]."
    system = f"IDENTITY: {agent_data['instruction']}\n\nEvidence:\n{context_block}\n\nRULES: {citation_rule} Make 2-3 arguments. Rebut previous claims. 200-350 words."
    user_msg = f"Round {round_num} — Debate:\n\n{chr(10).join(debate_history[-6:])}\n\nArgue: '{topic}'" if debate_history else f"Round {round_num} — Opening. Topic: '{topic}'"
    return parse_response(llm.invoke([SystemMessage(content=system), HumanMessage(content=user_msg)]))

def run_judge(topic, debate_history, judge_model):
    return parse_response(get_llm(judge_model).invoke([
        SystemMessage(content="Evaluate debate purely on transcripts. Cite using [Source: Title]."),
        HumanMessage(content=f"TOPIC: '{topic}'\n\nTRANSCRIPT:\n{chr(10).join(debate_history)}\n\nDeliver FINAL Verdict with Synthesis. 400-600 words.")
    ]))

AGENT_COLORS =["#6366F1", "#38BDF8", "#A855F7", "#10B981", "#F43F5E"]
AGENT_NAMES  =["AGENT I", "AGENT II", "AGENT III", "AGENT IV", "AGENT V"]

# --- UI FORMATTING ---
def format_professional_citations(text):
    text = re.sub(r'\[Source:\s*([^\]]+)\]', r'<span style="background-color: rgba(128,128,128,0.1); padding: 0.15rem 0.4rem; border-radius: 4px; font-size: 0.75em; font-weight: 600; border: 1px solid rgba(128,128,128,0.2);"><span style="opacity:0.7">📄</span> \1</span>', text, flags=re.IGNORECASE)
    text = re.sub(r'\[Web:\s*([^\]]+)\]', r'<span style="background-color: rgba(56,189,248,0.1); padding: 0.15rem 0.4rem; border-radius: 4px; font-size: 0.75em; font-weight: 600; border: 1px solid rgba(56,189,248,0.3);"><span style="opacity:0.7">🌐</span> \1</span>', text, flags=re.IGNORECASE)
    return text

def render_argument(speaker, model_name, text, color, query=None):
    st.markdown(f'<div style="display:flex; justify-content:space-between; margin-top:1.5rem; border-bottom: 2px solid {color}; margin-bottom: 1rem;"><span style="font-weight: 700; color:{color}">{speaker}</span><span style="font-size:0.75rem; opacity:0.7;">{model_name}</span></div>', unsafe_allow_html=True)
    st.markdown(format_professional_citations(text), unsafe_allow_html=True)
    if query: st.caption(f"🔍 **Research query:** {query}")

def render_verdict(text, model_name):
    st.markdown(f'<div style="display:flex; justify-content:space-between; margin-top:2.5rem; border-bottom: 2px solid var(--primary-color); margin-bottom: 1rem;"><span style="font-weight: 700; color:var(--primary-color);">⚖ Final Verdict</span><span style="font-size:0.75rem; color:var(--primary-color); opacity: 0.8;">{model_name}</span></div>', unsafe_allow_html=True)
    st.markdown(format_professional_citations(text), unsafe_allow_html=True)

def render_round_divider(r, total):
    st.markdown(f'<div class="round-divider"><div class="round-divider-line"></div><div class="round-divider-label">Round {r} of {total}</div><div class="round-divider-line"></div></div>', unsafe_allow_html=True)

# --- SIDEBAR & QUOTA UI ---
prefs = st.session_state.user_prefs 

with st.sidebar:
    st.markdown('<div class="sb-logo">⚖ DebateLM</div><div class="sb-logo-sub">The Open-Source NotebookLM Alternative</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-section">', unsafe_allow_html=True)
    st.button("+ New Debate", type="primary", use_container_width=True, on_click=lambda: st.session_state.update(current_view="new"))
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-section"><div class="sb-label">Configuration</div>', unsafe_allow_html=True)
    # 🔥 Bind values to cached preferences
    num_agents  = st.slider("Debaters", 2, 5, prefs.get("num_agents", 2))
    num_rounds  = st.slider("Rounds", 1, 5, prefs.get("num_rounds", 2))
    judge_model = st.selectbox("Judge Model", AVAILABLE_MODELS, index=get_model_idx(prefs.get("judge_model", DEFAULT_MODEL)))
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-section"><div class="sb-label">Knowledge Base</div>', unsafe_allow_html=True)
    more_files = st.file_uploader("Upload Documents (Auto-Syncs with Database)", type=["pdf","txt"], accept_multiple_files=True)
    
    current_file_names = list({f.name: f for f in (more_files or [])}.keys())
    added_names = [n for n in current_file_names if n not in st.session_state.uploaded_file_names]
    removed_names = [n for n in st.session_state.uploaded_file_names if n not in current_file_names]
    
    if added_names:
        process_documents([f for f in more_files if f.name in added_names])
        st.toast(f"✅ Auto-synced {len(added_names)} file(s)!")
    if removed_names:
        delete_documents(removed_names)
        st.toast(f"🗑️ Removed {len(removed_names)} file(s)!")
        
    st.session_state.uploaded_file_names = current_file_names
    st.markdown('</div>', unsafe_allow_html=True)

    pct = int(min(debates_used / MAX_DEBATES, 1.0) * 100)
    st.markdown(f'<div class="quota-row"><div class="quota-label"><span>Free Demo Quota</span><span>{debates_used}/{MAX_DEBATES}</span></div><div class="quota-track"><div class="quota-fill" style="width:{pct}%; background:{"#E11D48" if debates_used >= MAX_DEBATES else "var(--primary-color)"}"></div></div></div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-section"><div class="sb-label">History (Cloud Saved)</div>', unsafe_allow_html=True)
    if not st.session_state.past_debates: st.markdown('<div style="font-size:0.75rem; opacity:0.7;">No debates recorded.</div>', unsafe_allow_html=True)
    for rec in st.session_state.past_debates:
        if st.button(rec["topic"][:34] + "…" if len(rec["topic"])>34 else rec["topic"], key=f"h_{rec['id']}", use_container_width=True):
            st.session_state.current_view = "history"; st.session_state.selected_history = dict(rec); st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# --- MAIN APP UI ---
st.markdown('<div class="masthead"><div><div class="masthead-wordmark">DebateLM</div><div class="masthead-tagline">Cloud-Native Intelligence Debate · Isolated Guest Pass</div></div></div>', unsafe_allow_html=True)

if st.session_state.current_view == "new":
    can_debate = debates_used < MAX_DEBATES and bool(GEMINI_API_KEY)

    st.markdown('<div class="section-hd"><div class="section-hd-num">01</div><div><div class="section-hd-title">CONFIGURE DEBATERS</div></div></div>', unsafe_allow_html=True)
    agents_config = []
    saved_agents = prefs.get("agents", []) 
    
    cols = st.columns(min(num_agents, 3), gap="small")
    for i in range(num_agents):
        # 🔥 Safely extract saved agent state or use empty dictionary fallback
        sa = saved_agents[i] if i < len(saved_agents) else {}
        
        with cols[i % min(num_agents, 3)]:
            st.markdown(f'<div class="agent-card" style="border-left:4px solid {AGENT_COLORS[i]}"><div class="agent-id" style="color:{AGENT_COLORS[i]}">{AGENT_NAMES[i]}</div></div>', unsafe_allow_html=True)
            model_sel = st.selectbox("Model", AVAILABLE_MODELS, key=f"model_{i}", index=get_model_idx(sa.get("model", DEFAULT_MODEL)), label_visibility="collapsed")
            instr = st.text_area("Persona", value=sa.get("instruction", ""), key=f"inst_{i}", height=72, label_visibility="collapsed", placeholder="Define bias...")
            agents_config.append({"id": i, "instruction": instr if instr else "Be highly analytical and critical.", "model": model_sel})

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-hd"><div class="section-hd-num">02</div><div><div class="section-hd-title">THE MOTION</div></div></div>', unsafe_allow_html=True)
    
    # 🔥 Bind value to cached topic
    topic = st.text_area("Motion", value=prefs.get("topic", ""), placeholder="State the motion to be debated...", height=100, label_visibility="collapsed")
    use_web = st.toggle("Enable live web search grounding via Serper", value=prefs.get("use_web", bool(SERPER_API_KEY)), disabled=not SERPER_API_KEY)
    
    if debates_used >= MAX_DEBATES: st.error(f"🛑 You have reached the {MAX_DEBATES} debate limit.")
        
    button_placeholder = st.empty()
    launch = button_placeholder.button("Launch Debate", type="primary", disabled=not can_debate, use_container_width=True)

    if launch and topic.strip():
        # 🔥 INSTANT CACHE: Save user config immediately to the Database
        new_prefs = {
            "num_agents": num_agents, "num_rounds": num_rounds, "judge_model": judge_model,
            "topic": topic, "use_web": use_web, "agents": agents_config
        }
        st.session_state.user_prefs = new_prefs
        save_preferences(guest_id, new_prefs)
        
        button_placeholder.button("🛑 Stop Ongoing Debate (Saves Progress)", type="primary", use_container_width=True, key="stop_btn")

        debate_history, agent_research_logs = [],[]
        current_debate = save_new_debate(topic, debate_history, "⚠️ *Debate interrupted. No synthesis.*", agent_research_logs, judge_model, guest_id)

        # 🔥 QUALITY FIX: Generate tailored search queries based on agent personas
        with st.spinner("Formulating personalized research strategies..."):
            agent_queries = generate_agent_queries(topic, agents_config, get_llm(judge_model))

        for r in range(num_rounds):
            render_round_divider(r + 1, num_rounds)
            
            # 🔥 SPEED & QUALITY FIX: Pre-fetch tailored RAG and Web Context in parallel
            with st.spinner(f"Round {r+1} Intelligence Gathering..."):
                rag_contexts = {}
                web_contexts = {}
                with ThreadPoolExecutor(max_workers=min(num_agents * 2, 10)) as executor:
                    futures_rag = {executor.submit(retrieve_from_pinecone, agent_queries[i], get_llm(ag["model"]), ag["instruction"]): i for i, ag in enumerate(agents_config)}
                    futures_web = {executor.submit(serper_search, agent_queries[i], 3): i for i, ag in enumerate(agents_config)} if use_web and r == 0 else {}
                    
                    for future in as_completed(futures_rag):
                        rag_contexts[futures_rag[future]] = future.result()
                    for future in as_completed(futures_web):
                        web_contexts[futures_web[future]] = future.result()

            for i, ag in enumerate(agents_config):
                agent_llm = get_llm(ag["model"])
                with st.spinner(f"{AGENT_NAMES[i]} formulating argument..."):
                    rag_context = rag_contexts.get(i, "")
                    web_results = web_contexts.get(i, []) if r == 0 else []
                    argument = run_agent_turn(i, ag, topic, agent_queries[i], rag_context, web_results, debate_history, r + 1)
                    agent_research_logs.append({"round": r + 1, "agent": AGENT_NAMES[i], "query": agent_queries[i], "web": web_results, "rag_found": bool(rag_context)})
                    
                render_argument(AGENT_NAMES[i], ag["model"], argument, AGENT_COLORS[i % len(AGENT_COLORS)], agent_queries[i])
                debate_history.append(f"{AGENT_NAMES[i]}: {argument}")
                
                current_debate["history"] = list(debate_history)
                current_debate["research_logs"] = list(agent_research_logs)
                st.session_state.past_debates[0] = dict(current_debate)
                update_debate_chat(current_debate["id"], current_debate)

        # --- DELIBERATION PHASE (FIXED STATE ARCHITECTURE) ---
        st.markdown('<div class="round-divider"><div class="round-divider-line"></div><div class="round-divider-label">Deliberation</div><div class="round-divider-line"></div></div>', unsafe_allow_html=True)
        with st.spinner(f"Final Synthesizer ({judge_model}) deliberating..."):
            final_verdict_text = run_judge(topic, debate_history, judge_model)
            
        render_verdict(final_verdict_text, judge_model)

        current_debate["verdict"] = final_verdict_text
        current_debate["history"] = debate_history
        current_debate["research_logs"] = agent_research_logs
        
        update_debate_chat(current_debate["id"], current_debate)
        st.session_state.past_debates[0] = current_debate
        st.session_state.selected_history = current_debate
        st.session_state.current_view = "history"
        st.rerun()

elif st.session_state.current_view == "history":
    past = st.session_state.selected_history
    if st.button("← Return to Setup"):
        st.session_state.current_view = "new"; st.rerun()

    st.markdown(f'<div class="record-meta"><div class="record-date">{past["date"]}</div></div>', unsafe_allow_html=True)
    st.markdown(f"### Motion:\n{past['topic']}")
    st.markdown("<br>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Transcript", "Research Logs", "Follow-up"])

    with tab1:
        for msg in past.get("history", []):
            if ": " in msg:
                speaker, text = msg.split(": ", 1)
                idx = {"I":0, "II":1, "III":2, "IV":3, "V":4}.get(re.search(r"(V?I{0,3}|I{1,3}V?)$", speaker.strip()).group(), 0) if re.search(r"(V?I{0,3}|I{1,3}V?)$", speaker.strip()) else 0
                render_argument(speaker, "", text, AGENT_COLORS[idx % len(AGENT_COLORS)])
        render_verdict(past.get("verdict", ""), past.get("judge_model",""))

    with tab2:
        for log in past.get("research_logs",[]):
            st.markdown(f"**{log['agent']}** (Round {log['round']})")
            if log.get("rag_found"): st.markdown("- 📄 *Retrieved data from Pinecone Knowledge Base*")
            for w in log.get("web",[]): st.markdown(f"- 🌐 [{w['title']}]({w['link']})")
            st.divider()

    with tab3:
        for cm in past.get("post_chat", []):
            with st.chat_message(cm["role"]): 
                st.markdown(format_professional_citations(cm["content"]), unsafe_allow_html=True)
                
        if q := st.chat_input("Ask a follow-up question…"):
            past.setdefault("post_chat",[]).append({"role": "user", "content": q})
            with st.chat_message("user"): st.markdown(q)
            with st.chat_message("assistant"):
                with st.spinner("Deliberating…"):
                    msgs =[SystemMessage(content=f"Synthesizer. Debate: '{past['topic']}'. VERDICT: {past.get('verdict','')}. Cite sources if needed.")]
                    for cm in past["post_chat"]: msgs.append(HumanMessage(content=cm["content"]) if cm["role"]=="user" else AIMessage(content=cm["content"]))
                    ans = parse_response(get_llm(past.get("judge_model", DEFAULT_MODEL)).invoke(msgs))
                    st.markdown(format_professional_citations(ans), unsafe_allow_html=True)
                    
            past["post_chat"].append({"role": "assistant", "content": ans})
            st.session_state.selected_history = dict(past)
            update_debate_chat(past["id"], past)