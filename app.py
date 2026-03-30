import streamlit as st
import os, json, datetime, uuid, hashlib, time, re
from dotenv import load_dotenv

from supabase import create_client, Client
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.documents import Document
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

AVAILABLE_MODELS =[
    "gemini-3-flash-preview",
    "gemini-3.1-pro-preview",
    "gemini-3.1-flash-lite-preview",
]
DEFAULT_MODEL        = AVAILABLE_MODELS[0]
MAX_DEBATES          = 5
PARENT_CHUNK_SIZE    = 2000
PARENT_CHUNK_OVERLAP = 200
CHILD_CHUNK_SIZE     = 400
CHILD_CHUNK_OVERLAP  = 50
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
.record-meta { display:flex; gap:1rem; align-items:baseline; margin-bottom:0.8rem; padding-bottom:1rem; border-bottom:1px solid var(--secondary-background-color); }
.record-date { font-family:'Inter', sans-serif; font-size:0.8rem; font-weight: 500; color:var(--text-color); opacity:0.7; }
.record-topic { font-family:'Inter', sans-serif; font-size:1.6rem; font-weight:800; color:var(--text-color); line-height:1.3; margin-bottom:1rem; letter-spacing:-0.02em; word-wrap: break-word; }
</style>
""", unsafe_allow_html=True)

# --- 1. THE FRICTIONLESS GUEST PASS SYSTEM (DEVICE FINGERPRINTING) ---
def get_guest_id() -> str:
    # Step 1: Create a permanent Device Fingerprint (Survives page reloads & closed tabs!)
    try:
        # Grab the user's IP address and Browser info from Streamlit's headers
        ip = st.context.headers.get("X-Forwarded-For", "")
        user_agent = st.context.headers.get("User-Agent", "")
        
        if ip and user_agent:
            # Hash them together to create a secure, permanent ID for this device
            fingerprint = f"{ip.split(',')[0].strip()}-{user_agent}"
            guest_hash = "guest_" + hashlib.sha256(fingerprint.encode()).hexdigest()[:16]
            
            # Keep the URL clean but trackable
            st.query_params["uid"] = guest_hash
            st.session_state.guest_id = guest_hash
            return guest_hash
    except Exception:
        pass
        
    # Step 2: Fallback (Used if you are testing locally on your own computer)
    if "guest_id" in st.session_state:
        return st.session_state.guest_id
        
    if "uid" in st.query_params:
        uid = st.query_params["uid"]
    else:
        uid = "guest_" + str(uuid.uuid4()).replace("-", "")[:16]
        st.query_params["uid"] = uid
        
    st.session_state.guest_id = uid
    return uid

guest_id = get_guest_id()

# --- 2. SUPABASE INTEGRATION (Database & Quota) ---
@st.cache_resource
def init_supabase() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase = init_supabase()

def ensure_guest_exists(gid):
    # Upsert the guest session
    supabase.table("guest_sessions").upsert({"guest_id": gid}).execute()

def get_quota(gid):
    res = supabase.table("guest_sessions").select("debates_run").eq("guest_id", gid).execute()
    return res.data[0]["debates_run"] if res.data else 0

def increment_quota(gid):
    current = get_quota(gid)
    supabase.table("guest_sessions").update({"debates_run": current + 1, "last_active": "now()"}).eq("guest_id", gid).execute()

def load_past_debates(gid):
    res = supabase.table("debates").select("*").eq("guest_id", gid).order("created_at", desc=True).execute()
    debates = []
    for row in res.data:
        d = row["history"]
        d["id"] = row["debate_id"]
        debates.append(d)
    return debates

def save_new_debate(topic, history, verdict, research_logs, judge_model, gid):
    debate_id = str(uuid.uuid4())
    record = {
        "id": debate_id, 
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "topic": topic, 
        "research_logs": research_logs,
        "history": history, 
        "verdict": verdict, 
        "post_chat":[], 
        "judge_model": judge_model
    }
    # Save to Supabase
    supabase.table("debates").insert({
        "debate_id": debate_id,
        "guest_id": gid,
        "topic": topic,
        "verdict": verdict,
        "history": record
    }).execute()
    st.session_state.past_debates.insert(0, record)
    increment_quota(gid)
    return record

def update_debate_chat(debate_id, record):
    supabase.table("debates").update({"history": record}).eq("debate_id", debate_id).execute()

# --- RUN AUTO CLEANUP TO SAVE FREE TIER ---
try:
    supabase.rpc("cleanup_old_guests").execute()
except Exception:
    pass

# Initialize session state
ensure_guest_exists(guest_id)
if "past_debates" not in st.session_state: 
    st.session_state.past_debates = load_past_debates(guest_id)
if "current_view" not in st.session_state: 
    st.session_state.current_view = "new"
if "selected_history" not in st.session_state: 
    st.session_state.selected_history = None


# --- 3. PINECONE INTEGRATION (Vector Database) ---
@st.cache_resource
def init_pinecone():
    return Pinecone(api_key=PINECONE_API_KEY)

pc = init_pinecone()

def get_vectorstore():
    # Force 768 dimensions so it never crashes!
    emb = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", 
        google_api_key=GEMINI_API_KEY,
        output_dimensionality=768 
    )
    return PineconeVectorStore(index_name=PINECONE_INDEX_NAME, embedding=emb)

def process_documents(uploaded_files):
    TEMP_DIR = "/tmp/debatelm_docs"
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    p_split = RecursiveCharacterTextSplitter(chunk_size=PARENT_CHUNK_SIZE, chunk_overlap=PARENT_CHUNK_OVERLAP)
    c_split = RecursiveCharacterTextSplitter(chunk_size=CHILD_CHUNK_SIZE, chunk_overlap=CHILD_CHUNK_OVERLAP)
    
    raw_docs =[]
    for uf in uploaded_files:
        path = os.path.join(TEMP_DIR, uf.name)
        with open(path, "wb") as f: f.write(uf.getbuffer())
        raw_docs.extend(PyPDFLoader(path).load() if uf.name.lower().endswith(".pdf") else TextLoader(path).load())
        
    parent_docs = p_split.split_documents(raw_docs)
    child_docs = []
    
    for parent in parent_docs:
        children = c_split.split_documents([parent])
        for child in children:
            # THIS IS THE MAGIC: Tag chunk with Guest ID so users can't see each other's data
            child.metadata["guest_id"] = guest_id
            # Store parent text inside metadata so we don't need a local docstore.json
            child.metadata["parent_text"] = parent.page_content
            child.metadata["source"] = parent.metadata.get("source", "unknown")
        child_docs.extend(children)
        
    vs = get_vectorstore()
    
    # BATCHING: Prevent Google 429 Limit Crash
    BATCH_SIZE = 100
    for i in range(0, len(child_docs), BATCH_SIZE):
        batch = child_docs[i : i + BATCH_SIZE]
        vs.add_documents(batch)
        time.sleep(1) # Slight pause to respect Google's free tier
        
    return len(child_docs)

def parse_response(response) -> str:
    c = response.content
    if isinstance(c, list):
        return "\n".join(b.get("text","") for b in c if isinstance(b,dict) and "text" in b)
    return str(c)

def get_llm(model: str):
    return ChatGoogleGenerativeAI(model=model, google_api_key=GEMINI_API_KEY, temperature=0.7)

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

def retrieve_from_pinecone(query, llm):
    vs = get_vectorstore()
    try:
        # MAGIC: Only fetch documents that belong to this specific guest_id
        results = vs.similarity_search(query, k=TOP_K_RETRIEVAL, filter={"guest_id": guest_id})
        reranked = gemini_rerank(query, results, llm)
        
        # Reconstruct context using the parent_text we saved earlier
        parents = set()
        context_blocks =[]
        for doc in reranked:
            pt = doc.metadata.get("parent_text", doc.page_content)
            if pt not in parents:
                parents.add(pt)
                context_blocks.append(f"[Source: {doc.metadata.get('source','?')}]\n{pt}")
        return "\n\n---\n\n".join(context_blocks)
    except Exception as e:
        return ""

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

def run_agent_turn(agent_idx, agent_data, topic, query, rag_context, web_evidence, debate_history, round_num):
    llm = get_llm(agent_data["model"])
    context_block = ""
    if rag_context: context_block += f"\nINTERNAL KNOWLEDGE BASE:\n{rag_context}\n"
    if web_evidence: context_block += f"\nLIVE WEB EVIDENCE:\n{format_web_evidence(web_evidence)}\n"
    
    no_kb = not context_block.strip()
    citation_rule = "Do NOT fabricate citations. Base arguments on logic and first principles." if no_kb else "Cite empirical claims using[Source: ...] or [Web: URL]."
    
    system = (f"IDENTITY: {agent_data['instruction']}\n\n"
              f"You just researched: '{query}'. Based on this, you found:\n{context_block if context_block else 'No external documents or web evidence provided.'}\n\n"
              f"DEBATE RULES:\n- {citation_rule}\n- Make 2-3 sharp arguments per turn.\n- Rebut previous claims.\n- 200-350 words.")
    
    user_msg = f"Round {round_num} — Debate so far:\n\n{chr(10).join(debate_history[-6:])}\n\nArgue: '{topic}'" if debate_history else f"Round {round_num} — Opening statement. Topic: '{topic}'"
        
    return parse_response(llm.invoke([SystemMessage(content=system), HumanMessage(content=user_msg)]))

def run_judge(topic, debate_history, judge_model):
    llm = get_llm(judge_model)
    return parse_response(llm.invoke([
        SystemMessage(content="You are the Final Synthesizer. Evaluate the debate purely on the transcripts and citations provided."),
        HumanMessage(content=f"TOPIC: '{topic}'\n\nTRANSCRIPT:\n{chr(10).join(debate_history)}\n\nDeliver your FINAL VERDICT with Strongest Arguments, Weakest Arguments, and Synthesis. 400-600 words.")
    ]))


AGENT_COLORS =["#6366F1", "#38BDF8", "#A855F7", "#10B981", "#F43F5E"]
AGENT_NAMES  =["AGENT I", "AGENT II", "AGENT III", "AGENT IV", "AGENT V"]

def _esc(s): return s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace("\n","<br>")

def render_argument(speaker, model_name, text, color, query=None):
    query_html = f'<div class="arg-card-query">🔍 Research query: <i>{_esc(query)}</i></div>' if query else ""
    st.markdown(f"""<div class="arg-card">
        <div class="arg-card-accent" style="background:{color}"></div>
        <div class="arg-card-header"><span class="arg-card-speaker" style="color:{color}">{speaker}</span><span class="arg-card-model">{model_name}</span></div>
        <div class="arg-card-body">{_esc(text)}</div>{query_html}</div>""", unsafe_allow_html=True)

def render_verdict(text, model_name):
    st.markdown(f"""<div class="verdict-card">
        <div class="verdict-header"><span class="verdict-title">⚖ Final Verdict</span><span class="verdict-model">{model_name}</span></div>
        <div class="verdict-body">{_esc(text)}</div></div>""", unsafe_allow_html=True)

def render_round_divider(r, total):
    st.markdown(f'<div class="round-divider"><div class="round-divider-line"></div><div class="round-divider-label">Round {r} of {total}</div><div class="round-divider-line"></div></div>', unsafe_allow_html=True)

# --- SIDEBAR & QUOTA UI ---
with st.sidebar:
    st.markdown('<div class="sb-logo">⚖ DebateLM</div><div class="sb-logo-sub">The Open-Source NotebookLM Alternative</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-section">', unsafe_allow_html=True)
    st.button("+ New Debate", type="primary", use_container_width=True, on_click=lambda: st.session_state.update(current_view="new"))
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-section"><div class="sb-label">Configuration</div>', unsafe_allow_html=True)
    num_agents  = st.slider("Debaters", 2, 5, 2)
    num_rounds  = st.slider("Rounds", 1, 5, 2)
    judge_model = st.selectbox("Judge Model", AVAILABLE_MODELS, index=0)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-section"><div class="sb-label">Knowledge Base</div>', unsafe_allow_html=True)
    with st.expander("Upload Documents to DB"):
        more_files = st.file_uploader("PDFs / TXT", type=["pdf","txt"], accept_multiple_files=True, label_visibility="collapsed")
        if st.button("Add to Cloud DB", use_container_width=True):
            if more_files:
                with st.spinner("Indexing to Pinecone..."):
                    process_documents(more_files)
                st.success("Successfully added to your private Knowledge Base!")
    st.markdown('</div>', unsafe_allow_html=True)

    debates_used = get_quota(guest_id)
    pct = int(min(debates_used / MAX_DEBATES, 1.0) * 100)
    st.markdown(f"""
    <div class="quota-row">
        <div class="quota-label"><span>Free Demo Quota</span><span>{debates_used}/{MAX_DEBATES}</span></div>
        <div class="quota-track"><div class="quota-fill" style="width:{pct}%; background:{'#E11D48' if debates_used >= MAX_DEBATES else 'var(--primary-color)'}"></div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-section"><div class="sb-label">History (Cloud Saved)</div>', unsafe_allow_html=True)
    if not st.session_state.past_debates:
        st.markdown('<div style="font-family:\'Inter\',sans-serif;font-weight:500;font-size:0.75rem;color:var(--text-color);opacity:0.7;padding:0.3rem 0;">No debates recorded.</div>', unsafe_allow_html=True)
    for rec in st.session_state.past_debates:
        short = rec["topic"][:34] + "…" if len(rec["topic"]) > 34 else rec["topic"]
        if st.button(short, key=f"h_{rec['id']}", use_container_width=True):
            st.session_state.current_view = "history"; st.session_state.selected_history = rec; st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# --- MAIN APP UI ---
st.markdown(f"""
<div class="masthead">
    <div>
        <div class="masthead-wordmark">DebateLM</div>
        <div class="masthead-tagline">Cloud-Native Intelligence Debate · Isolated Guest Pass</div>
    </div>
</div>
""", unsafe_allow_html=True)

if st.session_state.current_view == "new":
    can_debate = debates_used < MAX_DEBATES and bool(GEMINI_API_KEY)

    st.markdown('<div class="section-hd"><div class="section-hd-num">01</div><div><div class="section-hd-title">CONFIGURE DEBATERS</div></div></div>', unsafe_allow_html=True)
    agents_config =[]
    cols = st.columns(min(num_agents, 3), gap="small")
    for i in range(num_agents):
        with cols[i % min(num_agents, 3)]:
            st.markdown(f'<div class="agent-card" style="border-left:4px solid {AGENT_COLORS[i]}"><div class="agent-id" style="color:{AGENT_COLORS[i]}">{AGENT_NAMES[i]}</div></div>', unsafe_allow_html=True)
            model_sel = st.selectbox("Model", AVAILABLE_MODELS, key=f"model_{i}", index=0, label_visibility="collapsed")
            instr = st.text_area("Persona", key=f"inst_{i}", height=72, label_visibility="collapsed", placeholder="Define bias... e.g. You are a skeptic.")
            agents_config.append({"id": i, "instruction": instr if instr else "Be highly analytical and critical.", "model": model_sel})

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-hd"><div class="section-hd-num">02</div><div><div class="section-hd-title">THE MOTION</div></div></div>', unsafe_allow_html=True)
    topic = st.text_area("Motion", placeholder="State the motion to be debated...", height=100, label_visibility="collapsed")
    use_web = st.toggle("Enable live web search grounding via Serper", value=bool(SERPER_API_KEY), disabled=not SERPER_API_KEY)
    
    if debates_used >= MAX_DEBATES:
        st.error(f"🛑 You have reached the {MAX_DEBATES} debate limit for this demo.")
        
    launch = st.button("Launch Debate", type="primary", disabled=not can_debate)

    if launch and topic.strip():
        debate_history, agent_research_logs = [],[]

        for r in range(num_rounds):
            render_round_divider(r + 1, num_rounds)
            for i, ag in enumerate(agents_config):
                agent_llm = get_llm(ag["model"])
                with st.spinner(f"{AGENT_NAMES[i]} researching & formulating..."):
                    # 1. Private Cloud RAG Retrieval (Pinecone)
                    rag_context = retrieve_from_pinecone(topic, agent_llm)
                    # 2. Web Retrieval (Serper)
                    web_results = serper_search(topic, 4) if use_web and r == 0 else[]
                    
                    argument = run_agent_turn(i, ag, topic, topic, rag_context, web_results, debate_history, r + 1)
                    
                    agent_research_logs.append({
                        "round": r + 1, "agent": AGENT_NAMES[i], "query": topic,
                        "web": web_results, "rag_found": bool(rag_context)
                    })
                    
                render_argument(AGENT_NAMES[i], ag["model"], argument, AGENT_COLORS[i % len(AGENT_COLORS)], topic)
                debate_history.append(f"{AGENT_NAMES[i]}: {argument}")

        st.markdown('<div class="round-divider"><div class="round-divider-line"></div><div class="round-divider-label">Deliberation</div><div class="round-divider-line"></div></div>', unsafe_allow_html=True)
        with st.spinner(f"Final Synthesizer ({judge_model}) deliberating..."):
            verdict = run_judge(topic, debate_history, judge_model)
        render_verdict(verdict, judge_model)

        # Saves securely to Supabase and increments quota!
        new_rec = save_new_debate(topic, debate_history, verdict, agent_research_logs, judge_model, guest_id)
        st.session_state.current_view = "history"; st.session_state.selected_history = new_rec; st.rerun()

elif st.session_state.current_view == "history":
    past = st.session_state.selected_history
    if st.button("← Return to Setup"):
        st.session_state.current_view = "new"; st.rerun()

    st.markdown(f'<div class="record-meta"><div class="record-date">{past["date"]}</div></div><div class="record-topic">{past["topic"]}</div>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["Transcript", "Research Logs", "Follow-up"])

    with tab1:
        for msg in past["history"]:
            if ": " in msg:
                speaker, text = msg.split(": ", 1)
                idx = {"I":0, "II":1, "III":2, "IV":3, "V":4}.get(re.search(r"(V?I{0,3}|I{1,3}V?)$", speaker.strip()).group(), 0) if re.search(r"(V?I{0,3}|I{1,3}V?)$", speaker.strip()) else 0
                render_argument(speaker, "", text, AGENT_COLORS[idx % len(AGENT_COLORS)])
        render_verdict(past["verdict"], past.get("judge_model",""))

    with tab2:
        for log in past.get("research_logs",[]):
            st.markdown(f"**{log['agent']}** (Round {log['round']})")
            if log.get("rag_found"): st.markdown("- 📄 *Retrieved data from Pinecone Knowledge Base*")
            for w in log.get("web",[]): st.markdown(f"- 🌐 [{w['title']}]({w['link']})")
            st.divider()

    with tab3:
        for cm in past.get("post_chat", []):
            with st.chat_message(cm["role"]): st.markdown(cm["content"])
        if q := st.chat_input("Ask a follow-up question…"):
            past.setdefault("post_chat",[]).append({"role": "user", "content": q})
            with st.chat_message("user"): st.markdown(q)
            with st.chat_message("assistant"):
                with st.spinner("Deliberating…"):
                    msgs =[SystemMessage(content=f"Synthesizer. Debate: '{past['topic']}'. VERDICT: {past['verdict']}")]
                    for cm in past["post_chat"]: msgs.append(HumanMessage(content=cm["content"]) if cm["role"]=="user" else AIMessage(content=cm["content"]))
                    ans = parse_response(get_llm(past.get("judge_model", DEFAULT_MODEL)).invoke(msgs))
                    st.markdown(ans)
            past["post_chat"].append({"role": "assistant", "content": ans})
            update_debate_chat(past["id"], past) # Syncs chat follow-up to Supabase