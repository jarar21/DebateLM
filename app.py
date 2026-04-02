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

st.set_page_config(page_title="DebateLM Workspaces", page_icon="⚖", layout="wide", initial_sidebar_state="expanded")

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
.project-card { border:1px solid var(--secondary-background-color); padding:1.5rem; border-radius:8px; background:var(--secondary-background-color); margin-bottom:1rem; transition:0.2s; }
.project-card:hover { border-color:var(--primary-color); transform:translateY(-2px); }
</style>
""", unsafe_allow_html=True)

# --- GUEST ID SYSTEM ---
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

# --- SUPABASE DATABASE ---
@st.cache_resource
def init_supabase() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase = init_supabase()

def is_valid_supabase_guest(gid: str) -> bool:
    if not gid: return False
    return bool(re.match(r"^guest_[a-z0-9]{16,}$", gid))

def get_guest_session_data(gid):
    if not is_valid_supabase_guest(gid): 
        return {"debates_run": 0, "preferences": {}}
    try:
        res = supabase.table("guest_sessions").select("debates_run, preferences").eq("guest_id", gid).execute()
        if res.data:
            return {"debates_run": res.data[0].get("debates_run", 0), "preferences": res.data[0].get("preferences") or {}}
        else:
            supabase.table("guest_sessions").upsert({"guest_id": gid, "preferences": {}}).execute()
            return {"debates_run": 0, "preferences": {}}
    except Exception: return {"debates_run": 0, "preferences": {}}

def increment_quota(gid):
    if is_valid_supabase_guest(gid):
        try:
            current = get_guest_session_data(gid)["debates_run"]
            supabase.table("guest_sessions").update({"debates_run": current + 1, "last_active": "now()"}).eq("guest_id", gid).execute()
        except Exception: pass

def load_projects(gid):
    if not is_valid_supabase_guest(gid): return []
    try:
        res = supabase.table("debates").select("*").eq("guest_id", gid).order("created_at", desc=True).execute()
        projs = []
        for row in res.data:
            p = row["history"]
            p["id"] = row["debate_id"]
            p["topic"] = row.get("topic", "Untitled Workspace")
            p["verdict"] = row.get("verdict", "")
            p.setdefault("files", [])
            projs.append(p)
        return projs
    except Exception: return []

def create_project(gid):
    pid = str(uuid.uuid4())
    record = {"id": pid, "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), "files": [], "history": [], "research_logs": [], "post_chat": [], "agents_config": []}
    if is_valid_supabase_guest(gid):
        try:
            supabase.table("debates").insert({"debate_id": pid, "guest_id": gid, "topic": "Untitled Workspace", "history": record}).execute()
            increment_quota(gid)
        except Exception as e: print("DB Error:", e)
    return record

def update_project_db(pid, record, topic, verdict):
    if is_valid_supabase_guest(guest_id):
        try: supabase.table("debates").update({"history": record, "topic": topic, "verdict": verdict}).eq("debate_id", pid).execute()
        except Exception: pass

def delete_project_db(pid):
    if is_valid_supabase_guest(guest_id):
        try: supabase.table("debates").delete().eq("debate_id", pid).execute()
        except Exception: pass

# --- INITIALIZE STATE ---
guest_data = get_guest_session_data(guest_id)
if "user_prefs" not in st.session_state: st.session_state.user_prefs = guest_data["preferences"]
debates_used = guest_data["debates_run"]

if "projects" not in st.session_state: st.session_state.projects = load_projects(guest_id)
if "current_view" not in st.session_state: st.session_state.current_view = "dashboard"
if "current_project_id" not in st.session_state: st.session_state.current_project_id = None

def get_model_idx(model_name): return AVAILABLE_MODELS.index(model_name) if model_name in AVAILABLE_MODELS else 0

# --- PINECONE INTEGRATION ---
@st.cache_resource
def init_pinecone(): return Pinecone(api_key=PINECONE_API_KEY)
pc = init_pinecone()

def get_vectorstore():
    emb = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=GEMINI_API_KEY, output_dimensionality=768)
    return PineconeVectorStore(index_name=PINECONE_INDEX_NAME, embedding=emb, pinecone_api_key=PINECONE_API_KEY)

def process_documents(uploaded_files, project_id):
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
            for d in file_docs:
                d.metadata["file_name"] = uf.name
                d.metadata["source"] = uf.name
            raw_docs.extend(file_docs)
        finally:
            if os.path.exists(tmp_path): os.remove(tmp_path)
            
    parent_docs = p_split.split_documents(raw_docs)
    child_docs = []
    for parent in parent_docs:
        children = c_split.split_documents([parent])
        for child in children:
            child.metadata.update({"guest_id": guest_id, "project_id": project_id, "file_name": parent.metadata.get("file_name", "unknown"), "parent_text": parent.page_content, "source": parent.metadata.get("source", "unknown")})
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
        try: vs.add_documents(batch)
        except Exception:
            progress_bar.progress(0.2 + (0.8 * ((i // BATCH_SIZE + 1) / total_batches)), text=f"⚠️ Network hiccup. Retrying batch {(i // BATCH_SIZE) + 1} in 2s...")
            time.sleep(2)
            vs.add_documents(batch)
            
    progress_bar.empty()
    return total_chunks

def delete_documents(file_names, project_id):
    index = pc.Index(PINECONE_INDEX_NAME)
    for fname in file_names:
        try: index.delete(filter={"file_name": {"$eq": fname}, "guest_id": {"$eq": guest_id}, "project_id": {"$eq": project_id}})
        except Exception: pass

def retrieve_from_pinecone(query, project_id):
    try:
        results = get_vectorstore().similarity_search(query, k=20, filter={"guest_id": guest_id, "project_id": project_id})
        parents, context_blocks = set(), []
        for doc in results:
            pt = doc.metadata.get("parent_text", doc.page_content)
            if pt not in parents:
                parents.add(pt)
                context_blocks.append(f"[Source: {doc.metadata.get('source','?')}]\n{pt}")
        return "\n\n---\n\n".join(context_blocks)
    except Exception: return ""

# --- CORE LLM LOGIC ---
def parse_response(response):
    c = response.content
    return "\n".join(b.get("text","") for b in c if isinstance(b,dict) and "text" in b) if isinstance(c, list) else str(c)

def get_llm(model: str):
    return ChatGoogleGenerativeAI(model=model, google_api_key=GEMINI_API_KEY, temperature=0.7)

def generate_agent_query(topic, agent_config):
    try:
        agent_llm = get_llm(agent_config["model"])
        prompt = f"Topic: '{topic}'\nYour Identity/Persona: {agent_config['instruction']}\n\nBased ONLY on your specific persona and the topic, what is the single most effective Google search query (under 8 words) you would run right now to find evidence supporting your unique perspective? Return ONLY the search string, no quotes, no explanation."
        raw = parse_response(agent_llm.invoke([SystemMessage(content="You are an expert researcher defining your own strategy."), HumanMessage(content=prompt)])).strip().replace('"', '')
        return raw if raw else topic
    except Exception as e:
        print(f"Query Gen Error for agent {agent_config.get('id', 'unknown')}:", e)
        return topic

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

# --- MAIN APP UI ROUTER ---

if st.session_state.current_view == "dashboard":
    with st.sidebar:
        st.markdown('<div class="sb-logo">⚖ DebateLM</div><div class="sb-logo-sub">Workspaces</div>', unsafe_allow_html=True)
        pct = int(min(debates_used / MAX_DEBATES, 1.0) * 100)
        st.markdown(f'<div class="quota-row"><div class="quota-label"><span>Free Demo Quota</span><span>{debates_used}/{MAX_DEBATES}</span></div><div class="quota-track"><div class="quota-fill" style="width:{pct}%; background:{"#E11D48" if debates_used >= MAX_DEBATES else "var(--primary-color)"}"></div></div></div>', unsafe_allow_html=True)

    st.markdown('<div class="masthead"><div><div class="masthead-wordmark">DebateLM Workspaces</div><div class="masthead-tagline">Your persistent intelligence projects</div></div></div>', unsafe_allow_html=True)
    
    c1, c2 = st.columns([0.8, 0.2])
    c1.markdown('<div class="section-hd"><div class="section-hd-num" style="min-width:0;"></div><div class="section-hd-title">ACTIVE PROJECTS</div></div>', unsafe_allow_html=True)
    if c2.button("➕ New Workspace", type="primary", use_container_width=True, disabled=debates_used >= MAX_DEBATES):
        new_p = create_project(guest_id)
        st.session_state.projects.insert(0, new_p)
        st.session_state.current_project_id = new_p["id"]
        st.session_state.current_view = "project"
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    if not st.session_state.projects:
        st.info("No workspaces yet. Create one to upload documents and launch a debate!")
    else:
        cols = st.columns(3, gap="medium")
        for i, p in enumerate(st.session_state.projects):
            with cols[i % 3]:
                st.markdown(f"""
                <div class="project-card">
                    <h4 style="margin:0 0 0.5rem 0; color:var(--text-color);">{p.get("topic", "Untitled Workspace")}</h4>
                    <div style="font-size:0.8rem; opacity:0.7; margin-bottom:1rem;">📁 {len(p.get("files",[]))} Files • {len(p.get("history",[]))//2} Rounds</div>
                </div>
                """, unsafe_allow_html=True)
                
                c_open, c_del = st.columns([0.75, 0.25])
                if c_open.button("Open Workspace", key=f"open_{p['id']}", use_container_width=True):
                    st.session_state.current_project_id = p["id"]
                    st.session_state.current_view = "project"
                    st.rerun()
                if c_del.button("🗑️", key=f"del_{p['id']}", help="Delete Workspace"):
                    delete_project_db(p["id"])
                    try: pc.Index(PINECONE_INDEX_NAME).delete(filter={"project_id": {"$eq": p["id"]}})
                    except: pass
                    st.session_state.projects = [x for x in st.session_state.projects if x["id"] != p["id"]]
                    st.rerun()

elif st.session_state.current_view == "project":
    proj = next((x for x in st.session_state.projects if x["id"] == st.session_state.current_project_id), None)
    if not proj:
        st.session_state.current_view = "dashboard"
        st.rerun()

    with st.sidebar:
        if st.button("← Back to Workspaces", use_container_width=True):
            st.session_state.current_view = "dashboard"
            st.rerun()
            
        st.divider()
        st.markdown('<div class="sb-label">📁 KNOWLEDGE BASE</div>', unsafe_allow_html=True)
        
        saved_files = proj.get("files", [])
        if saved_files:
            for fname in saved_files:
                c1, c2 = st.columns([0.85, 0.15])
                c1.markdown(f"<div style='font-size:0.8rem; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;'>📄 {fname}</div>", unsafe_allow_html=True)
                if c2.button("🗑️", key=f"del_f_{fname}_{proj['id']}"):
                    delete_documents([fname], proj["id"])
                    proj["files"].remove(fname)
                    update_project_db(proj["id"], proj, proj.get("topic", "Untitled Workspace"), proj.get("verdict", ""))
                    st.rerun()
                    
        st.markdown("<br>", unsafe_allow_html=True)
        new_files = st.file_uploader("Add Sources", type=["pdf","txt"], accept_multiple_files=True, key=f"up_{proj['id']}")
        if new_files:
            files_to_process = [f for f in new_files if f.name not in saved_files]
            if files_to_process:
                process_documents(files_to_process, proj["id"])
                proj.setdefault("files", []).extend([f.name for f in files_to_process])
                update_project_db(proj["id"], proj, proj.get("topic", "Untitled Workspace"), proj.get("verdict", ""))
                st.toast(f"✅ Synced {len(files_to_process)} file(s) to Workspace!")
                st.rerun()
                
    # --- WORKSPACE MAIN AREA ---
    new_title = st.text_input("Workspace Name", value=proj.get("topic", "Untitled Workspace"), label_visibility="collapsed", placeholder="Name your project or enter the motion...")
    if new_title != proj.get("topic", "Untitled Workspace"):
        proj["topic"] = new_title
        update_project_db(proj["id"], proj, new_title, proj.get("verdict", ""))

    if not proj.get("history"):
        st.markdown('<div class="section-hd"><div class="section-hd-num">01</div><div><div class="section-hd-title">CONFIGURE DEBATERS</div></div></div>', unsafe_allow_html=True)
        
        c_conf1, c_conf2, c_conf3 = st.columns(3)
        num_agents  = c_conf1.slider("Debaters", 2, 5, 2)
        num_rounds  = c_conf2.slider("Rounds", 1, 5, 2)
        judge_model = c_conf3.selectbox("Judge Model", AVAILABLE_MODELS, index=get_model_idx(DEFAULT_MODEL))
        
        agents_config = []
        cols = st.columns(min(num_agents, 3), gap="small")
        for i in range(num_agents):
            with cols[i % min(num_agents, 3)]:
                st.markdown(f'<div class="agent-card" style="border-left:4px solid {AGENT_COLORS[i]}"><div class="agent-id" style="color:{AGENT_COLORS[i]}">{AGENT_NAMES[i]}</div></div>', unsafe_allow_html=True)
                model_sel = st.selectbox("Model", AVAILABLE_MODELS, key=f"model_{i}", label_visibility="collapsed")
                instr = st.text_area("Persona", key=f"inst_{i}", height=72, label_visibility="collapsed", placeholder="Define bias...")
                agents_config.append({"id": i, "instruction": instr if instr else "Be highly analytical and critical.", "model": model_sel})

        use_web = st.toggle("Enable live web search grounding via Serper", value=bool(SERPER_API_KEY), disabled=not SERPER_API_KEY)
        
        button_placeholder = st.empty()
        launch = button_placeholder.button("Launch Debate", type="primary", disabled=not bool(GEMINI_API_KEY), use_container_width=True)

        if launch and new_title.strip() and new_title != "Untitled Workspace":
            proj["agents_config"] = agents_config
            debate_history, agent_research_logs = [], []
            
            button_placeholder.button("🛑 Stop Ongoing Debate", type="primary", use_container_width=True, key="stop_btn")

            with st.spinner("Agents are autonomously defining their research strategies..."):
                agent_queries = {}
                with ThreadPoolExecutor(max_workers=min(num_agents, 5)) as executor:
                    query_futures = {executor.submit(generate_agent_query, new_title, ag): i for i, ag in enumerate(agents_config)}
                    for future in as_completed(query_futures):
                        agent_queries[query_futures[future]] = future.result()

            for r in range(num_rounds):
                render_round_divider(r + 1, num_rounds)
                with st.spinner(f"Round {r+1} Intelligence Gathering..."):
                    rag_contexts, web_contexts = {}, {}
                    with ThreadPoolExecutor(max_workers=min(num_agents * 2, 10)) as executor:
                        futures_rag = {executor.submit(retrieve_from_pinecone, agent_queries[i], proj["id"]): i for i, ag in enumerate(agents_config)}
                        futures_web = {executor.submit(serper_search, agent_queries[i], 3): i for i, ag in enumerate(agents_config)} if use_web and r == 0 else {}
                        for future in as_completed(futures_rag): rag_contexts[futures_rag[future]] = future.result()
                        for future in as_completed(futures_web): web_contexts[futures_web[future]] = future.result()

                for i, ag in enumerate(agents_config):
                    with st.spinner(f"{AGENT_NAMES[i]} formulating argument..."):
                        rag_context = rag_contexts.get(i, "")
                        web_results = web_contexts.get(i, []) if r == 0 else []
                        argument = run_agent_turn(i, ag, new_title, agent_queries[i], rag_context, web_results, debate_history, r + 1)
                        agent_research_logs.append({"round": r + 1, "agent": AGENT_NAMES[i], "query": agent_queries[i], "web": web_results, "rag_found": bool(rag_context)})
                        
                    render_argument(AGENT_NAMES[i], ag["model"], argument, AGENT_COLORS[i % len(AGENT_COLORS)], agent_queries[i])
                    debate_history.append(f"{AGENT_NAMES[i]}: {argument}")
                    
                    proj["history"] = list(debate_history)
                    proj["research_logs"] = list(agent_research_logs)
                    update_project_db(proj["id"], proj, new_title, "⚠️ *Debate interrupted.*")

            st.markdown('<div class="round-divider"><div class="round-divider-line"></div><div class="round-divider-label">Deliberation</div><div class="round-divider-line"></div></div>', unsafe_allow_html=True)
            with st.spinner(f"Final Synthesizer ({judge_model}) deliberating..."):
                final_verdict_text = run_judge(new_title, debate_history, judge_model)
                
            render_verdict(final_verdict_text, judge_model)

            proj["verdict"] = final_verdict_text
            proj["history"] = debate_history
            proj["research_logs"] = agent_research_logs
            proj["judge_model"] = judge_model
            
            update_project_db(proj["id"], proj, new_title, final_verdict_text)
            st.rerun()

    else:
        # Show Transcript
        tab1, tab2, tab3 = st.tabs(["Transcript", "Research Logs", "Follow-up"])
        
        with tab1:
            for msg in proj.get("history", []):
                if ": " in msg:
                    speaker, text = msg.split(": ", 1)
                    idx = {"I":0, "II":1, "III":2, "IV":3, "V":4}.get(re.search(r"(V?I{0,3}|I{1,3}V?)$", speaker.strip()).group(), 0) if re.search(r"(V?I{0,3}|I{1,3}V?)$", speaker.strip()) else 0
                    render_argument(speaker, "", text, AGENT_COLORS[idx % len(AGENT_COLORS)])
            render_verdict(proj.get("verdict", ""), proj.get("judge_model",""))

        with tab2:
            for log in proj.get("research_logs",[]):
                st.markdown(f"**{log['agent']}** (Round {log['round']})")
                if log.get("rag_found"): st.markdown("- 📄 *Retrieved data from Pinecone Knowledge Base*")
                for w in log.get("web",[]): st.markdown(f"- 🌐 [{w['title']}]({w['link']})")
                st.divider()

        with tab3:
            for cm in proj.get("post_chat", []):
                with st.chat_message(cm["role"]): 
                    st.markdown(format_professional_citations(cm["content"]), unsafe_allow_html=True)
                    
            if q := st.chat_input("Ask a follow-up question…"):
                proj.setdefault("post_chat",[]).append({"role": "user", "content": q})
                with st.chat_message("user"): st.markdown(q)
                with st.chat_message("assistant"):
                    with st.spinner("Deliberating…"):
                        msgs =[SystemMessage(content=f"Synthesizer. Workspace: '{proj.get('topic','')}'. VERDICT: {proj.get('verdict','')}. Cite sources if needed.")]
                        for cm in proj["post_chat"]: msgs.append(HumanMessage(content=cm["content"]) if cm["role"]=="user" else AIMessage(content=cm["content"]))
                        ans = parse_response(get_llm(proj.get("judge_model", DEFAULT_MODEL)).invoke(msgs))
                        st.markdown(format_professional_citations(ans), unsafe_allow_html=True)
                        
                proj["post_chat"].append({"role": "assistant", "content": ans})
                update_project_db(proj["id"], proj, proj.get("topic", ""), proj.get("verdict", ""))
                st.rerun()