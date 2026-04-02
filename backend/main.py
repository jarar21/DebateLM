import os, json, datetime, uuid, time, re, tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

import requests
from supabase import create_client, Client
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()

# --- CONFIG & SECRETS ---
GEMINI_API_KEY      = os.getenv("GEMINI_API_KEY")
SERPER_API_KEY      = os.getenv("SERPER_API_KEY")
SUPABASE_URL        = os.getenv("SUPABASE_URL")
SUPABASE_KEY        = os.getenv("SUPABASE_KEY")
PINECONE_API_KEY    = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

AVAILABLE_MODELS = [
    "gemini-3-flash-preview",
    "gemini-3.1-pro-preview",
    "gemini-3.1-flash-lite-preview",
]
DEFAULT_MODEL = AVAILABLE_MODELS[0]
MAX_DEBATES = 10

# --- INITIALIZATION ---
app = FastAPI(title="DebateLM API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, lock this down to your GitHub Pages URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# --- PYDANTIC MODELS ---
class GuestIdParams(BaseModel):
    guest_id: str

class ProjectUpdate(BaseModel):
    guest_id: str
    topic: str
    agents_config: List[Dict[str, Any]]
    num_rounds: int
    judge_model: str
    use_web: bool

class ChatRequest(BaseModel):
    guest_id: str
    message: str

# --- HELPERS ---
def is_valid_guest(gid: str) -> bool:
    return bool(gid and re.match(r"^guest_[a-z0-9]{16,}$", gid))

def get_vectorstore():
    emb = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=GEMINI_API_KEY, output_dimensionality=768)
    return PineconeVectorStore(index_name=PINECONE_INDEX_NAME, embedding=emb, pinecone_api_key=PINECONE_API_KEY)

def get_llm(model: str):
    return ChatGoogleGenerativeAI(model=model, google_api_key=GEMINI_API_KEY, temperature=0.7)

def parse_response(response):
    c = response.content
    return "\n".join(b.get("text","") for b in c if isinstance(b,dict) and "text" in b) if isinstance(c, list) else str(c)

# --- DB FUNCTIONS ---
def get_or_create_guest(gid: str):
    if not is_valid_guest(gid): raise HTTPException(status_code=400, detail="Invalid Guest ID")
    res = supabase.table("guest_sessions").select("*").eq("guest_id", gid).execute()
    if res.data:
        return res.data[0]
    supabase.table("guest_sessions").upsert({"guest_id": gid, "debates_run": 0, "preferences": {}}).execute()
    return {"guest_id": gid, "debates_run": 0, "preferences": {}}

def increment_quota(gid: str):
    guest = get_or_create_guest(gid)
    supabase.table("guest_sessions").update({"debates_run": guest.get("debates_run", 0) + 1, "last_active": "now()"}).eq("guest_id", gid).execute()

# --- ROUTES ---

@app.get("/")
def healthcheck():
    return {"status": "online", "version": "2.0"}

@app.get("/api/guest/{guest_id}")
def get_guest_data(guest_id: str):
    return get_or_create_guest(guest_id)

@app.get("/api/projects/{guest_id}")
def get_projects(guest_id: str):
    if not is_valid_guest(guest_id): raise HTTPException(status_code=400, detail="Invalid Guest ID")
    res = supabase.table("debates").select("*").eq("guest_id", guest_id).order("created_at", desc=True).execute()
    
    projects = []
    for row in res.data:
        p = row.get("history", {})
        p["id"] = row["debate_id"]
        p["topic"] = row.get("topic", "Untitled Workspace")
        p["verdict"] = row.get("verdict", "")
        p.setdefault("files", [])
        projects.append(p)
    return {"projects": projects}

@app.post("/api/projects")
def create_project(payload: GuestIdParams):
    gid = payload.guest_id
    guest = get_or_create_guest(gid)
    if guest.get("debates_run", 0) >= MAX_DEBATES:
        raise HTTPException(status_code=403, detail="Quota exceeded")

    project_id = str(uuid.uuid4())
    record = {
        "id": project_id, "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "topic": "Untitled Workspace", "files": [], "history": [], "research_logs": [], 
        "post_chat": [], "agents_config": []
    }
    
    supabase.table("debates").insert({
        "debate_id": project_id, "guest_id": gid, "topic": "Untitled Workspace", "history": record
    }).execute()
    increment_quota(gid)
    return record

@app.delete("/api/projects/{project_id}")
def delete_project(project_id: str, guest_id: str):
    if not is_valid_guest(guest_id): raise HTTPException(status_code=400, detail="Invalid Guest ID")
    
    # DB Delete
    supabase.table("debates").delete().eq("debate_id", project_id).eq("guest_id", guest_id).execute()
    
    # Pinecone Vector Delete
    try:
        pc.Index(PINECONE_INDEX_NAME).delete(filter={"project_id": {"$eq": project_id}, "guest_id": {"$eq": guest_id}})
    except Exception as e:
        print("Pinecone delete error:", e)
        
    return {"status": "deleted"}

@app.post("/api/projects/{project_id}/files")
async def upload_files(project_id: str, guest_id: str = Form(...), files: List[UploadFile] = File(...)):
    if not is_valid_guest(guest_id): raise HTTPException(status_code=400, detail="Invalid Guest ID")
    
    # Fetch current project
    res = supabase.table("debates").select("history").eq("debate_id", project_id).eq("guest_id", guest_id).execute()
    if not res.data: raise HTTPException(status_code=404, detail="Project not found")
    proj = res.data[0]["history"]
    current_files = proj.get("files", [])
    
    p_split = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=400)
    c_split = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    
    raw_docs = []
    processed_names = []
    
    for uf in files:
        if uf.filename in current_files: continue
        
        suffix = ".pdf" if uf.filename.lower().endswith(".pdf") else ".txt"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            content = await uf.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
            
        try:
            file_docs = PyMuPDFLoader(tmp_path).load() if suffix == ".pdf" else TextLoader(tmp_path).load()
            for d in file_docs:
                d.metadata["file_name"] = uf.filename
                d.metadata["source"] = uf.filename
            raw_docs.extend(file_docs)
            processed_names.append(uf.filename)
        finally:
            if os.path.exists(tmp_path): os.remove(tmp_path)
            
    if not raw_docs:
        return {"status": "no new files"}

    parent_docs = p_split.split_documents(raw_docs)
    child_docs = []
    for parent in parent_docs:
        children = c_split.split_documents([parent])
        for child in children:
            child.metadata.update({
                "guest_id": guest_id, "project_id": project_id, 
                "file_name": parent.metadata.get("file_name", "unknown"), 
                "parent_text": parent.page_content, "source": parent.metadata.get("source", "unknown")
            })
            if "page" in child.metadata: del child.metadata["page"]
        child_docs.extend(children)
        
    vs = get_vectorstore()
    BATCH_SIZE = 150 
    for i in range(0, len(child_docs), BATCH_SIZE):
        batch = child_docs[i : i + BATCH_SIZE]
        try:
            vs.add_documents(batch)
        except Exception:
            time.sleep(2)
            vs.add_documents(batch)

    # Update DB
    proj.setdefault("files", []).extend(processed_names)
    supabase.table("debates").update({"history": proj}).eq("debate_id", project_id).execute()
    
    return {"status": "success", "added": processed_names}

@app.delete("/api/projects/{project_id}/files/{file_name}")
def remove_file(project_id: str, file_name: str, guest_id: str):
    if not is_valid_guest(guest_id): raise HTTPException(status_code=400, detail="Invalid Guest ID")
    
    # Remove from Pinecone
    try:
        pc.Index(PINECONE_INDEX_NAME).delete(filter={"file_name": {"$eq": file_name}, "guest_id": {"$eq": guest_id}, "project_id": {"$eq": project_id}})
    except Exception as e:
        print("Pinecone delete error:", e)

    # Remove from DB
    res = supabase.table("debates").select("history").eq("debate_id", project_id).eq("guest_id", guest_id).execute()
    if res.data:
        proj = res.data[0]["history"]
        if file_name in proj.get("files", []):
            proj["files"].remove(file_name)
            supabase.table("debates").update({"history": proj}).eq("debate_id", project_id).execute()
            
    return {"status": "deleted", "file": file_name}

# --- DEBATE LOGIC ---
def retrieve_from_pinecone(query, project_id, guest_id):
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

def serper_search(query, num_results=4):
    if not SERPER_API_KEY: return[]
    try:
        resp = requests.post("https://google.serper.dev/search", headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}, json={"q": query, "num": num_results, "hl": "en"}, timeout=10)
        return[{"title": r.get("title",""), "snippet": r.get("snippet",""), "link": r.get("link","")} for r in resp.json().get("organic",[])[:num_results]]
    except Exception: return[]

def format_web_evidence(results):
    return "\n".join(f"{i}. {r['title']}\n   {r['snippet']}\n   [{r['link']}]\n" for i, r in enumerate(results, 1))

def generate_agent_query(topic, agent_config):
    try:
        agent_llm = get_llm(agent_config["model"])
        prompt = f"Topic: '{topic}'\nYour Identity/Persona: {agent_config['instruction']}\n\nBased ONLY on your specific persona and the topic, what is the single most effective Google search query (under 8 words) you would run right now to find evidence supporting your unique perspective? Return ONLY the search string, no quotes, no explanation."
        raw = parse_response(agent_llm.invoke([SystemMessage(content="You are an expert researcher defining your own strategy."), HumanMessage(content=prompt)])).strip().replace('"', '')
        return raw if raw else topic
    except Exception:
        return topic

def run_agent_turn(agent_data, topic, rag_context, web_evidence, debate_history, round_num):
    llm = get_llm(agent_data["model"])
    context_block = ""
    if rag_context: context_block += f"\nINTERNAL KB:\n{rag_context}\n"
    if web_evidence: context_block += f"\nLIVE WEB:\n{format_web_evidence(web_evidence)}\n"
    
    citation_rule = "Do NOT fabricate citations." if not context_block.strip() else "Cite strictly using [Source: Title] or [Web: Domain]."
    system = f"IDENTITY: {agent_data['instruction']}\n\nEvidence:\n{context_block}\n\nRULES: {citation_rule} Make 2-3 arguments. Rebut previous claims. 200-350 words."
    user_msg = f"Round {round_num} — Debate:\n\n{chr(10).join(debate_history[-6:])}\n\nArgue: '{topic}'" if debate_history else f"Round {round_num} — Opening. Topic: '{topic}'"
    return parse_response(llm.invoke([SystemMessage(content=system), HumanMessage(content=user_msg)]))

@app.post("/api/projects/{project_id}/debate")
def run_debate_stream(project_id: str, config: ProjectUpdate):
    """
    Server-Sent Events (SSE) endpoint that runs the debate loop and streams events 
    live to the React frontend.
    """
    if not is_valid_guest(config.guest_id): raise HTTPException(status_code=400, detail="Invalid Guest ID")
    
    res = supabase.table("debates").select("history").eq("debate_id", project_id).eq("guest_id", config.guest_id).execute()
    if not res.data: raise HTTPException(status_code=404, detail="Project not found")
    proj = res.data[0]["history"]
    
    proj["topic"] = config.topic
    proj["agents_config"] = config.agents_config
    
    AGENT_NAMES = ["AGENT I", "AGENT II", "AGENT III", "AGENT IV", "AGENT V"]
    
    def event_stream():
        debate_history = []
        agent_research_logs = []
        
        yield f"data: {json.dumps({'type': 'status', 'msg': 'Agents formulating research strategies...'})}\n\n"
        
        agent_queries = {}
        with ThreadPoolExecutor(max_workers=min(len(config.agents_config), 5)) as executor:
            query_futures = {executor.submit(generate_agent_query, config.topic, ag): i for i, ag in enumerate(config.agents_config)}
            for future in as_completed(query_futures):
                agent_queries[query_futures[future]] = future.result()

        for r in range(config.num_rounds):
            yield f"data: {json.dumps({'type': 'status', 'msg': f'Round {r+1} Intelligence Gathering...'})}\n\n"
            
            rag_contexts, web_contexts = {}, {}
            with ThreadPoolExecutor(max_workers=min(len(config.agents_config) * 2, 10)) as executor:
                futures_rag = {executor.submit(retrieve_from_pinecone, agent_queries[i], project_id, config.guest_id): i for i, ag in enumerate(config.agents_config)}
                futures_web = {executor.submit(serper_search, agent_queries[i], 3): i for i, ag in enumerate(config.agents_config)} if config.use_web and r == 0 else {}
                
                for future in as_completed(futures_rag): rag_contexts[futures_rag[future]] = future.result()
                for future in as_completed(futures_web): web_contexts[futures_web[future]] = future.result()

            for i, ag in enumerate(config.agents_config):
                agent_name = AGENT_NAMES[i]
                yield f"data: {json.dumps({'type': 'status', 'msg': f'{agent_name} formulating argument...'})}\n\n"
                
                rag_context = rag_contexts.get(i, "")
                web_results = web_contexts.get(i, []) if r == 0 else []
                
                argument = run_agent_turn(ag, config.topic, rag_context, web_results, debate_history, r + 1)
                
                log_data = {"round": r + 1, "agent": agent_name, "query": agent_queries[i], "web": web_results, "rag_found": bool(rag_context)}
                agent_research_logs.append(log_data)
                
                debate_history.append(f"{agent_name}: {argument}")
                
                # Stream the argument to UI
                yield f"data: {json.dumps({'type': 'argument', 'speaker': agent_name, 'model': ag['model'], 'text': argument, 'query': agent_queries[i], 'log': log_data})}\n\n"

        yield f"data: {json.dumps({'type': 'status', 'msg': 'Final Synthesizer deliberating...'})}\n\n"
        
        judge_prompt = f"TOPIC: '{config.topic}'\n\nTRANSCRIPT:\n{chr(10).join(debate_history)}\n\nDeliver FINAL Verdict with Synthesis. 400-600 words."
        verdict = parse_response(get_llm(config.judge_model).invoke([
            SystemMessage(content="Evaluate debate purely on transcripts. Cite using [Source: Title]."),
            HumanMessage(content=judge_prompt)
        ]))
        
        yield f"data: {json.dumps({'type': 'verdict', 'model': config.judge_model, 'text': verdict})}\n\n"
        
        # Save to DB
        proj["history"] = debate_history
        proj["research_logs"] = agent_research_logs
        proj["verdict"] = verdict
        proj["judge_model"] = config.judge_model
        supabase.table("debates").update({"history": proj, "topic": config.topic, "verdict": verdict}).eq("debate_id", project_id).execute()
        
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.post("/api/projects/{project_id}/chat")
def follow_up_chat(project_id: str, payload: ChatRequest):
    if not is_valid_guest(payload.guest_id): raise HTTPException(status_code=400, detail="Invalid Guest ID")
    res = supabase.table("debates").select("history").eq("debate_id", project_id).eq("guest_id", payload.guest_id).execute()
    if not res.data: raise HTTPException(status_code=404, detail="Project not found")
    
    proj = res.data[0]["history"]
    proj.setdefault("post_chat", []).append({"role": "user", "content": payload.message})
    
    msgs = [SystemMessage(content=f"Synthesizer. Workspace: '{proj.get('topic','')}'. VERDICT: {proj.get('verdict','')}. Cite sources if needed.")]
    for cm in proj["post_chat"]:
        msgs.append(HumanMessage(content=cm["content"]) if cm["role"]=="user" else AIMessage(content=cm["content"]))
        
    ans = parse_response(get_llm(proj.get("judge_model", DEFAULT_MODEL)).invoke(msgs))
    proj["post_chat"].append({"role": "assistant", "content": ans})
    
    supabase.table("debates").update({"history": proj}).eq("debate_id", project_id).execute()
    return {"reply": ans}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)