import streamlit as st
import os
import json
import datetime
import uuid
import shutil
from dotenv import load_dotenv

# Import LangChain components for Google/Gemini
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Load environment variables (API Key)
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY") 

st.set_page_config(page_title="DebateLM", page_icon="🏛️", layout="wide")

# ==========================================
# 🟢 1. MULTI-USER ISOLATION
# ==========================================
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

USER_DIR = os.path.join("user_data", st.session_state.session_id)
os.makedirs(USER_DIR, exist_ok=True)

AUTOSAVE_FILE = os.path.join(USER_DIR, "autosave.json")
HISTORY_FILE = os.path.join(USER_DIR, "debate_history_db.json")
PERSIST_DIR = os.path.join(USER_DIR, "chroma_db") 
TEMP_DOCS_DIR = os.path.join(USER_DIR, "temp_docs")

# 🟢 2. THE EXCLUSIVE GEMINI 3.x MODELS
AVAILABLE_MODELS =[
    "gemini-3.1-pro-preview",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite-preview"
]

MAX_DEBATES = 7 # Protection Cap

# ==========================================
# 🟢 3. DATABASE & HISTORY FUNCTIONS
# ==========================================
def auto_save():
    data = {}
    if "debate_topic" in st.session_state:
        data["debate_topic"] = st.session_state["debate_topic"]
    for i in range(5): # Max 5 agents
        if f"inst_{i}" in st.session_state:
            data[f"inst_{i}"] = st.session_state[f"inst_{i}"]
    with open(AUTOSAVE_FILE, "w") as f:
        json.dump(data, f)

def load_past_debates():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            records = json.load(f)
            for i, record in enumerate(records):
                if "id" not in record:
                    record["id"] = f"legacy_{i}_{datetime.datetime.now().timestamp()}"
                if "research_brief" not in record:
                    record["research_brief"] = "Legacy Record"
                if "post_chat" not in record:
                    record["post_chat"] =[]
                if "judge_model" not in record:
                    record["judge_model"] = AVAILABLE_MODELS[1]
            return records
    return[]

def save_new_debate(topic, history, verdict, research_brief, judge_model):
    new_record = {
        "id": str(datetime.datetime.now().timestamp()),
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "topic": topic,
        "research_brief": research_brief,
        "history": history,
        "verdict": verdict,
        "post_chat":[],
        "judge_model": judge_model
    }
    st.session_state.past_debates.insert(0, new_record)
    with open(HISTORY_FILE, "w") as f:
        json.dump(st.session_state.past_debates, f)
    return new_record

# Initialize Session States
if "loaded_autosave" not in st.session_state:
    if os.path.exists(AUTOSAVE_FILE):
        with open(AUTOSAVE_FILE, "r") as f:
            saved_data = json.load(f)
            for key, value in saved_data.items():
                st.session_state[key] = value
    st.session_state["loaded_autosave"] = True

if "past_debates" not in st.session_state:
    st.session_state.past_debates = load_past_debates()

if "current_view" not in st.session_state:
    st.session_state.current_view = "new" 

if "vectorstore" not in st.session_state:
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        st.session_state.vectorstore = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)
        )
    else:
        st.session_state.vectorstore = None

# ==========================================
# 🟢 4. AI & RAG LOGIC
# ==========================================
def parse_gemini_response(response):
    content = response.content
    if isinstance(content, list):
        return "\n".join([block.get("text", "") for block in content if isinstance(block, dict) and "text" in block])
    return str(content)

def get_ai_persona(topic, agent_index, model_name):
    llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)
    prompt = f"The debate topic is: '{topic}'. Create a short (1 sentence) system instruction for 'Agent {agent_index+1}'. Give them a unique professional background and a specific bias."
    response = llm.invoke(prompt)
    return parse_gemini_response(response)

def generate_research_brief(topic, vectorstore, judge_model):
    if vectorstore is None:
        return "No external documents provided."
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={'k': 8, 'fetch_k': 15})
    docs = retriever.invoke(topic)
    context = "\n".join([f"Source: {d.metadata.get('source')} - Content: {d.page_content}" for d in docs])
    
    llm = ChatGoogleGenerativeAI(model=judge_model, google_api_key=api_key)
    prompt = f"""You are the Lead Researcher Agent. Read the following retrieved documents for: '{topic}'.
    Write a 'Research Brief' summarizing key findings and conflicting views. You MUST cite sources in brackets e.g.[Source: paper1.pdf].
    DOCUMENTS CONTEXT:
    {context}
    """
    response = llm.invoke(prompt)
    return parse_gemini_response(response)

def process_documents(uploaded_files, is_append=False):
    all_docs =[]
    os.makedirs(TEMP_DOCS_DIR, exist_ok=True)
        
    for uploaded_file in uploaded_files:
        file_path = os.path.join(TEMP_DOCS_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        loader = PyPDFLoader(file_path)
        all_docs.extend(loader.load())
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    splits = text_splitter.split_documents(all_docs)
    
    if is_append and st.session_state.vectorstore is not None:
        st.session_state.vectorstore.add_documents(splits)
        return st.session_state.vectorstore
    else:
        vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key),
            persist_directory=PERSIST_DIR 
        )
        return vectorstore

# ==========================================
# 🟢 5. BEAUTIFUL UI DESIGN
# ==========================================
st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h1 style='color: #2e6c80;'>🏛️ DebateLM</h1>
        <p style='font-size: 18px; color: gray;'>Upload knowledge, define AI personas, and watch them debate complex topics to find the truth.</p>
    </div>
    <hr>
""", unsafe_allow_html=True)

with st.sidebar:
    st.button("➕ Create New Debate", type="primary", use_container_width=True, on_click=lambda: st.session_state.update(current_view="new"))
    
    st.header("⚙️ Settings")
    # 🟢 Max 5 Agents, Max 5 Rounds
    num_agents = st.slider("Number of Agents", 2, 5, 2, help="Choose up to 5 distinct AI agents.")
    num_rounds = st.number_input("Number of Rounds", 1, 5, 2, help="Maximum 5 rounds per debate.")
    global_judge_model = st.selectbox("Judge & Researcher Model", AVAILABLE_MODELS, index=1)

    # 🟢 QUOTA COUNTER
    st.divider()
    st.header("📊 Usage Quota")
    debates_used = len(st.session_state.past_debates)
    progress_val = min(debates_used / MAX_DEBATES, 1.0)
    st.progress(progress_val)
    st.write(f"**Debates used:** {debates_used} / {MAX_DEBATES}")
    if debates_used >= MAX_DEBATES:
        st.error("🚨 You have reached the maximum limit of free debates for this session.")

    st.divider()
    st.header("📚 Private Knowledge Base")
    
    if st.session_state.vectorstore is not None:
        st.success("📁 Ready for Debate!")
        with st.expander("➕ Add more papers"):
            new_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True, key="append")
            if st.button("Add to Database 📥", use_container_width=True):
                if new_files:
                    with st.spinner("Processing..."):
                        st.session_state.vectorstore = process_documents(new_files, is_append=True)
                    st.rerun()
    else:
        uploaded_files = st.file_uploader("Upload PDFs to build brain", type="pdf", accept_multiple_files=True)
        if st.button("Process & Save 📚", use_container_width=True):
            if uploaded_files:
                with st.spinner("Embedding papers..."):
                    st.session_state.vectorstore = process_documents(uploaded_files, is_append=False)
                st.rerun() 

    if st.button("🚨 Clear My Documents", use_container_width=True):
        if os.path.exists(PERSIST_DIR):
            shutil.rmtree(PERSIST_DIR)
        st.session_state.vectorstore = None
        st.rerun()

    st.divider()
    st.header("🗄️ My Past Debates")
    st.caption("Only you can see these.")
    if len(st.session_state.past_debates) == 0:
        st.caption("No past debates yet.")
    else:
        for i, past_debate in enumerate(st.session_state.past_debates):
            short_topic = past_debate["topic"][:25] + "..." if len(past_debate["topic"]) > 25 else past_debate["topic"]
            if st.button(f"🗓️ {short_topic}", key=f"hist_btn_{past_debate['id']}", use_container_width=True):
                st.session_state.current_view = "history"
                st.session_state.selected_history = past_debate

# ==========================================
# MAIN SCREEN AREA
# ==========================================
if st.session_state.current_view == "new":
    
    st.subheader("1. 🤖 Define Your Debaters")
    st.caption(f"Configuring {num_agents} out of 5 possible agents.")
    agents_config =[]
    
    for i in range(num_agents):
        with st.expander(f"Agent {i+1} Configuration", expanded=(i==0)): 
            col1, col2 = st.columns([1, 2])
            
            with col1:
                selected_model = st.selectbox("Brain (Model)", AVAILABLE_MODELS, key=f"model_sel_{i}", index=1)
                mode = st.radio("Persona Generation",["Manual", "AI Generated"], key=f"mode_{i}")
                
            with col2:
                if mode == "Manual":
                    if f"inst_{i}" not in st.session_state:
                        st.session_state[f"inst_{i}"] = "You are a critical thinker. Argue your point logically."
                    instruction = st.text_area("System Prompt", key=f"inst_{i}", height=100, on_change=auto_save)
                else:
                    instruction = "AI_WILL_GENERATE"
                    st.info("✨ The AI will automatically generate a unique persona based on the topic.")
            
            agents_config.append({"id": i, "mode": mode, "instruction": instruction, "model": selected_model})

    st.subheader("2. 🎯 Enter The Debate Topic")
    if "debate_topic" not in st.session_state:
        st.session_state.debate_topic = ""

    topic = st.text_area(
        "What should the agents debate?", 
        key="debate_topic", 
        placeholder="Example: Act as a panel of experts and debate the economic impact of universal basic income...",
        height=100,
        on_change=auto_save
    )

    # 🟢 Disable the launch button if quota is reached
    can_launch = debates_used < MAX_DEBATES
    
    if st.button("Launch Debate 🚀", type="primary", use_container_width=True, disabled=not can_launch):
        if not topic:
            st.error("Please enter a topic first!")
        elif not api_key:
            st.error("Missing GEMINI_API_KEY in .env file!")
        else:
            with st.status("Initializing Debate Arena...", expanded=True) as status:
                
                # Step 1: Researcher
                st.write("🕵️‍♂️ Researcher Agent is analyzing documents...")
                research_brief = generate_research_brief(topic, st.session_state.vectorstore, global_judge_model)
                
                # Step 2: Finalize Personas
                st.write("🎭 Finalizing Agent Personas...")
                active_agents = []
                for i, agent in enumerate(agents_config):
                    if agent["mode"] == "AI Generated":
                        instr = get_ai_persona(topic, i, agent["model"])
                    else:
                        instr = agent["instruction"]
                    active_agents.append({"instruction": instr, "model": agent["model"]})
                
                status.update(label="Debate in Progress!", state="complete", expanded=False)

            with st.expander("📄 View Lead Researcher's Brief"):
                st.write(research_brief)

            debate_history =[] 
            
            for r in range(num_rounds):
                st.markdown(f"<h3 style='text-align:center; color:gray;'>--- Round {r+1} ---</h3>", unsafe_allow_html=True)
                
                for i, agent_data in enumerate(active_agents):
                    with st.chat_message(f"agent_{i+1}"):
                        with st.spinner(f"Agent {i+1} ({agent_data['model']}) is formulating argument..."):
                            
                            agent_llm = ChatGoogleGenerativeAI(model=agent_data['model'], google_api_key=api_key)
                            
                            # Dynamically decide if we should ask for citations
                            if research_brief == "No external documents provided.":
                                citation_rule = "- Base arguments on your general knowledge. DO NOT invent or hallucinate citations."
                            else:
                                citation_rule = "- Base arguments on the Research Brief and strictly cite sources in brackets."

                            system_prompt = f"""
                            SYSTEM INSTRUCTION: {agent_data['instruction']}
                            SHARED RESEARCH BRIEF: {research_brief}

                            INSTRUCTIONS:
                            - Address the topic: {topic}
                            {citation_rule}
                            - Directly address and rebut the previous speakers.
                            """
                            messages =[SystemMessage(content=system_prompt)]
                            
                            if len(debate_history) == 0:
                                user_prompt = f"The debate is starting. Provide your opening statement on: '{topic}'"
                            else:
                                history_text = "\n".join(debate_history)
                                user_prompt = f"History so far:\n{history_text}\n\nYour turn to rebut or argue."
                                
                            messages.append(HumanMessage(content=user_prompt))
                            
                            response = agent_llm.invoke(messages)
                            answer = parse_gemini_response(response)
                        
                        st.write(f"**Agent {i+1} says:**")
                        st.write(answer)
                        debate_history.append(f"Agent {i+1}: {answer}")

            # Step 3: Judge
            st.divider()
            st.header("⚖️ Final Synthesizer Verdict")
            with st.spinner("Judge is analyzing the debate..."):
                judge_llm = ChatGoogleGenerativeAI(model=global_judge_model, google_api_key=api_key)
                if research_brief == "No external documents provided.":
                    judge_citation_rule = "Rely on the transcript provided. Do not invent citations."
                else:
                    judge_citation_rule = "Strictly include citations from the Research Brief."

                verdict_prompt = f"""
                You are the Final Synthesizer.
                TOPIC: "{topic}"
                RESEARCH BRIEF: {research_brief}
                TRANSCRIPT: {chr(10).join(debate_history)}

                INSTRUCTIONS: Provide a final, balanced verdict. Speak directly to the user. {judge_citation_rule}
                """
                verdict = judge_llm.invoke(verdict_prompt)
                final_answer = parse_gemini_response(verdict)
                
                st.success("Conclusion Reached!")
                st.markdown(final_answer)
                
                # Save debate AND the model used to judge it
                new_record = save_new_debate(topic, debate_history, final_answer, research_brief, global_judge_model)
                st.session_state.current_view = "history"
                st.session_state.selected_history = new_record
                st.rerun()

elif st.session_state.current_view == "history":
    past_data = st.session_state.selected_history
    
    st.button("⬅️ Back to Setup", on_click=lambda: st.session_state.update(current_view="new"))
    
    st.info(f"📅 **Date Recorded:** {past_data['date']}")
    st.subheader(f"📝 Topic: {past_data['topic']}")
    
    with st.expander("🕵️‍♂️ View Original Research Brief"):
        st.write(past_data.get("research_brief", "No brief available."))
        
    st.divider()
    st.markdown("### 📜 Debate Transcript")
    for msg in past_data['history']:
        agent_name, text = msg.split(": ", 1)
        with st.chat_message(agent_name.lower().replace(" ", "_")):
            st.write(f"**{agent_name} says:**")
            st.write(text)
            
    st.divider()
    st.header("⚖️ Final Conclusion")
    st.markdown(past_data['verdict'])
    
    # POST-DEBATE CHAT
    st.divider()
    st.header("💬 Talk to the Synthesizer")
    st.caption("Ask follow-up questions about this debate without restarting it.")

    if "post_chat" not in past_data:
        past_data["post_chat"] =[]

    for chat_msg in past_data["post_chat"]:
        with st.chat_message(chat_msg["role"]):
            st.write(chat_msg["content"])

    if user_question := st.chat_input("Ask a follow-up question..."):
        past_data["post_chat"].append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.write(user_question)
            
        # 🟢 Post-chat now uses the exact same model that was used to judge the debate!
        chat_model = past_data.get("judge_model", AVAILABLE_MODELS[1])
        llm = ChatGoogleGenerativeAI(model=chat_model, google_api_key=api_key)
        
        if past_data.get('research_brief') == "No external documents provided.":
            follow_up_rule = "Answer the user's follow-up questions based on general knowledge and the debate."
        else:
            follow_up_rule = "Answer the user's follow-up questions. Continue citing sources from the brief."
        
        sys_context = f"""You are the Final Synthesizer. You just concluded a debate on '{past_data['topic']}'.
        RESEARCH BRIEF: {past_data.get('research_brief', 'N/A')}
        YOUR FINAL VERDICT: {past_data['verdict']}
        {follow_up_rule}"""
        
        messages =[SystemMessage(content=sys_context)]
        for m in past_data["post_chat"]:
            if m["role"] == "user":
                messages.append(HumanMessage(content=m["content"]))
            else:
                messages.append(AIMessage(content=m["content"]))
                
        with st.chat_message("assistant"):
            with st.spinner(f"Synthesizer ({chat_model}) is thinking..."):
                response = llm.invoke(messages)
                answer = parse_gemini_response(response)
                st.write(answer)
                
        past_data["post_chat"].append({"role": "assistant", "content": answer})
        
        for i, record in enumerate(st.session_state.past_debates):
            if record["id"] == past_data["id"]:
                st.session_state.past_debates[i] = past_data
                break
        with open(HISTORY_FILE, "w") as f:
            json.dump(st.session_state.past_debates, f)
