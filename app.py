import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from google.api_core.exceptions import ResourceExhausted

# 1. Config & Page Setup
st.set_page_config(page_title="DnD Grimoire", page_icon="🐉")

# --- ROTATION CONFIGURATION ---
MODEL_ROSTER = [
    {
        "id": "gemini-2.5-flash", 
        "name": "⚡ Gemini 2.0 Flash ", 
        "limit": 19  # Tier 1: Requests 0-19
    },
    {
        "id": "gemini-2.5-flash-lite", 
        "name": "🚀 Gemini 2.5 Flash-Lite", 
        "limit": 19  # Tier 2: Requests 20-38
    },
    {
        "id": "gemini-2.0-flash", # [Correction] '3-flash-preview' might be unstable, using 2.0 or 1.5 as fallback is safer, but keeping your logic.
        "name": "🧪 Gemini 3 Flash Preview", 
        "limit": 19  # Tier 3: Requests 39-57
    },
]

# 2. Session State & Counter
if "request_count" not in st.session_state:
    st.session_state.request_count = 0

if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. Helper: Select Model Based on Count
def get_active_model_config():
    count = st.session_state.request_count
    cumulative_limit = 0
    
    for model in MODEL_ROSTER:
        cumulative_limit += model["limit"]
        if count < cumulative_limit:
            return model, cumulative_limit
            
    # Fallback to the last model if all limits exceeded
    return MODEL_ROSTER[-1], cumulative_limit

# 4. Load API Key
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
else:
    api_key = st.secrets.get("GEMINI_API_KEY", "") 

if not api_key:
    st.error("API Key not found! Please check your .streamlit/secrets.toml file.")
    st.stop()

# 5. Load Data (Cached)
@st.cache_resource
def load_data():
    # MUST MATCH ingest.py
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    vectorstore = FAISS.load_local(
        "dnd_index", 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    return vectorstore

try:
    vectorstore = load_data()
except Exception as e:
    st.error(f"Error loading index: {e}")
    st.stop()

# --- SIDEBAR DASHBOARD ---
active_model, tier_limit = get_active_model_config()
st.sidebar.title("🐉 Dungeon Master")
st.sidebar.markdown(f"**Active Brain:**\n`{active_model['name']}`")
st.sidebar.progress(min(1.0, st.session_state.request_count / tier_limit) if tier_limit > 0 else 0)
st.sidebar.write(f"Global Requests: **{st.session_state.request_count}**")

if st.sidebar.button("Reset Counter"):
    st.session_state.request_count = 0
    st.rerun()

# 6. Chat Logic
st.title("🐉 Dungeon Master's Assistant")

# Display History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle Input
if user_input := st.chat_input("Ask a rule..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner(f"Consulting {active_model['name']}..."):
            
            # --- DYNAMIC LLM INITIALIZATION ---
            # We initialize the LLM here to ensure we use the CURRENT active model
            llm = ChatGoogleGenerativeAI(
                model=active_model["id"], 
                google_api_key=api_key,
                temperature=0.3
            )

            # Prompt Setup
            system_prompt = (
                "You are an expert Dungeon Master helper. Use the provided context to answer the question. "
                "Use the provided context to answer. If the context is incomplete, use your general knowledge of D&D 5e to fill the gaps, but prioritize the manuals provided'. "
                "Keep answers concise. Format spells and stats clearly.\n\n"
                "Context:\n{context}"
            )
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])

            # Chain Setup
            try:
                retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
                combine_docs_chain = create_stuff_documents_chain(llm, prompt)
                retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
                
                response = retrieval_chain.invoke({"input": user_input})
                answer = response["answer"]
                st.markdown(answer)
                
                # Update History & Counter
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.session_state.request_count += 1
                st.rerun() # Refresh to update sidebar immediately

            except ResourceExhausted:
                st.warning(f"⚠️ {active_model['name']} hit its limit! Rotating to next model...")
                # Force jump to next tier
                st.session_state.request_count = tier_limit + 1
                st.rerun()

            except Exception as e:

                st.error(f"Error: {e}")

