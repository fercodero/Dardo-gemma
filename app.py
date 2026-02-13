import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Config & Page Setup
st.set_page_config(page_title="DnD Grimoire", page_icon="🐉")

# 2. Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. Load API Key
api_key = st.secrets.get("GROQ_API_KEY", "")
if not api_key:
    st.error("API Key not found! Please add GROQ_API_KEY to your .streamlit/secrets.toml file.")
    st.stop()

# 4. Load Data (Cached)
@st.cache_resource
def load_data():
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

# 5. LLM Setup (single model, no rotation needed)
MODEL_ID = "llama-3.3-70b-versatile"  # Free, powerful, fast on Groq

llm = ChatGroq(
    model=MODEL_ID,
    api_key=api_key,
    temperature=0.3
)

# 6. Sidebar
st.sidebar.title("🐉 Dungeon Master")
st.sidebar.markdown(f"**Model:** `{MODEL_ID}`")
st.sidebar.markdown(f"**Messages:** {len(st.session_state.messages) // 2}")

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# 7. Chat UI
st.title("🐉 Dungeon Master's Assistant")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Ask a rule..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Consulting the Grimoire..."):

            system_prompt = (
                "You are an expert Dungeon Master helper. Use the provided context to answer the question. "
                "If the context is incomplete, use your general knowledge of D&D 5e to fill the gaps, but prioritize the manuals provided. "
                "Keep answers concise. Format spells and stats clearly.\n\n"
                "Context:\n{context}"
            )
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])

            try:
                retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
                combine_docs_chain = create_stuff_documents_chain(llm, prompt)
                retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

                response = retrieval_chain.invoke({"input": user_input})
                answer = response["answer"]
                st.markdown(answer)

                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.rerun()

            except Exception as e:
                st.error(f"Error: {e}")
