import streamlit as st
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
import os
import uuid
from datetime import datetime
import sqlite3
import json
from tavily import TavilyClient

# Set Groq API key
os.environ["GROQ_API_KEY"] = "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"

# Database setup
def init_db():
    conn = sqlite3.connect('chat_sessions.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS sessions
                 (session_id TEXT PRIMARY KEY,
                  start_time TEXT,
                  history TEXT)''')
    conn.commit()
    conn.close()

init_db()

# Session state initialization
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.history = []

if "conversation" not in st.session_state:
    st.session_state.conversation = None

# Custom prompt template
prompt_template = """You are ChatGPT. Follow these rules:
1. Provide clear, concise answers
2. Use Markdown formatting when appropriate
3. Admit when you don't know something

Chat history:
{history}
Human: {input}
AI:"""
prompt = PromptTemplate(
    input_variables=["history", "input"], 
    template=prompt_template
)

# Initialize Groq chat
def init_conversation():
    llm = ChatGroq(
        model_name="mixtral-8x7b-32768",
        temperature=0.7
    )
    memory = ConversationBufferMemory()
    
    # Load history from database
    conn = sqlite3.connect('chat_sessions.db')
    c = conn.cursor()
    c.execute("SELECT history FROM sessions WHERE session_id=?", 
             (st.session_state.session_id,))
    result = c.fetchone()
    
    if result:
        history = json.loads(result[0])
        for entry in history:
            if entry["role"] == "human":
                memory.chat_memory.add_user_message(entry["content"])
            else:
                memory.chat_memory.add_ai_message(entry["content"])
    
    conn.close()
    
    return ConversationChain(
        llm=llm,
        prompt=prompt,
        memory=memory
    )

# Save session to database
def save_session():
    conn = sqlite3.connect('chat_sessions.db')
    c = conn.cursor()
    history_json = json.dumps(st.session_state.history)
    c.execute('''INSERT OR REPLACE INTO sessions 
                 VALUES (?, ?, ?)''',
              (st.session_state.session_id,
               st.session_state.start_time,
               history_json))
    conn.commit()
    conn.close()

# Get all sessions
def get_all_sessions():
    conn = sqlite3.connect('chat_sessions.db')
    c = conn.cursor()
    c.execute("SELECT * FROM sessions ORDER BY start_time DESC")
    sessions = c.fetchall()
    conn.close()
    return sessions

# Initialize Tavily Client
def init_tavily_client():
    return TavilyClient(api_key="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")

# Improved search function with time-sensitive queries
def perform_tavily_search(query):
    tavily_client = init_tavily_client()
    try:
        # Add time context to the query
        dated_query = f"{query} - current - {datetime.now().strftime('%Y-%m-%d')}"
        response = tavily_client.search(
            query=dated_query,
            search_depth="advanced",
            max_results=3,
            include_answer=True
        )
        return response
    except Exception as e:
        return f"Error performing search: {str(e)}"

# Function to detect if query needs real-time data
def needs_realtime_data(query):
    time_keywords = ['yesterday', 'today', 'recent', 'latest', 'current', 'now', 'just']
    topic_keywords = ['match', 'score', 'result', 'news', 'update', 'happen']
    return any(word in query.lower() for word in time_keywords + topic_keywords)

# ChatGPT-like sidebar with New Chat button
with st.sidebar:
    st.header("ChatGPT")
    
    # New Chat button
    if st.button("+ New Chat", use_container_width=True):
        if st.session_state.history:
            save_session()
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.history = []
        st.session_state.conversation = None
        st.rerun()
    
    st.divider()
    
    st.subheader("Current Session")
    st.caption(f"Started: {st.session_state.start_time}")
    st.caption(f"ID: {st.session_state.session_id[:8]}...")
    
    st.divider()
    
    st.subheader("Previous Sessions")
    sessions = get_all_sessions()
    for session in sessions[:10]:
        session_id, start_time, history = session
        try:
            preview = json.loads(history)[0]["content"][:50] + "..." if history else "New chat"
        except:
            preview = "New chat"
            
        if st.button(
            f"{start_time} - {session_id[:8]}...",
            help=preview,
            use_container_width=True
        ):
            st.session_state.session_id = session_id
            st.session_state.start_time = start_time
            st.session_state.history = json.loads(history)
            st.session_state.conversation = init_conversation()
            st.rerun()

# Main chat interface
st.title("MALLI-CHANDU-MODEL")

# Initialize conversation
if st.session_state.conversation is None:
    st.session_state.conversation = init_conversation()

# Display chat history
for message in st.session_state.history:
    with st.chat_message("user" if message["role"] == "human" else "assistant"):
        st.markdown(message["content"])

# User input handling with improved real-time detection
if prompt := st.chat_input("Message Chat..."):
    st.session_state.history.append({"role": "human", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    try:
        with st.spinner("Thinking..."):
            # Generate initial response
            response = st.session_state.conversation.predict(input=prompt)
            
            # Check if real-time data needed
            if needs_realtime_data(prompt):
                search_results = perform_tavily_search(prompt)
                
                if isinstance(search_results, dict):
                    # Prepend direct answer if available
                    if search_results.get('answer'):
                        response = f"**Latest Update:**\n{search_results['answer']}\n\n{response}"
                    # Add sources section
                    if search_results.get('results'):
                        response += "\n\n**Verified Sources:**\n"
                        for idx, result in enumerate(search_results['results'][:3], 1):
                            response += f"{idx}. [{result['title']}]({result['url']})\n"
                    response += "\n\n*Updated with real-time information*"
                else:
                    response += f"\n\n{search_results}"

    except Exception as e:
        response = f"Error: {str(e)}"
        st.error(response)
    
    st.session_state.history.append({"role": "ai", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
    
    save_session()