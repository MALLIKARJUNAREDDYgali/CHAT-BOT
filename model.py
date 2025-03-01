import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
import os
import uuid
from datetime import datetime
import sqlite3
import json
import requests
import spacy
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import yake

# Download required NLTK data
nltk.download('vader_lexicon', quiet=True)

# Set API keys
os.environ["GROQ_API_KEY"] = "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"
GNEWS_API_KEY = "8257dcbd763ff3e1d8bb015fe0221112"
X_API_KEY = "1895700950368579585Maliarjun14"  # Replace with valid token

# Initialize NLP components
nlp = spacy.load("en_core_web_sm", disable=["ner"]) if spacy.util.is_package("en_core_web_sm") else None
sia = SentimentIntensityAnalyzer()
kw_extractor = yake.KeywordExtractor(lan="en", n=1, dedupLim=0.9, top=5)

# Intent classifier setup
intent_train_data = [
    ("news about", "news"), ("latest updates on", "news"), ("what's happening with", "news"),
    ("hi", "casual"), ("hello", "casual"), ("how are you", "casual"),
    ("what do you think about", "opinion"), ("opinion on", "opinion"),
    ("help me", "help"), ("not working", "help")
]
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform([x[0] for x in intent_train_data])
y_train = [x[1] for x in intent_train_data]
intent_clf = MultinomialNB().fit(X_train, y_train)

# Database setup
def init_db():
    conn = sqlite3.connect('chat_sessions.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS sessions
                 (session_id TEXT PRIMARY KEY,
                  start_time TEXT,
                  session_name TEXT,
                  history TEXT)''')
    conn.commit()
    conn.close()

init_db()

# Session state initialization
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.session_name = "New Chat"
    st.session_state.history = ChatMessageHistory()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}

# Load existing session history
def load_session_history(session_id):
    conn = sqlite3.connect('chat_sessions.db')
    c = conn.cursor()
    c.execute("SELECT history FROM sessions WHERE session_id=?", (session_id,))
    result = c.fetchone()
    history = ChatMessageHistory()
    if result and result[0]:
        try:
            messages = json.loads(result[0])
            for msg in messages:
                if msg["role"] == "human":
                    history.add_user_message(msg["content"])
                elif msg["role"] == "ai":
                    history.add_ai_message(msg["content"])
        except:
            pass
    conn.close()
    return history

# NLP Processing Functions
def process_query(query):
    nlp_data = {"entities": [], "sentiment": {"compound": 0.0}, "intent": "unknown", "keywords": []}
    if nlp:
        doc = nlp(query)
        nlp_data["entities"] = [(ent.text, ent.label_) for ent in doc.ents]
    if sia:
        nlp_data["sentiment"] = sia.polarity_scores(query)
    X = vectorizer.transform([query])
    nlp_data["intent"] = intent_clf.predict(X)[0]
    if kw_extractor:
        keywords = kw_extractor.extract_keywords(query)
        nlp_data["keywords"] = [kw[0] for kw in keywords]
    return nlp_data

def enhance_search(query, nlp_data):
    search_terms = [ent[0] for ent in nlp_data["entities"]] + nlp_data["keywords"]
    search_terms.append(datetime.now().strftime("%Y-%m-%d"))
    return f"{query} {' '.join(search_terms)}"

# Fetch news from GNews API
def fetch_gnews_articles(query, max_results=3):
    url = "https://gnews.io/api/v4/search"
    params = {"q": query, "token": GNEWS_API_KEY, "lang": "en", "max": max_results, "sortby": "publishedAt"}
    try:
        response = requests.get(url, params=params)
        return response.json().get("articles", [])
    except:
        return []

# Fetch recent X posts
def fetch_x_posts(query, max_results=5):
    url = "https://api.twitter.com/2/tweets/search/recent"
    headers = {"Authorization": f"Bearer {X_API_KEY}"}
    params = {
        "query": query,
        "max_results": max_results,
        "tweet.fields": "created_at,attachments",
        "expansions": "attachments.media_keys",
        "media.fields": "url",
    }
    try:
        response = requests.get(url, headers=headers, params=params)
        data = response.json()
        posts = []
        if "data" in data and "includes" in data and "media" in data["includes"]:
            media_map = {m["media_key"]: m["url"] for m in data["includes"]["media"]}
            for tweet in data["data"]:
                if "attachments" in tweet and "media_keys" in tweet["attachments"]:
                    image_url = media_map.get(tweet["attachments"]["media_keys"][0])
                    if image_url:
                        posts.append({
                            "text": tweet["text"],
                            "created_at": tweet["created_at"],
                            "image_url": image_url,
                            "url": f"https://twitter.com/i/status/{tweet['id']}"
                        })
        return posts
    except:
        return []

def needs_realtime_data(query):
    time_keywords = ['yesterday', 'today', 'recent', 'latest', 'current', 'now', 'just']
    topic_keywords = ['match', 'score', 'result', 'news', 'update', 'happen']
    return any(word in query.lower() for word in time_keywords + topic_keywords)

# Prompt template (Updated)
prompt_template = """You are an assistant. Consider these insights:
- User sentiment: {sentiment}
- Detected intent: {intent}
- Key entities: {entities}

Chat history:
{history}
Human: {input}
AI: """
prompt = PromptTemplate(
    input_variables=["history", "input", "sentiment", "intent", "entities"],
    template=prompt_template
)

# Initialize conversation
def init_conversation():
    llm = ChatGroq(model_name="mixtral-8x7b-32768", temperature=0.7)
    chain = prompt | llm
    
    def get_session_history(session_id: str) -> ChatMessageHistory:
        if session_id not in st.session_state.chat_history:
            st.session_state.chat_history[session_id] = load_session_history(session_id)
        return st.session_state.chat_history[session_id]
    
    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history"
    )

# Save session to database
def save_session(session_id, start_time, session_name, history):
    conn = sqlite3.connect('chat_sessions.db')
    c = conn.cursor()
    history_json = json.dumps([{"role": "human" if isinstance(m, HumanMessage) else "ai", 
                              "content": m.content} for m in history.messages])
    c.execute('''INSERT OR REPLACE INTO sessions 
                 VALUES (?, ?, ?, ?)''',
              (session_id, start_time, session_name, history_json))
    conn.commit()
    conn.close()

# Update session name
def update_session_name(session_id, new_name):
    conn = sqlite3.connect('chat_sessions.db')
    c = conn.cursor()
    c.execute("UPDATE sessions SET session_name = ? WHERE session_id = ?", (new_name, session_id))
    conn.commit()
    conn.close()
    if st.session_state.session_id == session_id:
        st.session_state.session_name = new_name

# Delete session
def delete_session(session_id):
    conn = sqlite3.connect('chat_sessions.db')
    c = conn.cursor()
    c.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()
    if session_id in st.session_state.chat_history:
        del st.session_state.chat_history[session_id]

# Get all sessions
def get_all_sessions():
    conn = sqlite3.connect('chat_sessions.db')
    c = conn.cursor()
    c.execute("SELECT session_id, start_time, session_name, history FROM sessions ORDER BY start_time DESC")
    sessions = c.fetchall()
    conn.close()
    return sessions

# Initialize conversation if not exists
if "conversation" not in st.session_state or st.session_state.conversation is None:
    st.session_state.conversation = init_conversation()
    st.session_state.history = load_session_history(st.session_state.session_id)

# Sidebar
with st.sidebar:
    st.header("Chat Sessions")
    if st.button("+ New Chat"):
        save_session(st.session_state.session_id, st.session_state.start_time, 
                    st.session_state.session_name, st.session_state.history)
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.session_name = "New Chat"
        st.session_state.history = ChatMessageHistory()
        st.session_state.conversation = init_conversation()
        st.rerun()

    st.subheader("Current Session")
    st.caption(f"Started: {st.session_state.start_time}")
    new_name = st.text_input("Session Name", value=st.session_state.session_name)
    if new_name != st.session_state.session_name:
        update_session_name(st.session_state.session_id, new_name)

    st.subheader("Previous Sessions")
    for session in get_all_sessions():
        session_id, start_time, session_name, history = session
        session_name = session_name or "Untitled Chat"
        preview = json.loads(history)[0]["content"][:50] + "..." if history and json.loads(history) else "Empty chat"
        
        with st.container():
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button(f"{session_name} ({start_time})", key=f"btn_{session_id}", help=preview):
                    save_session(st.session_state.session_id, st.session_state.start_time, 
                               st.session_state.session_name, st.session_state.history)
                    st.session_state.session_id = session_id
                    st.session_state.start_time = start_time
                    st.session_state.session_name = session_name
                    st.session_state.history = load_session_history(session_id)
                    st.session_state.conversation = init_conversation()
                    st.rerun()
            with col2:
                if st.button("X", key=f"delete_{session_id}"):
                    delete_session(session_id)
                    st.rerun()

# Main chat interface
st.title("MALLI-CHANDU-MODEL")

# Display chat history
for message in st.session_state.history.messages:
    with st.chat_message("user" if isinstance(message, HumanMessage) else "assistant"):
        st.markdown(message.content)

# User input handling
if prompt := st.chat_input("Message Chat..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    
    try:
        with st.spinner("Thinking..."):
            st.session_state.history.add_user_message(prompt)
            nlp_data = process_query(prompt)
            inputs = {
                "input": prompt,
                "sentiment": str(nlp_data["sentiment"]),
                "intent": nlp_data["intent"],
                "entities": str(nlp_data["entities"])
            }
            response = st.session_state.conversation.invoke(
                inputs,
                config={"configurable": {"session_id": st.session_state.session_id}}
            )
            response_text = response.content.strip()
            
            if needs_realtime_data(prompt) or nlp_data["intent"] == "news":
                search_query = enhance_search(prompt, nlp_data)
                gnews_results = fetch_gnews_articles(search_query)
                x_posts = fetch_x_posts(search_query)
                
                if gnews_results:
                    news_section = "\n\n**Latest News Articles:**\n"
                    for idx, article in enumerate(gnews_results, 1):
                        news_section += f"\n{idx}. **[{article.get('title', 'No title')}]({article.get('url', '#')})** " \
                                      f"(Published: {article.get('publishedAt', 'Unknown')})\n" \
                                      f"   - {article.get('description', 'No description')}\n"
                    response_text += news_section
                
                if x_posts:
                    x_section = "\n\n**Latest X Posts:**\n"
                    for idx, post in enumerate(x_posts, 1):
                        text = post["text"][:100] + "..." if len(post["text"]) > 100 else post["text"]
                        x_section += f"\n{idx}. **[{text}]({post['url']})** (Posted: {post['created_at']})\n"
                        if post.get("image_url"):
                            x_section += f"   - ![Image]({post['image_url']})\n"
                    response_text += x_section

            st.session_state.history.add_ai_message(response_text)
            with st.chat_message("assistant"):
                st.markdown(response_text)
            
            save_session(st.session_state.session_id, st.session_state.start_time, 
                        st.session_state.session_name, st.session_state.history)

    except Exception as e:
        st.error(f"Error: {str(e)}")