import streamlit as st
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
import os

# Set Groq API key
os.environ["GROQ_API_KEY"] = "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"

# Initialize session state for memory and history
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory()
if "history" not in st.session_state:
    st.session_state.history = []
if "conversation" not in st.session_state:
    st.session_state.conversation = None

# Custom prompt template for Groq
prompt_template = """You are a helpful AI assistant. Maintain conversation history and provide contextual responses.

History:
{history}

Current conversation:
Human: {input}
AI:"""
prompt = PromptTemplate(
    input_variables=["history", "input"],
    template=prompt_template
)

# Initialize Groq chat model
def init_conversation():
    llm = ChatGroq(
        model_name="mixtral-8x7b-32768",
        temperature=0.5
    )
    st.session_state.conversation = ConversationChain(
        llm=llm,
        prompt=prompt,
        memory=st.session_state.memory
    )

# Save conversation to file
def save_history():
    with open("chat_history.txt", "a") as f:
        for entry in st.session_state.history:
            f.write(f"{entry}\n")

# Load previous conversations
def load_history():
    try:
        with open("chat_history.txt", "r") as f:
            return f.read().split("\n")
    except FileNotFoundError:
        return []

# Streamlit UI
st.title("CHAT-MODEL-BASED-CHATBOT")

# Initialize conversation
if st.session_state.conversation is None:
    init_conversation()

# Display chat history
for message in load_history():
    if message.startswith("Human:"):
        with st.chat_message("user"):
            st.markdown(message.replace("Human: ", ""))
    elif message.startswith("AI:"):
        with st.chat_message("assistant"):
            st.markdown(message.replace("AI: ", ""))

# User input
if prompt := st.chat_input("Type your message..."):
    # Add user message to history
    st.session_state.history.append(f"Human: {prompt}")
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.spinner("Thinking..."):
        response = st.session_state.conversation.predict(input=prompt)
    
    # Add AI response to history
    st.session_state.history.append(f"AI: {response}")
    
    with st.chat_message("assistant"):
        st.markdown(response)
    
    # Save conversation to file
    save_history()