# src/app.py
import streamlit as st
from chatlogic import ChatLogic

st.title("I-sektionen Chatbot (RAG)")

# --- Setup ChatLogic with secrets
logic = ChatLogic(
    openai_key=st.secrets["OPENAI_API_KEY"],
    pinecone_key=st.secrets["PINECONE_API_KEY"],
    index_name="isektionen-rag-1536"
)

# --- Streamlit session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"
    logic.model = st.session_state["openai_model"]

# --- Render history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Handle new prompt
if prompt := st.chat_input("Vad vill du veta om dokumenten?"):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Show user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response (stream)
    with st.chat_message("assistant"):
        stream = logic.generate_response(st.session_state.messages, prompt)
        response = st.write_stream(stream)

    # Save assistant reply in session
    st.session_state.messages.append({"role": "assistant", "content": response})