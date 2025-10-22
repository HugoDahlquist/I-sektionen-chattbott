# src/app.py
import streamlit as st
from chatlogic import ChatLogic

#source venv/bin/activate

st.title("I-sektionen Chattbott med (RAG)")
st.markdown("""
Denna chattbot är tränad på dokument från I-sektionen och kan hjälpa dig med frågor relaterade till kurserna TDEI76 och TKMJ51. Den använder OpenAI för att generera svar baserat på innehållet i dokumenten. Dokumenten är diverse föreläsningsanteckningar, labbinstruktioner och artiklar kopplade till kurserna. 
            Den har bara tillgång till information som finns i dessa dokument, så om du frågar om något som inte täcks där, kommer den att meddela att det inte finns i materialet istället för att hitta på ett svar. Du kan just nu bara ställa frågor om en kurs i taget, välj kursen i dropdown-menyn ovanför chattfönstret.
""")
#change the url and icon
st.set_page_config(page_title="I-sektionen Chatbot", page_icon="💚")

# --- Setup ChatLogic with secrets
# app.py
logic = ChatLogic(
    openai_key=st.secrets["OPENAI_API_KEY"],
    pinecone_key=st.secrets["PINECONE_API_KEY"],
    index_name="isektionen-rag-1536"
)

#the user can select the course from a dropdown menu
course = st.selectbox(
    "Välj kurs:",
    ("TDEI76", "TDEI75", "TKMJ51"))

# --- Streamlit session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []


# --- Render history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Handle new prompt
if prompt := st.chat_input("Vad vill du veta om kursen?"):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Show user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response (stream)
    with st.chat_message("assistant"):
        stream = logic.generate_response(st.session_state.messages, prompt, course)
        response = st.write_stream(stream)

    # Save assistant reply in session
    st.session_state.messages.append({"role": "assistant", "content": response})