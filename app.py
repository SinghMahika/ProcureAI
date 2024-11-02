# app.py
import streamlit as st

st.cache_data.clear()

# Initialize the Streamlit app title and sidebar
st.set_page_config(page_title="ProcureAI Chatbot", page_icon="🤖")
st.title("ProcureAI Chatbot")

from bot_backend import qna

# Sidebar information and actions
with st.sidebar:
    st.image("ProcureAI.webp", width=200)
    st.header("About")
    st.write("ProcureAI is your trusted companion for procurement, offering clear insights and resources for employees. From basics to complex policies, ProcureAI simplifies procurement for everyone.")

# Initialize or retrieve chat history from session state
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# User input field
user_input = st.text_input("Ask me a question:", "")

# Check for user input
if user_input:
    # Get response from the backend
    # st.write(user_input)
    response = qna(user_input, st.session_state["chat_history"])
    # st.write(response)
    
    # Update chat history
    st.session_state["chat_history"].append({"user": user_input, "ProcureAI": response})
    
    # Display conversation history
    for message in st.session_state["chat_history"]:
        st.markdown(f"**You**: {message['user']}")
        st.markdown(f"**Bot**: {message['ProcureAI']}")

# Optionally clear chat history
if st.button("Clear Chat History"):
    st.session_state["chat_history"] = []
    st.success("Chat history cleared.")
