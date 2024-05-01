import streamlit as st
from retriever import retrieve

from langchain_core.messages import AIMessage, HumanMessage

st.title("Nikon Assistant!!!")
st.subheader("Ask me anything about Nikon!")

if "messages" not in st.session_state:
    st.session_state.messages = [AIMessage(content="Hello, how can I help you?")]  

for message in st.session_state.messages:
    if isinstance(message, AIMessage):
        with st.chat_message("AI",avatar = "🤖"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human",avatar ="🙋‍♀️"):
            st.write(message.content)

prompt = st.chat_input("How may I help you?")

if prompt is not None and prompt != "":
    with st.chat_message("user",avatar = "🙋‍♀️"):
        st.markdown(prompt)
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.spinner('Getting the information for you!!!!'):
        response = retrieve(prompt) 
        with st.chat_message("assistant", avatar = "🤖"):
                st.markdown(response)
        st.session_state.messages.append(AIMessage(content=response))
