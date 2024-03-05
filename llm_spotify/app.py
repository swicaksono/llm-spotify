import streamlit as st
from llm_spotify.rag import answer_with_rag

st.title("Spotify's LLM RAG Model Question Answering")

user_input = st.text_input("Ask a question:", "")
if st.button('Answer'):
    with st.spinner('Generating answer...'):
        answer, documents = answer_with_rag(question=user_input)
        st.write(answer)