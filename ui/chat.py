import os
import streamlit as st

from rag.pipeline import Pipeline

class Chat():
    def __init__(self):
        st.title("RAG App")

        # Initialize Pipeline
        self.pipe = Pipeline()

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

    def display_history(self):
        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    def start(self):
        self.display_history()
        # Accept user input
        if prompt := st.chat_input("What is up?"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                message, metadata = self.pipe.generate_response(prompt)
                if message:
                    for data in metadata:
                        message += f"\n\nDocument: {data.get('source', 'No source available')}, page: {data.get('page', 'N/A')}"
                st.write(message)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": message})