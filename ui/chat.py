import streamlit as st
from pydantic import BaseModel

class Chat(BaseModel):
    # Initialize chat history
    def model_post_init(self, __context=None):
        st.title("SGGW chat bot!")
        if "messages" not in st.session_state:
            st.session_state.messages = []

    def display_history(self):
        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    def chat(self):

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
                response = st.write('Hi')
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})


    