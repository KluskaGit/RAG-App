import os
import streamlit as st
import ollama
from openai import OpenAI
from dotenv import load_dotenv

from rag.pipeline import Pipeline

class Chat():
    def __init__(self):
        load_dotenv()
        st.title("RAG App")
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
                pipe = Pipeline(
                    embedding_model_name="google/embeddinggemma-300m",
                    api_key=os.environ["HF_TOKEN"]
                )
                pipe.save_data('data')
                context, metadata = pipe.generate_response(query=prompt)
                print(context)
                system_prompt = f"""
                    You are a helpful chatbot.
                    Use only the following pieces of context to answer the question. Don't make up any new information:
                    {context}
                """
                client = OpenAI(
                    base_url="https://router.huggingface.co/v1",
                    api_key=os.environ["HF_TOKEN"],
                )
                
                response = client.chat.completions.create(
                    model="openai/gpt-oss-20b:nebius",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Answer the question based on the context: {prompt}"},
                    ]
                )
                message = response.choices[0].message.content
                st.write(message, metadata)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": message})