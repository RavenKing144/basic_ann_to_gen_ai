import os
import sys
import streamlit as st

from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

## langsmith tracking

os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")    
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")  
os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING")  


  
## Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that helps people resolve queries. Please response to the user queries. Please do not answer anything else. Only answer the user"
        ),
        (
            "user",
            "Question:{question}"
        )
    ]
)


def generate(question: str, api_key: str, temperature: float, max_tokens: int, llm) -> str:
    llm = OllamaLLM(
        model=llm,
        temperature=temperature,
        max_tokens=max_tokens
    )
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke(
        {
            "question": question
        }
    )
    return answer


st.title("Enchanced Q&A Chatbot with Ollama and LangChain")
st.sidebar.title("Settings")
api_key = st.sidebar.text_input(
    "Enter your Ollama API key", type="password"
)
llm = st.sidebar.selectbox("Select Ollama model",
                           ("gemma2:latest", "phi3:latest", "tinyllama:latest", "gemma:2b"))
max_token = st.sidebar.slider("Max tokens", min_value=100, max_value=1000, value=150)
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.5)

st.write("Ask any question you want to ask the chatbot!")
user_input = st.text_input("Your question:")

if user_input:
    response = generate(user_input, api_key, temperature, max_token, llm)
    st.write(response)
else:
    st.write("Please enter your question.")
