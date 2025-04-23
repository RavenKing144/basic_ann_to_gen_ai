import os
import sys
import openai
import streamlit as st

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

## langsmith tracking

os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")    
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")  
os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING")  


  
## Prompt Template
prompt = ChatPromptTemplate(
    [
        (
            "system", "You are a helpful assistant that helps people resolve queries. Please response to the user queries. Please do not answer anything else. Only answer the user"
        ),
        (
            "user",
            "Question:{question}"
        )
    ]
)


def generate(question: str, api_key: str, temperature: float, max_tokens: int, llm) -> str:
    openai.api_key = api_key
    llm = ChatOpenAI(
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


st.title("Enchanced Q&A Chatbot with OpenAI and LangChain")
st.sidebar.title("Settings")
api_key = st.sidebar.text_input(
    "Enter your OpenAI API key", type="password"
)
llm = st.sidebar.selectbox("Select openai model", ("gpt-3.5-turbo", "gpt-4o", "gpt-4", "gpt-4-turbo",))
max_token = st.sidebar.slider("Max tokens", min_value=100, max_value=1000, value=150)
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.5)

st.write("Ask any question you want to ask the chatbot!")
user_input = st.text_input("Your question:")

if user_input and api_key:
    response = generate(user_input, api_key, temperature, max_token, llm)
    st.write(response)
else:
    st.write("Please enter your API key and question.")
