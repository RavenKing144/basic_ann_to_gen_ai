import os
import streamlit as st

from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_groq import ChatGroq
from langchain.agents import create_openai_tools_agent, AgentExecutor, initialize_agent, AgentType
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain.callbacks import StreamlitCallbackHandler

api_wrapper = WikipediaAPIWrapper(
    top_k=10,
    doc_content_chars_max=1000
)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)


arxiv_wrapper = ArxivAPIWrapper(
    top_k_results=10,
    doc_content_chars_max=1000
)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)

search = DuckDuckGoSearchRun(
    name="Search"
)

st.title("Langchain - Chat with the web search")

st.sidebar.title("Settings")
groq_api_key = st.sidebar.text_input("Groq Api key", type="password")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role":"assistant",
            "content": "Welcome to Langchain, you can ask me any question about the web!"
        }
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


if prompt:= st.chat_input(
    placeholder="Ask me anything..."
):
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )
    st.chat_message("user").write(prompt)
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model="Llama3-8b-8192",
        streaming=True,
    )
    
    tools = [search, arxiv_tool, wiki_tool]
    
    search_agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        handling_parsing_errors = True
    )
    
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(
            st.container(),
            expand_new_thoughts=True
        )
        response = search_agent.run(
            st.session_state.messages, callbacks=[st_cb])
        
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response
            }
        )
        st.write(response)
