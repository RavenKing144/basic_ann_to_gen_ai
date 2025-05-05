import streamlit as st

from langchain_groq import ChatGroq

from langchain.chains import LLMChain, LLMMathChain
from langchain.prompts import PromptTemplate
from langchain.agents.agent_types import AgentType
from langchain.agents import initialize_agent
from langchain.tools import Tool
from langchain.callbacks import StreamlitCallbackHandler

from langchain_community.utilities import WikipediaAPIWrapper


st.set_page_config(
    layout="wide",
    page_title="Text to Math problem solver"
)
st.title("Text to Math problem solver with Google:Gemma2")

api_key = st.sidebar.text_input(
    label="Groq API Key",
    type="password"
)

if not api_key:
    st.info("NO API KEY PROVIDED")
    st.stop()
    
llm = ChatGroq(
    api_key=api_key,
    model="Gemma2-9b-It"
)

wiki_wrapper = WikipediaAPIWrapper()
wiki_tool = Tool(
    name = "Wiki",
    func=wiki_wrapper.run,
    description="Useful for when you need to solve math problems. Input should be a math problem."
)

math_chain = LLMMathChain.from_llm(llm)

calculator = Tool(
    name="Calculator",
    func=math_chain.run,
    description="Useful for when you need to solve math problems. Input should be a math problem."
)

prompt = PromptTemplate(
    input_variables=["question"],
    template="Question: {question}"
)

chain = LLMChain(
    llm=llm,
    prompt=prompt
)

reasoning_tool = Tool(
    name="Reasoning",
    func=chain.run,
    description="Useful for when you need to reason about a math problem. Input should be a math problem."
)
tools = [wiki_tool, calculator, reasoning_tool]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that follows instructions extremely well. Make sure you follow them strictly. If you do not know the answer to a question, please say that you do not know or you will not be able to answer it."
        }
    ]
    
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])


def generateResponse(user_question):
    response = agent.invoke(
        {
            "question": user_question
        }
    )
    return response


user_question = st.text_input("Ask a question about math:")

if st.button("Ask"):
    if user_question:
        with st.spinner("Generating response..."):
            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": user_question
                }
            )
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = agent.run(
                st.session_state["messages"],
                callbacks = [st_cb]
            )
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response
                }
            )
            st.write(response)
            