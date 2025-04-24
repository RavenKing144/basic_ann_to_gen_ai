import os
import streamlit as st
from dotenv import load_dotenv

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_groq import ChatGroq

from langchain_huggingface import HuggingFaceEmbeddings




# Load .env variables
load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

embeddings = HuggingFaceEmbeddings(
    model_name = "all-MiniLM-L6-v2"
)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Prompt template
prompt_template = ChatPromptTemplate.from_template(
    """
    Answer the question below using the context provided. If the answer is not contained in the context, say "I don't know". 
    Please answer in the style of a friendly and helpful assistant. 
    <context>
    {context}
    </context>
    Question: {input}
    """
)

st.title("Conversational RAG with PDF uploads and chat history")

st.write("Enter Groq Api key")
groq_api_key = st.text_input("Groq Api key", type="password")

if not True:
    st.write("Please enter a Groq Api key")
else:
    session_id = st.text_input("Session ID", value="default")
    try:
        llm = ChatGroq(
            model="Llama3-8b-8192",
            groq_api_key=groq_api_key
        )
    except:
        st.write("Wrong API key. Please enter a valid key.")
        
    if "store" not in st.session_state:
        st.session_state.store = {}
        

    st.write("Upload PDF")
    uploaded_files = st.file_uploader(
        "Choose a file",
        type="pdf",
        accept_multiple_files=False
    )

    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            temp_pdf = "./temp.pdf"
            with open(temp_pdf, "wb") as f:
                f.write(uploaded_files.read())
                file_name = uploaded_files.name

            loader = PyPDFLoader(temp_pdf).load()
            documents = loader  # since you're loading from a single file

            retriever = Chroma.from_documents(
                text_splitter.split_documents(documents),
                embeddings
            ).as_retriever()
        
        contextualize_q_system_prompt = (
            """Given a chat history and the latest question which might reference context in the chat history, 
            formulate a standalone question which can be understood, without the chat history.
            Do NOT answer the question, just reformulate it if needed and otherwise return it as is."""
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    contextualize_q_system_prompt
                ),
                MessagesPlaceholder(
                    "chat_history"
                ),
                (
                    "human",
                    "{input}"
                )
            ]
        )
        
        hitory_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        contextualize_q_prompt
        )
        
        system_prompt = (
            """You are a helpful assistant. 
            You are given a chat history and the latest question which might reference context in the chat history.
            You will answer the question without the chat history. 
            Do NOT answer the question, just reformulate it if needed and otherwise return it as is.
            <context>
            {context}
            </context>
            """
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    system_prompt
                ),
                MessagesPlaceholder(
                    "chat_history"
                ),
                (
                    "human",
                    "{input}"
                )
            ]
        )
        
        question_answer_chain = create_stuff_documents_chain(
            llm,
            qa_prompt
        )
        rag_chain = create_retrieval_chain(
            hitory_aware_retriever,
            question_answer_chain
        )
        
        def get_session_history(session_id):
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            
            return st.session_state.store[session_id]
        
        
        conversation_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",           # ✅ corresponds to your input dict key
            history_messages_key="chat_history"   # ✅ correct param name
        )
        
        user_input = st.text_input("Ask a question", key="input")
        if user_input:
            session_history = get_session_history(session_id=session_id)
            response = conversation_rag_chain.invoke(
                {"input": user_input},
                config={
                    "configurable": {
                        "session_id": session_id
                    }
                }
            )

            # st.write(st.session_state.store)
            st.success(response["answer"])
            # st.write("History: ", session_history.messages)
        
        
        
        
