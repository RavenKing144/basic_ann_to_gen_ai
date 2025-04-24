import os
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# Load .env variables
load_dotenv()

# Set environment vars
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# LLM Setup
llm = ChatGroq(
    model="Llama3-8b-8192",
    api_key=os.environ["GROQ_API_KEY"]
)

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

# Embedding setup
def create_vector_embeddings():
    if "vectors" not in st.session_state:
        embeddings = OpenAIEmbeddings()
        loader = PyPDFDirectoryLoader("data")
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        final_documents = text_splitter.split_documents(docs)

        st.session_state.vectors = FAISS.from_documents(
            final_documents,
            embeddings
        )
        st.session_state.documents_loaded = True

# UI Input
st.title("Groq-Powered PDF QA Assistant")
user_prompt = st.text_input("Enter your question", key="question")

# Button to trigger embedding
if st.button("Create Document Embeddings"):
    create_vector_embeddings()
    st.success("✅ Embeddings created!")

# Question answer handling
if user_prompt:
    if "vectors" not in st.session_state:
        st.warning("Please create embeddings first using the button above.")
    else:
        st.write("🤖 Answering your question...")

        document_chain = create_stuff_documents_chain(
            llm=llm,
            prompt=prompt_template
        )

        retriever = st.session_state.vectors.as_retriever()

        rag_chain = (
            {"context": retriever, "input": RunnablePassthrough()} 
            | document_chain
        )

        # Run it
        answer = rag_chain.invoke(user_prompt)


        st.subheader("Answer:")
        st.write(answer)
