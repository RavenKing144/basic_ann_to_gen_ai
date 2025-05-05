from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

    
import os
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PubMedLoader

from openai import OpenAI


import openai
import tempfile

import ssl
import urllib.request

ssl._create_default_https_context = ssl._create_unverified_context


app = FastAPI(title="AI Healthcare Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    text: str


openai_api_key = "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"
client = OpenAI(api_key=openai_api_key)

def process_query(user_query: str) -> str:
    # db = FAISS.load_local("retrieval/pubmed_index", OpenAIEmbeddings(openai_api_key=openai_api_key), allow_dangerous_deserialization=True)
    llm = ChatOpenAI(model_name="gpt-4", temperature=0.2, openai_api_key=openai_api_key)

    prompt = PromptTemplate(
        input_variables=["question"],
        template="You are a helpful healthcare assistant. Answer the following medical query using reliable sources.\n\nQuery: {question}"
    )

    

    formatted_prompt = prompt.format(question=user_query)

    # Run LLM (use invoke instead of predict)
    response = llm.invoke(formatted_prompt)
    return response

def llm_diagnose(symptom_text):
    loader = PubMedLoader(symptom_text)  # or use UMLS/SNOMED terms
    docs = loader.load()

    # Step 2: Embed & store
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = FAISS.from_documents(docs, embeddings)

    # Step 3: Build QA chain
    retriever = db.as_retriever()
    llm = ChatOpenAI(model_name="gpt-4", temperature=0, openai_api_key=openai_api_key)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    

    prompt = PromptTemplate(
        input_variables=["question", "context"],
        template="""
    The patient describes the following symptoms: "{question}".
    Based on medical knowledge and the given context {context}, suggest the top 3 possible conditions.
    Reply in simple language.
    """
    )
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(),
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt}
    )
    return chain.run(symptom_text)

def diagnose(text):
    llm_diagnosis = llm_diagnose(text)
    return {
        "llm_diagnosis": llm_diagnosis
    }
    
    

def transcribe_audio(audio_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        with open(tmp.name, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                ).text

    return transcript

@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    if not file.filename.endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files supported")
    audio_bytes = await file.read()
    text = transcribe_audio(audio_bytes)
    # response = process_query(text)
    diagnosis = diagnose(text)
    print(diagnosis)
    return {
        # "response": response.content, 
        "transcription": text,
        "diagnosis": diagnosis["llm_diagnosis"]
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
