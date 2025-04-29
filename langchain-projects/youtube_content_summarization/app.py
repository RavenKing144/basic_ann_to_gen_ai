import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import validators
import traceback
import ssl
import urllib.request

ssl._create_default_https_context = ssl._create_unverified_context

from langchain_groq import ChatGroq

from langchain_core.documents import Document

from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain

from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
import re


def get_youtube_video_id(url):
    # Extracts video ID from typical YouTube URL formats
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    return match.group(1) if match else None




st.set_page_config(
    page_title = "Text Summarization from Youtube or Website Link"
)
st.title("Text Summarization from Youtube or Website Link")
st.subheader("Summarize URL")

with st.sidebar:
    api_key = st.text_input(
        type="password",
        value="",
        label="Enter your OpenAI API key",
        
    )
    
url = st.text_input(
    label="Enter URL",
    placeholder="Enter URL",
    value="",
    label_visibility="collapsed"
)
initial_prompt = PromptTemplate(
    template="Write a concise summary of the following text:\n\n{text}\n",
    input_variables=["text"]
)

refine_prompt = PromptTemplate(
    template="Your existing summary is:\n{existing_answer}\n\n"
             "Refine the summary with the following text:\n\n{text}\n",
    input_variables=["existing_answer", "text"]
)

if st.button("Summarize"):
    if not api_key.strip() or not url.strip():
        st.error("Please enter both API key and URL")
    else:
        if not validators.url(url):
            st.error("Invalid URL")
        else:
            try:
                with st.spinner("Summarizing..."):
                    ## loading data
                    if "youtube" in url.lower():
                        video_id = get_youtube_video_id(url)
                        if not video_id:
                            st.error("Invalid YouTube URL format.")
                            st.stop()
                        try:
                            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
                            full_transcript_text = " ".join([entry['text'] for entry in transcript_list])
                            docs = [Document(page_content=full_transcript_text)]
                        except Exception as e:
                            st.error(f"Failed to fetch YouTube transcript: {e}")
                            st.stop()
                    else:
                        loader = UnstructuredURLLoader(
                            urls=[url],
                            ssl_verify=False
                        )
                        docs = loader.load()
                    ## summarizing

                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200
                    ).split_documents(docs)
                    chain = load_summarize_chain(
                        llm=ChatGroq(
                            groq_api_key=api_key,
                            model_name="Llama3-8b-8192"
                        ),
                        chain_type="refine",
                        verbose=False
                    )
                    summary = chain.run(text_splitter)
                    st.success(summary)
            except Exception as e:
                tb_str = ''.join(traceback.format_exception(None, e, e.__traceback__))
                st.error(f"Exception occurred:\n\n{tb_str}")
                            