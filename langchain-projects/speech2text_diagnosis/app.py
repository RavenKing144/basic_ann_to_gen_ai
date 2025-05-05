import streamlit as st
import requests
import os

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Healthcare Assistant", page_icon="🩺")
st.title("🩺 AI Healthcare Virtual Assistant")

st.markdown("""
Upload your medical voice query (.wav), or type a question below.
The assistant will transcribe, analyze, and offer a preliminary response.
""")

audio_file = st.file_uploader("Upload .wav audio", type=["wav"])
if audio_file and st.button("Transcribe Audio"):
    with st.spinner("Transcribing..."):
        response = requests.post(f"{BACKEND_URL}/transcribe/", files={"file": (audio_file.name, audio_file, "audio/wav")})
        if response.status_code == 200:
            output = response.json()
            transcription = output["transcription"]
            st.success("Transcription complete")
            st.text_area("Transcribed Text", transcription, height=150, key="audio_text")
        else:
            st.error("Error in transcription")
    with st.spinner("Processing with LLM..."):
        st.markdown("### 🤖 Assistant's Response")
        st.write(output.get("response", "No response"))

    with st.spinner("Diagnosing symptoms..."):
        st.markdown("### LLM Differential Diagnosis")
        st.write(output.get("diagnosis", "No output"))