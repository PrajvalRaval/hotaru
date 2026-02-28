import streamlit as st
import os
import torch
import gc
import ollama
from hotaru.ui.fragments import render_system_stats

def render_sidebar(ollama_models):
    with st.sidebar:
        st.markdown("# 🏯 Hotaru")
        st.markdown("""
        **WhisperX Transcription & Translation**
        *Transcribe → Align → [Diarize] → Smart Split → Translate → Smooth → Generate SRT.*
        """)
        st.markdown("---")

        with st.expander("🖥️ System Status", expanded=True):
            render_system_stats()
            if st.button("Purge Ollama VRAM", use_container_width=True, key="purge_vram_btn"):
                try:
                    h = st.session_state.get("ollama_host", "http://localhost:11434")
                    client = ollama.Client(host=h)
                    try:
                        loaded = client.ps()
                        for m in loaded.get('models', []):
                            client.generate(model=m['name'], keep_alive=0)
                    except:
                        if "last_model" in st.session_state:
                            client.generate(model=st.session_state.last_model, keep_alive=0)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()
                    st.rerun()
                except: pass

        with st.expander("🗣️ Transcription", expanded=False):
            model_options = ["kotoba-tech/kotoba-whisper-v2.0-faster", "litagin/anime-whisper", "large-v3", "large-v2", "medium", "small"]
            st.selectbox("Whisper Model", model_options, index=0, key="model_size_val")

            # Anime-Whisper specific check
            if st.session_state.model_size_val == "litagin/anime-whisper":
                if not os.path.exists("models/anime-whisper-ct2/model.bin"):

                    st.warning("⚠️ anime-whisper not converted yet.")
                    if st.button("Convert to CTranslate2"):
                        import subprocess
                        st.info("Converting... Please check terminal.")
                        subprocess.run(["python", "scripts/convert_anime_whisper.py"])
                        st.success("Conversion complete! Please restart app.")

            st.toggle("👥 Enable Speaker ID (Diarization)", value=False, key="enable_diarization_toggle")
            if st.session_state.enable_diarization_toggle:
                st.text_input("🔑 HF Token", type="password", value=os.getenv("HF_TOKEN", ""), key="hf_token_input")
            
            c1, c2 = st.columns(2)
            c1.number_input("📏 Max Width", min_value=1, max_value=100, value=42, key="max_line_width_input")
            c2.number_input("📚 Max Lines", min_value=1, max_value=5, value=2, key="max_line_count_input")
            
            st.text_input("📐 Alignment Model", value="jonatasgrosman/wav2vec2-large-xlsr-53-japanese", key="align_model_input")
            st.number_input("✂️ Whisper Chunk Size (s)", min_value=1, max_value=60, value=30, key="whisper_chunk_size_input")
            st.slider("⏱️ Timing Offset (s)", -2.0, 2.0, 0.0, 0.05, key="timing_offset_slider")

        with st.expander("🌎 Translation", expanded=False):
            st.text_input("🏠 Ollama Host", value="http://localhost:11434", key="ollama_host")
            st.selectbox("Target LLM", ollama_models, index=0 if ollama_models else None, key="last_model")
