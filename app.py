import warnings
import logging
import os
import atexit
import threading
import time
from datetime import datetime
from collections import deque

import torch
import streamlit as st
import ollama
from dotenv import load_dotenv
from streamlit.runtime.scriptrunner_utils.script_run_context import add_script_run_ctx

# Internal Imports
from hotaru.common.constants import UPLOAD_DIR, LOG_DIR, OUTPUT_DIR
from hotaru.common.logger import setup_logging
from hotaru.common.state import load_task_state, save_task_state, remove_task
from hotaru.ui.fragments import render_log_tray
from hotaru.ui.sidebar import render_sidebar
from hotaru.app_logic import run_engine_thread

# Setup
warnings.filterwarnings("ignore", message=".*torchcodec.*")
for d in [LOG_DIR, UPLOAD_DIR, OUTPUT_DIR]: os.makedirs(d, exist_ok=True)
st.set_page_config(page_title="Hotaru - Transcription & Translation", layout="wide", page_icon="🏯", initial_sidebar_state="expanded")

# State
if 'GLOBAL_LOGS' not in st.session_state: st.session_state.GLOBAL_LOGS = deque(maxlen=1000)
if 'tasks' not in st.session_state: st.session_state.tasks = {}; load_task_state()
if 'SHUTDOWN_EVENT' not in st.session_state: st.session_state.SHUTDOWN_EVENT = threading.Event()

setup_logging()
load_dotenv()

# --- CSS ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    [data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; min-width: 336px !important; }
    .main .block-container { padding: 1rem !important; }
    div.stApp section[data-testid="stMain"] div[data-testid="stExpander"] {
        position: fixed !important; bottom: 0 !important; right: 0 !important; left: 336px !important;
        z-index: 10000 !important; background-color: #161b22 !important; border-top: 1px solid #30363d !important;
    }
    .log-container { background-color: #0d1117; color: #8b949e; font-family: 'SF Mono', monospace; height: 200px; overflow-y: scroll; font-size: 0.7rem; }
    .log-table { width: 100%; border-collapse: collapse; }
    .log-table td { padding: 2px 10px; border-bottom: 1px solid #21262d; white-space: nowrap; }
</style>
""", unsafe_allow_html=True)

# Callback for Uploads
def handle_upload():
    if st.session_state.uploader_key:
        for uploaded_file in st.session_state.uploader_key:
            if uploaded_file.name not in st.session_state.tasks:
                path = os.path.join(UPLOAD_DIR, uploaded_file.name)
                with open(path, "wb") as f: f.write(uploaded_file.getbuffer())
                st.session_state.tasks[uploaded_file.name] = {
                    "status": "Ready", "progress": 0, "stage": "Waiting", "file_path": path, "added": time.time()
                }
        save_task_state(st.session_state.tasks)

# Sidebar
@st.cache_data(ttl=10)
def get_ollama_models(host):
    try:
        client = ollama.Client(host=host)
        return [m.get('model', 'Unknown') for m in client.list().get('models', [])]
    except: return []

models = get_ollama_models(st.session_state.get("ollama_host", "http://localhost:11434"))
render_sidebar(models)

# --- INDEPENDENT TASK CARD FRAGMENT ---
@st.fragment(run_every="2s")
def render_task_card(name):
    """Truly independent fragment for a single task. Only this card reruns."""
    if name not in st.session_state.tasks: return
    task = st.session_state.tasks[name]
    
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([4, 2, 2, 1])
        c1.markdown(f"**{name}**")
        c1.caption(task["stage"])
        
        if task["status"] == "Processing":
            c2.progress(task["progress"] / 100.0)
            if c3.button("⏸ Pause", key=f"p_{name}", use_container_width=True):
                task["status"] = "Paused"; save_task_state(st.session_state.tasks); st.rerun(scope="fragment")
        elif task["status"] == "Completed":
            c2.markdown("✅ Complete")
            try:
                with open(task["output_path"], "rb") as f:
                    c3.download_button("💾 Download", f, file_name=task["output_name"], key=f"d_{name}", use_container_width=True)
            except: c3.error("SRT Missing")
        else:
            c2.markdown(f"Status: {task['status']}")
            if c3.button("▶ Start", key=f"s_{name}", use_container_width=True):
                task["status"] = "Processing"
                task["stage"] = "🎬 Initializing..."
                task["progress"] = 5
                save_task_state(st.session_state.tasks)
                
                config = {
                    "model_size": st.session_state.model_size_val,
                    "ollama_host": st.session_state.ollama_host,
                    "ollama_model": st.session_state.last_model,
                    "timing_offset": st.session_state.timing_offset_slider,
                    "max_width": st.session_state.max_line_width_input,
                    "max_lines": st.session_state.max_line_count_input,
                    "align_model": st.session_state.align_model_input,
                    "whisper_chunk": st.session_state.whisper_chunk_size_input,
                    "word_snapping": st.session_state.word_snapping_toggle
                }
                t = threading.Thread(target=run_engine_thread, args=(name, task, st.session_state.tasks, config, None, st.session_state.SHUTDOWN_EVENT), daemon=False)
                add_script_run_ctx(t); t.start()
                st.rerun(scope="fragment")

        if c4.button("🗑", key=f"r_{name}", use_container_width=True):
            remove_task(name)

# --- MAIN UI ---
st.title("🏯 Hotaru Dashboard")
st.file_uploader("Drop video files here", type=['mp4', 'mkv', 'mov', 'avi'], 
                 accept_multiple_files=True, key="uploader_key", on_change=handle_upload)

st.markdown("### 📂 Active Queue")
if not st.session_state.tasks:
    st.info("No active tasks. Upload a video above.")
else:
    # This loop ONLY runs once per main script execution.
    # The actual real-time updates happen inside the render_task_card fragment.
    for name in sorted(st.session_state.tasks.keys(), key=lambda x: st.session_state.tasks[x].get('added', 0), reverse=True):
        render_task_card(name)

# Independent Log Tray heartbeat
render_log_tray()
