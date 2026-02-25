import os
import logging
import warnings
from datetime import datetime
import atexit
import sys
import torch
import streamlit as st
from transcribe_engine import TranscribeEngine
import ollama
from dotenv import load_dotenv
import psutil
import re
import json
import shutil
import threading
import time
from collections import deque

# Enable TF32 for RTX 3090/4090 performance boost
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Directories
LOG_DIR = "logs"
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "output"
for d in [LOG_DIR, UPLOAD_DIR, OUTPUT_DIR]: os.makedirs(d, exist_ok=True)

TASKS_FILE = os.path.join(LOG_DIR, "tasks_state.json")
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(LOG_DIR, f"hotaru_{RUN_ID}.log")

# Global thread-safe log storage
if 'GLOBAL_LOGS' not in globals():
    GLOBAL_LOGS = deque(maxlen=1000)

# Logging Setup
logging.captureWarnings(True)
warnings.filterwarnings("ignore", message=".*Torchaudio's I/O functions.*")
warnings.filterwarnings("ignore", message=".*nvidia-ml-py.*")

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
for handler in root_logger.handlers[:]: root_logger.removeHandler(handler)
formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S')
file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
file_handler.setFormatter(formatter)
root_logger.addHandler(file_handler)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
root_logger.addHandler(console_handler)

class ThreadSafeLogHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            GLOBAL_LOGS.append(msg)
        except: pass

st_handler = ThreadSafeLogHandler()
st_handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S'))
root_logger.addHandler(st_handler)

logger = logging.getLogger("HotaruApp")
def log_stop(): logger.info("APP STOPPED")
atexit.register(log_stop)

load_dotenv()
def add_log(msg): logger.info(msg)

# --- CACHED ENGINE ---
@st.cache_resource
def get_engine(model_size, hf_token, ollama_host):
    return TranscribeEngine(model_size=model_size, hf_token=hf_token, ollama_host=ollama_host)

# Persistence Helpers
def save_task_state(tasks_dict):
    serializable = {name: {k: v for k, v in t.items() if k not in ['file', 'placeholders']} for name, t in tasks_dict.items()}
    with open(TASKS_FILE, "w") as f: json.dump(serializable, f)

def load_task_state():
    if os.path.exists(TASKS_FILE):
        try:
            with open(TASKS_FILE, "r") as f:
                data = json.load(f)
                for name, task in data.items():
                    path = os.path.join(UPLOAD_DIR, name)
                    if os.path.exists(path):
                        task['file_path'] = path
                        st.session_state.tasks[name] = task
        except: pass

st.set_page_config(page_title="Hotaru - Transcription & Translation", layout="wide", page_icon="🏯", initial_sidebar_state="expanded")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #e0e0e0; overflow: hidden; }
    [data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; min-width: 336px !important; max-width: 336px !important; width: 336px !important; }
    [data-testid="stSidebarCollapseButton"] { display: none !important; }
    [data-testid="stSidebar"] div[data-testid="stExpander"] { border: none !important; background-color: transparent !important; margin-bottom: 0 !important; }
    [data-testid="stSidebar"] div[data-testid="stExpander"] summary {
        color: #e0e0e0 !important; font-size: 0.85rem !important; text-transform: uppercase; letter-spacing: 0.5px; padding-left: 0 !important;
    }
    .main .block-container { padding: 1rem !important; max-width: 100% !important; height: 100vh; }
    
    div.stApp section[data-testid="stMain"] div[data-testid="stExpander"] {
        position: fixed !important; bottom: 0 !important; right: 0 !important;
        left: 336px !important; width: calc(100% - 336px) !important;
        z-index: 10000 !important;
        background-color: #161b22 !important; border: none !important; border-top: 1px solid #30363d !important; margin: 0 !important;
    }
    section[data-testid="stMain"] div[data-testid="stExpander"] summary {
        padding: 5px 20px !important; color: #00bfa5 !important; font-weight: 600 !important; background-color: #161b22 !important; min-height: 32px !important; display: flex; align-items: center;
    }
    section[data-testid="stMain"] div[data-testid="stExpander"] [data-testid="stExpanderDetails"] { padding: 0 !important; }
    .log-container { background-color: #0d1117; color: #8b949e; font-family: 'SF Mono', monospace; height: 200px; overflow-y: scroll; font-size: 0.7rem; }
    .log-table { width: 100%; border-collapse: collapse; }
    .log-table td { padding: 2px 10px; border-bottom: 1px solid #21262d; white-space: nowrap; }
    .log-time { color: #8b949e; width: 100px; }
    [data-testid="stFileUploader"] { border: 1px dashed #30363d !important; border-radius: 4px !important; padding: 10px !important; margin-bottom: 20px; }
    [data-testid="stFileUploader"] section + div { display: none !important; }
    [data-testid="stFileUploader"] section + div:has(div[role="progressbar"]) { display: block !important; }
    h1, h2, h3 { color: #00bfa5 !important; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.tasks = {}
    st.session_state.processing_now = False
    st.session_state.view_subtitle = None
    load_task_state()
    add_log("APP RUNNING")

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("# 🏯 Hotaru")
    st.markdown("""
    **WhisperX Transcription & Translation**
    *Transcribe → Align → [Diarize] → Smart Split → Translate → Smooth → Generate SRT.*
    """)
    st.markdown("---")

    with st.expander("🖥️ System Status", expanded=True):
        sys_metrics = st.empty()
        vram_bar = st.empty()
        def update_system_status_ui():
            cpu, ram = psutil.cpu_percent(), psutil.virtual_memory().percent
            with sys_metrics.container():
                c1, c2 = st.columns(2)
                c1.metric("CPU", f"{cpu}%")
                c2.metric("RAM", f"{ram}%")
            if torch.cuda.is_available():
                f, t = torch.cuda.mem_get_info()
                u, tg = (t-f)/1024**3, t/1024**3
                vram_bar.progress(u/tg, text=f"VRAM: {u:.1f}/{tg:.1f} GB")
            else: vram_bar.warning("⚠️ No GPU detected.")
        update_system_status_ui()
        if st.button("Purge Ollama VRAM", use_container_width=True, key="purge_vram_btn"):
            try:
                h = st.session_state.get("ollama_host", "http://localhost:11434")
                client = ollama.Client(host=h)
                if "last_model" in st.session_state: client.generate(model=st.session_state.last_model, keep_alive=0)
                st.rerun()
            except: pass

    with st.expander("🗣️ Transcription", expanded=False):
        model_size = st.selectbox("Whisper Model", ["large-v3", "large-v2", "medium", "small"], index=0)
        enable_diarization = st.toggle("👥 Enable Diarization", value=False)
        hf_token = st.text_input("HuggingFace Token", type="password") if enable_diarization else ""

    with st.expander("🦙 Ollama Config", expanded=False):
        ollama_host = st.text_input("🔗 Host URL", value="http://localhost:11434", key="ollama_host")
        try:
            client = ollama.Client(host=ollama_host); response = client.list()
            models = [m.get('model', 'Unknown') for m in response.get('models', [])]
            ollama_model = st.selectbox("🎯 Model", models, index=0 if models else None)
            if ollama_model: st.session_state.last_model = ollama_model
        except: st.error("⚠️ Ollama unreachable"); ollama_model = None

    with st.expander("🌎 Translation", expanded=False):
        skip_songs = st.toggle("🎵 Skip Song Translation", value=True)

# --- MAIN UI ---
uploaded_files = st.file_uploader("Add Files", accept_multiple_files=True, type=["mp4", "mkv", "mov", "wav", "mp3"], label_visibility="collapsed")
if uploaded_files:
    for f in uploaded_files:
        if f.name not in st.session_state.tasks:
            dest_path = os.path.join(UPLOAD_DIR, f.name)
            if not os.path.exists(dest_path):
                with open(dest_path, "wb") as out: out.write(f.getbuffer())
            st.session_state.tasks[f.name] = {"status": "Ready", "progress": 0, "stage": "⏳ Queued", "size": f"{f.size/1024**2:.1f} MB", "file_path": dest_path}
            save_task_state(st.session_state.tasks)

h_cols = st.columns([3, 1, 2, 2, 2])
h_cols[0].markdown("**NAME**"); h_cols[1].markdown("**SIZE**"); h_cols[2].markdown("**PROGRESS**"); h_cols[3].markdown("**STAGE**"); h_cols[4].markdown("**ACTIONS**")
st.markdown("---")

for name in list(st.session_state.tasks.keys()):
    task = st.session_state.tasks[name]
    r_cols = st.columns([3, 1, 2, 2, 2])
    r_cols[0].write(name); r_cols[1].write(task["size"])
    p_prog = r_cols[2].empty(); p_stage = r_cols[3].empty()
    p_prog.progress(task["progress"] / 100.0); p_stage.write(f"`{task['stage']}`")
    task['placeholders'] = (p_prog, p_stage)
    
    a_cols = r_cols[4].columns(3)
    if task["status"] == "Done":
        # Download Button
        srt_path = os.path.join(OUTPUT_DIR, os.path.splitext(name)[0] + ".srt")
        if os.path.exists(srt_path):
            with open(srt_path, "rb") as f:
                a_cols[0].download_button("⬇️", f, file_name=os.path.basename(srt_path), key=f"dl_{name}", help="Download SRT")
        
        # View Button
        if a_cols[1].button("👁️", key=f"view_{name}", help="View Subtitles"):
            st.session_state.view_subtitle = name

        # Delete Button
        if a_cols[2].button("🗑", key=f"del_{name}", help="Remove"):
            if os.path.exists(task["file_path"]): os.remove(task["file_path"])
            del st.session_state.tasks[name]; save_task_state(st.session_state.tasks); st.rerun()
    else:
        # Start Button
        if a_cols[0].button("▶", key=f"start_{name}", help="Start"):
            task["status"] = "Processing"; save_task_state(st.session_state.tasks); st.rerun()
        # Stop Button
        if a_cols[1].button("⏹", key=f"stop_{name}", help="Stop"):
            task["status"] = "Ready"; task["stage"] = "⏹ Stopped"; save_task_state(st.session_state.tasks); st.rerun()
        # Delete Button
        if a_cols[2].button("🗑", key=f"del_{name}", help="Remove"):
            if os.path.exists(task["file_path"]): os.remove(task["file_path"])
            del st.session_state.tasks[name]; save_task_state(st.session_state.tasks); st.rerun()

# Subtitle Preview
if st.session_state.view_subtitle:
    name = st.session_state.view_subtitle
    srt_path = os.path.join(OUTPUT_DIR, os.path.splitext(name)[0] + ".srt")
    if os.path.exists(srt_path):
        with st.expander(f"📄 Subtitle Preview: {name}", expanded=True):
            with open(srt_path, "r", encoding="utf-8") as f:
                st.text_area("SRT Content", f.read(), height=300)
            if st.button("Close Preview"):
                st.session_state.view_subtitle = None
                st.rerun()

# --- SYSTEM TASKS TRAY ---
tray_expander = st.expander("📋 System Tasks", expanded=False)
with tray_expander:
    tray_p = st.empty()

def render_logs_live_ui():
    log_html = '<div class="log-container"><table class="log-table"><tbody>'
    for log in list(GLOBAL_LOGS)[::-1]:
        try:
            match = re.search(r'\[(.*?)\]\s*(.*?):\s*(.*)', log)
            time_str, level, msg = match.groups() if match else ("-", "INFO", log)
            log_html += f'<tr><td class="log-time">[{time_str}]</td><td style="color:#00bfa5">{level}</td><td>{msg}</td></tr>'
        except: log_html += f'<tr><td>{log}</td></tr>'
    tray_p.markdown(log_html + '</tbody></table></div>', unsafe_allow_html=True)

render_logs_live_ui()

# --- BACKGROUND EXECUTION ---
def run_engine_thread(name, task_data, all_tasks, model_size, hf_token, host, model, skip, status_container):
    try:
        def engine_callback(msg):
            new_stage, new_prog = task_data["stage"], task_data["progress"]
            if "Transcribing" in msg: new_stage, new_prog = "🗣️ Transcribing", 15
            elif "Aligning" in msg: new_stage, new_prog = "📏 Aligning", 40
            elif "Segmenting" in msg: new_stage, new_prog = "✂️ Splitting", 60
            elif "Nuclear VRAM Reset" in msg: new_stage, new_prog = "🧹 Resetting VRAM", 65
            elif "Translating" in msg:
                if "chunk" in msg:
                    try:
                        chunk_info = msg.split("chunk ")[1].split("...")[0]
                        new_stage = f"🌎 Translating ({chunk_info})"
                        parts = chunk_info.split("/"); p = int(parts[0]); t = int(parts[1])
                        new_prog = 70 + int((p/t)*25)
                    except: new_stage, new_prog = "🌎 Translating", 80
                else: new_stage, new_prog = "🌎 Translating", 70
            task_data["stage"], task_data["progress"] = new_stage, new_prog
            save_task_state(all_tasks)
            logging.getLogger("HotaruApp").info(f"[{name}] {msg}")

        eng = TranscribeEngine(model_size=model_size, hf_token=hf_token, ollama_host=host)
        segments = eng.process_video(task_data["file_path"], ollama_model=model, log_callback=engine_callback, skip_songs=skip)
        srt_path = os.path.join(OUTPUT_DIR, os.path.splitext(name)[0] + ".srt")
        eng.generate_srt(segments, srt_path); task_data["status"], task_data["stage"], task_data["progress"] = "Done", "✅ Completed", 100
    except Exception as e:
        task_data["status"], task_data["stage"] = "Error", f"❌ Failed: {str(e)[:50]}"
        logging.getLogger("HotaruApp").error(f"Processing failed: {e}")
    finally:
        status_container['processing'] = False
        save_task_state(all_tasks)

if 'thread_status' not in st.session_state:
    st.session_state.thread_status = {'processing': False}

processing_task_name = next((n for n, t in st.session_state.tasks.items() if t["status"] == "Processing"), None)
if processing_task_name and not st.session_state.thread_status['processing']:
    st.session_state.thread_status['processing'] = True
    task = st.session_state.tasks[processing_task_name]
    t = threading.Thread(target=run_engine_thread, args=(
        processing_task_name, task, st.session_state.tasks, model_size, 
        hf_token if enable_diarization else None, ollama_host, ollama_model, skip_songs,
        st.session_state.thread_status
    ))
    t.start()

# --- HEARTBEAT LOOP ---
while True:
    update_system_status_ui()
    render_logs_live_ui()
    for name, t in st.session_state.tasks.items():
        p_prog, p_stage = t.get('placeholders', (None, None))
        if p_prog: p_prog.progress(t["progress"] / 100.0)
        if p_stage: p_stage.write(f"`{t['stage']}`")
    if st.session_state.thread_status['processing'] == False and processing_task_name:
        st.rerun()
    time.sleep(1)
