import warnings
import logging

# Silence noisy library-level warnings and logs
warnings.filterwarnings("ignore", message=".*torchcodec.*")
warnings.filterwarnings("ignore", message=".*Torchaudio's I/O functions.*")
warnings.filterwarnings("ignore", message=".*nvidia-ml-py.*")
logging.getLogger("httpx").setLevel(logging.WARNING) # Silence Ollama API spam

import os
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
from streamlit.runtime.scriptrunner_utils.script_run_context import add_script_run_ctx, get_script_run_ctx

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

# --- LOGGING CORE SETUP ---

# Global fallback buffer for logs when UI context is missing
BACKUP_LOGS = deque(maxlen=100)

class ThreadSafeLogHandler(logging.Handler):
    """Custom handler that pushes logs to Streamlit session_state safely."""
    def emit(self, record):
        try:
            # Check for valid Streamlit context
            ctx = get_script_run_ctx(suppress_warning=True)
            if ctx is None:
                BACKUP_LOGS.append(self.format(record))
                return

            if 'GLOBAL_LOGS' not in st.session_state:
                return

            msg = self.format(record)
            
            # De-duplicate based on the raw record message to ignore timestamp differences
            if hasattr(self, '_last_raw_msg') and self._last_raw_msg == record.msg:
                return
            
            self._last_raw_msg = record.msg
            st.session_state.GLOBAL_LOGS.append(msg)
            
            # Drain backup logs
            while BACKUP_LOGS:
                st.session_state.GLOBAL_LOGS.append(BACKUP_LOGS.popleft())
        except: pass

def setup_logging():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # CRITICAL: Check class name because Streamlit re-defines classes on rerun
    if any(h.__class__.__name__ == "ThreadSafeLogHandler" for h in root_logger.handlers):
        return

    # Clean existing handlers
    for handler in root_logger.handlers[:]: 
        root_logger.removeHandler(handler)

    # 1. File Handler (Full logs)
    file_formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S')
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # 2. Console Handler (Minimal clean output)
    console_formatter = logging.Formatter('%(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # 3. UI Handler
    st_handler = ThreadSafeLogHandler()
    st_handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S'))
    root_logger.addHandler(st_handler)

# Initialize logs in session state
if 'GLOBAL_LOGS' not in st.session_state:
    st.session_state.GLOBAL_LOGS = deque(maxlen=1000)

setup_logging()

# Global shutdown signal for background threads
if 'SHUTDOWN_EVENT' not in st.session_state:
    st.session_state.SHUTDOWN_EVENT = threading.Event()
SHUTDOWN_EVENT = st.session_state.SHUTDOWN_EVENT

# Pre-log startup branding (Only once per session)
if not st.session_state.get('init_logged', False):
    ASCII_ART = r"""
  _    _  ____ _______  _      _____  _    _ 
 | |  | |/ __ \__   __|/ \    |  __ \| |  | |
 | |__| | |  | | | |  / _ \   | |__) | |  | |
 |  __  | |  | | | | / ___ \  |  _  /| |  | |
 | |  | | |__| | | |/ /   \ \ | | \ \| |__| |
 |_|  |_|\____/  |_/_/     \_\|_|  \_\\____/ 
"""
    print(ASCII_ART) # Always show in terminal
    # Directly append to UI logs to bypass formatting/multiple-handlers issues for the branding
    st.session_state.GLOBAL_LOGS.append(f"[{datetime.now().strftime('%H:%M:%S')}] INFO: 🌟 Hotaru Engine Initialized and Ready.")
    st.session_state.init_logged = True

logger = logging.getLogger("HotaruApp")
def log_stop(): 
    logger.info("APP STOPPED - Sending shutdown signal to threads")
    SHUTDOWN_EVENT.set()
atexit.register(log_stop)

load_dotenv()
def add_log(msg): logger.info(msg)

# --- CACHED ENGINE ---
@st.cache_resource
def get_engine(model_size, hf_token, ollama_host):
    return TranscribeEngine(model_size=model_size, hf_token=hf_token, ollama_host=ollama_host)

@st.cache_data(ttl=10)
def get_ollama_models(host):
    try:
        client = ollama.Client(host=host)
        response = client.list()
        return [m.get('model', 'Unknown') for m in response.get('models', [])]
    except:
        return []

# Persistence Helpers
def remove_task(name):
    """Safely removes a task and its associated files."""
    if name in st.session_state.tasks:
        task = st.session_state.tasks[name]
        # Clean up uploaded video
        if os.path.exists(task["file_path"]): 
            try: os.remove(task["file_path"])
            except: pass
        # Clean up potential SRT output
        srt_path = os.path.join(OUTPUT_DIR, os.path.splitext(name)[0] + ".srt")
        if os.path.exists(srt_path):
            try: os.remove(srt_path)
            except: pass
        
        del st.session_state.tasks[name]
        save_task_state(st.session_state.tasks)
        st.rerun()

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

if 'tasks' not in st.session_state:
    st.session_state.tasks = {}
    load_task_state()

if 'view_subtitle' not in st.session_state:
    st.session_state.view_subtitle = None

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
</style>
""", unsafe_allow_html=True)

# --- UI FRAGMENTS ---
# These allow specific parts of the UI to refresh without rerunning the whole script.

@st.fragment(run_every="2s")
def render_system_status():
    """Isolated fragment for system metrics."""
    try:
        cpu, ram = psutil.cpu_percent(), psutil.virtual_memory().percent
        c1, c2 = st.columns(2)
        c1.metric("CPU", f"{cpu}%")
        c2.metric("RAM", f"{ram}%")
        if torch.cuda.is_available():
            f, t = torch.cuda.mem_get_info()
            u, tg = (t-f)/1024**3, t/1024**3
            progress = max(0.0, min(1.0, u/tg))
            st.progress(progress, text=f"VRAM: {u:.1f}/{tg:.1f} GB")
        else: st.warning("⚠️ No GPU detected.")
    except: pass

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("# 🏯 Hotaru")
    st.markdown("""
    **WhisperX Transcription & Translation**
    *Transcribe → Align → [Diarize] → Smart Split → Translate → Smooth → Generate SRT.*
    """)
    st.markdown("---")

    with st.expander("🖥️ System Status", expanded=True):
        render_system_status()
        if st.button("Purge Ollama VRAM", use_container_width=True, key="purge_vram_btn"):
            try:
                h = st.session_state.get("ollama_host", "http://localhost:11434")
                client = ollama.Client(host=h)
                
                # 1. Force unload all models currently in Ollama memory
                try:
                    loaded = client.ps()
                    for m in loaded.get('models', []):
                        client.generate(model=m['name'], keep_alive=0)
                        add_log(f"Unloaded Ollama model: {m['name']}")
                except:
                    # Fallback if ps() is not supported
                    if "last_model" in st.session_state:
                        client.generate(model=st.session_state.last_model, keep_alive=0)
                
                # 2. Local process cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
                
                add_log("VRAM Purge Requested.")
                st.rerun()
            except Exception as e:
                add_log(f"Purge failed: {e}")

    with st.expander("🗣️ Transcription", expanded=False):
        # Update model list to include anime-whisper
        model_options = ["kotoba-tech/kotoba-whisper-v2.0-faster", "litagin/anime-whisper", "large-v3", "large-v2", "medium", "small"]
        model_size = st.selectbox("Whisper Model", model_options, index=0, key="model_size_val")
        
        # Anime-Whisper specific check
        if model_size == "litagin/anime-whisper":
            if not os.path.exists("models/anime-whisper-ct2/model.bin"):
                st.warning("⚠️ anime-whisper not converted yet.")
                if st.button("Convert to CTranslate2"):
                    import subprocess
                    st.info("Converting... Please check terminal.")
                    subprocess.run(["python", "convert_anime_whisper.py"])
                    st.success("Conversion complete! Please restart app.")
        
        enable_diarization = st.toggle("👥 Enable Speaker ID (Diarization)", value=False, key="enable_diarization_toggle")
        if enable_diarization:
            hf_token = st.text_input("🔑 HF Token", type="password", value=os.getenv("HF_TOKEN", ""), key="hf_token_input", help="Required for Diarization")
        
        c1, c2 = st.columns(2)
        max_line_width = c1.number_input("📏 Max Width", min_value=1, max_value=100, value=42, key="max_line_width_input")
        max_line_count = c2.number_input("📚 Max Lines", min_value=1, max_value=5, value=2, key="max_line_count_input")
        
        align_model = st.text_input("📐 Alignment Model", value="jonatasgrosman/wav2vec2-large-xlsr-53-japanese", key="align_model_input", help="Hugging Face model ID for phoneme alignment.")
        
        whisper_chunk_size = st.number_input("✂️ Whisper Chunk Size (s)", min_value=1, max_value=60, value=30, key="whisper_chunk_size_input", help="Chunk size for merging VAD segments. Default is 30, reduce for more granular segments.")

        timing_offset = st.slider("⏱️ Timing Offset (s)", -2.0, 2.0, 0.0, 0.05, key="timing_offset_slider")

    with st.expander("🦙 Ollama Config", expanded=False):
        ollama_host = st.text_input("🔗 Host URL", value="http://localhost:11434", key="ollama_host")
        models = get_ollama_models(ollama_host)
        if models:
            ollama_model = st.selectbox("🎯 Model", models, index=0)
            st.session_state.last_model = ollama_model
        else:
            st.error("⚠️ Ollama unreachable")
            ollama_model = None

    with st.expander("🌎 Translation", expanded=False):
        no_chunking = st.toggle("🚀 Single Batch (No Chunking)", value=False, key="no_chunking_toggle", help="Send entire transcript at once. Best for 256K context models.")
        chunk_size_val = st.number_input("📦 Chunk Size (Lines)", min_value=1, max_value=500, value=25, step=1, key="chunk_size_input", disabled=no_chunking)
        translation_tolerance = st.slider("⚖️ Tolerance (%)", min_value=0, max_value=20, value=5, step=1, key="translation_tolerance_slider", help="Percentage of lines allowed to be missing/malformed before retrying. Lower is stricter.")

@st.fragment(run_every="1s")
def render_task_list_fragment():
    """Isolated fragment for the task dashboard and log tray."""
    # --- MAIN TASK DASHBOARD ---
    h_cols = st.columns([3, 1, 2, 2, 2])
    h_cols[0].markdown("**NAME**"); h_cols[1].markdown("**SIZE**"); h_cols[2].markdown("**PROGRESS**"); h_cols[3].markdown("**STAGE**"); 
    
    done_tasks = [n for n, t in st.session_state.tasks.items() if t["status"] == "Done"]
    if done_tasks:
        if h_cols[4].button("🧹 Clear Done", use_container_width=True, help="Remove all completed tasks"):
            for n in done_tasks: remove_task(n)
    else:
        h_cols[4].markdown("**ACTIONS**")
    st.markdown("---")

    for name in list(st.session_state.tasks.keys()):
        task = st.session_state.tasks[name]
        r_cols = st.columns([3, 1, 2, 2, 2])
        r_cols[0].write(name); r_cols[1].write(task["size"])
        
        # Render Progress and Stage directly from task state
        r_cols[2].progress(task["progress"] / 100.0)
        r_cols[3].write(f"`{task['stage']}`")
        
        a_cols = r_cols[4].columns(2)
        if task["status"] == "Done":
            # Action Column 1: Download & View (compact)
            c1, c2 = a_cols[0].columns(2)
            srt_path = os.path.join(OUTPUT_DIR, os.path.splitext(name)[0] + ".srt")
            if os.path.exists(srt_path):
                with open(srt_path, "rb") as f:
                    c1.download_button("⬇️", f, file_name=os.path.basename(srt_path), key=f"dl_{name}", help="Download SRT")
            if c2.button("👁️", key=f"view_{name}", help="View Subtitles"):
                st.session_state.view_subtitle = name

            # Action Column 2: Remove
            if a_cols[1].button("🗑", key=f"del_{name}", help="Remove"):
                remove_task(name)
        else:
            # Action Column 1: Start/Stop Toggle
            if task["status"] == "Processing":
                if a_cols[0].button("⏸", key=f"stop_{name}", help="Stop"):
                    task["status"] = "Ready"; task["stage"] = "⏹ Stopped"; save_task_state(st.session_state.tasks); st.rerun()
            else:
                if a_cols[0].button("▶", key=f"start_{name}", help="Start"):
                    task["status"] = "Processing"; save_task_state(st.session_state.tasks); st.rerun()

            # Action Column 2: Remove
            if a_cols[1].button("🗑", key=f"del_{name}", help="Remove"):
                remove_task(name)

    # --- SYSTEM TASKS TRAY (Nested in the same fragment for shared refresh cycle) ---
    st.markdown("<br>", unsafe_allow_html=True) # Spacer
    with st.expander("📋 System Tasks", expanded=False):
        log_html = '<div class="log-container"><table class="log-table"><tbody>'
        # Reverse to show newest logs first
        for log_entry in list(st.session_state.GLOBAL_LOGS)[::-1]:
            try:
                # Handle multiline logs (like ASCII art)
                if "\n" in log_entry:
                    log_html += f'<tr><td colspan="3"><pre style="color:#00bfa5; margin:0; font-size:10px; line-height:1">{log_entry}</pre></td></tr>'
                    continue

                match = re.search(r'\[(.*?)\]\s*(.*?):\s*(.*)', log_entry)
                if match:
                    time_str, level, msg = match.groups()
                    # Color coding for different levels
                    lvl_color = "#00bfa5" # INFO
                    if level == "WARNING": lvl_color = "#ff9800"
                    elif level == "ERROR": lvl_color = "#f44336"
                    
                    # Highlight section markers
                    if "PHASE" in msg or "══" in msg:
                        log_html += f'<tr style="background-color:rgba(0,191,165,0.1)"><td class="log-time">[{time_str}]</td><td style="color:{lvl_color}; font-weight:bold">{level}</td><td style="color:#ffffff; font-weight:bold">{msg}</td></tr>'
                    else:
                        log_html += f'<tr><td class="log-time">[{time_str}]</td><td style="color:{lvl_color}">{level}</td><td>{msg}</td></tr>'
                else:
                    log_html += f'<tr><td colspan="3">{log_entry}</td></tr>'
            except: 
                log_html += f'<tr><td colspan="3">{log_entry}</td></tr>'
        st.markdown(log_html + '</tbody></table></div>', unsafe_allow_html=True)

# --- MAIN UI ENTRY POINT ---
uploaded_files = st.file_uploader("Add Files", accept_multiple_files=True, type=["mp4", "mkv", "mov", "wav", "mp3"], label_visibility="collapsed")
if uploaded_files:
    for f in uploaded_files:
        if f.name not in st.session_state.tasks:
            dest_path = os.path.join(UPLOAD_DIR, f.name)
            if not os.path.exists(dest_path):
                with open(dest_path, "wb") as out: out.write(f.getbuffer())
            st.session_state.tasks[f.name] = {"status": "Ready", "progress": 0, "stage": "⏳ Queued", "size": f"{f.size/1024**2:.1f} MB", "file_path": dest_path}
            save_task_state(st.session_state.tasks)

render_task_list_fragment()

# --- SUBTITLE VIEWER MODAL ---
if st.session_state.view_subtitle:
    with st.expander(f"📖 Subtitle Preview: {st.session_state.view_subtitle}", expanded=True):
        srt_path = os.path.join(OUTPUT_DIR, os.path.splitext(st.session_state.view_subtitle)[0] + ".srt")
        if os.path.exists(srt_path):
            with open(srt_path, "r", encoding="utf-8") as f:
                st.text_area("SRT Content", f.read(), height=400)
        if st.button("Close Preview"):
            st.session_state.view_subtitle = None
            st.rerun()

# --- BACKGROUND EXECUTION ---
def run_engine_thread(name, task_data, all_tasks, model_size, hf_token, host, model, offset, chunk_sz, tolerance_p, status_container, max_width, max_lines, align_mod, whisper_chunk):
    try:
        def engine_callback(msg):
            # Check if task still exists in all_tasks; if not, we are in a stale thread
            if name not in all_tasks: return
            
            new_stage, new_prog = task_data["stage"], task_data["progress"]
            if "Transcribing" in msg: new_stage, new_prog = "🗣️ Transcribing", 15
            elif "Aligning" in msg: new_stage, new_prog = "📏 Aligning", 40
            elif "Nuclear VRAM Reset" in msg: new_stage, new_prog = "🧹 Resetting VRAM", 65
            elif "AI Polish" in msg or "Ollama Polish" in msg: new_stage, new_prog = "🪄 AI Polishing", 90
            elif "Translating" in msg:
                # Robust parsing for chunked progress: "chunk (\d+)/(\d+)"
                match = re.search(r"chunk (\d+)/(\d+)", msg)
                if match:
                    try:
                        p, t = int(match.group(1)), int(match.group(2))
                        new_stage = f"🌎 Translating ({p}/{t})"
                        new_prog = 70 + int((p/t)*25)
                    except:
                        new_stage, new_prog = "🌎 Translating", 80
                elif "entire transcript" in msg:
                    new_stage, new_prog = "🌎 Translating (Single Batch)", 85
                else:
                    new_stage, new_prog = "🌎 Translating", 75
            
            # Update background state (Safe for threads)
            task_data["stage"], task_data["progress"] = new_stage, new_prog
            # Disk persistence (Safe for threads)
            try:
                serializable = {n: {k: v for k, v in t.items() if k not in ['file', 'placeholders']} for n, t in all_tasks.items()}
                with open(TASKS_FILE, "w") as f: json.dump(serializable, f)
            except: pass
            
            # Use standard logging without redundant task name prefix (to prevent duplicates)
            logging.info(msg)

        def cancel_check():
            # Abort if:
            # 1. Task removed from global list
            # 2. Status changed from Processing (Stopped/Paused)
            # 3. Application is shutting down
            return name not in all_tasks or all_tasks[name].get("status") != "Processing" or SHUTDOWN_EVENT.is_set()

        eng = TranscribeEngine(model_size=model_size, hf_token=hf_token, ollama_host=host)
        segments = eng.process_video(
            task_data["file_path"], 
            ollama_model=model, 
            log_callback=engine_callback, 
            timing_offset=offset, 
            chunk_size=chunk_sz, 
            tolerance_pct=tolerance_p,
            cancel_check=cancel_check,
            max_line_width=max_width,
            max_line_count=max_lines,
            align_model=align_mod,
            whisper_chunk_size=whisper_chunk
        )
        
        srt_path = os.path.join(OUTPUT_DIR, os.path.splitext(name)[0] + ".srt")
        eng.generate_srt(segments, srt_path)
        task_data["status"], task_data["stage"], task_data["progress"] = "Done", "✅ Completed", 100
        
    except InterruptedError:
        logging.getLogger("HotaruApp").info(f"[{name}] Task successfully aborted and cleaned up.")
    except Exception as e:
        if name in all_tasks:
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
    
    # Get values from session state for robustness
    current_model = st.session_state.get("model_size_val", "kotoba-tech/kotoba-whisper-v2.0-faster")
    is_diarize = st.session_state.get("enable_diarization_toggle", False)
    token = st.session_state.get("hf_token_input", os.getenv("HF_TOKEN", ""))
    off = st.session_state.get("timing_offset_slider", 0.0)
    
    # Calculate effective chunk size from toggle + input
    is_no_chunk = st.session_state.get("no_chunking_toggle", False)
    current_chunk_size = 0 if is_no_chunk else st.session_state.get("chunk_size_input", 25)
    
    current_tolerance = st.session_state.get("translation_tolerance_slider", 5)
    
    current_max_width = st.session_state.get("max_line_width_input", 42)
    current_max_lines = st.session_state.get("max_line_count_input", 2)
    current_align_model = st.session_state.get("align_model_input", "jonatasgrosman/wav2vec2-large-xlsr-53-japanese")
    current_whisper_chunk = st.session_state.get("whisper_chunk_size_input", 30)
    
    # Ensure variables from sidebar are captured even if expanders are closed
    # (Streamlit keeps them in session_state)
    target_ollama_model = st.session_state.get("last_model")
    
    t = threading.Thread(target=run_engine_thread, args=(
        processing_task_name, task, st.session_state.tasks, current_model, 
        token if is_diarize else None, ollama_host, target_ollama_model,
        off, current_chunk_size, current_tolerance, st.session_state.thread_status,
        current_max_width, current_max_lines, current_align_model, current_whisper_chunk
    ), daemon=False)
    add_script_run_ctx(t)
    t.start()
