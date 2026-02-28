import os
import json
import streamlit as st
from hotaru.common.constants import TASKS_FILE, UPLOAD_DIR, OUTPUT_DIR

def save_task_state(tasks_dict):
    """Persists current tasks to disk."""
    serializable = {name: {k: v for k, v in t.items() if k not in ['file', 'placeholders']} for name, t in tasks_dict.items()}
    with open(TASKS_FILE, "w") as f:
        json.dump(serializable, f)

def load_task_state():
    """Loads tasks from disk into session state."""
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

def remove_task(name):
    """Safely removes a task and its associated files."""
    if name in st.session_state.tasks:
        task = st.session_state.tasks[name]
        if os.path.exists(task["file_path"]): 
            try: os.remove(task["file_path"])
            except: pass
        srt_path = os.path.join(OUTPUT_DIR, os.path.splitext(name)[0] + ".srt")
        if os.path.exists(srt_path):
            try: os.remove(srt_path)
            except: pass
        
        del st.session_state.tasks[name]
        save_task_state(st.session_state.tasks)
        st.rerun()
