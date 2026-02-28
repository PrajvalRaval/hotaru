import streamlit as st
import psutil
import torch
import torch.cuda

@st.fragment(run_every="1s")
def render_system_stats():
    """Independent fragment for real-time hardware monitoring."""
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
    except: pass

from hotaru.common.logger import BACKUP_LOGS

import html

@st.fragment(run_every="2s")
def render_log_tray():
    """Independent fragment for the log console. Drains background logs into UI."""
    if 'GLOBAL_LOGS' not in st.session_state: return
    
    # Proactively drain logs from background threads
    while BACKUP_LOGS:
        st.session_state.GLOBAL_LOGS.append(BACKUP_LOGS.popleft())
    
    with st.expander("📝 SYSTEM TASKS & LOGS", expanded=True):
        recent_logs = list(st.session_state.GLOBAL_LOGS)[-50:]
        if not recent_logs:
            st.caption("Waiting for engine activity...")
            return

        # Escape HTML to prevent broken tables from warnings/metadata
        log_html = "".join([f"<tr><td>{html.escape(line)}</td></tr>" for line in reversed(recent_logs)])
        st.markdown(f"""
        <div class="log-container">
            <table class="log-table">
                <tbody>
                    {log_html}
                </tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)
