import logging
import sys
import streamlit as st
from collections import deque
from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx
from hotaru.common.constants import LOG_FILE

# Global fallback buffer for logs when UI context is missing
BACKUP_LOGS = deque(maxlen=100)

class ThreadSafeLogHandler(logging.Handler):
    """Custom handler that pushes logs to Streamlit session_state safely with strict deduplication."""
    def emit(self, record):
        try:
            import time
            msg = self.format(record)
            now = time.time()
            
            # 1. Strict Deduplication: Ignore identical messages within 0.5s
            # This handles duplicates from multiple handlers or rapid-fire thread updates
            if hasattr(self, '_last_msg') and self._last_msg == msg:
                if hasattr(self, '_last_time') and (now - self._last_time) < 0.5:
                    return
            
            self._last_msg = msg
            self._last_time = now

            # 2. Context Check
            ctx = get_script_run_ctx(suppress_warning=True)
            if ctx is None:
                BACKUP_LOGS.append(msg)
                return

            if 'GLOBAL_LOGS' not in st.session_state:
                return

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

    # Standard Formatter: [HH:MM:SS] LEVEL: Message
    log_format = '[%(asctime)s] %(levelname)s: %(message)s'
    date_format = '%H:%M:%S'
    formatter = logging.Formatter(log_format, datefmt=date_format)

    # 1. File Handler (Full logs)
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # 2. Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # 3. UI Handler
    st_handler = ThreadSafeLogHandler()
    st_handler.setFormatter(formatter)
    root_logger.addHandler(st_handler)
    
    # Silence third-party spam
    for lib in ["httpx", "httpcore", "urllib3", "openai", "transformers", "faster_whisper"]:
        l = logging.getLogger(lib)
        l.setLevel(logging.WARNING)
        l.propagate = False 

    # CONFIGURE PROJECT LOGGERS (Prevent double-posting)
    for proj in ["HotaruEngine", "HotaruTranslator", "HotaruAudio", "HotaruLogic", "HotaruApp"]:
        l = logging.getLogger(proj)
        l.propagate = False # CRITICAL: Stop bubbling up to root
        l.setLevel(logging.INFO)
        # Re-attach handlers explicitly to these loggers
        if not l.handlers:
            l.addHandler(file_handler)
            l.addHandler(console_handler)
            l.addHandler(st_handler)
