import logging
import sys
import streamlit as st
from collections import deque
from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx
from hotaru.common.constants import LOG_FILE

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
    
    # Silence Ollama spam
    logging.getLogger("httpx").setLevel(logging.WARNING)
