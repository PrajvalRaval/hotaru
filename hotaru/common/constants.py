import os
from datetime import datetime

# Performance
TF32_ENABLED = True

# Directories
LOG_DIR = "logs"
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "output"

# State Files
TASKS_FILE = os.path.join(LOG_DIR, "tasks_state.json")

# Runtime
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(LOG_DIR, f"hotaru_{RUN_ID}.log")

# VRAM Guard
VRAM_MIN_THRESHOLD_GB = 1.5
