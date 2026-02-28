import os
import threading
import logging
import json
import re
import torch
from hotaru.engine.core import TranscribeEngine
from hotaru.common.constants import TASKS_FILE, OUTPUT_DIR
from hotaru.engine.subtitle_utils import generate_srt
from hotaru.common.state import save_task_state
from streamlit.runtime.scriptrunner_utils.script_run_context import add_script_run_ctx

logger = logging.getLogger("HotaruLogic")

def run_engine_thread(name, task_data, all_tasks, config, status_container, shutdown_event):
    """Background thread orchestrating the engine."""
    try:
        def engine_callback(msg):
            if name not in all_tasks: return
            
            # Robust stage/progress parsing from message
            new_stage, new_prog = task_data["stage"], task_data["progress"]
            if "PHASE 0" in msg or "Stripping SFX" in msg: new_stage, new_prog = "🎤 Vocal Isolation", 5
            elif "PHASE 1" in msg or "Loading audio" in msg: new_stage, new_prog = "🎧 Transcription", 10
            elif "Transcribing" in msg: new_stage, new_prog = "🗣️ Transcribing", 20
            elif "PHASE 2" in msg or "Aligning" in msg: new_stage, new_prog = "📏 Aligning", 40
            elif "manual timing offset" in msg: new_stage, new_prog = "⏱️ Offsetting", 50
            elif "Identifying speakers" in msg: new_stage, new_prog = "👥 Diarizing", 55
            elif "Resegmenting" in msg: new_stage, new_prog = "📦 Resegmenting", 60
            elif "PHASE 3" in msg or "Purging" in msg: new_stage, new_prog = "🧹 Resetting VRAM", 65
            elif "PHASE 4" in msg or "Localizing" in msg:
                match = re.search(r"chunk (\d+)/(\d+)", msg)
                if match:
                    p, t = int(match.group(1)), int(match.group(2))
                    new_stage = f"🌎 Localizing ({p}/{t})"
                    new_prog = 70 + int((p/t)*28)
                else: new_stage, new_prog = "🌎 Localizing", 75
            elif "Localization complete" in msg: new_stage, new_prog = "💾 Finalizing SRT", 98
            
            task_data["stage"], task_data["progress"] = new_stage, new_prog
            save_task_state(all_tasks)
            logging.info(msg)

        def cancel_check():
            return name not in all_tasks or all_tasks[name].get("status") != "Processing" or shutdown_event.is_set()

        eng = TranscribeEngine(model_size=config["model_size"], ollama_host=config["ollama_host"])
        
        # Pre-calculate target SRT path for streaming updates
        srt_name = os.path.splitext(name)[0] + ".srt"
        srt_path = os.path.join(OUTPUT_DIR, srt_name)

        segments = eng.process_video(
            task_data["file_path"], 
            ollama_model=config["ollama_model"],
            log_callback=engine_callback,
            timing_offset=config["timing_offset"],
            cancel_check=cancel_check,            max_line_width=config["max_width"],
            max_line_count=config["max_lines"],
            align_model=config["align_model"],
            whisper_chunk_size=config["whisper_chunk"],
            enable_word_snapping=config.get("word_snapping", False),
            srt_output_path=srt_path
        )
        
        # Final update
        task_data.update({
            "status": "Completed", "stage": "✅ Done", "progress": 100,
            "output_path": srt_path, "output_name": srt_name
        })
        save_task_state(all_tasks)
        logging.info(f"✨ Task '{name}' completed successfully.")

    except InterruptedError:
        logging.info(f"🛑 Task '{name}' aborted.")
    except Exception as e:
        logger.error(f"❌ Task '{name}' failed: {e}")
        if name in all_tasks:
            all_tasks[name].update({"status": "Failed", "stage": f"❌ Error: {str(e)[:30]}..."})
            save_task_state(all_tasks)
