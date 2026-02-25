# Hotaru - Development Context

This file contains foundational mandates and architectural state for Hotaru. These instructions take precedence over general defaults.

## 🎯 Core Project Goal
To provide the highest quality Japanese-to-English subtitles for anime using a hybrid pipeline of local AI models (WhisperX for timing, Ollama for translation).

## 🛠️ Architectural Mandates

### 1. VRAM & Resource Management
- **Nuclear VRAM Reset:** Between transcription (WhisperX) and translation (Ollama), ALL heavy models MUST be deleted and `torch.cuda.empty_cache()` called multiple times. This is non-negotiable to support 30B+ parameters on consumer GPUs.
- **Pre-emptive Unload:** Always ensure Ollama is unloaded (`keep_alive=0`) before starting WhisperX tasks to prevent OOM errors.
- **Manual Purge:** The UI provides a "Purge Ollama VRAM" button in System Status for emergency memory recovery.

### 2. Subtitle Timing & UI Dynamics
- **Translate-then-Split:** Translate full original segments first to preserve English grammar, THEN split based on word-level timestamps.
- **Anti-Flicker Logic:** 
    - Minimum subtitle duration: **1.2s**.
    - Minimum chunk size before allowed to split: **1.0s**.
    - Collision detection must prevent overlapping timestamps.
- **Smart Splitting:** Split segments only on punctuation (`。`, `！`, `？`) or silence gaps (`> 0.5s`).

### 3. Translation Quality (Ollama)
- **Batching:** Use a `chunk_size` of **25** lines.
- **Song Detection:** A heuristic `_is_likely_song` detects lyrics based on musical symbols (`♪`, `～`), text density, and vowel extensions. Detected songs are skipped (set to empty string) to prevent low-quality LLM hallucinations.
- **Prompting:** Instructs Ollama to output an empty line for any remaining lyrics to keep the SRT clean.
- **Cleaning:** Always strip `Line X:` prefixes and speaker labels from LLM output.

### 4. UI/UX Architecture (uTorrent + Proxmox Inspired)
- **Sidebar:** Fixed at **336px**, non-collapsible. Contains grouped expanders for System, Transcription, Ollama, and Translation settings.
- **Task List:** A row-based data-dense list showing ID, Size, Progress (animated bar), and detailed Stage (e.g., `🌎 Translating (1/11)`).
- **Persistence:** All tasks and uploaded files are persisted in `uploads/` and `logs/tasks_state.json`. Refreshing the browser does not lose progress.
- **System Tasks Tray:** A fixed, Proxmox-style status tray docked at the bottom of the main content area, aligned perfectly with the sidebar.

## 📝 Engineering Standards

### Logging & Warnings
- **Full Transparency:** Capture all library warnings and errors in session-specific log files (`logs/hotaru_*.log`).
- **Suppression:** Suppress non-fixable internal warnings (e.g., `torchaudio`, `nvidia-ml-py`).
- **Live Updates:** Use `st.empty()` placeholders to push engine updates to the UI in real-time.

## ⚠️ Known Version Constraints
- **torchcodec:** DO NOT install. It is currently incompatible with PyTorch 2.8.0 ABI.
- **pynvml:** Removed in favor of `nvidia-ml-py` to resolve deprecation warnings.
- **TF32:** Forced enable globally for RTX 3090/4090 performance.
