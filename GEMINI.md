# Hotaru - Development Context

This file contains foundational mandates and architectural state for Hotaru. These instructions take precedence over general defaults.

## 🎯 Core Project Goal
To provide the highest quality Japanese-to-English subtitles for anime using a hybrid pipeline of local AI models (WhisperX for timing, Ollama for translation).

## 🛠️ Architectural Mandates

### 1. Resource & Memory Orchestration
- **Nuclear VRAM Reset:** Between transcription (WhisperX) and translation (Ollama), ALL heavy models MUST be explicitly deleted. Calls to `torch.cuda.empty_cache()` and `torch.cuda.ipc_collect()` must be performed to ensure a clean slate for the LLM.
- **Fragmented UI Updates:** Global `st.rerun()` heartbeats are FORBIDDEN. All real-time telemetry (VRAM/RAM/Progress) MUST use `@st.fragment` to refresh isolated UI components, preventing script-wide re-execution and memory leaks.

### 2. Subtitle Timing & UI Dynamics
- **Start/Pause Toggle:** The task action button MUST toggle between **▶ (Start)** and **⏸ (Stop)** based on real-time task status.
- **Instant Removal & Abort:** Clicking **Remove (🗑)** must trigger immediate UI removal and raise an `InterruptedError` in the engine thread via a `cancel_check` callback that monitors task existence, status, and the global `SHUTDOWN_EVENT`.
- **Residue Cleanup:** Task removal MUST proactively delete both the source video from `uploads/` and any partial/full SRT output from `output/`.
- **Native Timing:** Rely on WhisperX's native segmentation and alignment outputs without additional manual splitting or post-merging logic.

### 3. Two-Pass Translation Architecture
- **Pass 1: Initial Translation:** Sent in batches (or single batch) with Speaker IDs provided for context. Focuses on accuracy and honorific retention.
- **Pass 2: The "Subber's Polish":** Sends the entire translated transcript back to the LLM. Focuses on resolving subject ambiguity, improving dialogue flow, and ensuring comfortable reading speeds.
- **"No Chunking" Mode:** For 256K context models (Qwen3), support a toggle-based `chunk_size=0` mode that sends the entire transcript in one batch for better narrative consistency and a ~3x speed boost.
- **Stable Batching:** When chunking is enabled, use a numerical input (default 25) instead of a slider to prevent rapid UI-state mutations.
- **Dynamic Tolerance:** Use a **Percentage-Based Tolerance** slider (default 5%) to determine acceptable missing lines before triggering a retry.
- **Verbose Telemetry:** Translation logs MUST include total character counts, estimated tokens, and detailed parsing results (matched vs unmatched lines).
- **Audio-Level Skipping:** Rely solely on Silero VAD (Voice Activity Detection) during the transcription phase to skip non-speech segments, including opening/ending themes and background music.

### 4. Model Specifics (Anime-Whisper)
- **Automatic Patching:** The `convert_anime_whisper.py` script MUST automatically patch the CTranslate2 `config.json` to use **128 Mel bins** (Whisper v3 requirement) and ensure all required preprocessor metadata is downloaded.

### 5. Mandatory Syntax Check
- **Zero-Tolerance for IndentationErrors:** After ANY modification to Python source files, the `./check_syntax.sh` script MUST be executed to verify structural integrity.
- **Pre-Execution Check:** Never finish a task without confirming all files pass the syntax check.

## 📝 Engineering Standards

### Logging & Thread Safety
- **Context Attachment:** All background threads MUST use `add_script_run_ctx` to link to the Streamlit session context for safe UI interaction.
- **Shutdown Handling:** Monitor a global `threading.Event()` for app shutdown to allow C++ extensions to unload gracefully, avoiding "terminate called without active exception" crashes.
- **httpx Silencing:** Set `logging.getLogger("httpx").setLevel(logging.WARNING)` to prevent Ollama API spam.
- **Cached Model Fetching:** Fetch Ollama models using a cached function with a **10s TTL**.

## ⚠️ Known Version Constraints
- **Streamlit >= 1.54.0:** Required for `@st.fragment` support.
- **PyTorch Stack >= 2.10.0:** Required for `torchcodec` 0.10.0 compatibility.
- **TF32:** Forced enable globally for RTX 3090/4090 performance.
