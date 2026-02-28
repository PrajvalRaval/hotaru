# Hotaru - Development Context

This file contains foundational mandates and architectural state for Hotaru. These instructions take precedence over general defaults.

## 🎯 Core Project Goal
To provide the highest quality Japanese-to-English subtitles for anime using a hybrid pipeline of local AI models (WhisperX for timing, Ollama for translation).

## 🛠️ Architectural Mandates

### 1. Resource & Memory Orchestration
- **Nuclear VRAM Reset:** Between transcription (WhisperX) and translation (Ollama), ALL heavy models MUST be explicitly deleted. Calls to `torch.cuda.empty_cache()` and `torch.cuda.ipc_collect()` must be performed to ensure a clean slate for the LLM.
- **Fragmented UI Updates:** Global `st.rerun()` heartbeats are FORBIDDEN. All real-time telemetry (VRAM/RAM/Progress) MUST use `@st.fragment` to refresh isolated UI components, preventing script-wide re-execution and memory leaks.

### 2. Subtitle Timing & UI Dynamics
- **Deterministic Resegmentation:** Use a **Buffer-and-Flush** algorithm that strictly respects Whisper segment boundaries. NEVER allow a subtitle to cross a VAD-detected silence gap.
- **Speaker-Aware Word Splitting:** Monitor speaker IDs at the **word level**. Trigger an instant buffer flush whenever the speaker changes to prevent dialogue merging and "spoiler" subtitles.
- **Localized Fallbacks:** For unaligned words, use rhythmic interpolation relative to neighboring words. NEVER snap timing to the beginning of a 30-second Whisper block.
- **Aggressive End-Padding:** Apply a tight (150-200ms) reading buffer only when speech is continuous. Clear the screen instantly if a silence gap (>0.4s) or segment boundary is reached.

### 3. Two-Pass Localization Architecture
- **Pass 1: Specialist Translation:** Sent in batches with Speaker IDs. Focuses on subject recovery and honorific retention.
- **Pass 2: Grammarian Polish:** Uses a **256K context window** (MoE optimized). Focuses on punctuation, flow, and linking split segments with ellipses (...) to maintain narrative continuity.
- **Density Guard:** Cap Japanese segment width at **24 characters** to ensure English translations adhere to professional subtitle layout standards.
- **Song Detection:** Utilize a text-based heuristic (symbols, repetition, density) combined with **0.50 VAD Onset** to proactively filter music and opening/ending themes.

### 4. Model Specifics (Anime-Whisper)
- **Automatic Patching:** The `convert_anime_whisper.py` script MUST automatically patch the CTranslate2 `config.json` to use **128 Mel bins** (Whisper v3 requirement) and ensure all required preprocessor metadata is downloaded.

### 5. Mandatory Syntax Check
- **Zero-Tolerance for IndentationErrors:** After ANY modification to Python source files, the `./check_syntax.sh` script MUST be executed to verify structural integrity.
- **Pre-Execution Check:** Never finish a task without confirming all files pass the syntax check.

## 📝 Engineering Standards

### Logging & Thread Safety
- **Context Attachment:** All background threads MUST use `add_script_run_ctx` to link to the Streamlit session context for safe UI interaction.
- **Shutdown Handling:** Monitor a global `threading.Event()` for app shutdown to allow C++ extensions to unload gracefully.
- **httpx Silencing:** Set `logging.getLogger("httpx").setLevel(logging.WARNING)` to prevent Ollama API spam.

## ⚠️ Known Version Constraints
- **Streamlit >= 1.54.0:** Required for `@st.fragment` support.
- **PyTorch Stack >= 2.10.0:** Required for `torchcodec` 0.10.0 compatibility.
- **TF32:** Forced enable globally for RTX 3090/4090 performance.
