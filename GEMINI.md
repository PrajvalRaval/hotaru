# Hotaru - Development Context

This file contains foundational mandates and architectural state for Hotaru. These instructions take precedence over general defaults.

## 🎯 Core Project Goal
To provide the highest quality Japanese-to-English subtitles for anime using a hybrid pipeline of local AI models (Vocal Isolation, WhisperX for timing, Ollama for translation).

## 🛠️ Architectural Mandates

### 0. The Teardown (Phase 0)
- **Vocal Isolation:** When enabled, the engine MUST pass the raw audio through **Demucs** before ingestion. WhisperX, VAD, and Alignment should then operate ONLY on the pristine isolated vocal track to ensure deterministic timing and zero hallucinations.

### 1. Resource & Memory Orchestration
- **Strategic Heartbeats:** Use `run_every` ONLY for independent hardware monitoring and individual task progress tracking. These MUST remain in separate fragments to avoid WebSocket flood.
- **UI Preservation:** Never modify or refactor already working UI components unless explicitly prompted by the user.
- **Documentation Integrity:** Never perform destructive updates to `README.md`. Updates should reflect the current state of the project but must preserve existing feature descriptions, setup instructions, and architectural overviews.
- **Nuclear VRAM Reset:** Between transcription (WhisperX) and translation (Ollama), ALL heavy models MUST be explicitly deleted.
- **Segmented Logic:** Keep the codebase modular. Logic belongs in the `hotaru/` package; `app.py` is strictly for coordination and UI assembly.

### 2. Subtitle Timing & UI Dynamics
- **Deterministic Resegmentation:** Use a **Buffer-and-Flush** algorithm that strictly respects Whisper segment boundaries. NEVER allow a subtitle to cross a VAD-detected silence gap.
- **Speaker-Aware Word Splitting:** Monitor speaker IDs at the **word level**. Trigger an instant buffer flush whenever the speaker changes to prevent dialogue merging and "spoiler" subtitles.
- **Localized Fallbacks:** For unaligned words, use rhythmic interpolation relative to neighboring words. NEVER snap timing to the beginning of a 30-second Whisper block.
- **Aggressive End-Padding:** Apply a tight (150-200ms) reading buffer only when speech is continuous. Clear the screen instantly if a silence gap (>0.4s) or segment boundary is reached.

### 3. One-Pass Localization Architecture
- **Direct-to-Fansub Pass:** Leverage the **Qwen3 30B MoE** parallel reasoning capabilities to perform translation and script doctoring in a SINGLE pass.
- **Context Utilization:** Use the **256K context window** to simultaneously handle subject recovery, honorific retention, and grammatical stitching (using ellipses `...` for split thoughts).
- **Density Guard:** Cap Japanese segment width at **24 characters** to ensure English translations adhere to professional subtitle layout standards.
- **Song Detection:** Utilize a text-based heuristic (symbols, repetition, density) combined with **0.50 VAD Onset** to proactively filter musical themes.

### 4. Mandatory Syntax Check
- **Zero-Tolerance for IndentationErrors:** After ANY modification to Python source files, the `scripts/check_syntax.sh` script MUST be executed to verify structural integrity.
- **Recursive Checking:** Syntax checks must cover the root directory and all sub-packages in `hotaru/`.

## 📝 Engineering Standards

### Logging & Thread Safety
- **Context Attachment:** All background threads MUST use `add_script_run_ctx` to link to the Streamlit session context for safe UI interaction.
- **Draining Backup Logs:** UI fragments must proactively drain the `BACKUP_LOGS` buffer to ensure background thread activity is visible without delay.
- **Ollama Persistence:** Pre-emptively unload Ollama models between phases to ensure a clean VRAM state for the next model load.

## ⚠️ Known Version Constraints
- **Streamlit >= 1.54.0:** Required for `@st.fragment` support.
- **PyTorch Stack >= 2.10.0:** Required for `torchcodec` 0.10.0 compatibility.
- **TF32:** Forced enable globally for RTX 3090/4090 performance.
