# 🏯 Hotaru (蛍) — Professional Japanese-to-English Anime Subtitling Engine

![Hotaru Dashboard - AI Anime Subtitle Generator](Screenshot.png)

### 🚀 High-Performance Local AI Pipeline for Precision Subtitling

[![Python 3.13+](https://img.shields.io/badge/Python-3.13+-00bfa5?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Ollama LLM](https://img.shields.io/badge/Ollama-256K_Context-teal?style=for-the-badge&logo=ollama)](https://ollama.com/)
[![WhisperX ASR](https://img.shields.io/badge/WhisperX-Phoneme_Alignment-00bfa5?style=for-the-badge)](https://github.com/m-bain/whisperX)
[![GPU Accelerated](https://img.shields.io/badge/RTX_4090-Optimized-orange?style=for-the-badge&logo=nvidia)](https://www.nvidia.com/)

**Hotaru** is a high-accuracy, locally-hosted AI subtitling tool designed to transform raw Japanese anime into professional-grade English subs. By bridging the gap between frame-perfect audio alignment (**WhisperX**) and context-aware dialogue translation (**Ollama/Qwen3**), Hotaru delivers "fansub quality" at machine speed without ever leaving your hardware.

---

## ✨ Key Features

*   **🎯 Word-Level Precision:** Powered by **WhisperX phoneme alignment** for frame-perfect subtitle timing that never drifts, even during rapid-fire dialogue or overlaps.
*   **👥 Speaker-Aware Timing:** Monitors speakers at the **word level**. If Person A is interrupted by Person B, Hotaru instantly flushes the buffer and starts a new subtitle block, preventing "spoiler" dialogue and merging.
*   **🌎 Two-Pass Localization:**
    *   **Pass 1 (Specialist):** Context-aware translation that recovers dropped subjects and preserves honorifics.
    *   **Pass 2 (Grammarian):** Refines punctuation, capitalization, and flow using a **256K context window** to link split segments with ellipses (...).
*   **🎵 Automated Song Filtering:** Integrated **Heuristic Song Detection** and strict **VAD (0.50 Onset)** to proactively skip opening/ending themes and background music hallucinations.
*   **✂️ Deterministic Resegmentation:** A robust **Buffer-and-Flush** algorithm that strictly respects Whisper segment boundaries and silence gaps (>0.4s) to ensure subtitles clear instantly when speech stops.
*   **📐 Custom Alignment Models:** Full support for specifying specialized Hugging Face model IDs (e.g., `jonatasgrosman/wav2vec2-large-xlsr-53-japanese`) for superior word-level timing.
*   **💾 Persistent Task Management:** A **uTorrent-inspired dashboard** that persists state to disk. Close your browser, refresh the page, or restart the app—your translation queue and progress stay exactly where you left them.
*   **🛠️ Proxmox-Style Monitoring:** Real-time system telemetry and an adaptive task tray for power users. Monitor **VRAM fluctuations**, CPU usage, and granular engine logs in a live console.
*   **☢️ Nuclear VRAM/RAM Reset:** Advanced memory orchestration designed for **30B+ parameter models** on consumer GPUs. Hotaru aggressively purges both GPU and System memory between phases.

---

## ⚡ Technical Optimizations

### 🚀 Professional Formatting Constraints
Configure `Max Width` and `Max Lines` directly from the UI. Hotaru's resegmenter ensures Japanese text is capped at **24 characters** before translation, resulting in English dialogue that fits perfectly within subtitle boundaries.

### ⚖️ Dynamic Translation Tolerance
Includes a configurable **Percentage-Based Tolerance** slider. Control exactly how many missing or malformed lines are acceptable before the engine triggers an automatic retry.

### 🛡️ System RAM Guard
The engine actively monitors host RAM availability. If system memory drops below 1.5GB, Hotaru pauses and triggers **Aggressive Garbage Collection** to prevent OOM termination.

---

## 🛠️ Installation & Setup

### 1. Prerequisites
*   **Hardware:** NVIDIA GPU (24GB VRAM recommended for 30B+ models).
*   **Backend:** [Ollama](https://ollama.com/) installed and running (`ollama serve`).
*   **System:** `ffmpeg` installed on your host OS.

### 2. Quick Start
```bash
# Clone the repository
git clone https://github.com/yourusername/hotaru.git
cd hotaru

# Install Python requirements
pip install -r requirements.txt

# Launch the dashboard
source venv/bin/activate
streamlit run app.py
```

---

## 🧠 The AI Pipeline
1.  **Extract:** Automated high-fidelity audio extraction from MP4/MKV containers.
2.  **ASR:** Japanese transcription via **WhisperX** with **0.50 VAD Onset** for noise filtering.
3.  **Align:** Phoneme-level refinement using standard or custom **Wav2Vec2** models.
4.  **Resegment:** Speaker-aware **Buffer-and-Flush** splitting based on natural pauses and density.
5.  **Translate:** Localization pass using **Anime Localization Specialist** persona.
6.  **Smooth:** Final refinement pass using **Anime Script Editor and Grammarian** persona with MoE-optimized context linking.

---

## 🔒 Privacy First
No cloud APIs. No subscription fees. No data harvesting.
**Everything stays on your machine.**

---

## 🤝 Contributing
Hotaru is an evolving project. Feel free to open issues or submit PRs to improve the translation heuristics, UI responsiveness, or VRAM management.

---

*“Subtitling isn't just about translation; it's about preserving the soul of the scene.”* — **Hotaru Team**
