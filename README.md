# 🏯 Hotaru (蛍) — Professional Japanese-to-English Anime Subtitling Engine

![Hotaru Dashboard - AI Anime Subtitle Generator](Screenshot.png)

### 🚀 High-Performance Local AI Pipeline for Precision Subtitling

[![Python 3.13+](https://img.shields.io/badge/Python-3.13+-00bfa5?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Ollama LLM](https://img.shields.io/badge/Ollama-256K_Context-teal?style=for-the-badge&logo=ollama)](https://ollama.com/)
[![WhisperX ASR](https://img.shields.io/badge/WhisperX-Phoneme_Alignment-00bfa5?style=for-the-badge)](https://github.com/m-bain/whisperX)
[![GPU Accelerated](https://img.shields.io/badge/RTX_4090-Optimized-orange?style=for-the-badge&logo=nvidia)](https://www.nvidia.com/)

**Hotaru** is a high-accuracy, locally-hosted AI subtitling tool designed to transform raw Japanese anime into professional-grade English subs. By bridging the gap between frame-perfect audio alignment (**WhisperX**) and context-aware dialogue translation (**Ollama/Qwen3**), Hotaru delivers "fansub quality" at machine speed without ever leaving your hardware.

---

## 🚀 Why Hotaru?

Stop waiting for fan-subs or settling for literal, "robotic" official translations. Hotaru gives you the power to create beautiful, context-aware subtitles on your own hardware.

*   **🎯 Precision Timing:** Frame-perfect alignment that snaps to speech using word-level phoneme data.
*   **🌎 Cultural Accuracy:** AI personas that understand honorifics, pro-drop subjects, and character "voice."
*   **🔒 Complete Privacy:** No cloud APIs, no data harvesting, no subscription fees.

---

## 🧠 The AI Pipeline
1.  **Isolate:** Vocal Isolation (Demucs) to strip BGM/SFX.
2.  **Transcribe:** Japanese transcription via **WhisperX** with **0.50 VAD Onset**.
3.  **Align:** Phoneme-level refinement using standard or custom **Wav2Vec2** models.
4.  **Resegment:** Speaker-aware **Buffer-and-Flush** splitting based on natural pauses and density.
5.  **Localize:** One-pass localization and polishing using the **Anime Localization Director** persona with MoE-optimized context linking and blind context inference.

---

## ✨ Key Features

*   **🎤 Vocal Isolation (Phase 0):** Integrated **Demucs (htdemucs)** to strip BGM and SFX. Feed pristine, voice-only tracks into WhisperX for 100% deterministic VAD and zero hallucinations.
*   **🎯 Word-Level Precision:** Powered by **WhisperX phoneme alignment** for frame-perfect subtitle timing that never drifts.
*   **🌎 One-Pass Localization:**
    *   **Direct-to-Fansub:** Context-aware translation that recovers dropped subjects and preserves honorifics.
    *   **Blind Context Inference:** Leverages a **256K context window** to deduce speaker changes and maintain narrative continuity without relying on hardcoded speaker tags.
*   **🛡️ Resilient LLM Parsing:**
    *   **Failsafe Parser:** Strict regex anchoring prevents the "Zip Desync Trap." If the LLM hallucinates and skips a line, it seamlessly falls back to Japanese, maintaining 100% frame-perfect subtitle alignment.
    *   **Smart Fallback:** Integrates dynamic chunk reduction (Divide & Conquer). If a token limit is exhausted, it automatically splits the workload to guarantee successful generation.
*   **🎵 Automated Song Filtering:** Integrated **Heuristic Song Detection** and strict **VAD (0.50 Onset)** to skip musical themes.

---

## 🛠️ Installation & Setup

### 1. Prerequisites
*   **Hardware:** NVIDIA GPU (24GB VRAM recommended for 30B+ models).
*   **Backend:** [Ollama](https://ollama.com/) installed and running.
*   **System:** `ffmpeg` installed on your host OS.

### 2. Quick Start
```bash
# Clone the repository
git clone https://github.com/yourusername/hotaru.git
cd hotaru

# Launch the dashboard
pip install -r requirements.txt

source venv/bin/activate
streamlit run app.py
```

---

## ⚡ Technical Optimizations

### 🚀 Segmented Architecture
Hotaru is built as a modular package. The UI, Engine, and Common utilities are strictly decoupled to ensure high performance and zero WebSocket noise.

### 🛡️ Resource Guard
Actively monitors host RAM and VRAM availability, triggering aggressive garbage collection to prevent OOM termination.

---

## 🎨 Sleek UI/UX

Hotaru features a professional **Teal-Dark Aesthetic** designed for long-session productivity:
*   **Sidebar:** Instant access to System Status, Transcription settings, and Ollama configuration.
*   **One-Click Preview:** View and edit your generated SRTs directly in the browser before downloading.

---

## 🔒 Privacy First
No cloud APIs. No data harvesting. **Everything stays on your machine.**

---