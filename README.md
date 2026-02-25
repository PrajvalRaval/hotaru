# 🏯 Hotaru (蛍)
### Precision Japanese-to-English Subtitling. Powered by Local AI.

[![Python](https://img.shields.io/badge/Python-3.10+-00bfa5?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Ollama](https://img.shields.io/badge/Ollama-LLM-teal?style=for-the-badge&logo=ollama)](https://ollama.com/)
[![WhisperX](https://img.shields.io/badge/WhisperX-ASR-00bfa5?style=for-the-badge)](https://github.com/m-bain/whisperX)
[![License: MIT](https://img.shields.io/badge/License-MIT-00bfa5?style=for-the-badge)](LICENSE)

**Hotaru** is a high-performance, locally-hosted AI pipeline designed to generate professional-grade English subtitles for Japanese anime. By bridging the gap between frame-perfect audio alignment (**WhisperX**) and natural, idiomatic dialogue translation (**Ollama/Qwen3**), Hotaru delivers fan-sub quality at AI speed.

---

## 🚀 Why Hotaru?

Stop waiting for fan-subs or settling for literal, "robotic" official translations. Hotaru gives you the power to create beautiful, context-aware subtitles on your own hardware.

### ✨ Key Features
*   **🎯 Word-Level Precision:** Powered by WhisperX phoneme alignment for frame-perfect subtitle timing that never drifts.
*   **🌎 Natural Dialogue:** Leveraging Ollama (Qwen3:30b optimized) to adapt Japanese idioms, honorifics, and context into punchy, natural English.
*   **🎵 Smart Lyric Detection:** Automatic heuristic detection of Opening/Ending themes. Hotaru skips the "hallucination-heavy" song segments to keep your SRTs clean.
*   **💾 Persistent Task Management:** A uTorrent-inspired dashboard that remembers your progress. Close the tab, refresh the page—your tasks stay queued and safe.
*   **🛠️ Proxmox-Style Monitoring:** Real-time system telemetry and an adaptive task tray for power users who need to see every log and VRAM fluctuation.
*   **☢️ Nuclear VRAM Reset:** Advanced memory management designed for 30B+ parameter models on consumer GPUs (optimized for RTX 3090/4090).

---

## 🎨 Sleek UI/UX
Hotaru features a professional **Teal-Dark Aesthetic** designed for long-session productivity:
*   **Fixed Sidebar:** Instant access to System Status, Transcription settings, and Ollama configuration.
*   **Adaptive Task Tray:** A docked, collapsible log console that moves with your sidebar.
*   **One-Click Preview:** View and edit your generated SRTs directly in the browser before downloading.

---

## 🛠️ Quick Start

### 1. Prerequisites
*   **NVIDIA GPU:** (8GB+ VRAM recommended, 24GB+ for 30B models)
*   **Python:** 3.10 or higher
*   **Ollama:** Installed and running (`ollama serve`)

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/hotaru.git
cd hotaru

# Install dependencies (WhisperX requires ffmpeg)
pip install -r requirements.txt
```

### 3. Launch
```bash
streamlit run app.py
```

---

## 🧠 The Pipeline
1.  **Transcribe:** Japanese audio to text via WhisperX.
2.  **Align:** Phoneme-level timestamp refinement.
3.  **Smart Split:** Logic-based segmenting at punctuation and silence gaps.
4.  **Reset:** Pre-emptive VRAM purging to clear the way for the LLM.
5.  **Translate:** Batch-processed LLM translation (25 lines per chunk).
6.  **Smooth:** Post-processing to eliminate duplicate frames and flicker.

---

## 🔒 Privacy First
No cloud APIs. No subscription fees. No data harvesting.
Everything stays on your machine.

---

## 🤝 Contributing
Hotaru is an evolving project. Feel free to open issues or submit PRs to improve the translation heuristics or VRAM management.

---

*“Subtitling isn't just about translation; it's about preserving the soul of the scene.”* — **Hotaru Team**
