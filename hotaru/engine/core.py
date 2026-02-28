import os
import torch
import whisperx
import gc
import time
import logging
import cv2
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable

from hotaru.engine.translator import OllamaTranslator
from hotaru.engine.subtitle_utils import generate_srt
from hotaru.engine.audio_utils import isolate_vocals
from janome.tokenizer import Tokenizer

logger = logging.getLogger("HotaruEngine")

class TranscribeEngine:
    """Core engine orchestrating WhisperX and Ollama."""
    def __init__(self, model_size: str = "kotoba-tech/kotoba-whisper-v2.0-faster", 
                 ollama_host: str = "http://localhost:11434"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        self.tokenizer = Tokenizer(wakati=True)
        
        # Local model path handling
        if model_size == "litagin/anime-whisper":
            model_size = os.path.join("models", "anime-whisper-ct2")
            if not os.path.exists(model_size):
                model_size = "kotoba-tech/kotoba-whisper-v2.0-faster"
        
        self.model_size = model_size
        self.translator = OllamaTranslator(host=ollama_host)
        
        # VAD Config (Truly Aggressive Initialization to prevent 30s blobs)
        vad_options = {
            "vad_onset": 0.700,
            "vad_offset": 0.650
        }
        
        self.model = whisperx.load_model(
            model_size, self.device, compute_type=self.compute_type,
            vad_method="silero", vad_options=vad_options
        )

    def get_free_vram(self) -> float:
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
            v_free, _ = torch.cuda.mem_get_info()
            return v_free / 1024**3
        return 0.0

    def process_video(self, video_path: str, ollama_model: str = "qwen3:30b",
                      log_callback: Optional[Callable[[str], None]] = None,
                      timing_offset: float = 0.0,
                      tolerance_pct: int = 5, cancel_check: Optional[Callable[[], bool]] = None,
                      max_line_width: int = 42, max_line_count: int = 2,
                      align_model: Optional[str] = None, whisper_chunk_size: int = 30) -> List[Dict[str, Any]]:        
        def log(msg: str):
            # The global logger now handles [HH:MM:SS] LEVEL prefixes
            if log_callback: log_callback(msg)
            else: logger.info(msg)

        def check_abort():
            time.sleep(0.1)
            if cancel_check and cancel_check():
                log("🚫 Task cancellation requested.")
                raise InterruptedError("Task cancelled by user.")

        # Initialize tracking variables for cleanup
        temp_vocal_path = None
        from hotaru.common.constants import UPLOAD_DIR, OUTPUT_DIR
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        
        try:
            # 0. Vocal Isolation (The Teardown)
            log("═"*40)
            log(f"🎤 PHASE 0: VOCAL ISOLATION - {os.path.basename(video_path)}")
            log("═"*40)
            log("🎬 Extracting high-fidelity audio for separation...")
            temp_vocal_path = isolate_vocals(video_path, UPLOAD_DIR, device=self.device)
            audio_source = temp_vocal_path
            check_abort()

            # 0.1 Pre-emptive Ollama Unload
            try:
                if ollama_model: self.translator.client.generate(model=ollama_model, keep_alive=0)
            except: pass

            log("═"*40)
            log("🎧 PHASE 1: TRANSCRIPTION")
            log("═"*40)
            
            # 1. Transcribe
            log(f"🎬 Loading audio from {os.path.basename(audio_source)}...")
            audio = whisperx.load_audio(audio_source)
            log(f"🗣️ Transcribing Japanese (VAD Onset: 0.50, Chunk: {whisper_chunk_size}s)...")
            result = self.model.transcribe(audio, batch_size=16, language="ja", chunk_size=whisper_chunk_size)
            check_abort()

            # --- DEBUG EXPORT: RAW TRANSCRIPTION ---
            try:
                raw_trans_path = os.path.join(OUTPUT_DIR, f"{base_name}_raw_transcription.srt")
                generate_srt(result["segments"], raw_trans_path)
                log(f"💾 Saved raw transcription: {os.path.basename(raw_trans_path)}")
            except Exception as e:
                logger.warning(f"Failed to save raw transcription debug SRT: {e}")
            
            # 1.5 Morphological Chunking (Janome)
            # We slice the long transcription segments into subtitle-sized chunks BEFORE alignment.
            log(f"🧪 Performing morphological chunking (Max Width: {max_line_width})...")
            chunked_segments = []
            limit = max_line_width * max_line_count
            
            for seg in result["segments"]:
                tokens = list(self.tokenizer.tokenize(seg["text"])) # Tokens are now strings (wakati=True)
                seg_text = seg["text"]
                seg_start = seg["start"]
                seg_end = seg["end"]
                seg_dur = seg_end - seg_start
                total_len = len(seg_text)
                
                if total_len == 0:
                    chunked_segments.append(seg)
                    continue
                
                current_tokens = []
                current_len = 0
                processed_len = 0
                
                for t in tokens:
                    t_surface = t
                    if current_len + len(t_surface) > limit and current_tokens:
                        c_start = seg_start + (processed_len - current_len) / total_len * seg_dur
                        c_end = seg_start + processed_len / total_len * seg_dur
                        chunked_segments.append({
                            "start": c_start, "end": c_end,
                            "text": " ".join(current_tokens)
                        })
                        current_tokens = [t]
                        current_len = len(t_surface)
                    else:
                        current_tokens.append(t)
                        current_len += len(t_surface)
                    processed_len += len(t_surface)
                
                if current_tokens:
                    c_start = seg_start + (processed_len - current_len) / total_len * seg_dur
                    chunked_segments.append({
                        "start": c_start, "end": seg_end,
                        "text": " ".join(current_tokens)
                    })
            
            # --- DEBUG EXPORT: MORPHOLOGICAL CHUNKING ---
            try:
                chunked_path = os.path.join(OUTPUT_DIR, f"{base_name}_morphological_chunked.srt")
                generate_srt(chunked_segments, chunked_path)
                log(f"💾 Saved morphological chunked SRT: {os.path.basename(chunked_path)}")
            except Exception as e:
                logger.warning(f"Failed to save morphological chunked debug SRT: {e}")

            # 2. Align
            log("═"*40)
            log("📏 PHASE 2: ALIGNMENT")
            log("═"*40)
            log(f"📐 Aligning phonemes using model: {align_model or 'default-ja'}...")
            model_a, metadata = whisperx.load_align_model(language_code="ja", device=self.device, model_name=align_model)
            result = whisperx.align(chunked_segments, model_a, metadata, audio, self.device, return_char_alignments=False)
            check_abort()

            # --- DEBUG EXPORT: RAW ALIGNMENT ---
            try:
                raw_align_path = os.path.join(OUTPUT_DIR, f"{base_name}_raw_alignment.srt")
                generate_srt(result["segments"], raw_align_path)
                log(f"💾 Saved raw alignment: {os.path.basename(raw_align_path)}")
            except Exception as e:
                logger.warning(f"Failed to save raw alignment debug SRT: {e}")
            
            # Use raw model segments directly (Removing Janome spaces for translation)
            segmented_ja = []
            for s in result["segments"]:
                s["text"] = s["text"].replace(" ", "")
                segmented_ja.append(s)

            if timing_offset != 0:
                log(f"⏱️ Applying manual timing offset: {timing_offset:+.3f}s")
                for seg in segmented_ja:
                    seg["start"] += timing_offset; seg["end"] += timing_offset
                    if "words" in seg:
                        for w in seg["words"]:
                            if "start" in w: w["start"] += timing_offset
                            if "end" in w: w["end"] += timing_offset

            # 5. VRAM Reset
            log("═"*40)
            log("☢️ PHASE 3: NUCLEAR VRAM RESET")
            log("═"*40)
            log("🧹 WhisperX pipeline complete. Purging GPU memory...")
            if hasattr(self, 'model'): del self.model
            if 'model_a' in locals(): del model_a
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache(); torch.cuda.ipc_collect()
                time.sleep(1)
                v_free, _ = torch.cuda.mem_get_info()
                log(f"✅ VRAM Reset Complete. Free VRAM: {v_free/1024**3:.2f}GB")

            # 6. Localization (One-Pass)
            log("═"*40)
            log("🌎 PHASE 4: OLLAMA LOCALIZATION (One-Pass)")
            log("═"*40)
            log(f"🌍 Localizing {len(segmented_ja)} segments using {ollama_model}...")
            
            # Dynamically fetch context length
            native_ctx = self.translator.get_model_context_length(ollama_model)
            
            # Calculate optimal num_ctx (cap at 32k to avoid OOM on 24GB VRAM, but allow smaller)
            eff_num_ctx = min(native_ctx, 32768)
            
            # Determine initial chunk size based on context window
            # Smaller context windows (like 40k) struggle heavily with attention drop-off on strict mapping
            if native_ctx <= 8192:
                eff_chunk_size = 4
            elif native_ctx <= 16384:
                eff_chunk_size = 5
            elif native_ctx <= 65536: # Catches models like qwen3:32b with 40k context
                eff_chunk_size = 6
            else:
                eff_chunk_size = 25 # Safe baseline for models with massive (128k+) context
                
            log(f"🧠 Model native context: {native_ctx}. Set effective num_ctx to {eff_num_ctx}, chunk size to {eff_chunk_size}.")

            translated_segments = []
            
            i = 0
            chunk_num = 1
            while i < len(segmented_ja):
                check_abort()
                
                # Estimate total chunks based on current effective chunk size
                remaining_lines = len(segmented_ja) - i
                estimated_remaining_chunks = (remaining_lines + eff_chunk_size - 1) // eff_chunk_size
                num_chunks = chunk_num + estimated_remaining_chunks - 1
                
                chunk = segmented_ja[i:i + eff_chunk_size]
                
                total_chars = sum(len(s.get("text", "")) for s in chunk)
                log(f"🌎 Localizing chunk {chunk_num}/{num_chunks} ({len(chunk)} segments, ~{total_chars} chars)...")
                
                if chunk:
                    localized_texts, new_chunk_size = self.translator.translate_batch(
                        chunk, ollama_model, tolerance_pct, cancel_check, log, num_ctx=eff_num_ctx
                    )
                    
                    if new_chunk_size < eff_chunk_size:
                        log(f"📉 Adjusting future batch size to {new_chunk_size} lines to prevent token exhaustion.")
                        eff_chunk_size = new_chunk_size
                    
                    ptr = 0
                    for seg in chunk:
                        seg["translated_text"] = localized_texts[ptr]
                        ptr += 1
                else:
                    for seg in chunk: seg["translated_text"] = ""
                
                translated_segments.extend(chunk)
                i += len(chunk)
                chunk_num += 1
            
            log("✅ Localization complete.")
            return translated_segments
            
        finally:
            if temp_vocal_path and os.path.exists(temp_vocal_path):
                try: os.remove(temp_vocal_path)
                except: pass
            try: self.translator.client.generate(model=ollama_model, keep_alive=0)
            except: pass
