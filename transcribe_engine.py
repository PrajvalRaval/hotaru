import warnings
# Silence torchcodec and torchaudio import warnings
warnings.filterwarnings("ignore", message=".*torchcodec.*")
warnings.filterwarnings("ignore", message=".*Torchaudio's I/O functions.*")

import os
import srt
import torch
import whisperx
import ollama
import psutil
from datetime import timedelta, datetime
import tempfile
from moviepy import VideoFileClip
import logging
import gc
import time
import re
from typing import List, Dict, Any, Optional, Callable

# Enable TF32 for RTX 3090/4090 performance boost
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Set up logging
logger = logging.getLogger("HotaruEngine")

class TranscribeEngine:
    """
    Core engine for transcription and translation using WhisperX and Ollama.
    Handles VRAM optimization, dynamic segment splitting, and translation post-processing.
    """
    
    def __init__(self, model_size: str = "kotoba-tech/kotoba-whisper-v2.0-faster", hf_token: Optional[str] = None, ollama_host: str = "http://localhost:11434"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Explicitly use float16 for CUDA as requested
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        
        # Point to converted local directory for anime-whisper
        if model_size == "litagin/anime-whisper":
            model_size = os.path.join("models", "anime-whisper-ct2")
            if not os.path.exists(model_size):
                logger.error(f"Anime-Whisper local path {model_size} not found. Ensure it is converted.")
                # Fallback to default if not found (though UI should prevent this)
                model_size = "kotoba-tech/kotoba-whisper-v2.0-faster"
        
        self.model_size = model_size
        self.hf_token = hf_token
        self.ollama_host = ollama_host
        # Disable timeout for LLM requests to allow for long processing times
        self.ollama_client = ollama.Client(host=ollama_host, timeout=None)
        
        logger.info(f"Initializing Hotaru Engine on {self.device} ({self.compute_type}) using model: {model_size}")
        
        # Explicit VAD options to skip music and intros:
        # onset/offset: lower values are more sensitive (more segments), 
        # higher values skip more (more aggressive).
        # Aggressive VAD to skip music and non-speech more effectively.
        vad_options = {
            "vad_onset": 0.50, # Increased from 0.363 to skip singing/melodic noise
            "vad_offset": 0.363,
            "min_silence_duration_ms": 1000,
            "speech_pad_ms": 400
        }
        
        self.model = whisperx.load_model(
            model_size, 
            self.device, 
            compute_type=self.compute_type,
            vad_method="silero", # Use Silero VAD
            vad_options=vad_options
        )

        self.diarize_model = None
        if hf_token:
            try:
                # Use correct module path for DiarizationPipeline
                self.diarize_model = whisperx.diarize.DiarizationPipeline(
                    model_name='pyannote/speaker-diarization-3.1', 
                    token=hf_token, 
                    device=self.device
                )
                logger.info("Diarization pipeline loaded successfully.")
            except Exception as e:
                logger.warning(f"Diarization load failed: {e}")

    def get_free_vram(self) -> float:
        """Returns free VRAM in GB."""
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
            v_free, v_total = torch.cuda.mem_get_info()
            return v_free / 1024**3
        return 0.0

    def process_video(self, video_path: str, ollama_model: str = "qwen3:30b", log_callback: Optional[Callable[[str], None]] = None, timing_offset: float = 0.0, chunk_size: int = 25, tolerance_pct: int = 5, cancel_check: Optional[Callable[[], bool]] = None, max_line_width: int = 42, max_line_count: int = 2, align_model: Optional[str] = None, whisper_chunk_size: int = 30) -> List[Dict[str, Any]]:
        """
        Full pipeline: Audio Extraction -> Transcription -> Alignment -> [Diarization] -> Pre-Splitting -> Translation -> Smoothing.
        """
        def log(msg: str):
            # Prepend timestamp and level for consistency across all logs (even headers)
            # This ensures the UI regex parser correctly identifies every line.
            ts = datetime.now().strftime("%H:%M:%S")
            formatted_msg = f"[{ts}] INFO: {msg}"
            
            # Only log internally if no callback is provided to prevent duplicates in the shared log stream
            if log_callback: 
                log_callback(formatted_msg)
            else: 
                logger.info(formatted_msg)

        def check_abort():
            # Brief sleep to allow other threads to breath and prevent tight loops
            time.sleep(0.1)
            if cancel_check and cancel_check():
                log("🚫 Task cancellation requested. Aborting...")
                raise InterruptedError("Task cancelled by user.")

        try:
            # 0. Pre-emptive Ollama Unload to ensure WhisperX has full VRAM
            try:
                if ollama_model: self.ollama_client.generate(model=ollama_model, keep_alive=0)
            except: pass

            log("═"*40)
            log(f"🎧 PHASE 1: TRANSCRIPTION - {os.path.basename(video_path)}")
            log("═"*40)
            
            free_vram = self.get_free_vram()
            log(f"📟 System Check: {free_vram:.2f}GB VRAM available.")
            check_abort()
            
            # 1. Load Audio
            log(f"🎬 Loading audio from video...")
            audio = whisperx.load_audio(video_path)
            check_abort()
            
            # 2. Transcription
            log(f"🗣️  Transcribing Japanese (Silero VAD, Chunk: {whisper_chunk_size}s)...")
            
            # Transcribe with VAD filtering enabled by default in WhisperX
            result = self.model.transcribe(audio, batch_size=16, language="ja", chunk_size=whisper_chunk_size)
            check_abort()
            
            # 3. Alignment (Word-level precision)
            log("═"*40)
            log("📏 PHASE 2: ALIGNMENT & DIARIZATION")
            log("═"*40)
            log(f"📐 Aligning phonemes using model: {align_model or 'default-ja'}...")
            # Use float16 for alignment as well
            model_a, metadata = whisperx.load_align_model(language_code="ja", device=self.device, model_name=align_model)
            result = whisperx.align(result["segments"], model_a, metadata, audio, self.device, return_char_alignments=False)
            check_abort()
            
            # Apply global timing offset if requested (fixes constant drift)
            if timing_offset != 0:
                log(f"⏱️  Applying manual timing offset: {timing_offset:+.3f}s")
                for seg in result["segments"]:
                    seg["start"] += timing_offset
                    seg["end"] += timing_offset
                    if "words" in seg:
                        for w in seg["words"]:
                            if "start" in w: w["start"] += timing_offset
                            if "end" in w: w["end"] += timing_offset

            # 4. Diarization (Optional)
            if self.diarize_model:
                log("👥 Identifying speakers using Pyannote...")
                # Ensure TF32 is still active
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                
                # Use the diarization pipeline
                diarize_segments = self.diarize_model(audio)
                check_abort()
                
                # Map diarization info to transcription segments
                result = whisperx.assign_word_speakers(diarize_segments, result)
                
                # Fill missing speakers for better UX
                for seg in result["segments"]:
                    if "speaker" not in seg:
                        seg["speaker"] = "UNKNOWN"

            # 5. Native WhisperX Segments
            log(f"📦 Resegmenting for subtitles (Width: {max_line_width}, Lines: {max_line_count})...")
            segmented_ja = self._resegment_results(result, max_line_width, max_line_count, language="ja")
            check_abort()

            # 6. Nuclear VRAM Reset: Unload heavy WhisperX models before loading LLM
            log("═"*40)
            log("☢️  PHASE 3: NUCLEAR VRAM RESET")
            log("═"*40)
            log("🧹 WhisperX pipeline complete. Purging GPU memory...")
            
            # Explicitly unload and delete models to free VRAM
            if hasattr(self, 'model'): del self.model
            if 'model_a' in locals(): del model_a
            if self.diarize_model:
                del self.diarize_model
                self.diarize_model = None

            # Standard cleanup
            gc.collect()
            
            # Clear CUDA Cache (VRAM)
            if self.device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                time.sleep(1)
                torch.cuda.empty_cache()
                
                # Check and log remaining memory
                v_free, v_total = torch.cuda.mem_get_info()
                log(f"✅ VRAM Reset Complete. Free VRAM: {v_free/1024**3:.2f}GB")

            # 7. Translation with Ollama (Batched for speed)
            log("═"*40)
            log("🌎 PHASE 4: OLLAMA TRANSLATION")
            log("═"*40)
            log(f"🌍 Translating {len(segmented_ja)} segments using {ollama_model}...")
            translated_segments = []
            
            # If chunk_size is 0, we translate everything in one go.
            # Ensure effective_chunk_size is at least 1 to prevent range() errors
            effective_chunk_size = chunk_size if chunk_size > 0 else max(1, len(segmented_ja))
            
            for i in range(0, len(segmented_ja), effective_chunk_size):
                check_abort()
                chunk = segmented_ja[i:i + effective_chunk_size]
                
                # Calculate metrics for logging
                total_chars = sum(len(s.get("text", "")) for s in chunk)
                est_tokens = int(total_chars * 1.3) # Rough estimate for Japanese tokens in LLM context
                
                if chunk_size == 0:
                    log(f"🚀 Translating entire transcript in one batch ({len(segmented_ja)} segments, ~{total_chars} chars).")
                else:
                    log(f"📦 Translating chunk {i//effective_chunk_size + 1}/{(len(segmented_ja) + effective_chunk_size - 1)//effective_chunk_size} ({len(chunk)} segments, ~{total_chars} chars).")
                
                # Filter out likely songs before sending to AI to save tokens and improve quality
                sub_chunk = []
                song_indices = []
                for idx, seg in enumerate(chunk):
                    if self._is_likely_song(seg.get("text", "")):
                        song_indices.append(idx)
                    else:
                        sub_chunk.append(seg)

                if sub_chunk:
                    translated_texts = self._translate_chunk_with_ollama(
                        sub_chunk, 
                        ollama_model, 
                        tolerance_pct=tolerance_pct, 
                        cancel_check=cancel_check,
                        log_callback=log
                    )
                    
                    # Re-map translated texts back to the original chunk, skipping songs
                    sub_ptr = 0
                    for idx in range(len(chunk)):
                        if idx in song_indices:
                            chunk[idx]["translated_text"] = "" # Discard song
                        else:
                            chunk[idx]["translated_text"] = translated_texts[sub_ptr]
                            sub_ptr += 1
                else:
                    # Entire chunk was songs
                    for seg in chunk:
                        seg["translated_text"] = ""
                
                for seg in chunk:
                    translated_segments.append(seg)
            
            check_abort()
            # 8. AI Smoothing Pass: The "Subber's Polish"
            log("═"*40)
            log("🪄  PHASE 5: AI POLISH (SUBBER'S POLISH)")
            log("═"*40)
            polished_texts = self._smooth_translation_with_ollama(
                translated_segments, 
                ollama_model, 
                cancel_check=cancel_check, 
                log_callback=log
            )
            for seg, polished in zip(translated_segments, polished_texts):
                seg["translated_text"] = polished

            log("🏁 Video processing complete.")
            return translated_segments
            
        finally:
            # 9. Final Ollama Unload (Always run to free VRAM)
            try:
                self.ollama_client.generate(model=ollama_model, keep_alive=0)
            except: pass

    def _translate_chunk_with_ollama(self, segments: List[Dict], model: str, tolerance_pct: int = 5, cancel_check: Optional[Callable[[], bool]] = None, log_callback: Optional[Callable[[str], None]] = None) -> List[str]:
        """Translates a batch of lines using Ollama with robust parsing and tolerance for missing lines."""
        if not model: return [s["text"] for s in segments]
        
        def log(msg: str):
            if log_callback: log_callback(msg)
            else: logger.info(msg)

        def check_abort():
            # Brief sleep to allow other threads to breath and prevent tight loops
            time.sleep(0.1)
            if cancel_check and cancel_check():
                raise InterruptedError("Translation cancelled by user.")

        original_lines = [s["text"].strip() for s in segments]
        # Include Speaker ID and Original JA in the specified format
        input_text = "\n".join([
            f"Line {idx+1} [{s.get('speaker', 'UNKNOWN')}] [{s['text']}]" 
            for idx, s in enumerate(segments)
        ])
        
        system_prompt = (
            "### ROLE\n"
            "You are an expert Anime Localization Specialist and Professional Fansubber. Your goal is to produce punchy, natural English dialogue that captures the emotional intent and character \"voice\" of the Japanese original.\n\n"
            "### INPUT FORMAT\n"
            "Lines are provided as: `Line X [ID] [JA_TEXT]`\n"
            "Example: `Line 1 [SPEAKER_00] [おい、何だよこれ！]`\n\n"
            "### CONSTITUTION & RULES\n"
            "1. CHARACTER VOICE: Use Speaker IDs and Japanese pronouns (ore, boku, watashi) to infer persona. Ensure a rude protagonist sounds rough/informal and a polite character sounds formal/gentle.\n"
            "2. SUBJECT RECOVERY: Japanese often omits subjects (I, you, he, she). Use Speaker IDs and previous context to fill in these gaps naturally in English.\n"
            "3. HONORIFICS: Retain all Japanese honorifics (-san, -kun, -chan, -sama, -senpai, -sensei) exactly as they are.\n"
            "4. LOCALIZATION: Avoid literal translation. Use culturally equivalent English slang, idioms, and contractions (don't, gonna, ain't) to make it feel like a modern anime.\n"
            "5. NON-VERBAL CUES: Keep cues like (Laughing), (Screaming), or (Sighing) in parentheses.\n"
            "6. SONG HANDLING: If a line is clearly a song/lyric, leave the translation empty: 'Line X: '.\n\n"
            "### OUTPUT CONSTRAINTS (STRICT)\n"
            "- FORMAT: Output the translated text in the format: 'Line X: [Translation]'\n"
            "- MAPPING: You MUST output exactly one translation for every input line provided.\n"
            "- NO TAGS: DO NOT include the Speaker ID, the [JA_TEXT] tag, or any metadata in the translation.\n"
            "- CLEANLINESS: No intros, no outros, no explanations. Output ONLY the Line X: lines."
        )
        
        user_prompt = f"Translate the following transcript:\n\n{input_text}\n\nOutput:"
        
        best_effort_cleaned = original_lines
        min_missing = len(segments)
        # Calculate dynamic tolerance based on chunk size (min 1 line)
        max_missing = max(1, int(len(segments) * (tolerance_pct / 100)))
        
        try:
            for attempt in range(3):
                check_abort()
                
                try:
                    log(f"Ollama Request (Attempt {attempt+1}/3): Sending {len(segments)} lines...")
                    
                    # On retries, use a slightly simpler prompt to reduce LLM confusion/load
                    current_system = system_prompt
                    if attempt > 0:
                        current_system += "\nSTRICT: Ensure your output is ONLY the Line X: lines. Do not add any conversational filler."

                    # Use chat with system message for better adherence
                    try:
                        response = self.ollama_client.chat(
                            model=model, 
                            messages=[
                                {'role': 'system', 'content': current_system},
                                {'role': 'user', 'content': user_prompt}
                            ],
                            options={
                                'num_ctx': 32768,    # Large context window for episode-long batches
                                'num_predict': 24576, # Increased headroom for long translation output
                                'temperature': 0.4, 
                                'top_p': 0.9
                            },
                            keep_alive="5m", 
                            stream=False
                        )
                    except Exception as e:
                        log(f"❌ Ollama Connection Error: {e}")
                        time.sleep(5)
                        continue
                    
                    # Robust response extraction
                    if hasattr(response, 'message') and hasattr(response.message, 'content'):
                        content = response.message.content.strip()
                    elif isinstance(response, dict):
                        content = response.get('message', {}).get('content', '').strip()
                    else:
                        content = str(response).strip()

                    done_reason = getattr(response, 'done_reason', 'unknown') if not isinstance(response, dict) else response.get('done_reason', 'unknown')
                    resp_chars = len(content)
                    
                    if done_reason == "length":
                        log(f"⚠️ Warning: Model reached output limit (98+ lines is likely too many). Please reduce 'Chunk Size' to 25.")

                    if resp_chars == 0:
                        log(f"⚠️ Ollama returned empty (Reason: {done_reason}). Retrying...")
                        time.sleep(5)
                        continue

                    results = [None] * len(segments)
                    lines = content.split('\n')
                    
                    unmatched_lines = []
                    match_count = 0
                    for line in lines:
                        line = line.strip()
                        if not line: continue
                        # Match: Line X [OPTIONAL_SPEAKER]: Translation
                        match = re.search(r'^(?:Line\s*)?(\d+)\s*(?:\[.*?\])?\s*[:\.]\s*(.*)', line, flags=re.IGNORECASE)
                        if match:
                            idx = int(match.group(1)) - 1
                            text = match.group(2).strip().strip('"')
                            if 0 <= idx < len(segments):
                                results[idx] = text
                                match_count += 1
                            else: unmatched_lines.append(line)
                        else: unmatched_lines.append(line)
                    
                    # Condensed log for UI
                    log(f"✅ Received {resp_chars} chars. Parsed {match_count}/{len(segments)} lines.")
                    
                    unmatched_ptr = 0
                    for k in range(len(results)):
                        if results[k] is None and unmatched_ptr < len(unmatched_lines):
                            clean_text = re.sub(r'^(?:Line\s*)?\d+\s*[:\.]\s*', '', unmatched_lines[unmatched_ptr], flags=re.IGNORECASE).strip()
                            results[k] = clean_text
                            unmatched_ptr += 1
                    
                    cleaned = []
                    missing_count = 0
                    for k, r in enumerate(results):
                        if r is None:
                            cleaned.append(original_lines[k])
                            missing_count += 1
                        else: cleaned.append(r)
                    
                    # Track best result so far
                    if missing_count < min_missing:
                        min_missing = missing_count
                        best_effort_cleaned = cleaned
                    
                    # Success criteria: missing lines within tolerance
                    if missing_count <= max_missing:
                        if missing_count > 0:
                            logger.info(f"Accepted translation with {missing_count} missing lines (Tolerance: {max_missing}).")
                        return cleaned
                    
                    logger.warning(f"Retry {attempt+1}: Missing {missing_count}/{len(segments)} translations (Tolerance: {max_missing}).")
                    time.sleep(1)
                except Exception as e:
                    if attempt == 2: break
                    time.sleep(2)
            
            logger.warning(f"Exhausted retries. Returning best effort with {min_missing} missing lines.")
            return best_effort_cleaned
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return original_lines

    def _smooth_translation_with_ollama(self, segments: List[Dict], model: str, cancel_check: Optional[Callable[[], bool]] = None, log_callback: Optional[Callable[[str], None]] = None) -> List[str]:
        """Pass 2: The 'Subber's Polish'. Refines translated dialogue for better flow and context."""
        if not model: return [s.get("translated_text", s["text"]) for s in segments]
        
        def log(msg: str):
            if log_callback: log_callback(msg)
            else: logger.info(msg)

        original_translations = [s.get("translated_text", s["text"]).strip() for s in segments]
        
        # We must chunk the polishing phase too, otherwise we hit output token limits with 100+ segments
        # 50 lines per chunk is usually the sweet spot for a 32K context model
        chunk_size = 50
        all_refined_texts = []
        
        system_prompt = (
            "### ROLE:\n"
            "You are an expert Anime Script Editor and Grammarian. You are receiving a series of short, \"pre-split\" translation chunks that have been aligned to exact audio timestamps.\n\n"
            "### THE CHALLENGE:\n"
            "Because the input text was \"chopped\" strictly for timing based on audio pauses, it often lacks punctuation. Some sentences may be split across segments.\n\n"
            "### TASK:\n"
            "Use your 256K context window to link these segments grammatically. If a character hasn't finished their thought at the end of a segment, use ellipses (...) to show continuity.\n\n"
            "### INPUT FORMAT:\n"
            "Each line is provided as: `Line X [SPEAKER_ID]: JA_TEXT -> EN_RAW_TEXT`\n\n"
            "### CRITICAL SMOOTHING RULES:\n"
            "1. RECONSTRUCT PUNCTUATION: Add periods, question marks, and exclamation points based on the JA_TEXT context.\n"
            "2. CONTINUITY: If a sentence spans across Line X and Line Y, ensure the grammar connects. Use ellipses (...) at the end of Line X if it is an incomplete thought.\n"
            "3. PRONOUN RECOVERY: Use SPEAKER_ID to fix missing subjects (I, You, He, She).\n"
            "4. CAPITALIZATION: Fix capitalization based on the sentence's position.\n"
            "5. FLOW & TONE: Ensure characters (Rude Male vs Polite Sister) maintain consistent voices.\n\n"
            "### OUTPUT RULES:\n"
            "- Return ONLY the final smoothed English lines in the format 'Line X: [Polished Translation]'.\n"
            "- You MUST return EXACTLY the same number of lines as provided.\n"
            "- ONE LINE PER CHUNK.\n"
            "- DO NOT add Speaker IDs, Japanese, or any introductory text."
        )

        for i in range(0, len(segments), chunk_size):
            if cancel_check and cancel_check(): raise InterruptedError()
            
            chunk = segments[i:i + chunk_size]
            input_text = "\n".join([
                f"Line {idx+i+1} [{s.get('speaker', 'UNKNOWN')}]: {s['text']} -> {s.get('translated_text', s['text'])}" 
                for idx, s in enumerate(chunk)
            ])
            
            log(f"Ollama Polish Request: Refining chunk {i//chunk_size + 1}/{(len(segments) + chunk_size - 1)//chunk_size} ({len(chunk)} lines)...")
            
            try:
                response = self.ollama_client.chat(
                    model=model,
                    messages=[
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': f"Review and polish these translated subtitles:\n\n{input_text}\n\nOutput:"}
                    ],
                    options={
                        'num_ctx': 32768,
                        'num_predict': 24576,
                        'temperature': 0.1,
                        'top_p': 0.9
                    },
                    keep_alive="5m",
                    stream=False
                )
                
                if hasattr(response, 'message') and hasattr(response.message, 'content'):
                    content = response.message.content.strip()
                elif isinstance(response, dict):
                    content = response.get('message', {}).get('content', '').strip()
                else:
                    content = str(response).strip()

                chunk_results = [None] * len(chunk)
                lines = content.split('\n')
                match_count = 0
                for line in lines:
                    line = line.strip()
                    if not line: continue
                    match = re.search(r'^(?:Line\s*)?(\d+)\s*[:\.]\s*(.*)', line, flags=re.IGNORECASE)
                    if match:
                        idx = int(match.group(1)) - (i + 1)
                        text = match.group(2).strip().strip('"')
                        if 0 <= idx < len(chunk):
                            chunk_results[idx] = text
                            match_count += 1
                
                for k, r in enumerate(chunk_results):
                    all_refined_texts.append(r if r is not None else chunk[k].get("translated_text", chunk[k]["text"]))
                
                log(f"✅ Refined {match_count}/{len(chunk)} lines in chunk.")
            except Exception as e:
                log(f"⚠️ Polish chunk failed: {e}")
                for seg in chunk:
                    all_refined_texts.append(seg.get("translated_text", seg["text"]))

        return all_refined_texts

    def _resegment_results(self, result: Dict, max_line_width: int, max_line_count: int, language: str = "ja") -> List[Dict]:
        """
        Precision 'Speaker-Aware' Resegmenter.
        Prevents dialogue merging and ensures subtitles only appear when the specific speaker is active.
        """
        if not result["segments"]: return []

        effective_width = 24 if language == "ja" else max_line_width
        new_segments = []
        punctuation = ("。", "！", "？", "!", "?", "…")

        for segment in result["segments"]:
            words = segment.get("words", [])
            # Initial speaker from segment level
            current_speaker = segment.get("speaker", "UNKNOWN")
            
            buffer_words = []
            line_count = 1
            line_len = 0
            
            if not words:
                if segment.get("text", "").strip():
                    new_segments.append({
                        "start": segment["start"], "end": segment["end"],
                        "text": segment["text"].strip(), "speaker": current_speaker
                    })
                continue

            for i, word in enumerate(words):
                w_text = word["word"]
                w_start = word.get("start")
                w_end = word.get("end")
                # Detect word-level speaker (added by diarization)
                w_speaker = word.get("speaker", current_speaker)
                
                # Triggers
                is_punc = any(p in w_text for p in punctuation)
                speaker_changed = (w_speaker != current_speaker and len(buffer_words) > 0)
                
                has_gap = False
                if i < len(words) - 1:
                    next_w = words[i+1]
                    if w_end and next_w.get("start"):
                        if (next_w["start"] - w_end) > 0.4: has_gap = True

                word_stripped = w_text.strip()
                needs_wrap = line_len > 0 and (line_len + len(word_stripped)) > effective_width
                
                # TRIGGER FLUSH: Punc, Gap, Max Lines, or SPEAKER CHANGE
                if len(buffer_words) > 0 and (is_punc or has_gap or speaker_changed or (needs_wrap and line_count >= max_line_count)):
                    # Determine start/end using only the words in THIS specific sub-block
                    # Fallback to current word timing instead of segment start to prevent 'hanging'
                    s_start = buffer_words[0].get("start", w_start if w_start else segment["start"])
                    s_end = buffer_words[-1].get("end", w_start if w_start else segment["end"])
                    
                    if language == "ja":
                        s_text = "".join([w["word"] for w in buffer_words]).replace(" ", "")
                    else:
                        s_text = " ".join([w["word"] for w in buffer_words]).replace("\n ", "\n")
                    
                    if s_text.strip():
                        new_segments.append({"start": s_start, "end": s_end, "text": s_text.strip(), "speaker": current_speaker})
                    
                    buffer_words = []
                    line_count = 1
                    line_len = 0
                    # Update active speaker tracking
                    current_speaker = w_speaker

                # Wrap logic
                if needs_wrap:
                    word["word"] = "\n" + word_stripped
                    line_count += 1
                    line_len = len(word_stripped)
                else:
                    line_len += len(w_text)
                
                buffer_words.append(word)

            # Final flush for segment
            if buffer_words:
                s_start = buffer_words[0].get("start", segment["start"])
                s_end = buffer_words[-1].get("end", segment["end"])
                if language == "ja":
                    s_text = "".join([w["word"] for w in buffer_words]).replace(" ", "")
                else:
                    s_text = " ".join([w["word"] for w in buffer_words]).replace("\n ", "\n")
                if s_text.strip():
                    new_segments.append({"start": s_start, "end": s_end, "text": s_text.strip(), "speaker": current_speaker})

        # Collision Guard
        for i in range(len(new_segments) - 1):
            if new_segments[i]["end"] > new_segments[i+1]["start"]:
                new_segments[i]["end"] = new_segments[i+1]["start"]

        return new_segments

    def _is_likely_song(self, text: str) -> bool:
        """Heuristic to detect if a Japanese segment is likely a song/lyric or music hallucination."""
        if not text: return False
        # 1. Musical symbols
        if any(char in text for char in ["♪", "♫", "〜", "~"]): return True
        
        # 2. Excessive repetition (common in music hallucinations)
        if re.search(r'(.)\1{4,}', text): return True # 5+ identical characters
        
        # 3. Low character variety / High density
        # Songs often produce long strings of repeated syllables
        if len(text) > 30 and len(set(text)) < 8: return True
        
        return False

    def generate_srt(self, segments: List[Dict], output_path: str):
        """Converts internal segment list to standard SRT format."""
        srt_segments = []
        for i, seg in enumerate(segments):
            content = seg.get("translated_text", seg["text"])
            if not content or content.strip() == "": continue
            
            # Remove any lingering speaker labels
            content = re.sub(r'\[SPEAKER_\d+\]\s*', '', content)
            # Remove 'Line X:' prefix from smoothing/translation phase
            content = re.sub(r'^(?:Line\s*)?\d+\s*[:\.]\s*', '', content, flags=re.IGNORECASE)
            
            srt_segments.append(srt.Subtitle(
                index=len(srt_segments)+1, 
                start=timedelta(seconds=seg["start"]),
                end=timedelta(seconds=seg["end"]), 
                content=content.strip()
            ))
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(srt.compose(srt_segments))
