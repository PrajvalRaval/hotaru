import os
import srt
import torch
import whisperx
import ollama
from datetime import timedelta
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
    
    def __init__(self, model_size: str = "large-v3", hf_token: Optional[str] = None, ollama_host: str = "http://localhost:11434"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if torch.cuda.is_available() else "int8"
        self.model_size = model_size
        self.hf_token = hf_token
        self.ollama_host = ollama_host
        self.ollama_client = ollama.Client(host=ollama_host)
        
        logger.info(f"Initializing Hotaru Engine on {self.device} ({self.compute_type})")
        self.model = whisperx.load_model(model_size, self.device, compute_type=self.compute_type)

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

    def process_video(self, video_path: str, ollama_model: str = "qwen3:30b", log_callback: Optional[Callable[[str], None]] = None, skip_songs: bool = True) -> List[Dict[str, Any]]:
        """
        Full pipeline: Audio Extraction -> Transcription -> Alignment -> [Diarization] -> Pre-Splitting -> Translation -> Smoothing.
        """
        def log(msg: str):
            logger.info(msg)
            if log_callback: log_callback(msg)

        # 0. Pre-emptive Ollama Unload to ensure WhisperX has full VRAM
        try:
            if ollama_model: self.ollama_client.generate(model=ollama_model, keep_alive=0)
        except: pass

        free_vram = self.get_free_vram()
        log(f"System Check: {free_vram:.2f}GB VRAM available.")
        
        # 1. Load Audio
        audio = whisperx.load_audio(video_path)
        
        # 2. Transcription
        log(f"Transcribing (Japanese)...")
        result = self.model.transcribe(audio, batch_size=32, language="ja")
        
        # 3. Alignment (Word-level precision)
        log("Aligning phonemes for precision...")
        model_a, metadata = whisperx.load_align_model(language_code="ja", device=self.device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, self.device, return_char_alignments=False)
        
        # 4. Diarization (Optional)
        if self.diarize_model:
            log("Identifying speakers...")
            # Ensure TF32 is still active (some libraries might toggle it)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            diarize_segments = self.diarize_model(audio)
            result = whisperx.assign_word_speakers(diarize_segments, result)
        
        # 5. Smart Pre-Splitting (Optimizes for Grammar + Dynamic UI)
        log("Segmenting Japanese into dynamic chunks...")
        segmented_ja = self._smart_split_japanese(result["segments"])

        # 6. Nuclear VRAM Reset: Unload heavy WhisperX models before loading LLM
        log("WhisperX pipeline complete. Performing Nuclear VRAM Reset for Ollama...")
        del self.model
        del model_a
        if self.diarize_model: del self.diarize_model
        
        if self.device == "cuda":
            for _ in range(3):
                torch.cuda.empty_cache()
                gc.collect()
                time.sleep(1)
            log(f"VRAM Reset Complete. Free VRAM: {self.get_free_vram():.2f}GB")

        # 7. Translation with Ollama (Batched for speed)
        log(f"Translating {len(segmented_ja)} segments using {ollama_model}...")
        translated_segments = []
        chunk_size = 25 
        
        for i in range(0, len(segmented_ja), chunk_size):
            chunk = segmented_ja[i:i + chunk_size]
            log(f"Translating chunk {i//chunk_size + 1}/{(len(segmented_ja) + chunk_size - 1)//chunk_size}...")
            
            # Identify segments to translate vs skip
            indices_to_translate = []
            for j, seg in enumerate(chunk):
                if skip_songs and self._is_likely_song(seg):
                    seg["translated_text"] = "" # Empty means it will be skipped in SRT generation
                    log(f"  > Skipping potential song/hallucination: {seg['text'][:30]}...")
                else:
                    indices_to_translate.append(j)
            
            if indices_to_translate:
                sub_chunk = [chunk[j] for j in indices_to_translate]
                translated_texts = self._translate_chunk_with_ollama(sub_chunk, ollama_model)
                
                for j, trans_text in zip(indices_to_translate, translated_texts):
                    chunk[j]["translated_text"] = trans_text
            
            for seg in chunk:
                translated_segments.append(seg)
        
        # 8. Post-Processing: Smooth out timings and repetitions
        log("Smoothing out subtitles and removing duplicates...")
        final_segments = self._smooth_subtitles(translated_segments)

        # 9. Final Ollama Unload
        try:
            self.ollama_client.generate(model=ollama_model, keep_alive=0)
        except: pass

        return final_segments

    def _is_likely_song(self, seg: Dict) -> bool:
        """Heuristic to detect if a segment is likely part of a song or lyric-heavy content."""
        text = seg.get("text", "").strip()
        duration = seg.get("end", 0) - seg.get("start", 0)
        
        if not text: return False
        
        # 1. Musical symbols (Common indicators of lyrics)
        if any(c in text for c in ["♪", "♫", "♬", "♩", "～", "〜"]):
            return True
            
        # 2. Text density (lyrics often get compressed by WhisperX in short segments)
        # > 30 characters in < 1.5s is extremely dense for dialogue.
        if duration < 1.5 and len(text) > 30:
            return True
            
        # 3. Repetitive characters (hallucinations or chorus repetition)
        if len(text) > 20 and len(set(text)) / len(text) < 0.25:
            return True
            
        # 4. Long vowel extensions (e.g., Japanese musical notation styles)
        if "ーー" in text or "！！" in text:
            return True

        return False

    def _smart_split_japanese(self, segments: List[Dict], max_duration: float = 5.0, min_gap: float = 0.5, min_display_time: float = 1.2) -> List[Dict]:
        """Splits segments based on punctuation and silence to ensure readability."""
        new_segments = []
        punctuation = ("。", "！", "？", "!", "?")
        
        for seg in segments:
            if "words" not in seg or not seg["words"]:
                if len(seg["text"].strip()) < 2: continue
                new_segments.append(seg)
                continue
            
            current_words = []
            
            def commit_mini_segment(words_to_add):
                start = words_to_add[0]["start"]
                end = words_to_add[-1]["end"]
                duration = end - start
                
                mini_text = "".join([w.get("word", "") for w in words_to_add]).strip()
                
                # Filter out hallucinations (single chars over long time)
                if len(mini_text) <= 2 and duration > 1.5: return 
                # Enforce minimum display time
                if duration < min_display_time: end = start + min_display_time

                if mini_text:
                    new_segments.append({
                        "start": start, "end": end, "text": mini_text,
                        "speaker": words_to_add[-1].get("speaker", seg.get("speaker", "UNKNOWN"))
                    })

            for word in seg["words"]:
                if "start" not in word:
                    if current_words: current_words.append(word)
                    continue
                
                should_split = False
                if current_words:
                    duration = word["end"] - current_words[0]["start"]
                    prev_end = current_words[-1].get("end")
                    gap = (word["start"] - prev_end) if prev_end else 0
                    
                    if duration > 1.0: # Minimum duration before splitting is allowed
                        if any(p in current_words[-1].get("word", "") for p in punctuation):
                            should_split = True
                        elif gap > min_gap:
                            should_split = True
                    
                    if duration > max_duration: should_split = True
                
                if should_split and current_words:
                    commit_mini_segment(current_words)
                    current_words = []
                current_words.append(word)
            
            if current_words: commit_mini_segment(current_words)
                
        # Collision detection: prevent overlapping timestamps
        for i in range(len(new_segments) - 1):
            if new_segments[i]["end"] > new_segments[i+1]["start"]:
                new_segments[i]["end"] = new_segments[i+1]["start"]

        return new_segments

    def _smooth_subtitles(self, segments: List[Dict]) -> List[Dict]:
        """Merges consecutive segments with identical text to prevent flashing."""
        if not segments: return []
        smoothed = []
        curr = segments[0]
        
        for i in range(1, len(segments)):
            nxt = segments[i]
            # Normalize for comparison
            text_a = re.sub(r'[^\w\s]', '', curr["translated_text"].lower().strip())
            text_b = re.sub(r'[^\w\s]', '', nxt["translated_text"].lower().strip())
            
            if text_a == text_b and text_a != "":
                curr["end"] = nxt["end"] # Extend duration
            else:
                smoothed.append(curr)
                curr = nxt
        smoothed.append(curr)
        return smoothed

    def _translate_chunk_with_ollama(self, segments: List[Dict], model: str) -> List[str]:
        """Translates a batch of lines using Ollama with robust parsing and tolerance for missing lines."""
        if not model: return [s["text"] for s in segments]
        original_lines = [s["text"].strip() for s in segments]
        input_text = "\n".join([f"Line {idx+1}: {text}" for idx, text in enumerate(original_lines)])
        
        prompt = (
            "You are a professional Anime Translator. Translate the following Japanese dialogue into natural, idiomatic, and punchy English.\n"
            "Rules:\n"
            "1. Output the translated text using the format: 'Line X: [Translation]'.\n"
            "2. Match line count EXACTLY. You MUST output a translation for every single line provided.\n"
            "3. If a line is a song (lyrics), leave it empty: 'Line X: '.\n"
            "4. NO intro, NO outro, NO explanations. Output ONLY the translated lines.\n"
            "5. Maintain honorifics context (senpai, san) but adapt to natural English dialogue flow.\n\n"
            f"Input:\n{input_text}\n\nOutput:"
        )
        
        best_effort_cleaned = original_lines
        min_missing = len(segments)
        
        try:
            for attempt in range(3):
                try:
                    response = self.ollama_client.chat(model=model, messages=[{'role': 'user', 'content': prompt}])
                    content = response['message']['content'].strip()
                    
                    results = [None] * len(segments)
                    lines = content.split('\n')
                    
                    unmatched_lines = []
                    for line in lines:
                        line = line.strip()
                        if not line: continue
                        match = re.search(r'^(?:Line\s*)?(\d+)\s*[:\.]\s*(.*)', line, flags=re.IGNORECASE)
                        if match:
                            idx = int(match.group(1)) - 1
                            text = match.group(2).strip().strip('"')
                            if 0 <= idx < len(segments):
                                results[idx] = text
                            else: unmatched_lines.append(line)
                        else: unmatched_lines.append(line)
                    
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
                    
                    # Success criteria: 0-2 missing lines allowed (tolerance)
                    if missing_count <= 2:
                        if missing_count > 0:
                            logger.info(f"Accepted translation with {missing_count} missing lines (within tolerance).")
                        return cleaned
                    
                    logger.warning(f"Retry {attempt+1}: Missing {missing_count}/{len(segments)} translations (Tolerance: 2).")
                    time.sleep(1)
                except Exception as e:
                    if attempt == 2: break
                    time.sleep(2)
            
            logger.warning(f"Exhausted retries. Returning best effort with {min_missing} missing lines.")
            return best_effort_cleaned
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return original_lines

    def generate_srt(self, segments: List[Dict], output_path: str):
        """Converts internal segment list to standard SRT format."""
        srt_segments = []
        for i, seg in enumerate(segments):
            content = seg.get("translated_text", seg["text"])
            if not content or content.strip() == "": continue
            
            # Remove any lingering speaker labels
            content = re.sub(r'\[SPEAKER_\d+\]\s*', '', content)
            
            srt_segments.append(srt.Subtitle(
                index=len(srt_segments)+1, 
                start=timedelta(seconds=seg["start"]),
                end=timedelta(seconds=seg["end"]), 
                content=content.strip()
            ))
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(srt.compose(srt_segments))
