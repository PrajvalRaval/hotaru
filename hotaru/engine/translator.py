import logging
import re
import time
import ollama
from typing import List, Dict, Optional, Callable

logger = logging.getLogger("HotaruTranslator")

class OllamaTranslator:
    """Handles all interaction with Ollama for translation and polishing."""
    
    def __init__(self, host: str = "http://localhost:11434"):
        self.client = ollama.Client(host=host, timeout=None)

    def translate_chunk(self, segments: List[Dict], model: str, tolerance_pct: int = 5, 
                        cancel_check: Optional[Callable[[], bool]] = None, 
                        log_callback: Optional[Callable[[str], None]] = None) -> List[str]:
        """Translates a batch of lines using Ollama with robust parsing."""
        if not model: return [s["text"] for s in segments]
        
        def log(msg: str):
            if log_callback: log_callback(msg)
            else: logger.info(msg)

        original_lines = [s["text"].strip() for s in segments]
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
            "1. CHARACTER VOICE: Use Speaker IDs and Japanese pronouns (ore, boku, watashi) to infer persona.\n"
            "2. SUBJECT RECOVERY: Japanese often omits subjects (I, you, he, she). Use Speaker IDs and context.\n"
            "3. HONORIFICS: Retain all Japanese honorifics (-san, -kun, -chan, -sama, -senpai, -sensei) exactly as they are.\n"
            "4. LOCALIZATION: Avoid literal translation. Use culturally equivalent English slang, idioms, and contractions.\n"
            "5. NON-VERBAL CUES: Keep cues like (Laughing), (Screaming) in parentheses.\n"
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
        max_missing = max(1, int(len(segments) * (tolerance_pct / 100)))
        
        for attempt in range(3):
            if cancel_check and cancel_check(): raise InterruptedError()
            
            try:
                log(f"Ollama Request (Attempt {attempt+1}/3): Sending {len(segments)} lines...")
                current_system = system_prompt
                if attempt > 0:
                    current_system += "\nSTRICT: Ensure your output is ONLY the Line X: lines. Do not add any conversational filler."

                response = self.client.chat(
                    model=model, 
                    messages=[{'role': 'system', 'content': current_system}, {'role': 'user', 'content': user_prompt}],
                    options={'num_ctx': 32768, 'num_predict': 24576, 'temperature': 0.4, 'top_p': 0.9},
                    keep_alive="5m", stream=False
                )
                
                # Extraction
                if hasattr(response, 'message') and hasattr(response.message, 'content'):
                    content = response.message.content.strip()
                elif isinstance(response, dict):
                    content = response.get('message', {}).get('content', '').strip()
                else: content = str(response).strip()

                if not content:
                    log(f"⚠️ Ollama returned empty. Retrying...")
                    time.sleep(5); continue

                results = [None] * len(segments)
                lines = content.split('\n')
                unmatched_lines = []
                match_count = 0
                for line in lines:
                    line = line.strip()
                    if not line: continue
                    match = re.search(r'^(?:Line\s*)?(\d+)\s*(?:\[.*?\])?\s*[:\.]\s*(.*)', line, flags=re.IGNORECASE)
                    if match:
                        idx = int(match.group(1)) - 1
                        if 0 <= idx < len(segments):
                            results[idx] = match.group(2).strip().strip('"')
                            match_count += 1
                        else: unmatched_lines.append(line)
                    else: unmatched_lines.append(line)
                
                log(f"✅ Received {len(content)} chars. Parsed {match_count}/{len(segments)} lines.")
                
                unmatched_ptr = 0
                for k in range(len(results)):
                    if results[k] is None and unmatched_ptr < len(unmatched_lines):
                        clean_text = re.sub(r'^(?:Line\s*)?\d+\s*[:\.]\s*', '', unmatched_lines[unmatched_ptr], flags=re.IGNORECASE).strip()
                        results[k] = clean_text
                        unmatched_ptr += 1
                
                cleaned = [r if r is not None else original_lines[k] for k, r in enumerate(results)]
                missing_count = sum(1 for r in results if r is None)
                
                if missing_count < min_missing:
                    min_missing = missing_count
                    best_effort_cleaned = cleaned
                
                if missing_count <= max_missing: return cleaned
                time.sleep(1)
            except Exception as e:
                log(f"❌ Ollama Connection Error: {e}")
                if attempt == 2: break
                time.sleep(2)
        
        return best_effort_cleaned

    def smooth_translation(self, segments: List[Dict], model: str, 
                           cancel_check: Optional[Callable[[], bool]] = None, 
                           log_callback: Optional[Callable[[str], None]] = None) -> List[str]:
        """Pass 2: The 'Subber's Polish'. Refines translated dialogue for better flow."""
        if not model: return [s.get("translated_text", s["text"]) for s in segments]
        
        def log(msg: str):
            if log_callback: log_callback(msg)
            else: logger.info(msg)

        original_translations = [s.get("translated_text", s["text"]).strip() for s in segments]
        chunk_size = 50
        all_refined_texts = []
        
        system_prompt = (
            "### ROLE:\n"
            "You are an expert Anime Script Editor and Grammarian. You are receiving a series of short, \"pre-split\" translation chunks aligned to exact audio timestamps.\n\n"
            "### THE CHALLENGE:\n"
            "Because the input text was \"chopped\" strictly for timing based on audio pauses, it often lacks punctuation. Some sentences may be split across segments.\n\n"
            "### TASK:\n"
            "Use your 256K context window to link these segments grammatically. If a character hasn't finished their thought, use ellipses (...) to show continuity.\n\n"
            "### INPUT FORMAT:\n"
            "Each line is provided as: `Line X [SPEAKER_ID]: JA_TEXT -> EN_RAW_TEXT`\n\n"
            "### CRITICAL SMOOTHING RULES:\n"
            "1. RECONSTRUCT PUNCTUATION based on JA_TEXT context.\n"
            "2. CONTINUITY: Use ellipses (...) at the end of Line X if it is an incomplete thought.\n"
            "3. PRONOUN RECOVERY: Use SPEAKER_ID to fix missing subjects.\n"
            "4. CAPITALIZATION: Fix based on sentence position.\n"
            "5. FLOW & TONE: Ensure characters maintain consistent voices.\n\n"
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
                response = self.client.chat(
                    model=model,
                    messages=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': f"Review and polish these subtitles:\n\n{input_text}\n\nOutput:"}],
                    options={'num_ctx': 32768, 'num_predict': 24576, 'temperature': 0.1, 'top_p': 0.9},
                    keep_alive="5m", stream=False
                )
                
                if hasattr(response, 'message') and hasattr(response.message, 'content'):
                    content = response.message.content.strip()
                elif isinstance(response, dict):
                    content = response.get('message', {}).get('content', '').strip()
                else: content = str(response).strip()

                chunk_results = [None] * len(chunk)
                lines = content.split('\n')
                for line in lines:
                    line = line.strip()
                    match = re.search(r'^(?:Line\s*)?(\d+)\s*[:\.]\s*(.*)', line, flags=re.IGNORECASE)
                    if match:
                        idx = int(match.group(1)) - (i + 1)
                        if 0 <= idx < len(chunk): chunk_results[idx] = match.group(2).strip().strip('"')
                
                for k, r in enumerate(chunk_results):
                    all_refined_texts.append(r if r is not None else chunk[k].get("translated_text", chunk[k]["text"]))
                
                match_count = sum(1 for r in chunk_results if r is not None)
                log(f"✅ Refined {match_count}/{len(chunk)} lines in chunk.")
            except Exception as e:
                log(f"⚠️ Polish chunk failed: {e}")
                for seg in chunk: all_refined_texts.append(seg.get("translated_text", seg["text"]))

        return all_refined_texts
