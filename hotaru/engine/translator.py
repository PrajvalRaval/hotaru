import logging
import re
import time
import ollama
from typing import List, Dict, Optional, Callable, Tuple

logger = logging.getLogger("HotaruTranslator")

class OllamaTranslator:
    """One-Pass Localization Engine optimized for Qwen3 30B MoE."""
    
    def __init__(self, host: str = "http://localhost:11434"):
        self.host = host.rstrip('/')
        self.client = ollama.Client(host=self.host, timeout=None)

    def get_model_context_length(self, model_name: str) -> int:
        """Fetches the native context window size of the loaded model from Ollama."""
        try:
            import requests
            response = requests.post(f"{self.host}/api/show", json={"name": model_name})
            data = response.json()
            
            if "model_info" in data:
                for key, value in data["model_info"].items():
                    if key.endswith(".context_length"):
                        return int(value)
        except Exception as e:
            logger.warning(f"Failed to fetch context length for {model_name}: {e}")
            
        return 8192 # Safe fallback if not found

    def translate_batch(self, segments: List[Dict], model: str, tolerance_pct: int = 5, 
                        cancel_check: Optional[Callable[[], bool]] = None, 
                        log_callback: Optional[Callable[[str], None]] = None,
                        num_ctx: int = 32768) -> Tuple[List[str], int]:
        """One-Pass 'Direct-to-Fansub' Localization. Handles translation and script doctoring in one pass."""
        if not model: return [s["text"] for s in segments], len(segments)
        
        def log(msg: str):
            if log_callback: log_callback(msg)
            else: logger.info(msg)

        original_lines = [s["text"].strip() for s in segments]
        input_text = "\n".join([
            f"Line {idx+1}: {s['text']}" 
            for idx, s in enumerate(segments)
        ])
        
        system_prompt = (
            "### ROLE\n"
            "You are an elite Anime Localization Director. Your objective is to translate and adapt Japanese anime dialogue into natural, cinematic English subtitles.\n\n"
            "### INPUT FORMAT\n"
            "You will receive a batch of chronologically ordered dialogue lines. "
            "Format: `Line [Index]: [JA_TEXT]`\n\n"
            "### CRITICAL RULES (THE CONSTITUTION)\n"
            "1. STRICT 1:1 MAPPING: You MUST output exactly ONE translated line for every single input line. "
            "Merging, dropping, or adding extra lines is strictly forbidden. The sequence numbers must match the input exactly.\n"
            "2. GRAMMAR STITCHING: A single sentence may be split across multiple lines due to audio pauses. "
            "Ensure the English flow connects across these breaks. Use ellipses (...) at the end of a line if the thought continues.\n"
            "3. BLIND CONTEXT INFERENCE: You do not have speaker tags. You MUST deduce speaker changes and subjects (I, You, He, She) using linguistic cues "
            "(e.g., 'ore' vs 'watashi', sentence-ending particles like '-ze' vs '-kashira', and politeness levels).\n"
            "4. CHARACTER CONTINUITY: Track characters through your context window. If a character speaks roughly in Line 1, maintain that tone if they speak again in Line 5.\n"
            "5. HONORIFICS & LOCALIZATION: Retain all honorifics (-san, -kun, -sama). Do not translate idioms literally; use culturally equivalent English phrases.\n\n"
            "### OUTPUT CONSTRAINTS (STRICT)\n"
            "- FORMAT: `Line [Index]: [Polished Translation]`\n"
            "- Output ONLY the localized English text with its index. No original Japanese, no metadata, no introductions.\n"
            "- DO NOT wrap your response in markdown code blocks (no backticks).\n\n"
            "### EXAMPLE\n"
            "Input:\n"
            "Line 45: おい、待てよ！\n"
            "Line 46: お前を絶対に...\n"
            "Line 47: 許さないからな！\n"
            "Line 48: ご、ごめんなさい！\n\n"
            "Output:\n"
            "Line 45: Hey, wait up!\n"
            "Line 46: I'm absolutely...\n"
            "Line 47: never going to forgive you!\n"
            "Line 48: I-I'm so sorry!"
        )
        
        user_prompt = f"Localize the following anime script:\n\n{input_text}\n\nOutput:"
        best_effort_cleaned = original_lines
        min_missing = len(segments)
        max_missing = max(2, int(len(segments) * (tolerance_pct / 100)))
        
        for attempt in range(3):
            if cancel_check and cancel_check(): raise InterruptedError()
            
            try:
                log(f"Ollama Request (Attempt {attempt+1}/3): Localizing {len(segments)} lines...")
                current_system = system_prompt
                if attempt > 0:
                    current_system += "\nSTRICT: Output ONLY the Line X: lines. Do not add any filler."

                num_predict = min(24576, max(2048, num_ctx // 2))

                response = self.client.chat(
                    model=model, 
                    messages=[{'role': 'system', 'content': current_system}, {'role': 'user', 'content': user_prompt}],
                    options={'num_ctx': num_ctx, 'num_predict': num_predict, 'temperature': 0.3, 'top_p': 0.9},
                    keep_alive="5m", stream=False
                )
                
                if hasattr(response, 'message') and hasattr(response.message, 'content'):
                    content = response.message.content.strip()
                elif isinstance(response, dict):
                    content = response.get('message', {}).get('content', '').strip()
                else: content = str(response).strip()

                if not content:
                    log(f"⚠️ Ollama returned empty. Retrying..."); time.sleep(5); continue

                # INDEX-LOCKED PARSING (Absolute Sync Guard)
                results = [None] * len(segments)
                lines = content.split('\n')
                match_count = 0
                line_pattern = re.compile(r'^Line\s*(\d+)\s*:\s*(.*)', re.IGNORECASE)

                for line in lines:
                    line = line.strip()
                    if not line: continue
                    
                    match = line_pattern.match(line)
                    if match:
                        idx = int(match.group(1)) - 1
                        if 0 <= idx < len(segments):
                            if results[idx] is None:
                                text = match.group(2).strip().strip('"')
                                # Strip hallucinated prefixes often added by MoE models
                                text = re.sub(r'^\[Polished Translation\]\s*', '', text, flags=re.IGNORECASE)
                                results[idx] = text
                                match_count += 1
                    else:
                        log(f"⚠️ Formatting mismatch ignored: {line}")

                log(f"✅ Received {len(content)} chars. Parsed {match_count}/{len(segments)} lines.")

                # Step 5: The Failsafe Assembler
                # We ensure NO SHIFTING occurs. If an index is missing, we use original JP.
                cleaned = []
                for k in range(len(segments)):
                    if results[k] is not None:
                        cleaned.append(results[k])
                    else:
                        log(f"🚨 FAILSAFE TRIGGERED: LLM skipped Line {k+1}. Falling back to raw Japanese.")
                        cleaned.append(f"[UNTRANSLATED] {original_lines[k]}")

                missing_count = sum(1 for r in results if r is None)

                if missing_count < min_missing:
                    min_missing = missing_count; best_effort_cleaned = cleaned
                
                if missing_count <= max_missing: return cleaned, len(segments)
                
                # --- SMART FALLBACK MECHANISM ---
                done_reason = ""
                if hasattr(response, 'done_reason'):
                    done_reason = response.done_reason
                elif isinstance(response, dict):
                    done_reason = response.get('done_reason', "")

                # Circuit Breaker: If we hit a hard token limit or failed the tolerance test
                if done_reason == "length" or missing_count > max_missing:
                    if done_reason == "length":
                        log(f"🛑 Circuit Breaker: Hit token limit (done_reason='length').")
                    
                    # Dynamic Chunk Reduction: Divide and Conquer
                    if len(segments) >= 4:
                        log(f"✂️ Chunk too complex (Missed {missing_count}). Splitting {len(segments)} lines into two smaller chunks...")
                        mid = len(segments) // 2
                        first_half, new_size_1 = self.translate_batch(segments[:mid], model, tolerance_pct, cancel_check, log_callback)
                        second_half, new_size_2 = self.translate_batch(segments[mid:], model, tolerance_pct, cancel_check, log_callback)
                        return first_half + second_half, min(new_size_1, new_size_2)
                    else:
                        log(f"⚠️ Chunk too small to split. Retrying same chunk...")
                
                time.sleep(1)
            except Exception as e:
                log(f"❌ Ollama Connection Error: {e}")
                if attempt == 2: break
                time.sleep(2)
        
        return best_effort_cleaned, len(segments)
