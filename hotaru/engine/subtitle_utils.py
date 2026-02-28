import re
import srt
from datetime import timedelta
from typing import List, Dict

def is_likely_song(text: str) -> bool:
    """Heuristic to detect if a Japanese segment is likely a song/lyric."""
    if not text: return False
    if any(char in text for char in ["♪", "♫", "〜", "~"]): return True
    if re.search(r'(.)\1{4,}', text): return True 
    if len(text) > 30 and len(set(text)) < 8: return True
    return False

def resegment_results(result: Dict, max_line_width: int, max_line_count: int, language: str = "ja") -> List[Dict]:
    """Precision 'Speaker-Aware' Resegmenter with Duration Guards."""
    if not result["segments"]: return []

    effective_width = 24 if language == "ja" else max_line_width
    new_segments = []
    punctuation = ("。", "！", "？", "!", "?", "…")

    for segment in result["segments"]:
        words = segment.get("words", [])
        current_speaker = segment.get("speaker", "UNKNOWN")
        buffer_words = []
        line_count = 1
        line_len = 0
        
        if not words:
            if segment.get("text", "").strip():
                # Ensure minimum 100ms duration
                s_end = max(segment["end"], segment["start"] + 0.1)
                new_segments.append({
                    "start": segment["start"], "end": s_end,
                    "text": segment["text"].strip(), "speaker": current_speaker
                })
            continue

        for i, word in enumerate(words):
            w_text = word["word"]
            w_start = word.get("start")
            w_end = word.get("end")
            w_speaker = word.get("speaker", current_speaker)
            
            is_punc = any(p in w_text for p in punctuation)
            speaker_changed = (w_speaker != current_speaker and len(buffer_words) > 0)
            
            # Gap detection (>0.3s)
            has_gap = False
            if i < len(words) - 1:
                next_w = words[i+1]
                next_start = next_w.get("start")
                if next_start is not None and w_end is not None:
                    # 0.3s is the sweet spot for clean vocal tracks
                    if (next_start - w_end) > 0.3: has_gap = True

            word_stripped = w_text.strip()
            needs_wrap = line_len > 0 and (line_len + len(word_stripped)) > effective_width
            
            if len(buffer_words) > 0 and (is_punc or has_gap or speaker_changed or (needs_wrap and line_count >= max_line_count)):
                s_start = buffer_words[0].get("start", segment["start"])
                # Use current word start as end fallback to prevent overlap
                s_end = buffer_words[-1].get("end", w_start if w_start is not None else segment["end"])
                # Final safety check
                if s_start is None: s_start = segment["start"]
                if s_end is None: s_end = segment["end"]
                s_end = max(s_end, s_start + 0.1)
                
                if language == "ja":
                    s_text = "".join([w["word"] for w in buffer_words]).replace(" ", "")
                else:
                    s_text = " ".join([w["word"] for w in buffer_words]).replace("\n ", "\n")
                
                if s_text.strip():
                    new_segments.append({"start": s_start, "end": s_end, "text": s_text.strip(), "speaker": current_speaker})
                
                buffer_words = []
                line_count = 1
                line_len = 0
                current_speaker = w_speaker

            if needs_wrap:
                word["word"] = "\n" + word_stripped
                line_count += 1
                line_len = len(word_stripped)
            else:
                line_len += len(w_text)
            buffer_words.append(word)

        if buffer_words:
            s_start = buffer_words[0].get("start", segment["start"])
            s_end = buffer_words[-1].get("end", s_start + 0.1)
            s_end = max(s_end, s_start + 0.1)
            
            if language == "ja":
                s_text = "".join([w["word"] for w in buffer_words]).replace(" ", "")
            else:
                s_text = " ".join([w["word"] for w in buffer_words]).replace("\n ", "\n")
            if s_text.strip():
                new_segments.append({"start": s_start, "end": s_end, "text": s_text.strip(), "speaker": current_speaker})

    # Collision Guard: Tighten gaps while maintaining validity
    for i in range(len(new_segments) - 1):
        if new_segments[i]["end"] > new_segments[i+1]["start"]:
            # Snap to next start, but keep at least 100ms duration
            new_segments[i]["end"] = max(new_segments[i+1]["start"], new_segments[i]["start"] + 0.1)

    return new_segments

def generate_srt(segments: List[Dict], output_path: str):
    """Converts internal segment list to standard SRT format."""
    srt_segments = []
    for i, seg in enumerate(segments):
        content = seg.get("translated_text", seg["text"])
        if not content or content.strip() == "": continue
        
        content = re.sub(r'\[SPEAKER_\d+\]\s*', '', content)
        content = re.sub(r'^(?:Line\s*)?\d+\s*[:\.]\s*', '', content, flags=re.IGNORECASE)
        
        srt_segments.append(srt.Subtitle(
            index=len(srt_segments)+1, 
            start=timedelta(seconds=seg["start"]),
            end=timedelta(seconds=seg["end"]), 
            content=content.strip()
        ))
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(srt.compose(srt_segments))
