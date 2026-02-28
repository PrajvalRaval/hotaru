import re
import srt
from datetime import timedelta
from typing import List, Dict

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

def snap_segments_to_words(segments: List[Dict], max_pause: float = 0.4, max_chars: int = 40, min_duration: float = 1.5) -> List[Dict]:
    """
    A robust, 4-pass Word Bounding algorithm that repairs missing timestamps,
    ignores segment boundaries, enforces strict silence cuts, and ensures
    minimum reading duration via look-ahead padding.
    """
    
    # --- PASS 1: Flatten and Clean ---
    # Strip artificial segment boundaries and create a continuous stream.
    all_words = []
    for seg in segments:
        seg_speaker = seg.get("speaker", "UNKNOWN")
        for w in seg.get("words", []):
            text = w.get("word", "").replace(" ", "")
            if not text: continue
            
            all_words.append({
                "text": text,
                "start": w.get("start"),
                "end": w.get("end"),
                "speaker": w.get("speaker", seg_speaker)
            })

    if not all_words:
        return segments

    # --- PASS 2: Repair and Interpolate ---
    # Fix missing timestamps by interpolating from neighbors.
    for i, word in enumerate(all_words):
        # Fix missing Start Time
        if word["start"] is None:
            if i > 0 and all_words[i-1]["end"] is not None:
                word["start"] = all_words[i-1]["end"] + 0.001
            else:
                # Fallback to the very beginning or next available start
                word["start"] = 0.0

        # Fix missing End Time
        if word["end"] is None:
            if i < len(all_words) - 1 and all_words[i+1]["start"] is not None:
                word["end"] = all_words[i+1]["start"] - 0.001
            else:
                word["end"] = word["start"] + 0.1 # 100ms fallback

        # Fix rounding errors (Start > End)
        if word["start"] > word["end"]:
            word["end"] = word["start"] + 0.05

    # --- PASS 3: Safe Chunking & Snapping ---
    # Rebuild the segments using the repaired continuous stream.
    raw_chunks = []
    current_chunk = []
    current_len = 0
    
    for i, word in enumerate(all_words):
        if current_chunk:
            last_word = current_chunk[-1]
            pause_duration = word["start"] - last_word["end"]
            
            # Triggers: Silence, Character Limit, or Speaker Change
            is_long_pause = pause_duration > max_pause
            is_char_limit = (current_len + len(word["text"])) > max_chars
            speaker_changed = word["speaker"] != current_chunk[0]["speaker"]
            
            if is_long_pause or is_char_limit or speaker_changed:
                # Flush block
                raw_chunks.append({
                    "start": current_chunk[0]["start"],
                    "end": current_chunk[-1]["end"],
                    "text": "".join([w["text"] for w in current_chunk]),
                    "speaker": current_chunk[0]["speaker"],
                    "words": current_chunk
                })
                current_chunk = []
                current_len = 0
                
        current_chunk.append(word)
        current_len += len(word["text"])
        
    if current_chunk:
        raw_chunks.append({
            "start": current_chunk[0]["start"],
            "end": current_chunk[-1]["end"],
            "text": "".join([w["text"] for w in current_chunk]),
            "speaker": current_chunk[0]["speaker"],
            "words": current_chunk
        })

    # --- PASS 4: Minimum Duration Padding & Overlap Protection ---
    # Enforce Netflix/BBC standards for reading speed.
    min_gap = 0.1 # 100ms blank screen between subtitles
    for i, chunk in enumerate(raw_chunks):
        original_duration = chunk["end"] - chunk["start"]
        
        if original_duration < min_duration:
            desired_end = chunk["start"] + min_duration
            
            # Check if stretching it will crash into the NEXT subtitle
            if i < len(raw_chunks) - 1:
                next_start = raw_chunks[i+1]["start"]
                max_allowed_end = next_start - min_gap
                
                # Stretch it to the minimum duration, OR right up until the next line starts
                chunk["end"] = min(desired_end, max_allowed_end)
            else:
                # Last line of the script can stretch freely
                chunk["end"] = desired_end
                
            # Failsafe: Ensure we never shrink the line
            chunk["end"] = max(chunk["end"], chunk["start"] + original_duration)
                
    return raw_chunks
