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

def snap_segments_to_words(segments: List[Dict], max_pause: float = 0.4, max_chars: int = 40) -> List[Dict]:
    """
    Refines segments by looking strictly at word-level timings.
    Forces a cut if silence exceeds max_pause or character count exceeds max_chars.
    This bypasses the loose 'segment bounding box' and snaps strictly to audio.
    """
    new_segments = []
    
    for seg in segments:
        words = seg.get("words", [])
        if not words:
            new_segments.append(seg)
            continue
            
        buffer = []
        current_len = 0
        
        for w in words:
            # Skip words without timing
            if "start" not in w or "end" not in w:
                buffer.append(w)
                continue
                
            if buffer:
                # Get the end time of the last timed word in buffer
                last_end = next((x["end"] for x in reversed(buffer) if "end" in x), None)
                
                # Triggers: Silence or Character Limit
                pause_too_long = last_end is not None and (w["start"] - last_end > max_pause)
                word_text = w.get("word", "").strip().replace(" ", "")
                char_limit_hit = (current_len + len(word_text)) > max_chars
                
                if pause_too_long or char_limit_hit:
                    # Flush the current buffer into a perfectly snapped segment
                    flush_start = next((x["start"] for x in buffer if "start" in x), seg["start"])
                    flush_end = last_end if last_end is not None else w["start"]
                    flush_text = "".join([x.get("word", "") for x in buffer]).replace(" ", "").strip()
                    
                    if flush_text:
                        new_segments.append({
                            "start": flush_start,
                            "end": flush_end,
                            "text": flush_text,
                            "words": buffer,
                            "speaker": seg.get("speaker", "UNKNOWN")
                        })
                    buffer = []
                    current_len = 0
            
            buffer.append(w)
            current_len += len(w.get("word", "").strip().replace(" ", ""))
            
        if buffer:
            flush_start = next((x["start"] for x in buffer if "start" in x), seg["start"])
            flush_end = next((x["end"] for x in reversed(buffer) if "end" in x), seg["end"])
            flush_text = "".join([x.get("word", "") for x in buffer]).replace(" ", "").strip()
            if flush_text:
                new_segments.append({
                    "start": flush_start,
                    "end": flush_end,
                    "text": flush_text,
                    "words": buffer,
                    "speaker": seg.get("speaker", "UNKNOWN")
                })
                
    return new_segments
