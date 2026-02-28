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
