"""
Standalone SRT Polishing Script
================================

This script uses a local Ollama model to polish machine-translated English SRT subtitles, 
fixing grammar, punctuation, and flow while strictly preserving the original timestamps.

Usage:
  # Activate your python environment first
  source venv/bin/activate
  
  # Basic execution (defaults to qwen3:30b and chunk size 25)
  python scripts/polish_srt.py path/to/your_subs.srt

  # Advanced execution (custom model, custom output, custom chunk size)
  python scripts/polish_srt.py path/to/your_subs.srt -m "qwen3.5:35b" -c 15 -o path/to/polished_subs.srt
"""

import os
import re
import argparse
import time
import ollama

def parse_srt(file_path):
    """Parses an SRT file into a list of subtitle dictionaries."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    blocks = content.strip().split('\n\n')
    subtitles = []
    for block in blocks:
        lines = block.split('\n')
        if len(lines) >= 3:
            index = lines[0]
            timestamp = lines[1]
            text = '\n'.join(lines[2:])
            subtitles.append({
                'index': index,
                'timestamp': timestamp,
                'text': text
            })
    return subtitles

def write_srt(subtitles, output_path):
    """Writes a list of subtitle dictionaries back to an SRT file format."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, sub in enumerate(subtitles):
            f.write(f"{sub['index']}\n")
            f.write(f"{sub['timestamp']}\n")
            f.write(f"{sub['text']}\n")
            if i < len(subtitles) - 1:
                f.write("\n")

def polish_chunk(chunk, model, client):
    """Sends a batch of subtitles to Ollama for grammar polishing."""
    # Flatten multi-line subtitles into single lines for the LLM prompt
    input_text = "\n".join([f"Line {i+1}: {sub['text'].replace(chr(10), ' ')}" for i, sub in enumerate(chunk)])
    
    system_prompt = (
        "### ROLE\n"
        "You are an expert English copyeditor and cinematic subtitler. You will receive machine-translated English anime subtitles.\n"
        "Your task is to polish the grammar, punctuation, and flow to make the dialogue natural, idiomatic, and dramatic. Fix awkward phrasing while preserving the original meaning.\n\n"
        "### CRITICAL RULES\n"
        "1. STRICT 1:1 MAPPING: You MUST output exactly ONE translated line for every single input line.\n"
        "2. FORMAT: `Line [Index]: [Polished Text]`\n"
        "3. DO NOT merge lines. DO NOT skip lines. The sequence numbers must match the input exactly.\n"
        "4. DO NOT wrap your response in markdown blocks. Output ONLY the numbered localized English text.\n"
    )
    
    user_prompt = f"Polish the following subtitles:\n\n{input_text}\n\nOutput:"

    try:
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            options={
                "temperature": 0.3, 
                "num_ctx": 8192,
                "num_predict": 4096
            },
            stream=False
        )
        content = response.get("message", {}).get("content", "")
    except Exception as e:
        print(f"❌ Error calling Ollama: {e}")
        return [sub['text'] for sub in chunk]

    # strict Index-Locked parsing to prevent array desync
    lines = content.split('\n')
    results = [None] * len(chunk)
    line_pattern = re.compile(r'^Line\s*(\d+)\s*:\s*(.*)', re.IGNORECASE)

    for line in lines:
        line = line.strip()
        if not line: continue
        
        match = line_pattern.match(line)
        if match:
            idx = int(match.group(1)) - 1
            if 0 <= idx < len(chunk):
                if results[idx] is None:
                    text = match.group(2).strip().strip('"')
                    # Remove common hallucinated prefixes
                    text = re.sub(r'^\[Polished Text\]\s*', '', text, flags=re.IGNORECASE)
                    results[idx] = text
    
    # Assembly and Failsafe
    cleaned = []
    for k in range(len(chunk)):
        if results[k] is not None:
            cleaned.append(results[k])
        else:
            print(f"🚨 Failsafe Triggered: LLM skipped Line {k+1} in this chunk. Retaining original unpolished text to preserve timing.")
            cleaned.append(chunk[k]['text'])
            
    return cleaned

def main():
    parser = argparse.ArgumentParser(description="Standalone script to polish machine-translated SRT files using local Ollama models.")
    parser.add_argument("input_srt", help="Path to the input SRT file")
    parser.add_argument("-o", "--output", help="Optional: Path to the output SRT file. Defaults to appending '_polished' to the filename.")
    parser.add_argument("-m", "--model", default="qwen3:30b", help="Ollama model to use (default: qwen3:30b)")
    parser.add_argument("-c", "--chunk_size", type=int, default=25, help="Lines to process per batch (default: 25)")
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama host URL (default: http://localhost:11434)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_srt):
        print(f"❌ Error: File '{args.input_srt}' not found.")
        return

    output_path = args.output
    if not output_path:
        base = args.input_srt.rsplit('.', 1)
        if len(base) == 2:
            output_path = f"{base[0]}_polished.{base[1]}"
        else:
            output_path = f"{args.input_srt}_polished"
            
    print(f"🎬 Loading '{args.input_srt}'...")
    subtitles = parse_srt(args.input_srt)
    print(f"✅ Found {len(subtitles)} subtitle blocks.")
    
    client = ollama.Client(host=args.host)
    chunk_size = args.chunk_size
    total_chunks = (len(subtitles) + chunk_size - 1) // chunk_size
    
    print(f"🚀 Starting polishing process using {args.model} ({total_chunks} chunks of {chunk_size} lines)...")
    
    for i in range(0, len(subtitles), chunk_size):
        chunk = subtitles[i:i+chunk_size]
        chunk_num = i // chunk_size + 1
        
        print(f"🪄 Polishing chunk {chunk_num}/{total_chunks} ({len(chunk)} lines)...")
        polished_texts = polish_chunk(chunk, args.model, client)
        
        # Inject the polished text back into the dictionaries
        for j, text in enumerate(polished_texts):
            subtitles[i+j]['text'] = text
            
        # Give Ollama a short breather between batches
        time.sleep(1)
        
    print(f"💾 Writing polished subtitles to '{output_path}'...")
    write_srt(subtitles, output_path)
    print("✨ Done!")

if __name__ == "__main__":
    main()
