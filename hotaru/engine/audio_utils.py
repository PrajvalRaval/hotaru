import os
import torch
import torchaudio
import logging
import subprocess
from demucs.apply import apply_model
from demucs.pretrained import get_model
from demucs.audio import save_audio

logger = logging.getLogger("HotaruAudio")

def isolate_vocals(input_path: str, output_dir: str, device: str = "cuda") -> str:
    """
    Phase 0: The Teardown (Ironclad Sync Version).
    1. Extract 44.1kHz Stereo for Demucs.
    2. Separate Stems using htdemucs.
    3. Downsample Vocals to 16kHz Mono for WhisperX.
    """

    # STEP 1: Extract high-quality audio for Demucs (44.1kHz, Stereo)
    temp_hq = os.path.join(output_dir, "temp_hq_for_demucs.wav")
    extract_cmd = [
        'ffmpeg', '-y', '-i', input_path,
        '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2',
        temp_hq
    ]
    subprocess.run(extract_cmd, capture_output=True, check=True)

    # STEP 2: Load and Separate
    model = get_model("htdemucs")
    model.to(device)
    model.eval()

    # Load HQ track
    wav, sr = torchaudio.load(temp_hq)
    
    # THE FIX: Demucs apply_model expects [batch, channels, time]
    # torchaudio.load returns [channels, time], so we must unsqueeze.
    wav_batched = wav.unsqueeze(0)
    
    logger.info("⏳ Separating vocals from background stems (htdemucs)...")
    with torch.no_grad():
        # returns [batch, stems, channels, time]
        sources = apply_model(model, wav_batched, device=device, split=True, overlap=0.25)
    
    # Get first batch [0] and vocals stem [3]
    vocals = sources[0][3]
    
    # Save isolated vocals at 44.1kHz
    temp_vocals_hq = os.path.join(output_dir, "temp_vocals_hq.wav")
    save_audio(vocals.cpu(), temp_vocals_hq, samplerate=model.samplerate)

    # STEP 3: Downsample strictly for WhisperX (16kHz, Mono)
    vocal_filename = f"vocals_{os.path.basename(input_path)}.wav"
    final_vocal_path = os.path.join(output_dir, vocal_filename)
    
    downsample_cmd = [
        'ffmpeg', '-y', '-i', temp_vocals_hq,
        '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
        final_vocal_path
    ]
    logger.info("📏 Downsampling isolated track to 16kHz Mono...")
    subprocess.run(downsample_cmd, capture_output=True, check=True)
    
    # CLEANUP
    for f in [temp_hq, temp_vocals_hq]:
        if os.path.exists(f): os.remove(f)
    
    logger.info(f"✅ Vocal separation complete: {vocal_filename}")
    
    # Memory Cleanup
    del model, sources, wav, wav_batched
    torch.cuda.empty_cache()
    
    return final_vocal_path
