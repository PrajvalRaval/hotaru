import logging
import torchaudio

# --- TORCHAUDIO COMPATIBILITY PATCH (PyTorch 2.10+ / torchaudio 2.10+) ---
# torchaudio 2.10+ removes .info and .AudioMetaData in favor of torchcodec.
# We restore them here so pyannote-audio, speechbrain, and whisperx continue to function.
if not hasattr(torchaudio, 'AudioMetaData') or not hasattr(torchaudio, 'info'):
    try:
        import torchcodec.decoders as decoders
        from dataclasses import dataclass

        @dataclass
        class AudioMetaData:
            sample_rate: int
            num_frames: int
            num_channels: int
            bits_per_sample: int
            encoding: str

        def info_patch(uri, format=None, buffer_size=4096, backend=None):
            # torchcodec expects string path
            decoder = decoders.AudioDecoder(str(uri))
            m = decoder.metadata
            frames = int(m.duration_seconds * m.sample_rate) if m.duration_seconds else 0
            return AudioMetaData(
                sample_rate=int(m.sample_rate),
                num_frames=frames,
                num_channels=int(m.num_channels),
                bits_per_sample=16, # Fallback for legacy API
                encoding=m.codec or "unknown"
            )

        if not hasattr(torchaudio, 'AudioMetaData'):
            torchaudio.AudioMetaData = AudioMetaData
        if not hasattr(torchaudio, 'info'):
            torchaudio.info = info_patch
        
        # Some libraries look for it in torchaudio.backend.common
        import sys
        from types import ModuleType
        
        if 'torchaudio.backend' not in sys.modules:
            backend_mod = ModuleType('torchaudio.backend')
            sys.modules['torchaudio.backend'] = backend_mod
            torchaudio.backend = backend_mod
            
        if 'torchaudio.backend.common' not in sys.modules:
            common_mod = ModuleType('torchaudio.backend.common')
            sys.modules['torchaudio.backend.common'] = common_mod
            torchaudio.backend.common = common_mod
            common_mod.AudioMetaData = AudioMetaData

        logging.getLogger("HotaruPatch").info("✅ Applied torchaudio 2.10+ compatibility patch.")
    except Exception as e:
        logging.getLogger("HotaruPatch").error(f"❌ Failed to apply torchaudio patch: {e}")
