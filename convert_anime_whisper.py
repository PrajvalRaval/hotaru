import os
import subprocess
import sys

def convert_model():
    model_id = "litagin/anime-whisper"
    output_dir = "models/anime-whisper-ct2"
    
    print(f"Converting {model_id} to CTranslate2 format...")
    
    # Identify the correct converter binary
    converter_bin = "ct2-transformers-converter"
    venv_bin = os.path.join("venv", "bin", converter_bin)
    if os.path.exists(venv_bin):
        converter_bin = venv_bin
    
    # Command to convert whisper model to CTranslate2
    cmd = [
        converter_bin,
        "--model", model_id,
        "--output_dir", output_dir,
        "--quantization", "float16",
        "--force"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Successfully converted model to {output_dir}")
        
        # POST-PROCESSING: Fix config.json for Whisper v3 (128 Mel bins)
        config_path = os.path.join(output_dir, "config.json")
        if os.path.exists(config_path):
            import json
            with open(config_path, "r") as f:
                config = json.load(f)
            
            # Anime-whisper is v3-based, requires 128 mel bins
            if config.get("num_mel_bins") != 128:
                print("Patching config.json: setting num_mel_bins to 128...")
                config["num_mel_bins"] = 128
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=2)
        
        # ENSURE PREPROCESSOR CONFIG EXISTS (Crucial for WhisperX/Faster-Whisper)
        preprocessor_path = os.path.join(output_dir, "preprocessor_config.json")
        if not os.path.exists(preprocessor_path):
            print("Downloading missing preprocessor_config.json...")
            try:
                import urllib.request
                url = f"https://huggingface.co/{model_id}/raw/main/preprocessor_config.json"
                urllib.request.urlretrieve(url, preprocessor_path)
            except Exception as e:
                print(f"Warning: Could not download preprocessor_config.json: {e}")

    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)

if __name__ == "__main__":
    convert_model()
