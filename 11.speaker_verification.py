from pathlib import Path
import numpy as np
from resemblyzer import preprocess_wav, VoiceEncoder
from dotenv import load_dotenv
import os

def load_model(root, device):
    """Load model"""
    model = VoiceEncoder()
    return model

def calculate_score(model, target_files, converted_files, threshold_path, **kwargs):
    # Load optimal threshold from text file
    with open(threshold_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith("Optimal Threshold:"):
                threshold = float(line.split(":")[1].strip())
                break

    # Initialize counters
    n_accept = 0
    n_total = len(target_files)

    # Process each pair of original and converted files
    for original_file, converted_file in zip(target_files, converted_files):
        wav_original = preprocess_wav(original_file)
        source_emb = model.embed_utterance(wav_original)

        wav_converted = preprocess_wav(converted_file)
        target_emb = model.embed_speaker([wav_converted])

        cosine_similarity = (
            np.inner(source_emb, target_emb)
            / np.linalg.norm(source_emb)
            / np.linalg.norm(target_emb)
        )

        if cosine_similarity > threshold:
            n_accept += 1

    svar = n_accept / n_total if n_total > 0 else 0
    print(f"[INFO]: Speaker verification accept rate: {svar:.2f}")

def main():
    # Load environment variables from .env file
    load_dotenv()

    # Configuration variables
    Target_Path = Path(os.getenv('Target_Path'))
    converted_dir = Path(os.getenv('Converted_Path'))
    threshold_path = os.getenv('THRESHOLD_PATH')

    # Load model
    model = load_model(root=None, device=None)

    # Gather all .wav files in the directories
    target_files = list(Target_Path.glob("*.wav"))
    converted_files = [converted_dir / file.name for file in target_files]

    # Calculate score for all pairs
    calculate_score(model, [str(file) for file in target_files], [str(file) for file in converted_files], threshold_path)

if __name__ == "__main__":
    main()
