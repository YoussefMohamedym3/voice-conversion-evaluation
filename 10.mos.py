"""MBNet for MOS prediction"""
from pathlib import Path
import numpy as np
from tqdm import tqdm
import librosa
import sox
import torch
import sys
import os
from mos_model_and_checkpoints.model import MBNet
from dotenv import load_dotenv



def load_model(root, device):
    """Load model"""

    root = Path(root) / "mos_model_and_checkpoints"
    model_paths = sorted(list(root.glob("MBNet*")))
    models = []
    for model_path in model_paths:
        model = MBNet(num_judges=5000)
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(model_path))
        else:
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        models.append(model.to(device))

    return models



def do_MBNet(model, wavs, device):
    """Do MBNet."""
    mean_scores = 0
    with torch.no_grad():
        for wav in wavs:
            wav = wav.to(device)
            mean_score = model.only_mean_inference(spectrum=wav)
            mean_scores += mean_score.cpu().tolist()[0]

    return mean_scores / len(wavs)




def calculate_score(model, device, data_dir, **kwargs):
    """Calculate score"""

    file_paths = librosa.util.find_files(data_dir)
    tfm = sox.Transformer()
    tfm.norm(-3.0)

    wavs = []
    for file_path in tqdm(file_paths):
        wav, _ = librosa.load(file_path, sr=16000)
        wav = tfm.build_array(input_array=wav, sample_rate_in=16000)
        wav = np.abs(librosa.stft(wav, n_fft=512)).T
        wav = torch.from_numpy(wav).unsqueeze(0).unsqueeze(0)
        wavs.append(wav)

    mean_scores = []
    for m in tqdm(model):
        mean_score = do_MBNet(m, wavs, device)
        mean_scores.append(mean_score)

    average_score = np.mean(mean_scores)

    print(f"[INFO]: All mean opinion scores: {mean_scores}")
    print(f"[INFO]: Average mean opinion score: {average_score}")


def main():
    # Load environment variables from .env file
    load_dotenv()

    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_dir = os.getenv("ROOT_DIR")
    models = load_model(root_dir, device)

    # Calculate and save the scores
    Converted_Path = os.getenv("Converted_Path")
    calculate_score(models, device, Converted_Path)

if __name__ == "__main__":
    main()
