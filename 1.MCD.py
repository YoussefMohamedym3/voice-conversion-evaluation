import os
import math
import glob
import librosa
import scipy.io.wavfile
import pyworld
import pysptk
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# Constants
SAMPLING_RATE = 22050
FRAME_PERIOD = 5.0

Target_Path = os.getenv('Target_Path')
Converted_Path = os.getenv('Converted_Path')

def load_wav(wav_file, sr=SAMPLING_RATE):
    """Load a WAV file using librosa or scipy as a fallback."""
    try:
        wav, _ = librosa.load(wav_file, sr=sr, mono=True)
        return wav
    except Exception as e:
        print(f"Librosa failed to load {wav_file}: {e}")
        try:
            _ , wav = scipy.io.wavfile.read(wav_file)
            return wav
        except Exception as e:
            print(f"Scipy failed to load {wav_file}: {e}")
            raise

def MCD(x, y):
    """Calculate the Mel-cepstral distortion (MCD) between two sequences."""
    log_spec_dB_const = 10.0 / math.log(10.0) * math.sqrt(2.0)
    diff = x - y
    return log_spec_dB_const * math.sqrt(np.inner(diff, diff))

def MCEP(wav_file, mcep_target_directory, alpha=0.65, fft_size=512, mcep_size=24):
    """Extract Mel-cepstral coefficients (MCEP) from a WAV file and save as a .npy file."""
    if not os.path.exists(mcep_target_directory):
        os.makedirs(mcep_target_directory)
    try:
        loaded_wav_file = load_wav(wav_file, sr=SAMPLING_RATE)
        _, spectral_envelop, _ = pyworld.wav2world(
            loaded_wav_file.astype(np.double), fs=SAMPLING_RATE,
            frame_period=FRAME_PERIOD, fft_size=fft_size
        )
        mcep = pysptk.sptk.mcep(
            spectral_envelop, order=mcep_size, alpha=alpha, maxiter=0,
            etype=1, eps=1.0E-8, min_det=0.0, itype=3
        )
        fname = os.path.basename(wav_file).split('.')[0]
        np.save(os.path.join(mcep_target_directory, fname + '.npy'), mcep, allow_pickle=False)
    except Exception as e:
        print(f"Error processing {wav_file}: {e}")

def mcd_cal(mcep_target_files, mcep_converted_files):
    """Calculate the average MCD between Target and Converted MCEP files."""
    min_cost_tot = 0.0
    total_frames = 0

    for target_file in mcep_target_files:
        for converted_file in mcep_converted_files:
            split_target_file = os.path.basename(target_file).split('_')
            split_converted_file = os.path.basename(converted_file).split('_')
            target_speaker, target_speaker_id = split_target_file[0], split_target_file[-1]
            converted_speaker, converted_speaker_id = split_converted_file[0], split_converted_file[-1]

            if target_speaker == converted_speaker and target_speaker_id == converted_speaker_id:
                target_mcep_npy = np.load(target_file)
                frame_no = len(target_mcep_npy)
                converted_mcep_npy = np.load(converted_file)

                min_cost, _ = librosa.sequence.dtw(
                    target_mcep_npy[:, 1:].T, converted_mcep_npy[:, 1:].T, metric=MCD
                )
                min_cost_tot += np.mean(min_cost)
                total_frames += frame_no

    mcd = min_cost_tot / total_frames
    return mcd, total_frames

def main():
    target_wav_files = glob.glob(os.path.join(Target_Path, '*.wav'))
    converted_wav_files = glob.glob(os.path.join(Converted_Path, '*.wav'))

    alpha = 0.65
    fft_size = 512
    mcep_size = 24

    target_mcep_dir = os.path.join(Target_Path, 'mceps_trg')
    converted_mcep_dir = os.path.join(Converted_Path, 'mceps_conv')

    for wav in target_wav_files:
        MCEP(wav, target_mcep_dir, fft_size=fft_size, mcep_size=mcep_size)

    for wav in converted_wav_files:
        MCEP(wav, converted_mcep_dir, fft_size=fft_size, mcep_size=mcep_size)

    target_mcep_files = glob.glob(os.path.join(target_mcep_dir, '*.npy'))
    converted_mcep_files = glob.glob(os.path.join(converted_mcep_dir, '*.npy'))

    mcd, frames_used = mcd_cal(target_mcep_files, converted_mcep_files)

    print(f'MCD = {mcd} dB and total frames = {frames_used}')


if __name__ == "__main__":
    main()