from scipy.io import wavfile
import pysptk
import pyworld
import os
import numpy as np
import glob
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Constants
ms_fftlen = 4096


def compute_static_features(path):
    fs, x = wavfile.read(path)
    if len(x.shape) > 1:
        x = x.mean(axis=1)  # Convert stereo to mono
    x = x.astype(np.float64)
    f0, timeaxis = pyworld.dio(x, fs, frame_period=5.0)
    f0 = pyworld.stonemask(x, f0, timeaxis, fs)
    spectrogram = pyworld.cheaptrick(x, f0, timeaxis, fs)
    aperiodicity = pyworld.d4c(x, f0, timeaxis, fs)
    alpha = pysptk.util.mcepalpha(fs)
    mc = pysptk.sp2mc(spectrogram, order=24, alpha=alpha)
    c0, mc = mc[:, 0], mc[:, 1:]
    return mc

def modspec(x, n=4096, norm=None):
    s_complex = np.fft.rfft(x, n=n, axis=0, norm=norm)
    R, im = s_complex.real, s_complex.imag
    ms = R * R + im * im
    return ms

def mean_modspec(paths):
    mss = []
    for wav in paths:
        mgc = compute_static_features(wav)
        ms = np.log(modspec(mgc, n=ms_fftlen))
        mss.append(ms)
    return np.mean(np.array(mss), axis=(0,))

def main():
    Target_Path = os.getenv('Target_Path')
    Converted_Path = os.getenv('Converted_Path')

    Target_paths = glob.glob(os.path.join(Target_Path, '*.wav'))
    Converted_paths = glob.glob(os.path.join(Converted_Path, '*.wav'))

    ms_into2out_target = mean_modspec(Target_paths)
    ms_into2out_converted = mean_modspec(Converted_paths)

    new = 0
    for i in range(24):
        diff = np.mean(np.absolute(ms_into2out_target[i, :].T - ms_into2out_converted[i,:].T))
        diff = np.inner(diff, diff)
        new += diff

    MSD = np.sqrt(1 / len(mean_modspec(Target_paths).T)) * np.sqrt(new)
    print(MSD)

if __name__ == "__main__":
    main()
