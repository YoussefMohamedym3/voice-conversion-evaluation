from scipy.io import wavfile
import pysptk
import pyworld
import glob
import os
import numpy as np
import collections
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

Target_Path = os.getenv('Target_Path')
Converted_Path = os.getenv('Converted_Path')

def compute_static_features(wav):
    fs, x = wavfile.read(wav)
    if len(x.shape) > 1:
        x = np.mean(x, axis=1)  # Convert to mono by averaging channels
    x = x.astype(np.float64)
    f0, timeaxis = pyworld.dio(x, fs, frame_period=5.0)
    f0 = pyworld.stonemask(x, f0, timeaxis, fs)
    spectrogram = pyworld.cheaptrick(x, f0, timeaxis, fs)
    alpha = pysptk.util.mcepalpha(fs)
    mc = pysptk.sp2mc(spectrogram, order=24, alpha=alpha)
    _, mc = mc[:, 0], mc[:, 1:]
    gv = np.var(mc, axis=0)
    return gv  # [x1, x2, ..., x24]

def calc_rmse(x, y):
    return np.sqrt(((x - y) ** 2).mean())

def vis_gv(paths):
    # Define labels and markers for visualization
    labels = ["Target", "Converted"]
    markers = ['x', '+']

    gv_dict = collections.defaultdict(list)

    # Compute static features for each path
    for i, path_list in enumerate(paths):
        for wav in path_list:
            gv_dict[f'path_{i+1}'].append(compute_static_features(wav))

    # Calculate minimum differences
    min_differences = []
    for i in range(len(gv_dict['path_1'])):
        min_diff = sum(calc_rmse(gv_dict['path_1'][i], gv_dict[f'path_{j+1}'][i]) for j in range(1, len(paths)))
        min_differences.append(min_diff)

    # Find index of minimum difference
    min_index = np.argmin(min_differences)

    # Plot GV data
    figure(figsize=(16, 6))
    for j, (key, values) in enumerate(gv_dict.items()):
        plt.plot(values[min_index], marker=markers[j], linewidth=2, label=labels[j])

    plt.legend(prop={"size": 18})
    plt.yscale("log")
    plt.ylabel("GV", fontsize=16)
    plt.xlabel("Index of Mel-cepstral coefficient", fontsize=16)
    plt.savefig('GV.png')

def main():
    target_wav_files = glob.glob(os.path.join(Target_Path, '*.wav'))
    converted_wav_files = glob.glob(os.path.join(Converted_Path, '*.wav'))

    paths = [target_wav_files, converted_wav_files]
    vis_gv(paths)

if __name__ == "__main__":
    main()
