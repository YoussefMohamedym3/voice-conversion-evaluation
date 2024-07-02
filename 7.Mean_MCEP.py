from scipy.io import wavfile
import pysptk
import pyworld
import glob
import os
import numpy as np
import collections
from matplotlib import pyplot as plt
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def compute_static_features(path):
    fs, x = wavfile.read(path)
    if len(x.shape) > 1:
        x = np.mean(x, axis=1)  # Convert to mono by averaging channels
    x = x.astype(np.float64)
    f0, timeaxis = pyworld.dio(x, fs, frame_period=5.0)
    f0 = pyworld.stonemask(x, f0, timeaxis, fs)
    spectrogram = pyworld.cheaptrick(x, f0, timeaxis, fs)
    alpha = pysptk.util.mcepalpha(fs)
    mc = pysptk.sp2mc(spectrogram, order=24, alpha=alpha)
    _, mc = mc[:, 0], mc[:, 1:]
    return mc

def calc_rmse(x, y):
    return np.sqrt(((x - y) ** 2).mean())

def get_mc(paths):
    labels = ["Target", "Converted"]
    n = len(paths)
    mcep_dict = collections.defaultdict(list)
    
    for i in range(n):
        path = glob.glob(os.path.join(paths[i], '*.wav'))
        for wav in sorted(path):
            mc = compute_static_features(wav)
            mcep_dict[f'path_{i+1}'].append(mc)

    min_differences = []
    for i in range(len(mcep_dict['path_1'])):
        min_dif = sum(
            calc_rmse(mcep_dict['path_1'][i], mcep_dict[f'path_{j+1}'][i])
            for j in range(1, n)
            if mcep_dict['path_1'][i].shape == mcep_dict[f'path_{j+1}'][i].shape
        )
        min_differences.append(min_dif if min_dif != 0 else 1)
    
    ind = min_differences.index(min(min_differences))
    return mcep_dict, labels, ind

def vis(lists, labels, ind):
    plt.rcParams.update({'font.size': 20})
    fig, ax = plt.subplots(figsize=(24, 8))
    markers = ['x', '+']
    
    for j, key in enumerate(lists.keys()):
        arr = np.asarray(lists[key][ind]).T
        ax.plot(np.mean(arr, axis=1), linewidth=2, label=labels[j], marker=markers[j])
    
    ax.legend(fontsize=18)
    plt.ylabel("Mel Cepstrum Coefficients", fontsize=22)
    plt.xlabel("Dimensions", fontsize=22)
    plt.title("MCEP Distribution for all Dimensions", fontsize=24)
    plt.savefig('Mean MCEP for Dimension.png')

def main():
    target_path = os.getenv('Target_Path')
    converted_path = os.getenv('Converted_Path')
    paths = [target_path, converted_path]
    lists, labels, ind = get_mc(paths)
    vis(lists, labels, ind)

if __name__ == "__main__":
    main()
