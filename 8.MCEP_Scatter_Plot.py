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
    aperiodicity = pyworld.d4c(x, f0, timeaxis, fs)
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

    mini = []
    for i in range(len(mcep_dict['path_1'])):
        min_dif = 0
        for j in range(1, n):
            if mcep_dict['path_1'][i].shape == mcep_dict[f'path_{j+1}'][i].shape:
                min_dif += calc_rmse(mcep_dict['path_1'][i], mcep_dict[f'path_{j+1}'][i])
        mini.append(min_dif if min_dif > 0 else 1)
    
    ind = mini.index(min(mini))
    return mcep_dict, labels, ind

def vis(lists, labels, ind):  
    dims = [8, 13, 23]  # You can use any dimensions you want from 1 to 24

    for d in dims:
        fig, ax = plt.subplots(figsize=(8, 8))
        for j, key in enumerate(lists.keys()):
            arr = np.asarray(lists[key][ind]).T
            ax.scatter(arr[0], arr[d-1], linewidth=2, label=labels[j])
            ax.legend()
            ax.set_ylabel(f"Mel Cepstrum Coefficient for Dimension {d}", fontsize=14)
            ax.set_xlabel("Mel Cepstrum Coefficient for Dimension 1", fontsize=14)
            ax.set_title(f"MCEP Distribution for Dimension - {d} vs Dimension - 1:", fontsize=16)
        plt.savefig(f'MCEP_Scatter_Plot for Dimension{d}.png')

def main():
    Target_Path = os.getenv('Target_Path')
    Converted_Path = os.getenv('Converted_Path')
    paths = [Target_Path, Converted_Path]

    lists, labels, ind = get_mc(paths)
    vis(lists, labels, ind)

if __name__ == "__main__":
    main()
