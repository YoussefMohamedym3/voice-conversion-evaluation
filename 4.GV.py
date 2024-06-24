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
    aperiodicity = pyworld.d4c(x, f0, timeaxis, fs)
    alpha = pysptk.util.mcepalpha(fs)
    mc = pysptk.sp2mc(spectrogram, order=24, alpha=alpha)
    c0, mc = mc[:, 0], mc[:, 1:]
    gv = np.var(mc, axis=0)
    return gv  # [x1, x2, ..., x24]



def calc_rmse(x,y):
    min=np.sqrt(((x - y) ** 2).mean())
    return min





def vis_gv(paths):
    # Define default labels and markers
    default_labels = ["Target", "Converted"] # Change the lables to your desired names
    default_markers = ['x', '+'] # Change the markers to your desired shapes 
    
    # Use only as many labels and markers as there are paths
    n = len(paths)
    labels = default_labels[:n]
    markers = default_markers[:n]

    # Initialize the dictionary to store GV data and file paths
    gv_dict = collections.defaultdict(list)
    file_paths = {}

    # Read the files and compute static features
    for i in range(n):
        for wav in paths[i]:  # Iterate through each path in paths[i]
            gv_dict[f'path_{i+1}'].append(compute_static_features(wav))

    # Calculate the minimum differences
    mini = []
    for i in range(len(gv_dict['path_1'])):
        min_dif = sum(calc_rmse(gv_dict['path_1'][i], gv_dict[f'path_{j+1}'][i]) for j in range(1, n))
        mini.append(min_dif)

    # Find the index with the minimum difference
    ind = mini.index(min(mini))

    # Plot the GV data
    figure(figsize=(16, 6))
    for j, (key, values) in enumerate(gv_dict.items()):
        plt.plot(values[ind], marker=markers[j], linewidth=2, label=labels[j])
    
    plt.legend(prop={"size": 18})
    plt.yscale("log")  # This is part of matplotlib.pyplot
    plt.ylabel("GV", fontsize=16)
    plt.xlabel("Index of Mel-cepstral coefficient", fontsize=16)
    plt.savefig('GV.png')
    plt.show()






def main():
    target_wav_files = glob.glob(os.path.join(Target_Path, '*.wav'))
    converted_wav_files = glob.glob(os.path.join(Converted_Path, '*.wav'))

    paths=[target_wav_files,converted_wav_files]

    vis_gv(paths)
if __name__ == "__main__":
    main()