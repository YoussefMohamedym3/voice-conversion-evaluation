from scipy.io import wavfile
import pysptk
import pyworld
import glob
import os
import numpy as np
import collections
from matplotlib import pyplot as plt

# Constants
ms_fftlen = 4096

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
    c0, mc = mc[:, 0], mc[:, 1:]
    return mc

def modspec(x, n=4096, norm=None):
    s_complex = np.fft.rfft(x, n=n, axis=0, norm=norm)
    assert s_complex.shape[0] == n // 2 + 1
    R, im = s_complex.real, s_complex.imag
    ms = R * R + im * im
    return ms

def mean_modspec(files):
    mss = []
    for file_path in files:
        mgc = compute_static_features(file_path)
        ms = np.log(modspec(mgc, n=ms_fftlen))
        mss.append(ms)
    return np.mean(np.array(mss), axis=(0,))

def calc_rmse(x, y):
    return np.sqrt(((x - y) ** 2).mean())

def get_mc(paths):
    labels = ["Target", "Converted"]
    dims = [8, 13, 23]
    indx = []
    n = len(paths)
    mcep_dict = collections.defaultdict(list)
    for i in range(n):
        path = glob.glob(os.path.join(paths[i], '*.wav'))
        path = sorted(path)
        for wav in path:
            mc = mean_modspec([wav])
            mcep_dict[f'path_{i+1}'].append(mc)
    
    for d in dims:
        mini = []
        for k in range(len(mcep_dict['path_1'])):
            min_dif = 0
            for j in range(1, n):
                min_dif += calc_rmse(mcep_dict['path_1'][k][:, d], mcep_dict[f'path_{j+1}'][k][:, d])
            mini.append(min_dif)
        indx.append(mini.index(min(mini)))
    
    return mcep_dict, labels, indx, dims

def vis(lists, labels, ind, a, dims):
    for d in range(len(dims)):
        fig, ax = plt.subplots(figsize=(24, 8))
        for j, key in enumerate(lists.keys()):
            arr = lists[key][ind[d]][:, dims[d]]
            ax.plot(a, arr, linewidth=2, label=labels[j])
        
        ax.set_xticks(np.arange(len(a)), minor=True)
        ax.set_xticks(np.arange(0, len(a), 50))
        ax.set_xticklabels(np.arange(len(a))[np.arange(0, len(a), 50)])
        
        plt.xscale("log")
        plt.xlabel("Modulation frequency index", fontsize=16)
        plt.ylabel("MS[dB]", fontsize=16)
        plt.legend(loc='center left', prop={'size': 16}, bbox_to_anchor=(1, 1))
        plt.title(f"Modulation Spectrum for dimension - {dims[d]}", fontsize=18)
        
        plt.tight_layout()
        plt.savefig(f'MCEP_Trajectory_Dimension{int(dims[d])}.png')
        plt.close(fig)

def main():
    Target_Path = os.getenv('Target_Path')
    Converted_Path = os.getenv('Converted_Path')
    paths = [Target_Path, Converted_Path]

    fs = 16000
    frame_period = 5.0
    hop_length = int(fs * (frame_period * 0.001))
    modfs = fs / hop_length
    ms_freq = np.arange(ms_fftlen // 2 + 1) / ms_fftlen * modfs

    lists, labels, ind, dims = get_mc(paths)
    vis(lists, labels, ind, ms_freq.tolist(), dims)

if __name__ == "__main__":
    main()
