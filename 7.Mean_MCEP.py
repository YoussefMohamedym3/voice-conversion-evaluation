from scipy.io import wavfile
import pysptk
import pyworld
import glob
import os
import numpy as np
import collections
from numpy.core.fromnumeric import shape
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
    c0, mc = mc[:, 0], mc[:, 1:]
    # print(shape(mc))
    return mc



def calc_rmse(x,y):
    min=np.sqrt(((x - y) ** 2).mean())
    return min



from numpy.core.fromnumeric import shape
def get_mc(paths):
    labels = ["Target", "Converted"]
    n=len(paths)


    mcep_dict = collections.defaultdict(list)
    file_paths={}
    for i in range(n):
        # Get all .wav files directly inside the directory
        path = glob.glob(os.path.join(paths[i], '*.wav'))
        file_paths[f'path_{i+1}'] =sorted(path)
        for wav in file_paths[f'path_{i+1}']:
            mc=compute_static_features(wav)
            mcep_dict[f'path_{i+1}'].append(mc)

    #(gv_dict[0][1]))
    mini=[]
    x=1
    for i in range(len(mcep_dict['path_1'])):
        j=1
        min_dif=0
        while (j+1)<=n:
            if shape(mcep_dict['path_1'][i])==shape(mcep_dict[f'path_{j+1}'][i]):
                min_dif+=(calc_rmse(mcep_dict['path_1'][i],mcep_dict[f'path_{j+1}'][i]))
            #print(min_dif,j)
            else:
                x=0
            j+=1
        #print(min_dif)
        mini.append(min_dif)
    for i in range(len(mini)):
        if mini[i]==0:
            mini[i]=1
    ind=mini.index(min(mini))
    return mcep_dict,labels,ind



def vis(lists,labels,ind,):  
    plt.rcParams.update({'font.size': 20})
    fig, ax = plt.subplots(figsize=(24,8))
    marker=['x', '+']
    # n=len(labels)


    j=0
    for i in lists.keys():
        arr=np.asarray(lists[i][ind]).T

        #plt.plot(gv_dict[i][1], marker=marker[j] ,linewidth=2, label=labels[j])
        ax.plot(np.mean(arr,axis=1), linewidth=2, label=labels[j], marker=marker[j])
        ax.legend(fontsize=18)
        plt.ylabel(f"Mel Cepstrum Coeeficients ", fontsize=22)
        plt.xlabel("Dimensions ",fontsize=22)
        plt.title(f"Mean MCEP Distribution for all Dimensions :",fontsize=24)
        j+=1
    plt.savefig(f'Mean MCEP for Dimension.png')




def main():

    Target_Path = os.getenv('Target_Path')
    Converted_Path = os.getenv('Converted_Path')


    paths=[Target_Path,Converted_Path]






    lists,labels,ind=get_mc(paths)

    vis(lists,labels,ind)
if __name__ == "__main__":
    main()