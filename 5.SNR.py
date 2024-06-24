import os
import math
import glob
import librosa
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

Source_Path = os.getenv('Source_Path')
Converted_Path = os.getenv('Converted_Path')

# Get list of .wav files
source_paths = glob.glob(os.path.join(Source_Path, '*.wav'))
converted_paths = glob.glob(os.path.join(Converted_Path, '*.wav'))

# Sort paths to ensure order
source_paths.sort()
converted_paths.sort()

# SNR[dB]=10*log(RMSE(Signal)/RMSE(Noise))
length=len(source_paths)
snr=[]
for i in range(length):
    source_wavform,_=librosa.load(source_paths[i], sr=800)
    converted_wavform,_=librosa.load(converted_paths[i],sr=800)
    source_rms=math.sqrt(np.mean(source_wavform**2))
    converted_rms=math.sqrt(np.mean(converted_wavform**2))
    snr.append(10*np.log10(source_rms/converted_rms))
print(np.mean(snr))  

