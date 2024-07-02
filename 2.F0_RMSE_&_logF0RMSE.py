import os
import math
import glob
import librosa
import pyworld
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Constants
SAMPLING_RATE = 16000
frame_period = 5.0

Source_Path = os.getenv('Source_Path')
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

def world_encode_data(wavs, fs, frame_period = 5.0):

    f0s = []
    log_f0s_concatenated0=[]
    for i in range(len(wavs)):
        wav = wavs[i]
        wav = wav.astype(np.float64)
        f0, _  = pyworld.harvest(wav, fs, frame_period = frame_period, f0_floor = 71.0, f0_ceil = 800.0)
        f0s.append(f0)
        log_f0s_concatenated0.append(np.ma.log(f0s[i]))
        
        

    return f0s,log_f0s_concatenated0 








def main():

    # Get list of .wav files
    source_paths = glob.glob(os.path.join(Source_Path, '*.wav'))
    converted_paths = glob.glob(os.path.join(Converted_Path, '*.wav'))

    # Sort paths to ensure order
    source_paths.sort()
    converted_paths.sort()

    # Check if filenames match
    for i in range(len(source_paths)):
        source_filename = os.path.basename(source_paths[i])
        converted_filename = os.path.basename(converted_paths[i])
        print(f"{source_filename}   {converted_filename}")
        assert source_filename == converted_filename

    print("All .wav files matched successfully.")

    wavs_source=[]
    wavs_converted=[]

    for i in range(len(source_paths)):
        if os.path.basename(source_paths[i])==os.path.basename(converted_paths[i]):
            wavs_source.append(load_wav(wav_file = source_paths[i], sr = SAMPLING_RATE))
            wavs_converted.append(load_wav(wav_file = converted_paths[i], sr = SAMPLING_RATE))

    print(f"Loaded {len(wavs_source)} source and {len(wavs_converted)} converted wav files.")

    # Encode data
    f0s_source,log_f0s_source = world_encode_data(wavs_source, SAMPLING_RATE, frame_period)
    f0s_converted,log_f0s_converted = world_encode_data(wavs_converted, SAMPLING_RATE, frame_period)

    # Calculate F0_RMSE 
    min_cost_tot=[]
    for i in range(len(wavs_source)):
        frame_len=0
        def logf0_rmse(x, y): # method to calculate cost
            log_spec_dB_const = 1/len(frame_len)
            diff = x - y
            return log_spec_dB_const * math.sqrt(np.inner(diff, diff))
        
        
        if len(f0s_source[i])<len(f0s_converted[i]):
            frame_len=f0s_source[i]
        else:
            frame_len=f0s_converted[i]

        cost_function = logf0_rmse
        min_cost, _ = librosa.sequence.dtw(f0s_source[i][:].T, f0s_converted[i][:].T, 
                                                        metric=cost_function)
        #print(len(min_cost))
        
        min_cost_tot.append(np.mean(min_cost))

    # Calculate and print F0 RMSE
    F0RMSE = np.mean(min_cost_tot)
    print(f"F0_RMSE = {F0RMSE}")
    


    # Calculate logF0_RMSE 
    min_cost_tot=[]
    for i in range(len(wavs_source)):
        frame_len=0
        def logf0_rmse(x, y): # method to calculate cost
            log_spec_dB_const = 1/len(frame_len)
            diff = x - y
            return log_spec_dB_const * math.sqrt(np.inner(diff, diff))
        
        
        if len(log_f0s_source[i])<len(log_f0s_converted[i]):
            frame_len=log_f0s_source[i]
        else:
            frame_len=log_f0s_converted[i]

        cost_function = logf0_rmse
        min_cost, _ = librosa.sequence.dtw(log_f0s_source[i][:].T, log_f0s_converted[i][:].T, 
                                                        metric=cost_function)
        #print(len(min_cost))
        
        min_cost_tot.append(np.mean(min_cost))


    logF0RMSE=sum(min_cost_tot)/len(min_cost_tot)
    print(f"logF0_RMSE = {logF0RMSE}")
if __name__ == "__main__":
    main()
