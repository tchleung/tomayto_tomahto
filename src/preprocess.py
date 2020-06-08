import numpy as np
import pandas as pd
import librosa

def wav_to_img(path):
    '''
    Turn wav files into melspectrograms
    Input:
        Path of the audio file (.wav format)
    Output:
        128x256 shaped Mel-spectograms
    '''
    
    audio, sr = librosa.load(path,duration=2.97) # 2.97 second clip generates exactly 256 pixel time series 
    # parameters for calculating spectrogram in mel scale
    fmax = 10000 # maximum frequency considered
    fft_window_points = 512
    fft_window_dur = fft_window_points * 1.0 / sr
    hop_size = int(fft_window_points/ 2)
    n_mels = 128 # setting this too high would create distortion
    # generate the mel-spectrogram
    spec = librosa.feature.melspectrogram(audio, sr=sr, n_mels=n_mels, n_fft=fft_window_points, hop_length=hop_size, fmax=fmax)
    spec_gram = librosa.power_to_db(spec, np.max)
    # some clips are shorter than 2.97 seconds, we can pad the front and back of clip
    try:
        return librosa.util.pad_center(spec_gram, size = 256, axis = 1)
    except:
        return spec_gram
    
def picturized(directory1,path1,lang1,directory2,path2,lang2,empty_list):
    '''
    Loop over a directory of all the wav clips, turn them all into mel-spectrograms, then return it as a feature matrix with label
    
    Input:
        1. List of all the .wav files of the first language (directory1)
        2. Directory path to those files (path1)
        3. Name of the first language (lang1)
        4. List of all the .wav files of the second language (directory2)
        5. Directory path to those files (path2)
        6. Name of the second language (lang2)
        7. An empty list (empty_list)
        
    Output:
        An array with feature column (column 0) and the label (column 1)
    '''
    for f in directory1:
        path_to_file = os.path.join(path1,f)
        result = wav_to_img(path_to_file)
        empty_list.append([result,lang1])
    for f2 in directory2:
        path_to_file2 = os.path.join(path2,f2)
        result = wav_to_img(path_to_file2)
        empty_list.append([result,lang2])
    return empty_list

if __name__ == "__main__":
    # modify to your local directory of where the .wav files are located
    paths = [
        ['./processed_audio/train/cn/', './processed_audio/train/tw/', 'train'],
        ['./processed_audio/test/cn/', './processed_audio/test/tw/', 'test'],
        ['./processed_audio/hold_out/cn/', './processed_audio/hold_out/tw/', 'hold_out']]

    
    '''
    Loops through all the directory, turn all .wav into mel-spectrograms features
    Then save the features and labels as a pandas dataframe, and save as pickle
    '''
    vectorized = []
    for i in paths:
        directory1 = os.listdir(i[0])
        path1 = i[0]
        lang1 = 'cn'
        directory2 = os.listdir(i[1])
        path2 = i[1]
        lang2 = 'tw'
        result = picturized(directory1,path1,lang1,directory2,path2,vectorized,lang2)
        df = pd.DataFrame(result,columns=['features','lang'])
        filename = i[2] + '.pkl'
        df.to_pickle(filename)