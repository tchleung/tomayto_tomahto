import numpy as np
import pandas as pd
import librosa
from pydub import AudioSegment

def get_meta(path):
    '''
    input: directory of the metadata file (path)
    output: metadata in pandas df
    '''
    df = pd.read_csv(path,sep='\t')
    return df
    
def get_random_sample(df, df1):
    '''
    This creates a randomized, gender-balanced train, test, and hold_out data sets (8:1:1 ratio)
    input: metadata pandas df (df, df1)
    output: Lists of file IDs that will be used as sample
    '''
    # Create the boolean masks
    cn_female_mask = df[df['gender'] == 'female']['path'].tolist()
    tw_female_mask = df1[df1['gender'] == 'female']['path'].tolist()
    cn_male_mask = df[df['gender'] == 'male']['path'].tolist()
    tw_male_mask = df1[df1['gender'] == 'male']['path'].tolist()
    
    # Random select the needed number into the curated dataset
    num_sample = 1851 # based on the smallest subgroup (cn female)
    cn_f_sample = np.random.choice(cn_female_mask, size = num_sample, replace = False) #this is pointless, just to make the variables consistent
    tw_f_sample = np.random.choice(tw_female_mask, size = num_sample, replace = False)
    cn_m_sample = np.random.choice(cn_male_mask, size = num_sample, replace = False)
    tw_m_sample = np.random.choice(tw_male_mask, size = num_sample, replace = False)
    
    # Shuffle the sample before we split into train/test/hold-out
    np.random.shuffle(cn_f_sample)
    np.random.shuffle(tw_f_sample)
    np.random.shuffle(cn_m_sample)
    np.random.shuffle(tw_m_sample)
    
    # Split into train/test/hold-out, then combine them by language
    cn_f_tr = cn_f_sample[0:1481].tolist()
    cn_f_ts = cn_f_sample[1481:1666].tolist()
    cn_f_ho = cn_f_sample[1666:].tolist()

    tw_f_tr = tw_f_sample[0:1481].tolist()
    tw_f_ts = tw_f_sample[1481:1666].tolist()
    tw_f_ho = tw_f_sample[1666:].tolist()

    cn_m_tr = cn_m_sample[0:1481].tolist()
    cn_m_ts = cn_m_sample[1481:1666].tolist()
    cn_m_ho = cn_m_sample[1666:].tolist()

    tw_m_tr = tw_m_sample[0:1481].tolist()
    tw_m_ts = tw_m_sample[1481:1666].tolist()
    tw_m_ho = tw_m_sample[1666:].tolist()

    cn_tr = cn_f_tr + cn_m_tr
    cn_ts = cn_f_ts + cn_m_ts
    cn_ho = cn_f_ho + cn_m_ho

    tw_tr = tw_f_tr + tw_m_tr
    tw_ts = tw_f_ts + tw_m_ts
    tw_ho = tw_f_ho + tw_m_ho
    
    return cn_tr,  cn_ts, cn_ho, tw_tr, tw_ts, ts_ho

def mp3_to_wav(source_dir, dest_dir, sample_set, language, file_id):
    '''
    input:
        Directory of where the .mp3 source audios are located (source_dir)
        Directory of where the converted .wav audios should be saved (dest_dir)
        The sample set that is being processed, e.g. 'train', 'test', 'hold_out' (sample_set)
        The language being processed, e.g. 'cn', 'tw' (language)
        The file ID that is being processed (file_id)
    output:
        .wav formatted audio clips saved in the destination directory
    '''
    source = source_dir + language + "/" + file_id
    dest = dest_dir + sample_set + "/" + language + "/"+ file_id[:-4] + '.wav'
    audio = AudioSegment.from_mp3(source)
    audio.export(dest, format="wav")
    
    
if __name__ == "__main__":
    # Load the Mozilla metadata file
    path1 = "../raw_audio/cn/validated.tsv" # update to where the Mozilla's validated tsv file is located
    path2 = "../raw_audio/tw/validated.tsv"
    df = get_meta(path1)
    df1 = get_meta(path2)
    
    # Get the file ID of the samples 
    cn_tr,  cn_ts, cn_ho, tw_tr, tw_ts, ts_ho = get_random_sample(df, df1)
    
    # The follow loops will convert all the samples into .wav files
    for track in cn_tr:
        mp3towav('../raw_audio/', '../processed_audio', 'train', 'cn', track)
    for track in cn_ts:
        mp3towav('../raw_audio/', '../processed_audio', 'test','cn', track)
    for track in cn_ho:
        mp3towav('../raw_audio/', '../processed_audio', 'hold_out','cn', track)
    for track in tw_tr:
        mp3towav('../raw_audio/', '../processed_audio', 'train','tw', track)
    for track in tw_ts:
        mp3towav('../raw_audio/', '../processed_audio', 'test','tw', track)
    for track in tw_ho:
        mp3towav('../raw_audio/', '../processed_audio', 'hold_out','tw', track)
