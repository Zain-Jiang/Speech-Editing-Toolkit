from multiprocessing import Pool
import librosa
import torch
import numpy as np

from glob import glob
from tqdm import tqdm
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from python_speech_features import mfcc

MAX_WAV_VALUE = 32768.1
trainset_config = {'sampling_rate': 22050, 'filter_length': 1024, 'win_length': 1024, 'hop_length': 256, 'mel_fmin': 55,
                   'mel_fmax': 7600}


def load_wav_to_torch(full_path):
    """
    Loads wavdata into torch array
    """
    data, sr = librosa.core.load(full_path, sr=trainset_config['sampling_rate'])
    return torch.from_numpy(data).float(), sr


def cal_mcd(wav_pair, use_dtw=False):
    ref, est = wav_pair
    est_audio, _ = librosa.core.load(est, sr=trainset_config['sampling_rate'])
    ref_audio, _ = librosa.core.load(ref, sr=trainset_config['sampling_rate'])
    est_mfcc = librosa.feature.mfcc(y=est_audio, sr=trainset_config['sampling_rate'], 
                                    n_fft=1024, win_length=1024, fmin=55, fmax=7600, n_mels=80,
                                    hop_length=256, n_mfcc=34, htk=True)
    ref_mfcc = librosa.feature.mfcc(y=ref_audio, sr=trainset_config['sampling_rate'], 
                                    n_fft=1024, win_length=1024, fmin=55, fmax=7600, n_mels=80,
                                    hop_length=256, n_mfcc=34, htk=True)
    
    diff2sum = np.sum((est_mfcc - ref_mfcc) ** 2, 1)
    mcd = np.mean(10.0 / np.log(10.0) * np.sqrt(2 * diff2sum), 0) / ref_mfcc.shape[1]
    # mcd = (np.sum((ref_mfcc - est_mfcc) ** 2., 1) ** .5).mean() / ref_mfcc.shape[1]  # _mcd(ref_mfcc, est_mfcc)

    if use_dtw:
        distance, path = fastdtw(ref_mfcc, est_mfcc, dist=euclidean)
        pathx = list(map(lambda l: l[0], path))
        pathy = list(map(lambda l: l[1], path))
        x, y = ref_mfcc[pathx], est_mfcc[pathy]
        frames = x.shape[0]
        z = x - y
        mcd = np.sqrt((z * z).sum(-1)).mean()
    return mcd


def cal_mcd_with_wave_batch(waves_dir):
    wavs = glob(waves_dir)
    
    wave_pairs = []
    for item in wavs:
        if '[G_SEG]' in item:
            wave_pairs.append((item, item.replace('G_SEG', 'P_SEG')))

    mcd_total, audio_num = 0, 0
    processes = 4
    with Pool(processes) as pool:
        for result_item in tqdm(pool.imap(cal_mcd, wave_pairs)):
            mcd = result_item
            mcd_total += mcd
            audio_num += 1
    MCD_value = float(mcd_total) / float(audio_num)
    return MCD_value


if __name__ == '__main__':
    waves_dir = 'checkpoints/spec_denoiser_libritts_dur_pitch_masked_256/generated_568000_/wavs/*'
    MCD_value = cal_mcd_with_wave_batch(waves_dir)
    print("MCD = {:f}".format(MCD_value))
