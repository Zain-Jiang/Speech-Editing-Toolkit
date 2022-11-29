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
trainset_config = {'sampling_rate': 22050, 'filter_length': 1024, 'win_length': 1024, 'hop_length': 256, 'mel_fmin':80, 'mel_fmax':8000}


def load_wav_to_torch(full_path):
    """
    Loads wavdata into torch array
    """
    data, sr = librosa.core.load(full_path, sr=trainset_config['sampling_rate'])
    return torch.from_numpy(data).float(), sr

def cal_mcd(wav_pair):
    ref, est = wav_pair
    est_audio, _ = load_wav_to_torch(est)
    ref_audio, _ = load_wav_to_torch(ref)
    min_length = min(ref_audio.size(0), est_audio.size(0))
    ref_audio = ref_audio[:min_length]
    est_audio = est_audio[:min_length]
    est_audio = est_audio.unsqueeze(0).unsqueeze(0)
    ref_audio = ref_audio / MAX_WAV_VALUE
    est_audio = est_audio / MAX_WAV_VALUE

    est = est_audio.cpu().numpy()
    ref = ref_audio.cpu().numpy()
    ref_mfcc = mfcc(ref, nfft=trainset_config['filter_length'],
                    winlen=trainset_config['win_length'] / trainset_config['sampling_rate'],
                    winstep=trainset_config['hop_length'] / trainset_config['sampling_rate'], appendEnergy=False)
    est_mfcc = mfcc(est, nfft=trainset_config['filter_length'],
                    winlen=trainset_config['win_length'] / trainset_config['sampling_rate'],
                    winstep=trainset_config['hop_length'] / trainset_config['sampling_rate'], appendEnergy=False)

    distance, path = fastdtw(ref_mfcc, est_mfcc, dist=euclidean)
    pathx = list(map(lambda l: l[0], path))
    pathy = list(map(lambda l: l[1], path))
    x, y = ref_mfcc[pathx], est_mfcc[pathy]
    frames = x.shape[0]
    z = x - y
    mcd = np.sqrt((z * z).sum(-1)).mean()
    # mcd = (np.sum((ref_mfcc - est_mfcc) ** 2., 1) ** .5).mean() / x.shape[0]  # _mcd(ref_mfcc, est_mfcc)
    return mcd, frames

if __name__ == '__main__':
    _logdb_const = 10.0 / np.log(10.0) * np.sqrt(2.0)
    # wavs = glob('checkpoints/spec_denoiser_vctk/generated_300000_/wavs/*')
    wavs = glob('checkpoints/spec_denoiser_vctk_dur_pitch/generated_300000_/wavs/*')
    # wavs = glob('checkpoints/a3t_vctk_0.8/generated_300000_/wavs/*')
    # wavs = glob('checkpoints/campnet_vctk_ali_0.8/generated_514000_/wavs/*')
    wav_pred, wav_gt = [], []
    for item in wavs:
        if '[G]' in item:
            wav_gt.append(item)
        else:
            wav_pred.append(item)
    wav_gt.sort()
    wav_pred.sort()
    wave_pairs = []
    for i in range(200):
        wave_pairs.append((wav_gt[i], wav_pred[i]))


    mcd_total, frames_total = 0, 0
    processes = 4
    with Pool(processes) as pool:
        for result_item in tqdm(pool.imap(cal_mcd, wave_pairs)):
            mcd, frames = result_item
            mcd_total += mcd
            frames_total += frames
    MCD_value = _logdb_const * float(mcd_total) / float(frames_total)
    print("MCD = : {:f}".format(MCD_value))
    # mcd_total = 0
    # for idx in tqdm(range(len(wav_gt))):
    #     mcd_total += mcd(wav_pred[idx], wav_gt[idx])
    # print(mcd_total/len(wav_gt))
    # mcd()