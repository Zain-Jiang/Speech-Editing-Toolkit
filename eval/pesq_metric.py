import librosa
import torch
import numpy as np

from glob import glob
from pesq import pesq
from multiprocessing import Pool
from tqdm import tqdm

config = {'sampling_rate': 16000, 'filter_length': 1024, 'win_length': 1024, 'hop_length': 256, 'mel_fmin': 80,
          'mel_fmax': 8000}
MAX_WAV_VALUE = 32768.1


def load_wav_to_torch(full_path):
    """
    Loads wavdata into torch array
    """
    data, sr = librosa.core.load(full_path, sr=config['sampling_rate'])
    return torch.from_numpy(data).float(), sr


def cal_pesq(wave_pair):
    ref_path, gen_path = wave_pair
    ref_wave, rate = load_wav_to_torch(ref_path)
    gen_wave, _ = load_wav_to_torch(gen_path)
    min_length = min(ref_wave.size(0), gen_wave.size(0))
    ref_wave = ref_wave[:min_length]
    gen_wave = gen_wave[:min_length]
    ref_wave = ref_wave / MAX_WAV_VALUE
    gen_wave = gen_wave / MAX_WAV_VALUE

    ref = ref_wave.cpu().numpy()
    gen = gen_wave.cpu().numpy()
    try:
        score = pesq(config['sampling_rate'], ref, gen, 'nb')
    except:
        score = None
    return score

def cal_pesq_with_waves_batch(waves_dir):
    wavs = glob(waves_dir)
    
    wave_pairs = []
    for item in wavs:
        if '[G_SEG]' in item:
            wave_pairs.append((item, item.replace('G_SEG', 'P_SEG')))

    pesq_total, audio_num = 0, 0
    processes = 4
    with Pool(processes) as pool:
        for result_item in tqdm(pool.imap(cal_pesq, wave_pairs)):
            score = result_item
            if score is not None:
                pesq_total += score
                audio_num += 1
    PESQ_value = float(pesq_total) / float(audio_num)
    return PESQ_value


if __name__ == '__main__':
    # wavs = glob('checkpoints/spec_denoiser_libritts_dur_pitch_masked_256/generated_568000_/wavs/*')
    # wavs = glob('checkpoints/spec_denoiser_stutterset_dur_pitch_masked/generated_300000_/wavs/*')
    # wavs = glob('checkpoints/spec_denoiser_vctk/generated_300000_/wavs/*')
    # wavs = glob('checkpoints/a3t_vctk_0.8/generated_300000_/wavs/*')
    # wavs = glob('checkpoints/campnet_vctk_ali_0.8/generated_300000_/wavs/*')
    # wavs = glob('checkpoints/campnet_vctk/generated_2000000_/wavs/*')
    wavs = 'checkpoints/spec_denoiser_libritts_dur_pitch_masked_256/generated_568000_/wavs/*' 
    PESQ_value = cal_pesq_with_waves_batch(wavs)   
    print("PESQ = {:f}".format(PESQ_value))
