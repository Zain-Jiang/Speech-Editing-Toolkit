import parselmouth
import numpy as np
from glob import glob
from numpy import mean
from tqdm import tqdm

import utils.commons.single_thread_env
from utils.audio import librosa_wav2spec

# text2mel parameters
text2mel_params = {'fft_size': 1024, 'hop_size': 256, 'win_size': 1024,
                        'audio_num_mel_bins': 80, 'fmin': 55, 'fmax': 7600,
                        'f0_min': 80, 'f0_max': 600, 'pitch_extractor': 'parselmouth',
                        'audio_sample_rate': 22050, 'loud_norm': False,
                        'mfa_min_sil_duration': 0.1, 'trim_eos_bos': False,
                        'with_align': True, 'text2mel_params': False,
                        'dataset_name': 'vctk',
                        'with_f0': True, 'min_mel_length': 64}

def process_audio(wav_fn, text2mel_params):
        wav2spec_dict = librosa_wav2spec(
            wav_fn,
            fft_size=text2mel_params['fft_size'],
            hop_size=text2mel_params['hop_size'],
            win_length=text2mel_params['win_size'],
            num_mels=text2mel_params['audio_num_mel_bins'],
            fmin=text2mel_params['fmin'],
            fmax=text2mel_params['fmax'],
            sample_rate=text2mel_params['audio_sample_rate'],
            loud_norm=text2mel_params['loud_norm'])
        mel = wav2spec_dict['mel']
        wav = wav2spec_dict['wav'].astype(np.float16)
        return wav

def parselmouth_pitch(wav_data, hop_size, audio_sample_rate, f0_min, f0_max,
                      voicing_threshold=0.6, *args, **kwargs):
    time_step = hop_size / audio_sample_rate * 1000
    n_mel_frames = int(len(wav_data) // hop_size)
    f0_pm = parselmouth.Sound(wav_data, audio_sample_rate).to_pitch_ac(
        time_step=time_step / 1000, voicing_threshold=voicing_threshold,
        pitch_floor=f0_min, pitch_ceiling=f0_max).selected_array['frequency']
    pad_size = (n_mel_frames - len(f0_pm) + 1) // 2
    f0 = np.pad(f0_pm, [[pad_size, n_mel_frames - len(f0_pm) - pad_size]], mode='constant')
    return f0


#wavs_dir = 'checkpoints/spec_denoiser_vctk_normal/generated_300000_/wavs/*'
#wavs_dir = 'checkpoints/spec_denoiser_vctk_dur_pitch_masked_0.8/generated_300000_/wavs/*'
#wavs_dir = 'checkpoints/campnet_vctk/generated_2000000_/wavs/*'
#wavs_dir = 'checkpoints/campnet_vctk_ali_0.8/generated_1000000_/wavs/*'
#wavs_dir = 'checkpoints/editspeech_orig_0.3_vctk/generated_100000_/wavs/*'
#wavs_dir = 'checkpoints/a3t_vctk_0.8/generated_800000_/wavs/*'
wavs_dir = 'checkpoints/yq_wo_spec_vctk_copy/generated_132000_/wavs/*'

wavs = glob(wavs_dir)
wave_pairs = []
for item in wavs:
    if '[G_SEG]' in item:
        wave_pairs.append((item, item.replace('G_SEG', 'P_SEG')))

pitch_mse = []
for (gt, predict) in tqdm(wave_pairs):
    def cal_f0(wav_fn):
        wav = process_audio(wav_fn, text2mel_params)
        f0 = parselmouth_pitch(wav, text2mel_params['hop_size'],  text2mel_params['audio_sample_rate'],
                            text2mel_params['f0_min'], text2mel_params['f0_max'])
        return f0
    gt_f0 = cal_f0(gt)
    predict_f0 = cal_f0(predict)
    error = []
    for i in range(len(gt_f0)):
        error.append((gt_f0[i] - predict_f0[i]) ** 2)
    pitch_mse.append(mean(error))

print(mean(pitch_mse))