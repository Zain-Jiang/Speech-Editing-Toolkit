import os
import numpy as np
from utils.audio.align import get_mel2ph
from utils.audio.pitch_extractors import extract_pitch
from utils.audio.pitch.utils import f0_to_coarse, norm_interp_f0, denorm_f0

def get_align_from_mfa_output(tg_fn, ph, ph_token, mel, text2mel_params={'hop_size':256,'audio_sample_rate':22050, 'mfa_min_sil_duration':0.1}):
    if tg_fn is not None and os.path.exists(tg_fn):
        mel2ph, dur = get_mel2ph(tg_fn, ph, mel, text2mel_params['hop_size'], text2mel_params['audio_sample_rate'],
                                    text2mel_params['mfa_min_sil_duration'])
    else:
        raise Exception(f"Align not found")
    if np.array(mel2ph).max() - 1 >= len(ph_token):
        raise Exception(
            f"Align does not match: mel2ph.max() - 1: {np.array(mel2ph).max() - 1}, len(phone_encoded): {len(ph_token)}")
    return mel2ph, dur

def extract_f0_uv(wav, mel):
    T = mel.shape[0]
    f0 = extract_pitch('parselmouth', wav, 256, 22050, f0_min=80, f0_max=600)
    assert len(mel) == len(f0), (len(mel), len(f0))
    f0, uv = norm_interp_f0(f0[:T])
    return f0, uv