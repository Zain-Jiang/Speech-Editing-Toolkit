import os
import numpy as np
import re
from utils.audio.align import get_mel2ph
from utils.audio.pitch_extractors import extract_pitch
from utils.audio.pitch.utils import f0_to_coarse, norm_interp_f0, denorm_f0
from utils.text.text_encoder import is_sil_phoneme, PUNCS

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

def get_words_region_from_origintxt_region(words,region_list):
    word_id = 0
    region_id = 0
    words_region = [[0,0] for i in range(len(region_list))]
    assert len(region_list) >= 1, f'length of region_list is {len(region_list)}'
    for i,word in enumerate(words):
        if is_sil_phoneme(word) and word in ['|','<BOS>','<pad>']:
            continue
        word_id+=1
        if word_id == region_list[region_id][0]:
            words_region[region_id][0] = i+1
        if word_id == region_list[region_id][1]:
            words_region[region_id][1] = i+1
            region_id+=1
        if region_id == len(region_list):
            break
    return words_region


def parse_region_list_from_str(region_str):    
    pattern = '\[([1-9]\d*),([1-9]\d*)]'
    region_list = []
    region_str_list = re.findall(pattern,region_str)
    region_list = [[int(region_str[0]),int(region_str[1])] for region_str in region_str_list]
    region_list = sorted(region_list, key = lambda x:x[0])
    return region_list

        
    