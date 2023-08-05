from glob import glob
import pandas as pd
from eval.mcd import cal_mcd_with_wave_batch
from eval.stoi import cal_stoi_with_waves_batch
from eval.pesq_metric import cal_pesq_with_waves_batch


if __name__ == '__main__':
    
    # LibriTTS
    # wavs_dir = 'checkpoints/spec_denoiser_libritts_dur_pitch_masked_256/generated_568000_/wavs/*'
    # wavs_dir = 'checkpoints/spec_denoiser_libritts_normal_256/generated_800000_/wavs/*'
    # wavs_dir = 'checkpoints/spec_denoiser_libritts_wo_pitch_256/generated_742000_/wavs/*'
    # wavs_dir = 'checkpoints/campnet_libri_ali_0.8_256/generated_2000000_/wavs/*'
    # wavs_dir = 'checkpoints/campnet_libri_orig_0.3_256/generated_2000000_/wavs/*'
    # wavs_dir = 'checkpoints/campnet_libri_orig_0.12_256/generated_2000000_/wavs/*'
    # wavs_dir = 'checkpoints/a3t_libritts_0.8/generated_800000_/wavs/*'
    # wavs_dir = 'checkpoints/editspeech_orig_0.3_libritts/generated_100000_/wavs/*'

    
    # VCTK
    # wavs_dir = 'checkpoints/campnet_vctk/generated_2000000_/wavs/*'
    # wavs_dir = 'checkpoints/campnet_vctk_ali_0.8/generated_1000000_/wavs/*'
    # wavs_dir = 'checkpoints/spec_denoiser_vctk_wo_pitch/generated_300000_/wavs/*'
    # wavs_dir = 'checkpoints/spec_denoiser_vctk_normal/generated_300000_/wavs/*'
    # wavs_dir = 'checkpoints/spec_denoiser_vctk_dur_pitch_masked_0.8/generated_300000_/wavs/*'
    # wavs_dir = 'checkpoints/editspeech_orig_0.3_vctk/generated_100000_/wavs/*'
    # wavs_dir = 'checkpoints/a3t_vctk_0.8/generated_800000_/wavs/*'
    # wavs_dir = 'checkpoints/yq_wo_spec_vctk/generated_132000_/wavs/*'

    # stutter_set
    # wavs_dir = 'checkpoints/spec_denoiser_stutter_set_normal/generated_462000_/wavs/*'
    # wavs_dir = 'checkpoints/stutter_speech_stutter_set/generated_120000_/wavs/*'
    # wavs_dir = 'checkpoints/a3t_stutter_set_0.8/generated_600000_/wavs/*'
    # wavs_dir = 'checkpoints/spec_denoiser_stutter_set_wo_pitch/generated_300000_/wavs/*'
    # wavs_dir = 'checkpoints/spec_denoiser_stutterset_dur_pitch_masked/generated_300000_/wavs/*'
    wavs_dir = 'checkpoints/campnet_stutter_set_ali_0.8/generated_1000000_/wavs/*'
    # wavs_dir = 'checkpoints/campnet_stutter_set_orig_0.3/generated_1000000_/wavs/*'
    
    
    mcd_values = cal_mcd_with_wave_batch(wavs_dir)
    stoi_values = cal_stoi_with_waves_batch(wavs_dir)
    pesq_values = cal_pesq_with_waves_batch(wavs_dir)
    
    print(f"MCD = {mcd_values}; STOI = {stoi_values}; PESQ = {pesq_values}.")