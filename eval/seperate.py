import os
from glob import glob
from shutil import copyfile

wav_path = 'checkpoints/yq_wo_spec_vctk/generated_132000_'

gt_wav_path = os.path.join(wav_path, 'gt')
gen_wav_path = os.path.join(wav_path, 'gen')
os.makedirs(gt_wav_path, exist_ok=True)
os.makedirs(gen_wav_path, exist_ok=True)
wavs = glob(os.path.join(wav_path, 'wavs', '*.wav'))
for wav in wavs:
    print(wav)
    if(wav.find('[P]') != -1):
        file_name = wav.split('/')[-1]
        copyfile(wav, os.path.join(gen_wav_path, file_name))
    elif(wav.find('[G]') != -1):
        file_name = wav.split('/')[-1]
        copyfile(wav, os.path.join(gt_wav_path, file_name))
    
