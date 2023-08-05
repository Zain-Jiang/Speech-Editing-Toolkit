import pandas as pd
from numpy import mean

#tb = pd.read_csv('checkpoints/editspeech_orig_0.3_vctk/generated_100000_/meta.csv') 
#tb = pd.read_csv('checkpoints/spec_denoiser_vctk_normal/generated_300000_/meta.csv')   
tb = pd.read_csv('checkpoints/spec_denoiser_vctk_dur_pitch_masked_0.8/generated_300000_/meta.csv') 


print(mean(tb['dur_loss']))