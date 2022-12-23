<p align="center">
    <br>
    <img src="assets/logo.png" width="200"/>
    <br>
</p>

<h2 align="center">
<p> Speech-Editing-Toolkit</p>
</h2>

<div align="center">

[![GitHub Stars](https://img.shields.io/github/stars/Zain-Jiang/Speech-Editing-Toolkit?style=social)](https://github.com/Zain-Jiang/Speech-Editing-Toolkit)
![visitors](https://visitor-badge.glitch.me/badge?page_id=Zain-Jiang/Speech-Editing-Toolkit)

</div>

This repo contains unofficial PyTorch implementation of:

- [CampNet: Context-Aware Mask Prediction for End-to-End Text-Based Speech Editing](https://arxiv.org/pdf/2202.09950) (ICASSP 2022)  
[Demo page](https://hairuo55.github.io/CampNet)
- [A3T: Alignment-Aware Acoustic and Text Pretraining for Speech Synthesis and Editing](https://proceedings.mlr.press/v162/bai22d/bai22d.pdf) (ICML 2022)  
[Demo page](https://educated-toothpaste-462.notion.site/Demo-b0edd300e6004c508744c6259369a468) | [Official code](https://github.com/richardbaihe/a3t)
- [EditSpeech: A text based speech editing system using partial inference and bidirectional fusion](https://arxiv.org/pdf/2107.01554) (ASRU 2021)  
[Demo page](https://daxintan-cuhk.github.io/EditSpeech/) (Still Working)
- Our implementation of a generator-based spectrogram denoise diffusion models for speech editing.

## Supported Datasets
Our framework supports the following datasets:

- VCTK
- LibriTTS
- StammerSpeech Dataset (We will publish it later)

## Install Dependencies
Please install the latest numpy, torch and tensorboard first. Then run the following commands:
```bash
export PYTHONPATH=.
# install requirements.
pip install -U pip
pip install -r requirements.txt
sudo apt install -y sox libsox-fmt-mp3
```
Finally, install Montreal Forced Aligner following the link below:

`https://montreal-forced-aligner.readthedocs.io/en/latest/`

## Download the pre-trained vocoder
```
mkdir pretrained
mkdir pretrained/hifigan_hifitts
```
download `model_ckpt_steps_2168000.ckpt`, `config.yaml`, from https://drive.google.com/drive/folders/1n_0tROauyiAYGUDbmoQ__eqyT_G4RvjN?usp=sharing to `pretrained/hifigan_hifitts`

## Data Preprocess
```bash
python data_gen/tts/base_preprocess.py
python data_gen/tts/run_mfa_train_aligh.sh
python data_gen/tts/base_binarizer.py
```

## Train
```bash
# example run
CUDA_VISIBLE_DEVICES=2 python tasks/run.py --config egs/campnet_vctk.yaml --exp_name campnet_vctk --reset
```

## Inference
```bash
# run with full inference
CUDA_VISIBLE_DEVICES=2 python tasks/run.py --config egs/campnet_vctk.yaml --exp_name campnet_vctk --infer

# run with one example
python inference/tts/campnet.py --config=/checkpoints/campnet_vctk/config.yaml --exp_name campnet_vctk --hparams='work_dir=checkpoints/campnet_vctk'
```

## Citation

If you find this useful for your research, please star our repo.


## License and Agreement
Any organization or individual is prohibited from using any technology mentioned in this paper to generate someone's speech without his/her consent, including but not limited to government leaders, political figures, and celebrities. If you do not comply with this item, you could be in violation of copyright laws.
