<p align="center">
    <br>
    <img src="assets/logo.png" width="200"/>
    <br>
</p>

<h2 align="center">
<p> Speech-Editing-Toolkit</p>
</h2>

This repo contains official PyTorch implementations of:

- [FluentSpeech: Stutter-Oriented Automatic Speech Editing with Context-Aware Diffusion Models](https://github.com/Zain-Jiang/Speech-Editing-Toolkit) (ACL 2023) 
[Demo page](https://speechai-demo.github.io/FluentSpeech/)
<p align="center">
    <br>
    <img src="assets/spec_denoiser.gif" width="400" height="180"/>
    <br>
</p>

This repo contains unofficial PyTorch implementations of:

- [CampNet: Context-Aware Mask Prediction for End-to-End Text-Based Speech Editing](https://arxiv.org/pdf/2202.09950) (ICASSP 2022)  
[Demo page](https://hairuo55.github.io/CampNet)
- [A3T: Alignment-Aware Acoustic and Text Pretraining for Speech Synthesis and Editing](https://proceedings.mlr.press/v162/bai22d/bai22d.pdf) (ICML 2022)  
[Demo page](https://educated-toothpaste-462.notion.site/Demo-b0edd300e6004c508744c6259369a468) | [Official code](https://github.com/richardbaihe/a3t)
- [EditSpeech: A text based speech editing system using partial inference and bidirectional fusion](https://arxiv.org/pdf/2107.01554) (ASRU 2021)  
[Demo page](https://daxintan-cuhk.github.io/EditSpeech/)



## Supported Datasets
Our framework supports the following datasets:

- VCTK
- LibriTTS
- SASE Dataset (We will publish it later)

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
# You can set the 'self.dataset_name' in these files as 'vctk' or 'libritts' to process these datasets. And you should also set the ``BASE_DIR`` value in ``run_mfa_train_align.sh`` to the corresponding directory. 
# The default dataset is ``vctk``.
python data_gen/tts/base_preprocess.py
python data_gen/tts/run_mfa_train_align.sh
python data_gen/tts/base_binarizer.py
```

## Train (FluentSpeech)
```bash
# Example run for FluentSpeech.
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config egs/spec_denoiser.yaml --exp_name spec_denoiser --reset
```

## Train (Baselines)
```bash
# Example run for CampNet.
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config egs/campnet.yaml --exp_name campnet --reset
# Example run for A3T.
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config egs/a3t.yaml --exp_name a3t --reset
# Example run for EditSpeech.
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config egs/editspeech.yaml --exp_name editspeech --reset
```

## Pretrained Checkpoint
Here, we provide the pretrained checkpoint of fluentspeech. To start, please put the `config.yaml` and `xxx.ckpt` at `./checkpoints/spec_denoiser/`.

|  model   | dataset  | url | checkpoint name |
| -- | -- | -- | -- |
|  FluentSpeech  | libritts-clean  | https://drive.google.com/drive/folders/1saqpWc4vrSgUZvRvHkf2QbwWSikMTyoo?usp=sharing | model_ckpt_steps_568000.ckpt |


## Inference
We provide the data structure of inference in inference/example.csv. `text` and `edited_text` refer to the original text and target text. `region` refers to the word idx range (start from 1 ) that you want to edit. `edited_region` refers to the word idx range of the edited_text.

|  id   | item_name  | text | edited_text| wav_fn_orig | edited_region| region|
| -- | -- | -- | -- | -- | -- | -- |
|  0  | 1  | "this is a libri vox recording" | "this is a funny joke shows." | inference/audio_backup/1.wav | [3,6] | [3,6] |

```bash
# run with one example
python inference/tts/spec_denoiser.py --exp_name spec_denoiser
```

## Citation

If you find this useful for your research, please star our repo.


## License and Agreement
Any organization or individual is prohibited from using any technology mentioned in this paper to generate someone's speech without his/her consent, including but not limited to government leaders, political figures, and celebrities. If you do not comply with this item, you could be in violation of copyright laws.


## Tips
1. If you find the ``mfa_dict.txt``, ``mfa_model.zip``, ``phone_set.json``, or ``word_set.json`` are missing in inference, you need to run the preprocess script in our repo to get them. You can also download all of these files you need for inferencing the pre-trained model from
``https://drive.google.com/drive/folders/1BOFQ0j2j6nsPqfUlG8ot9I-xvNGmwgPK?usp=sharing`` and put them in ``data/processed/libritts``. 
2. Please specify the MFA version as 2.0.0rc3.

If you find any other problems, please contact me.
