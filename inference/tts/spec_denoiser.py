import os
import numpy as np
import torch
from data_gen.tts.base_preprocess import BasePreprocessor
from inference.tts.base_tts_infer import BaseTTSInfer
from modules.speech_editing.spec_denoiser import GaussianDiffusion
from modules.speech_editing.diffnet import DiffNet
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.hparams import hparams
from utils.spec_aug.time_mask import generate_time_mask
from utils.text.text_encoder import is_sil_phoneme
from utils.audio.align import get_mel2ph


DIFF_DECODERS = {
    'wavenet': lambda hp: DiffNet(hp['audio_num_mel_bins']),
}

class StutterSpeechInfer(BaseTTSInfer):
    def __init__(self, hparams, device=None):
        if device is None:
            # device = 'cuda' if torch.cuda.is_available() else 'cpu'
            device = 'cpu'
        self.hparams = hparams
        self.device = device
        self.data_dir = hparams['binary_data_dir']
        self.preprocessor = BasePreprocessor()
        self.ph_encoder, self.word_encoder = self.preprocessor.load_dict(self.data_dir)
        self.spk_map = self.preprocessor.load_spk_map(self.data_dir)
        self.model = self.build_model()
        self.model.eval()
        self.model.to(self.device)
        self.vocoder = self.build_vocoder()
        self.vocoder.eval()
        self.vocoder.to(self.device)

    def build_model(self):
        ph_dict_size = len(self.ph_encoder)
        word_dict_size = len(self.word_encoder)
        model = GaussianDiffusion(
            phone_encoder=self.ph_encoder,
            out_dims=hparams['audio_num_mel_bins'], denoise_fn=DIFF_DECODERS[hparams['diff_decoder_type']](hparams),
            timesteps=hparams['timesteps'], time_scale=hparams['timescale'],
            loss_type=hparams['diff_loss_type'],
            spec_min=hparams['spec_min'], spec_max=hparams['spec_max'],
        )
        load_ckpt(model, hparams['work_dir'], 'model')
        model.to(self.device)
        model.eval()
        return model

    def forward_model(self, inp):
        sample = self.input_to_batch(inp)
        with torch.no_grad():
            output = self.model(sample['txt_tokens'], time_mel_masks=sample['time_mel_masks'], mel2ph=sample['mel2ph'], spk_embed=None,
                       ref_mels=sample['mel'], f0=None, uv=None, energy=None, infer=True)
            mel_out = output['mel_out'] * sample['time_mel_masks'] + sample['mel'] * (1-sample['time_mel_masks'])
            wav_out = self.run_vocoder(mel_out)
            wav_gt = self.run_vocoder(sample['mel'])

        wav_out = wav_out.cpu().numpy()
        wav_gt = wav_gt.cpu().numpy()
        mel_out = mel_out.cpu().numpy()
        mel_gt = sample['mel'].cpu().numpy()

        return wav_out[0], wav_gt[0], mel_out[0], mel_gt[0]

    def preprocess_input(self, inp):
        """

        :param inp: {'text': str, 'item_name': (str, optional), 'spk_name': (str, optional)}
        :return:
        """
        preprocessor = self.preprocessor
        text_raw = inp['text']
        item_name = inp.get('item_name', '<ITEM_NAME>')
        spk_name = inp.get('spk_name', '<SINGLE_SPK>')
        ph, txt, _, _, ph_gb_word = preprocessor.txt_to_ph(
            preprocessor.txt_processor, text_raw)
        ph_token = self.ph_encoder.encode(ph)
        # spk_id = self.spk_map[spk_name]
        
        # masked prediction related
        wav = inp['wav']
        mel = inp['mel']

        # Generate forced alignment
        tg_fn = f'inference/audio/{item_name}.lab'
        ph_gb_word_nosil = "_".join(ph.split(' ')[1:-1]).replace('_|_', ' ')
        with open(tg_fn, 'w') as f_txt:
            f_txt.write(ph_gb_word_nosil)
        print("Generating forced alignments with mfa. Please wait for about 1 minutes.")
        os.system('mfa align --clean inference/audio data/processed/vctk/mfa_dict.txt data/processed/vctk/mfa_model.zip inference/audio/mfa_out')
        mfa_textgrid = f'inference/audio/mfa_out/{item_name}.TextGrid'
        mel2ph, dur = self.process_align(mfa_textgrid, ph, ph_token, mel)

        # Obtain ph-level mask
        mel2ph = np.array(mel2ph)
        ph_mask = np.zeros((mel2ph.max()+1).item())
        ph_mask[15:18] = 1.0
        # Obtain mel-level mask
        mel2ph_ = torch.from_numpy(mel2ph).long()
        ph_mask = torch.from_numpy(ph_mask).float()
        ph_mask = torch.nn.functional.pad(ph_mask, [1, 0])
        time_mel_masks = torch.gather(ph_mask, 0, mel2ph_).numpy()  # [B, T, H]

        item = {'item_name': item_name, 'text': txt, 'ph': ph,
                'ph_token': ph_token, 'mel2ph': mel2ph,
                'mel': mel, 'wav': wav, 'time_mel_masks': time_mel_masks}
        item['ph_len'] = len(item['ph_token'])
        return item

    def process_align(self, tg_fn, ph, ph_token, mel, text2mel_params={'hop_size':256,'audio_sample_rate':22050, 'mfa_min_sil_duration':0.1}):
        if tg_fn is not None and os.path.exists(tg_fn):
            mel2ph, dur = get_mel2ph(tg_fn, ph, mel, text2mel_params['hop_size'], text2mel_params['audio_sample_rate'],
                                     text2mel_params['mfa_min_sil_duration'])
        else:
            raise BinarizationError(f"Align not found")
        if np.array(mel2ph).max() - 1 >= len(ph_token):
            raise BinarizationError(
                f"Align does not match: mel2ph.max() - 1: {np.array(mel2ph).max() - 1}, len(phone_encoded): {len(ph_token)}")
        return mel2ph, dur

    def input_to_batch(self, item):
        item_names = [item['item_name']]
        text = [item['text']]
        ph = [item['ph']]
        mel2ph = torch.LongTensor(item['mel2ph'])[None, :].to(self.device)
        txt_tokens = torch.LongTensor(item['ph_token'])[None, :].to(self.device)
        txt_lengths = torch.LongTensor([txt_tokens.shape[1]]).to(self.device)
        # spk_ids = torch.LongTensor(item['spk_id'])[None, :].to(self.device)

        # masked prediction related
        mel = torch.FloatTensor(item['mel'])[None, :].to(self.device)
        wav = torch.FloatTensor(item['wav'])[None, :].to(self.device)
        time_mel_masks = torch.FloatTensor(item['time_mel_masks'])[None, :, None].to(self.device)
        batch = {
            'item_name': item_names,
            'text': text,
            'ph': ph,
            'mel2ph': mel2ph,
            'txt_tokens': txt_tokens,
            'txt_lengths': txt_lengths,
            # 'spk_ids': spk_ids,
            'mel': mel, 
            'wav': wav, 
            'time_mel_masks': time_mel_masks
        }
        return batch

    @classmethod
    def example_run(cls):
        from utils.commons.hparams import set_hparams
        from utils.commons.hparams import hparams as hp
        from utils.audio.io import save_wav
        from utils.audio import librosa_wav2spec
        from utils.plot.plot import plot_mel

        set_hparams()
        wav2spec_res = librosa_wav2spec('inference/audio/1.wav', fmin=55, fmax=7600, sample_rate=22050)
        inp = {
            'text': 'Nice to meet you Trump.',
            'item_name': '1',
            # 'text': 'And several new measures to protect Our security and prosperity.',
            'mel': wav2spec_res['mel'],
            'wav': wav2spec_res['wav'],
        }
        infer_ins = cls(hp)
        wav_out, wav_gt, mel_out, mel_gt = infer_ins.infer_once(inp)

        os.makedirs('infer_out', exist_ok=True)
        save_wav(wav_out, f'inference/out/wav_out.wav', hp['audio_sample_rate'])
        save_wav(wav_gt, f'inference/out/wav_gt.wav', hp['audio_sample_rate'])

        plot_mel(mel_out, 'inference/out/mel_out.png')
        plot_mel(mel_gt, 'inference/out/mel_gt.png')


if __name__ == '__main__':
    StutterSpeechInfer.example_run()
