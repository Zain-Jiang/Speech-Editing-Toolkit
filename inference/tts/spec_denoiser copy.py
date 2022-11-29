import os
import numpy as np
import torch
from data_gen.tts.base_preprocess import BasePreprocessor
from inference.tts.base_tts_infer import BaseTTSInfer
from modules.speech_editing.spec_denoiser.spec_denoiser import GaussianDiffusion
from modules.speech_editing.spec_denoiser.diffnet import DiffNet
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.hparams import hparams
from utils.spec_aug.time_mask import generate_time_mask
from utils.text.text_encoder import is_sil_phoneme
from utils.audio.align import get_mel2ph
from resemblyzer import VoiceEncoder


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
        self.spk_embeding = VoiceEncoder(device='cpu')

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

        # forward the edited txt to the encoder
        edited_txt_tokens = sample['edited_txt_tokens']
        mel = sample['mel']
        mel2ph = sample['mel2ph']
        mel2word = sample['mel2word']
        edited_word_idx = 14
        changed_idx = [13,25]

        ret = {}
        encoder_out = self.model.fs.encoder(edited_txt_tokens)  # [B, T, C]
        src_nonpadding = (edited_txt_tokens > 0).float()[:, :, None]
        style_embed = self.model.fs.forward_style_embed(sample['spk_embed'], None)
        
        # forward duration model to get the duration for whole edited text seq
        dur_inp = (encoder_out + style_embed) * src_nonpadding
        masked_mel2ph = mel2ph
        masked_mel2ph[mel2ph==edited_word_idx] = 0
        time_mel_masks = torch.zeros_like(mel2ph).to(self.device)
        time_mel_masks[mel2ph==edited_word_idx] = 1.0
        edited_mel2ph = self.model.fs.forward_dur(dur_inp, time_mel_masks, masked_mel2ph, edited_txt_tokens, ret)
        edited_mel2word = torch.Tensor([sample['edited_ph2word'][0].numpy()[p - 1] for p in edited_mel2ph[0]]).to(self.device)[None, :]
        
        # get mel2ph of the edited region by concating the head and tial of the original mel2ph
        # length_edited = edited_mel2word[(edited_mel2word>=changed_idx[0]) & (edited_mel2word<=changed_idx[1])].size(0) - mel2word[mel2word==edited_word_idx].size(0)
        # edited_mel2ph_ = torch.zeros((1, mel2ph.size(1)+length_edited)).to(self.device)
        # head_idx = mel2word[mel2word<edited_word_idx].size(0)
        # tail_idx = mel2word[mel2word<=edited_word_idx].size(0) + length_edited
        # edited_mel2ph_[:, :head_idx] = mel2ph[:, :head_idx]
        # edited_mel2ph_[:, head_idx:tail_idx] = edited_mel2ph[(edited_mel2word>=changed_idx[0]) & (edited_mel2word<=changed_idx[1])]
        # edited_mel2ph_[:, tail_idx:] = mel2ph[mel2word>edited_word_idx] - mel2ph[mel2word>edited_word_idx].min() + edited_mel2ph[(edited_mel2word>=changed_idx[0]) & (edited_mel2word<=changed_idx[1])].max() + 2
        # edited_mel2ph = edited_mel2ph_.long()

        # create new ref mel
        ref_mels = torch.zeros((1, edited_mel2ph.size(1), mel.size(2))).to(self.device)
        T = min(ref_mels.size(1), mel.size(1))
        ref_mels[:, :head_idx, :] = mel[:, :head_idx, :]
        ref_mels[:, tail_idx:, :] = mel[mel2word>edited_word_idx]
        # create time mask 
        time_mel_masks = torch.zeros((1, edited_mel2ph.size(1), 1)).to(self.device)
        time_mel_masks[:, head_idx:tail_idx] = 1.0
        
        with torch.no_grad():
            output = self.model(sample['edited_txt_tokens'], time_mel_masks=time_mel_masks, mel2ph=edited_mel2ph, spk_embed=sample['spk_embed'],
                       ref_mels=ref_mels, f0=None, uv=None, energy=None, infer=True)
            mel_out = output['mel_out'] * time_mel_masks + ref_mels * (1-time_mel_masks)
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
        ph, txt, _, ph2word, ph_gb_word = preprocessor.txt_to_ph(
            preprocessor.txt_processor, text_raw)
        ph_token = self.ph_encoder.encode(ph)
        
        # get ph for edited txt
        edited_text_raw = inp['edited_text']
        edited_ph, _, _, edited_ph2word, _ = preprocessor.txt_to_ph(
            preprocessor.txt_processor, edited_text_raw)
        edited_ph_token = self.ph_encoder.encode(edited_ph)
        
        # Generate forced alignment
        wav = inp['wav']
        mel = inp['mel']
        tg_fn = f'inference/audio/{item_name}.lab'
        ph_gb_word_nosil = " ".join(["_".join([p for p in w.split("_") if not is_sil_phoneme(p)])
                                for w in ph_gb_word.split(" ") if not is_sil_phoneme(w)])
        with open(tg_fn, 'w') as f_txt:
            f_txt.write(ph_gb_word_nosil)
        with open('data/processed/vctk/mfa_dict.txt', 'r') as f: # update mfa dict for unseen word
            lines = f.readlines()
        with open('data/processed/vctk/mfa_dict.txt', 'a+') as f:
            for item in ph_gb_word_nosil.split(" "):
                item = item + ' ' + ' '.join(item.split('_')) + '\n'
                if item not in lines:
                    f.writelines([item])
        print("Generating forced alignments with mfa. Please wait for about 1 minutes.")
        os.system('mfa align --clean inference/audio data/processed/vctk/mfa_dict.txt data/processed/vctk/mfa_model.zip inference/audio/mfa_out')
        mfa_textgrid = f'inference/audio/mfa_out/{item_name}.TextGrid'
        mel2ph, dur = self.process_align(mfa_textgrid, ph, ph_token, mel)
        mel2word = [ph2word[p - 1] for p in mel2ph] # [T_mel]

        item = {'item_name': item_name, 'text': txt, 'ph': ph, 
                'ph2word': ph2word, 'edited_ph2word': edited_ph2word,
                'ph_token': ph_token, 'edited_ph_token': edited_ph_token, 
                'mel2ph': mel2ph, 'mel2word': mel2word,
                'mel': mel, 'wav': wav}
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
        ph2word = torch.LongTensor(item['ph2word'])[None, :].to(self.device)
        edited_ph2word = torch.LongTensor(item['edited_ph2word'])[None, :].to(self.device)
        mel2ph = torch.LongTensor(item['mel2ph'])[None, :].to(self.device)
        mel2word = torch.LongTensor(item['mel2word'])[None, :].to(self.device)
        txt_tokens = torch.LongTensor(item['ph_token'])[None, :].to(self.device)
        txt_lengths = torch.LongTensor([txt_tokens.shape[1]]).to(self.device)
        edited_txt_tokens = torch.LongTensor(item['edited_ph_token'])[None, :].to(self.device)
        # spk_ids = torch.LongTensor(item['spk_id'])[None, :].to(self.device)

        # masked prediction related
        mel = torch.FloatTensor(item['mel'])[None, :].to(self.device)
        wav = torch.FloatTensor(item['wav'])[None, :].to(self.device)

        # get spk embed
        spk_embed = self.spk_embeding.embed_utterance(item['wav'].astype(float))
        spk_embed = torch.FloatTensor(spk_embed[None, :]).to(self.device)

        batch = {
            'item_name': item_names,
            'text': text,
            'ph': ph,
            'ph2word': ph2word,
            'edited_ph2word': edited_ph2word,
            'mel2ph': mel2ph,
            'mel2word': mel2word,
            'txt_tokens': txt_tokens,
            'txt_lengths': txt_lengths,
            'edited_txt_tokens': edited_txt_tokens,
            # 'spk_ids': spk_ids,
            'mel': mel, 
            'wav': wav,
            'spk_embed': spk_embed,
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
        wav2spec_res = librosa_wav2spec('inference/audio/trump.wav', fmin=55, fmax=7600, sample_rate=22050)
        inp = {
            'text': 'And several new measures to protect American security and prosperity.',
            'edited_text': 'And several new measures to protect China China China Let\'s Say China security and prosperity.',
            'item_name': 'trump',
            # 'text': 'This is a LibriVox recording.',
            # 'edited_text': 'This is a god damn online course recording.',
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
