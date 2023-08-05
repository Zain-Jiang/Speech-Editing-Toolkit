import os
import numpy as np
import torch
from data_gen.tts.base_preprocess import BasePreprocessor
from inference.tts.base_tts_infer import BaseTTSInfer
from inference.tts.infer_utils import get_align_from_mfa_output, extract_f0_uv
from modules.speech_editing.spec_denoiser.spec_denoiser_normal import GaussianDiffusion
from modules.speech_editing.spec_denoiser.diffnet import DiffNet
from modules.tts.commons.align_ops import clip_mel2token_to_multiple, expand_states
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.hparams import hparams
from utils.spec_aug.time_mask import generate_time_mask
from utils.text.text_encoder import is_sil_phoneme
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

        # Forward the edited txt to the encoder
        edited_txt_tokens = sample['edited_txt_tokens']
        mel = sample['mel']
        mel2ph = sample['mel2ph']
        f0 = sample['f0']
        uv = sample['uv']
        ref_mels = mel

        # Create time mask
        time_mel_masks = sample['time_mask']
        
        with torch.no_grad():
            output = self.model(edited_txt_tokens, time_mel_masks=time_mel_masks, mel2ph=mel2ph, spk_embed=sample['spk_embed'],
                       ref_mels=mel, f0=None, uv=None, energy=None, infer=True)
            mel_out = output['mel_out'] * time_mel_masks + ref_mels * (1-time_mel_masks)
            wav_out = self.run_vocoder(mel_out)
            wav_gt = self.run_vocoder(sample['mel'])

        wav_out = wav_out.cpu().numpy()
        wav_gt = wav_gt.cpu().numpy()
        mel_out = mel_out.cpu().numpy()
        mel_gt = sample['mel'].cpu().numpy()
        masked_mel_out = ref_mels.cpu().numpy()
        masked_mel_gt = sample['mel'].cpu().numpy()

        return wav_out[0], wav_gt[0], mel_out[0], mel_gt[0], masked_mel_out[0], masked_mel_gt[0]

    def preprocess_input(self, inp):
        """

        :param inp: {'text': str, 'item_name': (str, optional), 'spk_name': (str, optional)}
        :return:
        """
        # Get ph for original txt
        preprocessor = self.preprocessor
        text_raw = inp['text']
        item_name = inp.get('item_name', '<ITEM_NAME>')
        spk_name = inp.get('spk_name', '<SINGLE_SPK>')
        ph, txt, _, ph2word, ph_gb_word = preprocessor.txt_to_ph(
            preprocessor.txt_processor, text_raw)
        ph_token = self.ph_encoder.encode(ph)
        # Get ph for edited txt
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
        os.system('mfa align -j 2 --clean inference/audio data/processed/vctk/mfa_dict.txt data/processed/vctk/mfa_model.zip inference/audio/mfa_out')
        mfa_textgrid = f'inference/audio/mfa_out/{item_name}.TextGrid'
        mel2ph, dur = get_align_from_mfa_output(mfa_textgrid, ph, ph_token, mel)
        mel2word = [ph2word[p - 1] for p in mel2ph] # [T_mel]

        # Extract frame-level f0 and uv (pitch info)
        f0, uv = extract_f0_uv(wav, mel)

        time_mask = np.zeros(mel.shape[0])
        time_mask[120:195] = 1.0

        item = {'item_name': item_name, 'text': txt, 'ph': ph, 
                'ph2word': ph2word, 'edited_ph2word': edited_ph2word,
                'ph_token': ph_token, 'edited_ph_token': edited_ph_token, 
                'mel2ph': mel2ph, 'mel2word': mel2word, 'dur': dur,
                'f0': f0, 'uv': uv,
                'mel': mel, 'wav': wav,
                'time_mask': time_mask}
        return item

    def input_to_batch(self, item):
        item_names = [item['item_name']]
        text = [item['text']]
        ph = [item['ph']]
        ph2word = torch.LongTensor(item['ph2word'])[None, :].to(self.device)
        edited_ph2word = torch.LongTensor(item['edited_ph2word'])[None, :].to(self.device)
        mel2ph = torch.LongTensor(item['mel2ph'])[None, :].to(self.device)
        dur = torch.LongTensor(item['dur'])[None, :].to(self.device)
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

        # get frame-level f0 and uv (pitch info)
        f0 = torch.FloatTensor(item['f0'])[None, :].to(self.device)
        uv = torch.FloatTensor(item['uv'])[None, :].to(self.device)

        time_mask = torch.FloatTensor(item['time_mask'])[None, :, None].to(self.device)

        batch = {
            'item_name': item_names,
            'text': text,
            'ph': ph,
            'ph2word': ph2word,
            'edited_ph2word': edited_ph2word,
            'mel2ph': mel2ph,
            'mel2word': mel2word,
            'dur': dur,
            'txt_tokens': txt_tokens,
            'txt_lengths': txt_lengths,
            'edited_txt_tokens': edited_txt_tokens,
            # 'spk_ids': spk_ids,
            'mel': mel, 
            'wav': wav,
            'spk_embed': spk_embed,
            'f0': f0,
            'uv': uv,
            'time_mask': time_mask
        }
        return batch

    @classmethod
    def example_run(cls):
        from utils.commons.hparams import set_hparams
        from utils.commons.hparams import hparams as hp
        from utils.audio.io import save_wav
        from utils.audio import librosa_wav2spec
        from inference.tts.infer_utils import plot_mel

        set_hparams()
        wav2spec_res = librosa_wav2spec('inference/audio/p323_290.wav', fmin=55, fmax=7600, sample_rate=22050)
        inp = {
            'text': 'we didnt enjoy the first game , but today they were excellent .',
            'edited_text': 'we didnt enjoy the first game , but today they were excellent .',
            'item_name': 'p323_290',
            # 'text': 'This is a LibriVox recording.',
            # 'edited_text': 'This is a god damn online course recording.',
            'mel': wav2spec_res['mel'],
            'wav': wav2spec_res['wav'],
        }
        infer_ins = cls(hp)
        wav_out, wav_gt, mel_out, mel_gt, masked_mel_out, masked_mel_gt = infer_ins.infer_once(inp)

        os.makedirs('infer_out', exist_ok=True)
        save_wav(wav_out, f'inference/out/wav_out.wav', hp['audio_sample_rate'])
        save_wav(wav_gt, f'inference/out/wav_gt.wav', hp['audio_sample_rate'])

        plot_mel(mel_out, 'inference/out/mel_out.png')
        plot_mel(mel_gt, 'inference/out/mel_gt.png')
        plot_mel(masked_mel_out, 'inference/out/masked_mel_out.png')
        plot_mel(masked_mel_gt, 'inference/out/masked_mel_gt.png')


if __name__ == '__main__':
    StutterSpeechInfer.example_run()
