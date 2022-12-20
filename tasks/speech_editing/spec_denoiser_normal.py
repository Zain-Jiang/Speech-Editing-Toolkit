import torch

import utils
from utils.commons.hparams import hparams
from utils.audio.pitch.utils import denorm_f0
from modules.speech_editing.spec_denoiser.spec_denoiser_normal import GaussianDiffusion
from modules.speech_editing.spec_denoiser.diffnet import DiffNet
from tasks.speech_editing.speech_editing_base import SpeechEditingBaseTask
from tasks.tts.vocoder_infer.base_vocoder import get_vocoder_cls, BaseVocoder
from tasks.speech_editing.dataset_utils import StutterSpeechDataset


DIFF_DECODERS = {
    'wavenet': lambda hp: DiffNet(hp['audio_num_mel_bins']),
}


class SpeechDenoiserNormalTask(SpeechEditingBaseTask):
    def __init__(self):
        super(SpeechDenoiserNormalTask, self).__init__()
        self.dataset_cls = StutterSpeechDataset
        self.vocoder: BaseVocoder = get_vocoder_cls(hparams['vocoder'])()

    def build_model(self):
        self.build_tts_model()
        utils.nn.model_utils.num_params(self.model)
        return self.model

    def build_tts_model(self):
        self.model = GaussianDiffusion(
            phone_encoder=self.token_encoder,
            out_dims=hparams['audio_num_mel_bins'], denoise_fn=DIFF_DECODERS[hparams['diff_decoder_type']](hparams),
            timesteps=hparams['timesteps'], time_scale=hparams['timescale'],
            loss_type=hparams['diff_loss_type'],
            spec_min=hparams['spec_min'], spec_max=hparams['spec_max'],
        )


    def run_model(self, sample, infer=False, *args, **kwargs):
        txt_tokens = sample['txt_tokens']  # [B, T_t]
        target = sample['mels']  # [B, T_s, 80]
        mel2ph = sample['mel2ph']
        f0 = sample['f0']
        uv = sample['uv']
        energy = None
        time_mel_masks = sample['time_mel_masks'][:,:,None]
        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        output = self.model(txt_tokens, time_mel_masks, mel2ph=mel2ph, spk_embed=spk_embed,
                       ref_mels=target, f0=f0, uv=uv, energy=energy, infer=infer)

        losses = {}
        self.add_mel_loss(output['mel_out']*time_mel_masks, target*time_mel_masks, losses, postfix="_coarse")
        output['mel_out'] = output['mel_out']*time_mel_masks + target*(1-time_mel_masks)
        self.add_dur_loss(output['dur'], mel2ph, txt_tokens, losses=losses)
        if hparams['use_pitch_embed']:
            self.add_pitch_loss(output, sample, losses)
        # if hparams['use_energy_embed']:
        #     self.add_energy_loss(output['energy_pred'], energy, losses)
        if not infer:
            return losses, output
        else:
            return output

    def validation_step(self, sample, batch_idx):
        outputs = {}
        txt_tokens = sample['txt_tokens']  # [B, T_t]
        target = sample['mels']

        energy = None
        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        mel2ph = sample['mel2ph']
        f0 = sample['f0']
        uv = sample['uv']
        time_mel_masks = sample['time_mel_masks'][:,:,None]

        outputs['losses'] = {}
        outputs['losses'], output = self.run_model(sample, infer=False)
        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs['nsamples'] = sample['nsamples']
        outputs = utils.commons.tensor_utils.tensors_to_scalars(outputs)
        if batch_idx < hparams['num_valid_plots']:
            model_out = self.model(
                txt_tokens, time_mel_masks, spk_embed=spk_embed, mel2ph=mel2ph, f0=f0, uv=uv, energy=energy, ref_mels=target, infer=True)
            model_out['mel_out'] = model_out['mel_out']*time_mel_masks + target*(1-time_mel_masks)
            # gt_f0 = denorm_f0(sample['f0'], sample['uv'], hparams)
            self.plot_wav(batch_idx, sample['mels'], model_out['mel_out'], is_mel=True, gt_f0=None, f0=None)
            self.plot_mel(batch_idx, sample['mels'], model_out['mel_out'])
        return outputs

    ############
    # validation plots
    ############
    def plot_wav(self, batch_idx, gt_wav, wav_out, is_mel=False, gt_f0=None, f0=None, name=None):
        gt_wav = gt_wav[0].cpu().numpy()
        wav_out = wav_out[0].cpu().numpy()
        gt_f0 = None
        f0 = None
        if is_mel:
            gt_wav = self.vocoder.spec2wav(gt_wav, f0=gt_f0)
            wav_out = self.vocoder.spec2wav(wav_out, f0=f0)
        self.logger.add_audio(f'gt_{batch_idx}', gt_wav, sample_rate=hparams['audio_sample_rate'], global_step=self.global_step)
        self.logger.add_audio(f'wav_{batch_idx}', wav_out, sample_rate=hparams['audio_sample_rate'], global_step=self.global_step)
