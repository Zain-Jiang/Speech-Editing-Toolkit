import torch
import torch.nn as nn

import utils
from utils.commons.hparams import hparams
from utils.audio.pitch.utils import denorm_f0
from modules.speech_editing.stutter_speech.spec_denoiser import GaussianDiffusion
from modules.speech_editing.stutter_speech.diffnet import DiffNet
from modules.speech_editing.stutter_speech.stutter_predictor import MultiFocalLoss
from tasks.speech_editing.speech_editing_base import SpeechEditingBaseTask
from tasks.tts.vocoder_infer.base_vocoder import get_vocoder_cls, BaseVocoder
from tasks.speech_editing.dataset_utils import StutterSpeechDataset


DIFF_DECODERS = {
    'wavenet': lambda hp: DiffNet(hp['audio_num_mel_bins']),
}


class StutterSpeechTask(SpeechEditingBaseTask):
    def __init__(self):
        super(StutterSpeechTask, self).__init__()
        self.dataset_cls = StutterSpeechDataset
        self.vocoder: BaseVocoder = get_vocoder_cls(hparams['vocoder'])()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=hparams.get('stutter_pad_idx', 2))
        self.focal_loss = MultiFocalLoss(ignore_index=hparams.get('stutter_pad_idx', 2))

    def build_model(self):
        self.build_tts_model()
        utils.nn.model_utils.num_params(self.model)
        # utils.nn.model_utils.num_params(self.model.fs.encoder)
        # print('------')
        # utils.nn.model_utils.num_params(self.model.fs.spk_embed_proj)
        # utils.nn.model_utils.num_params(self.model.fs.dur_embed)
        # utils.nn.model_utils.num_params(self.model.fs.dur_predictor)
        # utils.nn.model_utils.num_params(self.model.fs.pitch_embed)
        # utils.nn.model_utils.num_params(self.model.fs.pitch_predictor)
        # utils.nn.model_utils.num_params(self.model.mel_encoder)
        # utils.nn.model_utils.num_params(self.model.stutter_embed)
        # utils.nn.model_utils.num_params(self.model.stutter_predictor)
        # print('------')
        # utils.nn.model_utils.num_params(self.model.denoise_fn)
        # import sys
        # sys.exit(0)
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

        # Time mel mask
        time_mel_masks = sample['time_mel_masks'][:,:,None]

        # Stutter mask label
        stutter_mel_masks = sample['stutter_mel_masks']
        # Construct the blocked stutter mask
        # B, T = stutter_mel_masks.shape
        # block_size = hparams['stutter_block_size']
        # stutter_mel_masks = stutter_mel_masks.reshape(B, T//block_size, block_size) # [B, T//block_size, block_size]
        # stutter_mel_masks = stutter_mel_masks.sum(-1) # [B, T//block_size]
        stutter_mel_masks[stutter_mel_masks>0] = 1.0
        stutter_mel_masks[stutter_mel_masks<0] = 2.0

        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        output = self.model(txt_tokens, time_mel_masks, stutter_mel_masks, mel2ph=mel2ph, spk_embed=spk_embed,
                       ref_mels=target, f0=f0, uv=uv, energy=energy, infer=infer)

        losses = {}
        self.add_mel_loss(output['mel_out']*time_mel_masks, target*time_mel_masks, losses, postfix="_coarse")
        self.add_dur_loss(output['dur'], mel2ph, txt_tokens, losses=losses)
        self.add_stutter_loss(output['stutter_predictor_out'], stutter_mel_masks, losses=losses)

        output['mel_out'] = output['mel_out']*time_mel_masks + target*(1-time_mel_masks)
        if hparams['use_pitch_embed']:
            self.add_pitch_loss(output, sample, losses)
        if not infer:
            return losses, output
        else:
            return output
    
    def add_stutter_loss(self, stutter_predictor_out, stutter_mel_masks, losses):
        losses['ce'] = self.ce_loss(stutter_predictor_out.transpose(1,2), stutter_mel_masks)
        losses['focal'] = self.focal_loss(stutter_predictor_out.transpose(1,2), stutter_mel_masks)
    
    def _training_step(self, sample, batch_idx, opt_idx):
        loss_output, _ = self.run_model(sample)
        loss_weights = {
            'ce': 8e-3 + 5e-3 *  (self.global_step + 1) / 100000,
            'focal': 1 + 2 *  (self.global_step + 1) / 100000,
        }
        total_loss = sum([loss_weights.get(k, 1) * v for k, v in loss_output.items() if isinstance(v, torch.Tensor) and v.requires_grad])
        loss_output['batch_size'] = sample['txt_tokens'].size()[0]
        return total_loss, loss_output

    def validation_step(self, sample, batch_idx):
        outputs = {}
        txt_tokens = sample['txt_tokens']  # [B, T_t]
        target = sample['mels']

        energy = None
        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        mel2ph = sample['mel2ph']
        f0 = sample['f0']
        uv = sample['uv']
        # Time mel mask
        time_mel_masks = sample['time_mel_masks'][:,:,None]
        # Stutter mask label
        stutter_mel_masks = sample['stutter_mel_masks']
        # Construct the blocked stutter mask
        # B, T = stutter_mel_masks.shape
        # block_size = hparams['stutter_block_size']
        # stutter_mel_masks = stutter_mel_masks.reshape(B, T//block_size, block_size) # [B, T//block_size, block_size]
        # stutter_mel_masks = stutter_mel_masks.sum(-1) # [B, T//block_size]
        stutter_mel_masks[stutter_mel_masks>0] = 1.0
        stutter_mel_masks[stutter_mel_masks<0] = 2.0

        outputs['losses'] = {}
        outputs['losses'], output = self.run_model(sample, infer=False)

        # Cal Acc (Ignore the pad token)
        _, pred_idx = output['stutter_predictor_out'].max(dim=-1)
        acc = ( ((pred_idx==stutter_mel_masks) & (pred_idx==0)).float().sum() \
            + ((pred_idx==stutter_mel_masks) & (pred_idx==1)).float().sum() ) \
            / stutter_mel_masks.numel()
        outputs['losses']['acc'] = acc
        # Cal Acc_1
        if (stutter_mel_masks[stutter_mel_masks == 1]).numel() != 0:
            acc_1 = ((pred_idx[stutter_mel_masks == 1] == 1).float()).sum() / (stutter_mel_masks[stutter_mel_masks == 1]).sum()
            outputs['losses']['acc_1'] = acc_1

        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs['nsamples'] = sample['nsamples']
        outputs = utils.commons.tensor_utils.tensors_to_scalars(outputs)
        if batch_idx < hparams['num_valid_plots']:
            model_out = self.model(
                txt_tokens, time_mel_masks, stutter_mel_masks, spk_embed=spk_embed, mel2ph=mel2ph, f0=f0, uv=uv, energy=energy, ref_mels=target, infer=True)
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
