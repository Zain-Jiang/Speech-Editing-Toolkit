import os

import numpy as np
import torch
import torch.nn.functional as F
from modules.speech_editing.a3t.a3t import A3T
from tasks.speech_editing.dataset_utils import StutterSpeechDataset
from tasks.speech_editing.speech_editing_base import SpeechEditingBaseTask
from torch import nn
from utils.audio.align import mel2token_to_dur
from utils.commons.hparams import hparams
from utils.metrics.diagonal_metrics import (get_diagonal_focus_rate,
                                            get_focus_rate,
                                            get_phone_coverage_rate)
from utils.nn.model_utils import num_params
from utils.plot.plot import spec_to_figure
from utils.text.text_encoder import build_token_encoder


class A3TTask(SpeechEditingBaseTask):
    def __init__(self):
        super().__init__()
        data_dir = hparams['binary_data_dir']
        self.word_encoder = build_token_encoder(f'{data_dir}/word_set.json')
        self.mse_loss_fn = torch.nn.MSELoss()
        self.dataset_cls = StutterSpeechDataset
        
    def build_tts_model(self):
        ph_dict_size = len(self.token_encoder)
        word_dict_size = len(self.word_encoder)
        self.model = A3T(ph_dict_size, word_dict_size, hparams)
        self.gen_params = list(self.model.parameters())
    
    def on_train_start(self):
        super().on_train_start()
        for n, m in self.model.named_children():
            num_params(m, model_name=n)

    def run_model(self, sample, infer=False, *args, **kwargs):
        txt_tokens = sample['txt_tokens']
        spk_embed = sample.get('spk_embed')
        spk_id = sample.get('spk_ids')
        stutter_mel_masks = None
        time_mel_masks = sample['time_mel_masks'][:, :, None]
        if not infer:
            output = self.model(txt_tokens,
                                mel2ph=sample['mel2ph'],
                                pitch=sample.get('pitch'),
                                spk_embed=spk_embed,
                                spk_id=spk_id,
                                mels=sample['mels'],
                                stutter_mel_masks=stutter_mel_masks,
                                time_mel_masks=time_mel_masks,
                                infer=False,
                                global_step=self.global_step,
                                )
            losses = {}
            self.add_mel_loss(output['mel_out_decoder'] * time_mel_masks, sample['mels'] * time_mel_masks, losses, postfix="_coarse")
            self.add_mel_loss(output['mel_out_postnet'] * time_mel_masks, sample['mels'] * time_mel_masks, losses, postfix="_fine")
            output['mel_out'] = output['mel_out_postnet'] * time_mel_masks + sample['mels'] * (1-time_mel_masks)
            # super(A3TTask, self).add_dur_loss(output['dur'], sample['mel2ph'], sample['txt_tokens'], losses)
            return losses, output
        else:
            output = self.model(
                txt_tokens,
                mel2ph=sample['mel2ph'],
                pitch=sample.get('pitch'),
                infer=True,
                spk_embed=spk_embed,
                spk_id=spk_id,
                mels=sample['mels'],
                stutter_mel_masks=stutter_mel_masks,
                time_mel_masks=time_mel_masks,
            )
            output['mel_out'] = output['mel_out_postnet'] * time_mel_masks + sample['mels'] * (1-time_mel_masks)
            return output

    def validation_step(self, sample, batch_idx):
        return super().validation_step(sample, batch_idx)
    
    def save_valid_result(self, sample, batch_idx, model_out):
        sr = hparams['audio_sample_rate']
        f0_gt = None
        mel_out = model_out['mel_out']
        self.plot_mel(batch_idx, sample['mels'], mel_out, f0s=f0_gt)
        if self.global_step > 0:
            wav_pred = self.vocoder.spec2wav(mel_out[0].cpu(), f0=f0_gt)
            self.logger.add_audio(f'wav_val_{batch_idx}', wav_pred, self.global_step, sr)
        # gt wav
        if self.global_step <= hparams['valid_infer_interval']:
            mel_gt = sample['mels'][0].cpu()
            wav_gt = self.vocoder.spec2wav(mel_gt, f0=f0_gt)
            self.logger.add_audio(f'wav_gt_{batch_idx}', wav_gt, self.global_step, sr)
        if self.global_step > 0 and hparams['dur_level'] == 'word':
            if model_out.get('attn') is not None:
                self.logger.add_figure(f'attn_{batch_idx}', spec_to_figure(model_out['attn'][0]), self.global_step)

    def get_attn_stats(self, attn, sample, logging_outputs, prefix=''):
        # diagonal_focus_rate
        txt_lengths = sample['txt_lengths'].float()
        mel_lengths = sample['mel_lengths'].float()
        src_padding_mask = sample['txt_tokens'].eq(0)
        target_padding_mask = sample['mels'].abs().sum(-1).eq(0)
        src_seg_mask = sample['txt_tokens'].eq(self.seg_idx)
        attn_ks = txt_lengths.float() / mel_lengths.float()

        focus_rate = get_focus_rate(attn, src_padding_mask, target_padding_mask).mean().data
        phone_coverage_rate = get_phone_coverage_rate(
            attn, src_padding_mask, src_seg_mask, target_padding_mask).mean()
        diagonal_focus_rate, diag_mask = get_diagonal_focus_rate(
            attn, attn_ks, mel_lengths, src_padding_mask, target_padding_mask)
        logging_outputs[f'{prefix}fr'] = focus_rate.mean().data
        logging_outputs[f'{prefix}pcr'] = phone_coverage_rate.mean().data
        logging_outputs[f'{prefix}dfr'] = diagonal_focus_rate.mean().data

    def build_optimizer(self, model):
        optimizer_gen = torch.optim.AdamW(
            self.model.parameters(),
            # [ param for name, param in self.model.named_parameters() if (('fvae.decoder' in name) or 'spk' in name)],
            lr=hparams['lr'],
            betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
            weight_decay=hparams['weight_decay'])
        return [optimizer_gen]

    def build_scheduler(self, optimizer):
        return [
            SpeechEditingBaseTask.build_scheduler(self, optimizer[0]), # Generator Scheduler
        ]

    def on_after_optimization(self, epoch, batch_idx, optimizer, optimizer_idx):
        if self.scheduler is not None:
            self.scheduler[0].step(self.global_step // hparams['accumulate_grad_batches'])