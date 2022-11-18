import os

import numpy as np
import torch
import torch.nn.functional as F
from modules.tts.ps_adv.multi_window_disc import Discriminator
from modules.tts.ps_adv.stutter_speech_adv import StutterSpeech_adv
from tasks.tts.dataset_utils import StutterSpeechDataset
from tasks.tts.fs import FastSpeechTask
from torch import nn
from utils.audio.align import mel2token_to_dur
from utils.commons.hparams import hparams
from utils.metrics.diagonal_metrics import (get_diagonal_focus_rate,
                                            get_focus_rate,
                                            get_phone_coverage_rate)
from utils.nn.model_utils import num_params
from utils.plot.plot import spec_to_figure
from utils.text.text_encoder import build_token_encoder


class StutterSpeechAdvTask(FastSpeechTask):
    def __init__(self):
        super().__init__()
        data_dir = hparams['binary_data_dir']
        self.word_encoder = build_token_encoder(f'{data_dir}/word_set.json')
        self.build_disc_model()
        self.mse_loss_fn = torch.nn.MSELoss()
        self.dataset_cls = StutterSpeechDataset
        
    def build_tts_model(self):
        ph_dict_size = len(self.token_encoder)
        word_dict_size = len(self.word_encoder)
        self.model = StutterSpeech_adv(ph_dict_size, word_dict_size, hparams)
        self.gen_params = list(self.model.parameters())
    
    def build_disc_model(self):
        disc_win_num = hparams['disc_win_num']
        h = hparams['mel_disc_hidden_size']
        self.mel_disc = Discriminator(
            time_lengths=[32, 64, 128][:disc_win_num],
            freq_length=80, hidden_size=h, kernel=(3, 3)
        )
        self.disc_params = list(self.mel_disc.parameters())

    def on_train_start(self):
        super().on_train_start()
        for n, m in self.model.named_children():
            num_params(m, model_name=n)
        if hasattr(self.model, 'fvae'):
            for n, m in self.model.fvae.named_children():
                num_params(m, model_name=f'fvae.{n}')

    def _training_step(self, sample, batch_idx, optimizer_idx):
        loss_output = {}
        loss_weights = {}
        disc_start = self.global_step >= hparams["disc_start_steps"] and hparams['lambda_mel_adv'] > 0
        if optimizer_idx == 0:
            #######################
            #      Generator      #
            #######################
            loss_output, model_out = self.run_model(sample, infer=False)
            self.model_out_gt = self.model_out = \
                {k: v.detach() for k, v in model_out.items() if isinstance(v, torch.Tensor)}
            if disc_start:
                mel_p = model_out['mel_out']
                if hasattr(self.model, 'out2mel'):
                    mel_p = self.model.out2mel(mel_p)
                o_ = self.mel_disc(mel_p)
                p_, pc_ = o_['y'], o_['y_c']
                if p_ is not None:
                    loss_output['a'] = self.mse_loss_fn(p_, p_.new_ones(p_.size()))
                    loss_weights['a'] = hparams['lambda_mel_adv']
                if pc_ is not None:
                    loss_output['ac'] = self.mse_loss_fn(pc_, pc_.new_ones(pc_.size()))
                    loss_weights['ac'] = hparams['lambda_mel_adv']
        else:
            #######################
            #    Discriminator    #
            #######################
            if disc_start and self.global_step % hparams['disc_interval'] == 0:
                model_out = self.model_out_gt
                mel_g = sample['mels']
                mel_p = model_out['mel_out']
                o = self.mel_disc(mel_g)
                p, pc = o['y'], o['y_c']
                o_ = self.mel_disc(mel_p)
                p_, pc_ = o_['y'], o_['y_c']
                if p_ is not None:
                    loss_output["r"] = self.mse_loss_fn(p, p.new_ones(p.size()))
                    loss_output["f"] = self.mse_loss_fn(p_, p_.new_zeros(p_.size()))
                if pc_ is not None:
                    loss_output["rc"] = self.mse_loss_fn(pc, pc.new_ones(pc.size()))
                    loss_output["fc"] = self.mse_loss_fn(pc_, pc_.new_zeros(pc_.size()))
            if len(loss_output) == 0:
                return None
        total_loss = sum([loss_weights.get(k, 1) * v for k, v in loss_output.items() if isinstance(v, torch.Tensor) and v.requires_grad])
        loss_output['batch_size'] = sample['txt_tokens'].size()[0]
        return total_loss, loss_output

    def run_model(self, sample, infer=False, *args, **kwargs):
        txt_tokens = sample['txt_tokens']
        word_tokens = sample['word_tokens']
        spk_embed = sample.get('spk_embed')
        spk_id = sample.get('spk_ids')
        stutter_mel_masks = sample['stutter_mel_masks']
        time_mel_masks = sample['time_mel_masks'][:, :, None]
        if not infer:
            output = self.model(txt_tokens, word_tokens,
                                ph2word=sample['ph2word'],
                                word_len=sample['word_lengths'].max(),
                                tgt_mels=sample['mels'],
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
            self.add_mel_loss(output['mel_out_coarse'] * time_mel_masks, sample['mels'] * time_mel_masks, losses, postfix="_coarse")
            self.add_mel_loss(output['mel_out_fine'] * time_mel_masks, sample['mels'] * time_mel_masks, losses, postfix="_fine")
            output['mel_out'] = output['mel_out_fine'] * time_mel_masks + sample['mels'] * (1-time_mel_masks)
            if hparams['use_pitch_embed']:
                self.add_pitch_loss(output, sample, losses)
            # if hparams['dur_level'] == 'word':
            #     self.add_dur_loss(
            #         output['dur'], sample['mel2word'], sample['word_lengths'], sample['txt_tokens'], losses)
            #     self.get_attn_stats(output['attn'], sample, losses)
            # else:
            #     super(StutterSpeechAdvTask, self).add_dur_loss(output['dur'], sample['mel2ph'], sample['txt_tokens'], losses)
            return losses, output
        else:
            output = self.model(
                txt_tokens, word_tokens,
                ph2word=sample['ph2word'],
                word_len=sample['word_lengths'].max(),
                pitch=sample.get('pitch'),
                tgt_mels=sample['mels'],
                infer=True,
                spk_embed=spk_embed,
                spk_id=spk_id,
                mels=sample['mels'],
                stutter_mel_masks=stutter_mel_masks,
                time_mel_masks=time_mel_masks,
            )
            output['mel_out'] = output['mel_out'] * output_mel_masks + sample['mels'] * (1-output_mel_masks)
            return output

    # def add_dur_loss(self, dur_pred, mel2token, word_len, txt_tokens, losses=None):
    #     T = word_len.max()
    #     dur_gt = mel2token_to_dur(mel2token, T).float()
    #     nonpadding = (torch.arange(T).to(dur_pred.device)[None, :] < word_len[:, None]).float()
    #     dur_pred = dur_pred * nonpadding
    #     dur_gt = dur_gt * nonpadding
    #     wdur = F.l1_loss((dur_pred + 1).log(), (dur_gt + 1).log(), reduction='none')
    #     wdur = (wdur * nonpadding).sum() / nonpadding.sum()
    #     if hparams['lambda_word_dur'] > 0:
    #         losses['wdur'] = wdur * hparams['lambda_word_dur']
    #     if hparams['lambda_sent_dur'] > 0:
    #         sent_dur_p = dur_pred.sum(-1)
    #         sent_dur_g = dur_gt.sum(-1)
    #         sdur_loss = F.l1_loss(sent_dur_p, sent_dur_g, reduction='mean')
    #         losses['sdur'] = sdur_loss.mean() * hparams['lambda_sent_dur']

    def validation_step(self, sample, batch_idx):
        return super().validation_step(sample, batch_idx)
    
    def save_valid_result(self, sample, batch_idx, model_out):
        sr = hparams['audio_sample_rate']
        f0_gt = None
        mel_out = model_out['mel_out']
        if sample.get('f0') is not None:
            f0_gt = denorm_f0(sample['f0'][0].cpu(), sample['uv'][0].cpu())
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

        optimizer_disc = torch.optim.AdamW(
            self.disc_params,
            lr=hparams['disc_lr'],
            betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
            **hparams["discriminator_optimizer_params"]) if len(self.disc_params) > 0 else None

        return [optimizer_gen, optimizer_disc]

    def build_scheduler(self, optimizer):
        return [
            FastSpeechTask.build_scheduler(self, optimizer[0]), # Generator Scheduler
            torch.optim.lr_scheduler.StepLR(optimizer=optimizer[1], # Discriminator Scheduler
                **hparams["discriminator_scheduler_params"]),
        ]

    def on_after_optimization(self, epoch, batch_idx, optimizer, optimizer_idx):
        if self.scheduler is not None:
            self.scheduler[0].step(self.global_step // hparams['accumulate_grad_batches'])
            self.scheduler[1].step(self.global_step // hparams['accumulate_grad_batches'])


    ############
    # infer
    ############
    def test_start(self):
        super().test_start()
        if hparams.get('save_attn', False):
            os.makedirs(f'{self.gen_dir}/attn', exist_ok=True)
        self.model.store_inverse_all()

    def test_step(self, sample, batch_idx):
        assert sample['txt_tokens'].shape[0] == 1, 'only support batch_size=1 in inference'
        outputs = self.run_model(sample, infer=True)
        text = sample['text'][0]
        item_name = sample['item_name'][0]
        tokens = sample['txt_tokens'][0].cpu().numpy()
        mel_gt = sample['mels'][0].cpu().numpy()
        mel_pred = outputs['mel_out'][0].cpu().numpy()
        mel2ph = sample['mel2ph'][0].cpu().numpy()
        mel2ph_pred = None
        str_phs = self.token_encoder.decode(tokens, strip_padding=True)
        base_fn = f'[{batch_idx:06d}][{item_name.replace("%", "_")}][%s]'
        if text is not None:
            base_fn += text.replace(":", "$3A")[:80]
        base_fn = base_fn.replace(' ', '_')
        gen_dir = self.gen_dir
        wav_pred = self.vocoder.spec2wav(mel_pred)
        self.saving_result_pool.add_job(self.save_result, args=[
            wav_pred, mel_pred, base_fn % 'P', gen_dir, str_phs, mel2ph_pred])
        if hparams['save_gt']:
            wav_gt = self.vocoder.spec2wav(mel_gt)
            self.saving_result_pool.add_job(self.save_result, args=[
                wav_gt, mel_gt, base_fn % 'G', gen_dir, str_phs, mel2ph])
        if hparams.get('save_attn', False):
            attn = outputs['attn'][0].cpu().numpy()
            np.save(f'{gen_dir}/attn/{item_name}.npy', attn)
        print(f"Pred_shape: {mel_pred.shape}, gt_shape: {mel_gt.shape}")
        return {
            'item_name': item_name,
            'text': text,
            'ph_tokens': self.token_encoder.decode(tokens.tolist()),
            'wav_fn_pred': base_fn % 'P',
            'wav_fn_gt': base_fn % 'G',
        }
