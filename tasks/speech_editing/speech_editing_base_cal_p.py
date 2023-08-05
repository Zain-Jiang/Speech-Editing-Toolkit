import numpy as np
import torch
import torch.distributions
import torch.nn.functional as F
import torch.optim
import torch.utils.data

from tasks.speech_editing.dataset_utils import StutterSpeechDataset
from tasks.tts.speech_base import SpeechBaseTask
from utils.audio.align import mel2token_to_dur
from utils.audio.pitch.utils import denorm_f0
from utils.commons.hparams import hparams
from utils.eval.mcd import get_metrics_mels


import pandas as pd
from numpy import mean
from tqdm import tqdm

class SpeechEditingBaseTask(SpeechBaseTask):
    def __init__(self):
        super().__init__()
        self.dataset_cls = StutterSpeechDataset
        self.sil_ph = self.token_encoder.sil_phonemes()

        self.mcd_dict = {'mcd_total': 0, 'num': 0}
        
        self.metadata = pd.read_csv('data/processed/vctk/metadata.csv')
        self.metadata.index = self.metadata['item_name']

    def build_tts_model(self):
        dict_size = len(self.token_encoder)
        self.model = SpeechEditingBaseTask(dict_size, hparams)

    def run_model(self, sample, infer=False, *args, **kwargs):
        txt_tokens = sample['txt_tokens']  # [B, T_t]
        spk_embed = sample.get('spk_embed')
        spk_id = sample.get('spk_ids')
        if not infer:
            target = sample['mels']  # [B, T_s, 80]
            mel2ph = sample['mel2ph']  # [B, T_s]
            f0 = sample.get('f0')
            uv = sample.get('uv')
            output = self.model(txt_tokens, mel2ph=mel2ph, spk_embed=spk_embed, spk_id=spk_id,
                                f0=f0, uv=uv, infer=False)
            losses = {}
            self.add_mel_loss(output['mel_out'], target, losses)
            self.add_dur_loss(output['dur'], mel2ph, txt_tokens, losses=losses)
            if hparams['use_pitch_embed']:
                self.add_pitch_loss(output, sample, losses)
            return losses, output
        else:
            use_gt_dur = kwargs.get('infer_use_gt_dur', hparams['use_gt_dur'])
            use_gt_f0 = kwargs.get('infer_use_gt_f0', hparams['use_gt_f0'])
            mel2ph, uv, f0 = None, None, None
            if use_gt_dur:
                mel2ph = sample['mel2ph']
            if use_gt_f0:
                f0 = sample['f0']
                uv = sample['uv']
            output = self.model(txt_tokens, mel2ph=mel2ph, spk_embed=spk_embed, spk_id=spk_id,
                                f0=f0, uv=uv, infer=True)
            return output

    def add_dur_loss(self, dur_pred, mel2ph, txt_tokens, losses=None):
        """

        :param dur_pred: [B, T], float, log scale
        :param mel2ph: [B, T]
        :param txt_tokens: [B, T]
        :param losses:
        :return:
        """
        B, T = txt_tokens.shape
        nonpadding = (txt_tokens != 0).float()
        dur_gt = mel2token_to_dur(mel2ph, T).float() * nonpadding
        is_sil = torch.zeros_like(txt_tokens).bool()
        for p in self.sil_ph:
            is_sil = is_sil | (txt_tokens == self.token_encoder.encode(p)[0])
        is_sil = is_sil.float()  # [B, T_txt]
        losses['pdur'] = F.mse_loss((dur_pred + 1).log(), (dur_gt + 1).log(), reduction='none')
        losses['pdur'] = (losses['pdur'] * nonpadding).sum() / nonpadding.sum()
        losses['pdur'] = losses['pdur'] * hparams['lambda_ph_dur']
        # use linear scale for sentence and word duration
        if hparams['lambda_word_dur'] > 0:
            word_id = (is_sil.cumsum(-1) * (1 - is_sil)).long()
            word_dur_p = dur_pred.new_zeros([B, word_id.max() + 1]).scatter_add(1, word_id, dur_pred)[:, 1:]
            word_dur_g = dur_gt.new_zeros([B, word_id.max() + 1]).scatter_add(1, word_id, dur_gt)[:, 1:]
            wdur_loss = F.mse_loss((word_dur_p + 1).log(), (word_dur_g + 1).log(), reduction='none')
            word_nonpadding = (word_dur_g > 0).float()
            wdur_loss = (wdur_loss * word_nonpadding).sum() / word_nonpadding.sum()
            losses['wdur'] = wdur_loss * hparams['lambda_word_dur']
        if hparams['lambda_sent_dur'] > 0:
            sent_dur_p = dur_pred.sum(-1)
            sent_dur_g = dur_gt.sum(-1)
            sdur_loss = F.mse_loss((sent_dur_p + 1).log(), (sent_dur_g + 1).log(), reduction='mean')
            losses['sdur'] = sdur_loss.mean() * hparams['lambda_sent_dur']

    def add_pitch_loss(self, output, sample, losses):
        mel2ph = sample['mel2ph']  # [B, T_s]
        f0 = sample['f0']
        uv = sample['uv']
        nonpadding = (mel2ph != 0).float() if hparams['pitch_type'] == 'frame' \
            else (sample['txt_tokens'] != 0).float()
        p_pred = output['pitch_pred']
        assert p_pred[..., 0].shape == f0.shape
        if hparams['use_uv'] and hparams['pitch_type'] == 'frame':
            assert p_pred[..., 1].shape == uv.shape, (p_pred.shape, uv.shape)
            losses['uv'] = (F.binary_cross_entropy_with_logits(
                p_pred[:, :, 1], uv, reduction='none') * nonpadding).sum() \
                           / nonpadding.sum() * hparams['lambda_uv']
            nonpadding = nonpadding * (uv == 0).float()
        f0_pred = p_pred[:, :, 0]
        losses['f0'] = (F.l1_loss(f0_pred, f0, reduction='none') * nonpadding).sum() \
                       / nonpadding.sum() * hparams['lambda_f0']

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
            # with gt duration
            model_out = self.run_model(sample, infer=True, infer_use_gt_dur=True)
            dur_info = self.get_plot_dur_info(sample, model_out)
            del dur_info['dur_pred']
            wav_pred = self.vocoder.spec2wav(model_out['mel_out'][0].cpu(), f0=f0_gt)
            self.logger.add_audio(f'wav_gdur_{batch_idx}', wav_pred, self.global_step, sr)
            self.plot_mel(batch_idx, sample['mels'], model_out['mel_out'][0], f'mel_gdur_{batch_idx}',
                          dur_info=dur_info, f0s=f0_gt)

            # with pred duration
            if not hparams['use_gt_dur']:
                model_out = self.run_model(sample, infer=True, infer_use_gt_dur=False)
                dur_info = self.get_plot_dur_info(sample, model_out)
                self.plot_mel(batch_idx, sample['mels'], model_out['mel_out'][0], f'mel_pdur_{batch_idx}',
                              dur_info=dur_info, f0s=f0_gt)
                wav_pred = self.vocoder.spec2wav(model_out['mel_out'][0].cpu(), f0=f0_gt)
                self.logger.add_audio(f'wav_pdur_{batch_idx}', wav_pred, self.global_step, sr)
        # gt wav
        if self.global_step <= hparams['valid_infer_interval']:
            mel_gt = sample['mels'][0].cpu()
            wav_gt = self.vocoder.spec2wav(mel_gt, f0=f0_gt)
            self.logger.add_audio(f'wav_gt_{batch_idx}', wav_gt, self.global_step, sr)

    def get_plot_dur_info(self, sample, model_out):
        T_txt = sample['txt_tokens'].shape[1]
        dur_gt = mel2token_to_dur(sample['mel2ph'], T_txt)[0]
        dur_pred = model_out['dur'] if 'dur' in model_out else dur_gt
        txt = self.token_encoder.decode(sample['txt_tokens'][0].cpu().numpy())
        txt = txt.split(" ")
        return {'dur_gt': dur_gt, 'dur_pred': dur_pred, 'txt': txt}

    def test_step(self, sample, batch_idx):
        assert sample['txt_tokens'].shape[0] == 1, 'only support batch_size=1 in inference'
        outputs = self.run_model(sample, infer=True)
        text = sample['text'][0]
        item_name = sample['item_name'][0]
        tokens = sample['txt_tokens'][0].cpu().numpy()
        mel_gt = sample['mels'][0].cpu().numpy()
        mel_pred = outputs['mel_out'][0].cpu().numpy()
        mel2ph = sample['mel2ph'][0].cpu().numpy()
        time_mel_masks = sample['time_mel_masks'][0].cpu().numpy()
        mel2ph_pred = None
        str_phs = self.token_encoder.decode(tokens, strip_padding=True)
        base_fn = f'[{batch_idx:06d}][{item_name.replace("%", "_")}][%s]'
        if text is not None:
            base_fn += text.replace(":", "$3A")[:80]
        base_fn = base_fn.replace(' ', '_')
        gen_dir = self.gen_dir
        wav_pred = self.vocoder.spec2wav(mel_pred)
        '''
        self.saving_result_pool.add_job(self.save_result, args=[
            wav_pred, mel_pred, base_fn % 'P', gen_dir, str_phs, mel2ph_pred])
        mel_pred_seg = mel_pred[time_mel_masks==1]
        wav_pred_seg =self.vocoder.spec2wav(mel_pred_seg)
        self.saving_result_pool.add_job(self.save_result, args=[
            wav_pred_seg, mel_pred_seg, base_fn % 'P_SEG', gen_dir, None, None])
        if hparams['save_gt']:
            wav_gt = self.vocoder.spec2wav(mel_gt)
            self.saving_result_pool.add_job(self.save_result, args=[
                wav_gt, mel_gt, base_fn % 'G', gen_dir, str_phs, mel2ph])
            mel_gt_seg = mel_gt[time_mel_masks==1]
            wav_gt_seg =self.vocoder.spec2wav(mel_gt_seg)
            self.saving_result_pool.add_job(self.save_result, args=[
                wav_gt_seg, mel_gt_seg, base_fn % 'G_SEG', gen_dir, None, None])
        '''

        
        ###################dur compute#######################
        #print(item_name)
        ph2word = self.metadata.loc[item_name, 'ph2word']
        word = self.metadata.loc[item_name, 'word']
        word_list = word.split()
        #print(word_list)
        ph2word = ph2word[1:-1]
        ph2word = ph2word.split(',')
        ph2word = [int(i) for i in ph2word]
        #print("ph2word")
        #print(ph2word)
    
        dur_info = self.get_plot_dur_info(sample, outputs)
        #print("gt dur")
        dur_gt_ph = dur_info['dur_gt'].cpu().numpy()
        #print(dur_gt_ph)
        #print("predict dur")
        dur_pred_ph = dur_info['dur_pred'][0].cpu().numpy()
        #print(dur_pred_ph)
        #print(sample)
        
        def ph2word_dur(phdur, ph2word):
            word_dur = [0] * ph2word[-1]
            for dur, index in zip(phdur, ph2word):
                word_dur[index-1] += dur
            return word_dur
        dur_gt_word = ph2word_dur(dur_gt_ph, ph2word)
        dur_pred_word = ph2word_dur(dur_pred_ph, ph2word)
        #print(dur_gt_word)
        #print(dur_pred_word)
        dur_pred_word_process = []
        dur_gt_word_process = []
        for word, gt,pred in zip(word_list, dur_gt_word, dur_pred_word):
            if(word == '<BOS>' or word[0].isalpha()):
                dur_pred_word_process.append(pred)
                dur_gt_word_process.append(gt)
            elif(word == ',' or word == '|' or word == '<EOS>'):
                gt += dur_gt_word_process[-1]
                pred += dur_pred_word_process[-1]
                del dur_pred_word_process[-1]
                del dur_gt_word_process[-1]
                dur_pred_word_process.append(pred)
                dur_gt_word_process.append(gt)
            else:
                print("error")
                print(word)
                exit(1)
        #print(dur_gt_word_process)
        #print(dur_pred_word_process)


        error = []
        for i in range(len(dur_gt_word_process)):
            error.append((dur_pred_word_process[i] - dur_gt_word_process[i]) ** 2)
        dur_MSE = mean(error)
        
        '''
        ############## pitch  calculate####################
        print(sample)
        print(outputs)
        exit(1)
        uv = sample['uv']
        gt_pitch = sample['f0']
        predict_pitch = outputs['f0_denorm_pred']
        ###########frame level################
        frame_uv = uv[0].cpu().numpy()
        gt_pitch_frame = denorm_f0(gt_pitch, None)[0].cpu().numpy()
        predict_pitch_frame = predict_pitch[0].cpu().numpy()
        error_frame = []
        for i in range(len(gt_pitch_frame)):
            if(frame_uv[i]==1.):
                continue
            else:
                error_frame.append((predict_pitch_frame[i] - gt_pitch_frame[i]) ** 2)
        frame_pitch_MSE = mean(error_frame)
        ##############ph level################
        ph_token = sample['txt_tokens'][0]
        mel2ph = sample['mel2ph'][0]
        gt_pitch_ph = denorm_f0(gt_pitch, None)[0]
        predict_pitch_ph = predict_pitch[0]
        uv_ph = uv[0]
        f0_phlevel_sum = torch.zeros_like(ph_token).float().scatter_add(0, mel2ph - 1, gt_pitch_ph)
        f0_phlevel_num = torch.zeros_like(ph_token).float().scatter_add(
                    0, mel2ph - 1, torch.ones_like(gt_pitch_ph)).clamp_min(1)
        gt_pitch_ph = f0_phlevel_sum / f0_phlevel_num
        pre_f0_phlevel_sum = torch.zeros_like(ph_token).float().scatter_add(0, mel2ph - 1, predict_pitch_ph)
        pre_f0_phlevel_num = torch.zeros_like(ph_token).float().scatter_add(
                    0, mel2ph - 1, torch.ones_like(predict_pitch_ph)).clamp_min(1)
        predict_pitch_ph = pre_f0_phlevel_sum / pre_f0_phlevel_num
        uv_phlevel_sum = torch.zeros_like(ph_token).float().scatter_add(0, mel2ph - 1, uv_ph)
        uv_phlevel_num = torch.zeros_like(ph_token).float().scatter_add(
                    0, mel2ph - 1, torch.ones_like(uv_ph)).clamp_min(1)
        uv_ph = uv_phlevel_sum / uv_phlevel_num
        print(gt_pitch_ph)
        print(predict_pitch_ph)
        print(uv_ph)
        exit(1)
        '''

        # print(f"Pred_shape: {mel_pred.shape}, gt_shape: {mel_gt.shape}")
        return {
            'item_name': item_name,
            'text': text,
            'ph_tokens': self.token_encoder.decode(tokens.tolist()),
            'wav_fn_pred': base_fn % 'P',
            'wav_fn_gt': base_fn % 'G',
            'wav_fn_orig': sample['wav_fn'][0],
            'dur_loss': dur_MSE,
        }
    
   