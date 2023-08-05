import os
import filecmp
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d

import utils
from utils.commons.hparams import hparams
from utils.commons.dataset_utils import data_loader, BaseConcatDataset
from utils.commons.ckpt_utils import get_last_checkpoint, load_ckpt
from utils.audio.pitch.utils import denorm_f0
from utils.nn.model_utils import num_params
from modules.speech_editing.spec_denoiser.spec_denoiser import GaussianDiffusion
from modules.speech_editing.spec_denoiser.diffnet import DiffNet
from modules.speech_editing.spec_denoiser.stutter_predictor import StutterPredictor
from tasks.speech_editing.speech_editing_base import SpeechEditingBaseTask
from tasks.tts.vocoder_infer.base_vocoder import get_vocoder_cls, BaseVocoder
from tasks.speech_editing.dataset_utils import StutterSpeechDataset
from tasks.speech_editing.spec_denoiser import SpeechDenoiserTask


DIFF_DECODERS = {
    'wavenet': lambda hp: DiffNet(hp['audio_num_mel_bins']),
}

class MultiFocalLoss(torch.nn.Module):
    def __init__(self, num_class=3, gamma=5, reduction='mean', ignore_index=-1):
        super(MultiFocalLoss, self).__init__()
        self.num_class = num_class
        self.gamma = gamma
        self.reduction = reduction
        self.smooth = 1e-6
        self.ignore_index = ignore_index

        self.alpha = torch.Tensor([1e-3, 1, 0.0]) # [fluent, stutter, pad]

    def forward(self, logit, target):
        alpha = self.alpha.to(logit.device)
        logit = logit.transpose(1,2).contiguous().view(-1, self.num_class)
        logit_softmax = F.softmax(logit, dim=1) # [B*T, num_class]
        logit_log_softmax = torch.log(logit_softmax) # [B*T, num_class]

        target = target.view(-1, 1) # [B*T, 1]

        logit_softmax = logit_softmax.gather(1, target).view(-1) + self.smooth  # Avoid nan
        logit_log_softmax = logit_log_softmax.gather(1, target).view(-1) + self.smooth  # Avoid nan

        alpha_weight = alpha[target.squeeze().long()].view(-1)
        loss = -alpha_weight * torch.pow(torch.sub(1.0, logit_softmax), self.gamma) * logit_log_softmax
        # loss[target==self.ignore_index] = 0.0 # Ingore pad seq

        if self.reduction == 'mean':
            loss = loss.mean()

        return loss

class StutterPredictorTask(SpeechEditingBaseTask):
    def __init__(self):
        super(StutterPredictorTask, self).__init__()
        self.dataset_cls = StutterSpeechDataset
        self.vocoder: BaseVocoder = get_vocoder_cls(hparams['vocoder'])()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=hparams.get('stutter_pad_idx', -1))
        self.focal_loss = MultiFocalLoss(ignore_index=hparams.get('stutter_pad_idx', -1))

    def build_model(self):
        self.build_stutter_predictor_model()
        utils.nn.model_utils.num_params(self.model)
        return self.model

    def build_transition_net_task(self):
        self.spec_denoiser_task = SpeechDenoiserTask()
        self.spec_denoiser_task.build_model()
        spec_denoiser_work_dir = hparams.get("spec_denoiser_work_dir", "checkpoints/spec_denoiser_stutterset_dur_pitch_masked")
        load_ckpt(self.spec_denoiser_task.model, spec_denoiser_work_dir, 'model')
        for p in self.spec_denoiser_task.parameters():
            p.requires_grad = False
        self.spec_denoiser_task.eval()

    def build_stutter_predictor_model(self):
        self.model = StutterPredictor(len(self.token_encoder), hparams)
        # Load txt_encoder from the pretrained speech editing model
        checkpoint, _ = get_last_checkpoint('checkpoints/spec_denoiser_stutterset_dur_pitch_masked')
        model_dict = self.model.state_dict()
        for key in checkpoint['state_dict']['model']:
            if 'fs.encoder' in key:
                key_ = key.replace('fs.encoder', 'txt_encoder')
                model_dict[key_] = checkpoint['state_dict']['model'][key]
        self.model.load_state_dict(model_dict)

    def on_train_start(self):
        super().on_train_start()
        for n, m in self.model.named_children():
            num_params(m, model_name=n)

    def run_model(self, sample, infer=False, *args, **kwargs):
        txt_tokens = sample['txt_tokens']  # [B, T_t]
        mels = sample['mels']  # [B, T_s, 80]
        mel2ph = sample['mel2ph']
        time_mel_masks = sample['time_mel_masks'][:,:,None]

        # Construct the blocked stutter mask
        block_size = hparams['stutter_block_size']
        stutter_mel_masks = sample['stutter_mel_masks']
        B, T = stutter_mel_masks.shape
        stutter_mel_masks = stutter_mel_masks.reshape(B, T//block_size, block_size) # [B, T//block_size, block_size]
        stutter_mel_masks = stutter_mel_masks.sum(-1) # [B, T//block_size]
        stutter_mel_masks[stutter_mel_masks>0] = 1.0
        stutter_mel_masks[stutter_mel_masks<0] = 2.0

        output = self.model(txt_tokens, mels, mel2ph, infer=infer)
        losses = {}
        losses['ce'] = self.ce_loss(output['logits'].transpose(1,2), stutter_mel_masks)
        losses['focal'] = self.focal_loss(output['logits'].transpose(1,2), stutter_mel_masks)
        output['stutter_label'] = stutter_mel_masks
        # self.add_mel_loss(output['mel_out']*time_mel_masks, target*time_mel_masks, losses, postfix="_coarse")
        # output['mel_out'] = output['mel_out']*time_mel_masks + target*(1-time_mel_masks)
        # self.add_dur_loss(output['dur'], mel2ph, txt_tokens, losses=losses)
        # if hparams['use_pitch_embed']:
        #     self.add_pitch_loss(output, sample, losses)
        # if hparams['use_energy_embed']:
        #     self.add_energy_loss(output['energy_pred'], energy, losses)
        if not infer:
            return losses, output
        else:
            return output
    
    def _training_step(self, sample, batch_idx, opt_idx):
        loss_output, _ = self.run_model(sample)
        loss_weights = {
            'ce': min(1e-2, 1e-2 * 6000 / (self.global_step+1e-9)),
            'focal': 1
        }
        total_loss = sum([loss_weights.get(k, 1) * v for k, v in loss_output.items() if isinstance(v, torch.Tensor) and v.requires_grad])
        loss_output['batch_size'] = sample['txt_tokens'].size()[0]
        return total_loss, loss_output

    def validation_step(self, sample, batch_idx):
        outputs = {}
        txt_tokens = sample['txt_tokens']  # [B, T_t]
        mels = sample['mels']

        mel2ph = sample['mel2ph']

        outputs['losses'] = {}
        outputs['losses'], output = self.run_model(sample, infer=False)
        
        _, pred_idx = output['logits'].max(dim=-1)
        # Cal Acc (Ignore the pad token)
        acc = ( ((pred_idx==output['stutter_label']) & (pred_idx==0)).float().sum() \
            + ((pred_idx==output['stutter_label']) & (pred_idx==1)).float().sum() ) \
            / output['stutter_label'].numel()
        outputs['losses']['acc'] = acc
        # Cal Acc_1
        if (output['stutter_label'][output['stutter_label'] == 1]).numel() != 0:
            acc_1 = ((pred_idx[output['stutter_label'] == 1] == 1).float()).sum() / (output['stutter_label'][output['stutter_label'] == 1]).sum()
            outputs['losses']['acc_1'] = acc_1

        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs['nsamples'] = sample['nsamples']
        outputs = utils.commons.tensor_utils.tensors_to_scalars(outputs)
        if batch_idx < hparams['num_valid_plots']:
            model_out = self.model(
                txt_tokens, mels, mel2ph, infer=True)
            # gt_f0 = denorm_f0(sample['f0'], sample['uv'], hparams)
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


    ##########################
    # datasets
    ##########################
    @data_loader
    def train_dataloader(self):
        if hparams['train_sets'] != '':
            train_sets = hparams['train_sets'].split("|")
            # check if all train_sets have the same spk map and dictionary
            binary_data_dir = hparams['binary_data_dir']
            file_to_cmp = ['phone_set.json']
            if os.path.exists(f'{binary_data_dir}/word_set.json'):
                file_to_cmp.append('word_set.json')
            if hparams['use_spk_id']:
                file_to_cmp.append('spk_map.json')
            for f in file_to_cmp:
                for ds_name in train_sets:
                    base_file = os.path.join(binary_data_dir, f)
                    ds_file = os.path.join(ds_name, f)
                    assert filecmp.cmp(base_file, ds_file), \
                        f'{f} in {ds_name} is not same with that in {binary_data_dir}.'
            train_dataset = BaseConcatDataset([
                self.dataset_cls(prefix='train', shuffle=True, data_dir=ds_name) for ds_name in train_sets])
        else:
            train_dataset = self.dataset_cls(prefix=hparams['train_set_name'], shuffle=True)
        
        # sampler = self.creater_sampler(train_dataset)
        return self.build_dataloader(train_dataset, True, self.max_tokens, self.max_sentences,
                                     endless=hparams['endless_ds'], sampler=None)
    
    def creater_sampler(self,train_set):
        from torch.utils.data.sampler import WeightedRandomSampler
        print("| Building Weighted Sampler: ")
        sample_weights = []
        for item in tqdm(train_set):
            stutter_mel_mask = item['stutter_mel_mask']
            stutter_mel_mask[stutter_mel_mask>0] = 1.0
            zero_frames = stutter_mel_mask[stutter_mel_mask == 0].numel()
            nonzero_frames = stutter_mel_mask[stutter_mel_mask >= 0].numel()
            sample_weights.append((10 + nonzero_frames) / (zero_frames + nonzero_frames))
        num_sample=int(len(train_set))
        sampler = WeightedRandomSampler(sample_weights, num_sample, replacement=True)
        return sampler
    
    def test_step(self, sample, batch_idx):
        assert sample['txt_tokens'].shape[0] == 1, 'only support batch_size=1 in inference'
        outputs = self.run_model(sample, infer=True)
        text = sample['text'][0]
        item_name = sample['item_name'][0]
        tokens = sample['txt_tokens'][0].cpu().numpy()
        
        mel_gt = sample['mels']
        stutter_label = outputs['stutter_label']
        logits = outputs['logits']
        _, pred_idx = logits.max(dim=-1)
        pred_idx = F.interpolate(pred_idx[None,:,:].float(), scale_factor=(16), mode='nearest')[0][:,:,None]
        stutter_label = F.interpolate(stutter_label[None,:,:].float(), scale_factor=(16), mode='nearest')[0][:,:,None]
        mel_pred = mel_gt * (1-pred_idx)
        mel_s_gt = mel_gt * (1-stutter_label)
        mel_gt = mel_gt[0].cpu().numpy()
        mel_pred = mel_pred[0].cpu().numpy()
        mel_s_gt = mel_s_gt[0].cpu().numpy()

        mel2ph = sample['mel2ph'][0].cpu().numpy()
        str_phs = self.token_encoder.decode(tokens, strip_padding=True)
        base_fn = f'[{batch_idx:06d}][{item_name.replace("%", "_")}][%s]'
        if text is not None:
            base_fn += text.replace(":", "$3A")[:80]
        base_fn = base_fn.replace(' ', '_')
        gen_dir = self.gen_dir
        if hparams['save_gt']:
            wav_gt = self.vocoder.spec2wav(mel_gt)
            self.saving_result_pool.add_job(self.save_result, args=[
                wav_gt, mel_gt, base_fn % 'G', gen_dir, str_phs, mel2ph])
            self.saving_result_pool.add_job(self.save_result, args=[
                wav_gt, mel_pred, base_fn % 'P', gen_dir, str_phs, mel2ph])
            self.saving_result_pool.add_job(self.save_result, args=[
                wav_gt, mel_s_gt, base_fn % 'S', gen_dir, str_phs, mel2ph])
        
        # print(f"Pred_shape: {mel_pred.shape}, gt_shape: {mel_gt.shape}")
        return {
            'item_name': item_name,
            'text': text,
            'ph_tokens': self.token_encoder.decode(tokens.tolist()),
            'wav_fn_pred': base_fn % 'P',
            'wav_fn_gt': base_fn % 'G',
            'wav_fn_orig': sample['wav_fn'][0],
        }

    def test_end(self, outputs):
        pass