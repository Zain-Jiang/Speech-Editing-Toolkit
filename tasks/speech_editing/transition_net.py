import torch
import random
import utils
from utils.commons.hparams import hparams
from utils.audio.pitch.utils import denorm_f0
from modules.speech_editing.transition_net.transition_net import SingleWindowDisc
from modules.speech_editing.spec_denoiser.diffnet import DiffNet
from tasks.speech_editing.speech_editing_base import SpeechEditingBaseTask
from tasks.tts.vocoder_infer.base_vocoder import get_vocoder_cls, BaseVocoder
from tasks.tts.dataset_utils import StutterSpeechDataset


DIFF_DECODERS = {
    'wavenet': lambda hp: DiffNet(hp['audio_num_mel_bins']),
}


class TransitionNetTask(SpeechEditingBaseTask):
    def __init__(self):
        super(TransitionNetTask, self).__init__()
        self.dataset_cls = StutterSpeechDataset

    def build_model(self):
        self.build_tts_model()
        utils.nn.model_utils.num_params(self.model)
        return self.model

    def build_tts_model(self):
        self.model = SingleWindowDisc(time_length=64)


    def run_model(self, sample, infer=False, batch_size=200, clip_length_range=50, *args, **kwargs):
        txt_tokens = sample['txt_tokens']  # [B, T_t]
        target = sample['mels']  # [B, T_s, 80]
        mel2ph = sample['mel2ph']
        f0 = sample['f0']
        uv = sample['uv']
        energy = None
        time_mel_masks = sample['time_mel_masks'][:,:,None]
        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')

        mel = sample['mels']
        mel_len = sample['mel_lengths']
        mel_list, label_list = [], []
        clip_length = 64
        inserted_clip_length = random.randint(a=10, b=30)
        while len(mel_list) < batch_size:
            for i in range(mel.shape[0]):
                if mel[i].shape[0] <= 80:
                    continue
                if not infer:
                    is_pos_sample = random.choice([True, False])
                else:
                    is_pos_sample = True
                clip_idx = random.randint(a=0, b=mel_len[i]-1-clip_length)
                mel_clip = mel[i, clip_idx: clip_idx+clip_length]
                assert mel_clip.shape[0]==clip_length, f"exp_idx={clip_idx},mel_len={mel_len[i]}"
                if is_pos_sample:
                    mel_list.append(mel_clip)
                    label_list.append(1.)
                else:
                    # The first case (random shift)
                    # if random.random() < 0.25:
                    inserted_clip_idx = random.randint(a=0, b=mel_len[i]-1-inserted_clip_length)
                    # while (inserted_clip_idx >= clip_idx-3) and (inserted_clip_idx <= clip_idx+3):
                        # inserted_clip_idx = random.randint(a=0, b=mel_len[i]-1-inserted_clip_length)
                    insert_to_idx = random.randint(a=0, b=clip_length-1-inserted_clip_length)
                    wrong_mel_clip = mel_clip.clone()
                    wrong_mel_clip[insert_to_idx:insert_to_idx+inserted_clip_length] = mel[i, inserted_clip_idx: inserted_clip_idx + inserted_clip_length]
                    mel_list.append(wrong_mel_clip)
                    label_list.append(0.)
        mel_clips = torch.stack(mel_list)
        labels = torch.tensor(label_list).to(mel_clips.device)[:,None]
        pred = self.model(mel_clips[:, None, :, :])
        bce_loss = self.model.bce_loss(pred, labels)

        losses = {}
        model_out = {}
        if not infer:
            losses['bce'] = bce_loss
            return losses, model_out
        else:
            model_out['bce'] = bce_loss
            return model_out

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
        return outputs

    def _training_step(self, sample, batch_idx, optimizer_idx):
        loss_output, model_out = self.run_model(sample, infer=False)
        loss_weights = {
        }
        total_loss = sum([loss_weights.get(k, 1) * v for k, v in loss_output.items() if isinstance(v, torch.Tensor) and v.requires_grad])

        return total_loss, loss_output

    def validation_start(self):
        pass

    @torch.no_grad()
    def validation_step(self, sample, batch_idx):
        outputs = {}
        outputs['losses'] = {}
        outputs['losses'], model_out = self.run_model(sample, infer=False)
        outputs = utils.commons.tensor_utils.tensors_to_scalars(outputs)
        return outputs

    def validation_end(self, outputs):
        return super().validation_end(outputs)
        
    #####################
    # Testing
    #####################
    def test_start(self):
        pass

    @torch.no_grad()
    def test_step(self, sample, batch_idx):
        """
        :param sample:
        :param batch_idx:
        :return:
        """
        outputs = {}
        return None

    def test_end(self, outputs):
        pass

    @staticmethod
    def save_result(exp_arr, base_fname, gen_dir):
        pass
    
    def get_grad(self, opt_idx):
        # grad_dict = {
        #     'grad/model': get_grad_norm(self.model),
        # }
        # return grad_dict
        pass