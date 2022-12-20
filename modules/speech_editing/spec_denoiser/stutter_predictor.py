import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear

from modules.commons.conv import ConvBlocks, TextConvEncoder
from modules.commons.layers import Embedding
from modules.speech_editing.commons.transformer import TransformerEncoder, TransformerDecoder, MultiheadAttention
from modules.tts.commons.align_ops import clip_mel2token_to_multiple, expand_states
from modules.commons.wavenet import WN
from modules.tts.fs import FS_DECODERS, FastSpeech


class ConvMelPrenet(nn.Module):
    def __init__(self, input_dim=80, hidden_size=192):
        super(ConvMelPrenet, self).__init__()
        self.pre_net = nn.Sequential(
            nn.Conv1d(input_dim, hidden_size, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Linear function (out)
        self.fc_out = nn.Linear(hidden_size, hidden_size)  

    def forward(self, x):
        out = self.pre_net(x.transpose(1,2)).transpose(1,2)
        # Linear function (out)
        out = self.fc_out(out)
        return out


class StutterPredictor(FastSpeech):
    def __init__(self, ph_dict_size, hparams, out_dims=None):
        super().__init__(ph_dict_size, hparams, out_dims)
        self.txt_encoder = TextConvEncoder(ph_dict_size, hparams['hidden_size'], hparams['hidden_size'],
                                            hparams['enc_dilations'], hparams['enc_kernel_size'],
                                            layers_in_block=hparams['layers_in_block'],
                                            norm_type=hparams['enc_dec_norm'],
                                            post_net_kernel=hparams.get('enc_post_net_kernel', 3))
        if hparams['num_spk'] > 1:
            if self.hparams['use_spk_embed']:
                self.spk_embed_proj = nn.Linear(256, self.hidden_size)
            elif self.hparams['use_spk_id']:
                self.spk_embed_proj = Embedding(hparams['num_spk'], self.hidden_size)
        self.mel_encoder = nn.Sequential(*[
            ConvMelPrenet(input_dim=80),
            ConvBlocks(
            self.hidden_size, self.hidden_size,
            [1] * 5, kernel_size=5, layers_in_block=2)
        ])
        self.encoder_attn = MultiheadAttention(
            self.hidden_size, 2, encoder_decoder_attention=True, dropout=0.1, bias=False,
        )
        self.decoder_text_prenet = ConvMelPrenet(input_dim=192)
        # self.decoder = ConvBlocks(
        #     self.hidden_size, self.hidden_size,
        #     [1] * 5, kernel_size=5, layers_in_block=2)
        self.decoder = WN(self.hidden_size, 5, 1, n_layers=4, c_cond=self.hidden_size, p_dropout=0.3,is_BTC=True)
        self.mel_out = Linear(self.hidden_size, 3, bias=False)
        self.text_drop = nn.Dropout(p=0.3)
        self.mel_drop = nn.Dropout(p=0.3)
        del self.encoder
        del self.pitch_embed
        del self.pitch_predictor
        del self.spk_embed_proj
        del self.dur_predictor
        del self.length_regulator

    def forward(self, txt_tokens, mels, mel2ph, 
                infer=False, global_step=None, *args, **kwargs):     
        ret = {}
        txt_embed, txt_nonpadding = self.run_text_encoder(txt_tokens)
        mel_embed, mel_nonpadding = self.run_mel_encoder(mels, mel2ph)
        txt_embed, mel_embed = self.text_drop(txt_embed), self.mel_drop(mel_embed)
        decoder_out, enc_dec_attn = self.run_decoder(mel_embed, txt_embed, mel2ph, mel_nonpadding, ret)
        ret['logits'] = decoder_out
        ret['attn'] = enc_dec_attn
        return ret

    def run_text_encoder(self, txt_tokens):
        # phone encoder
        txt_nonpadding = (txt_tokens > 0).float()[:, :, None]
        txt_embed = self.txt_encoder(txt_tokens) * txt_nonpadding
        return txt_embed, txt_nonpadding

    def run_mel_encoder(self, mels, mel2ph):
        block_size = 16
        B, T = mel2ph.shape
        mel_nonpadding = (mel2ph > 0).float()
        mel_nonpadding = mel_nonpadding.reshape(B, T//block_size, block_size).sum(dim=-1)[:,:,None] # [B, T//block_size, 1]
        mel_nonpadding[mel_nonpadding!=0] = 1.0
        mel_embed = self.mel_encoder(mels) * mel_nonpadding
        return mel_embed, mel_nonpadding

    def run_decoder(self, mel_encoder_out, txt_encoder_out, mel2ph, mel_nonpadding, ret):
        # coarse decoder
        # decoder_out = self.decoder(decoder_inp) * mel_nonpadding
        txt_encoder_out = expand_states(txt_encoder_out, mel2ph)
        condition = self.decoder_text_prenet(txt_encoder_out) * mel_nonpadding
        key_padding_mask = txt_encoder_out.abs().sum(-1).eq(0).data
        # condition, enc_dec_attn = self.encoder_attn(
        #         query=mel_encoder_out.transpose(0, 1),
        #         key=txt_encoder_out.transpose(0, 1),
        #         value=txt_encoder_out.transpose(0, 1),
        #         key_padding_mask=key_padding_mask,
        #         incremental_state=None,
        #         static_kv=True,
        #         enc_dec_attn_constraint_mask=None,
        #         reset_attn_weight=None
        #     )
        # condition = condition.transpose(0, 1) * mel_nonpadding
        decoder_out = self.decoder(x=mel_encoder_out, cond=condition)
        decoder_out = self.mel_out(decoder_out) * mel_nonpadding
        return decoder_out, None