import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear

from modules.commons.conv import ConvBlocks
from modules.commons.layers import Embedding
from modules.speech_editing.commons.transformer import TransformerEncoder, TransformerDecoder
from modules.speech_editing.commons.mel_encoder import MelEncoder
from modules.tts.fs import FS_DECODERS, FastSpeech


class CampNet(FastSpeech):
    def __init__(self, ph_dict_size, word_dict_size, hparams, out_dims=None):
        super().__init__(ph_dict_size, hparams, out_dims)
        self.encoder = TransformerEncoder(
            ph_dict_size, self.hidden_size, num_layers=3,
            kernel_size=hparams['dec_ffn_kernel_size'], num_heads=2)
        if hparams['num_spk'] > 1:
            if self.hparams['use_spk_embed']:
                self.spk_embed_proj = nn.Linear(256, self.hidden_size)
            elif self.hparams['use_spk_id']:
                self.spk_embed_proj = Embedding(hparams['num_spk'], self.hidden_size)
        self.mel_encoder = MelEncoder(hidden_size=self.hidden_size)
        self.decoder_coarse = TransformerDecoder(
            self.hidden_size, num_layers=6, 
            ffn_kernel_size=hparams['dec_ffn_kernel_size'], num_heads=2)
        self.decoder_fine = ConvBlocks(
            self.hidden_size, self.hidden_size,
            [1] * 5, kernel_size=5, layers_in_block=2)
        self.mel_out_coarse = Linear(self.hidden_size, self.out_dims, bias=False)
        self.mel_out_fine = Linear(self.hidden_size, self.out_dims, bias=False)
        self.mask_emb = torch.nn.Parameter(torch.zeros(1, 1, 80), requires_grad=True)
        del self.decoder
        del self.spk_embed_proj
        del self.dur_predictor
        del self.length_regulator

    def forward(self, txt_tokens,
                spk_embed=None, spk_id=None, mels=None, stutter_mel_masks=None, time_mel_masks=None, 
                infer=False, global_step=None, *args, **kwargs):     
        ret = {}
        x, src_nonpadding = self.run_text_encoder(txt_tokens, ret)
        ret['mel_out_coarse'], ret['mel_out_fine'], ret['attn'] = self.run_decoder(x, mels, time_mel_masks, ret, global_step)
        return ret

    def run_text_encoder(self, txt_tokens, ret):
        # phone encoder
        src_nonpadding = (txt_tokens > 0).float()[:, :, None]
        ph_encoder_out = self.encoder(txt_tokens) * src_nonpadding
        x = ph_encoder_out * src_nonpadding
        return x, src_nonpadding

    def run_decoder(self, encoder_out, mels, time_mel_masks, ret, global_step=0):
        mel_nonpadding = (mels.abs().sum(-1) > 0).float()[:, :, None]
        # coarse decoder
        mel_input_coarse = mels * (1-time_mel_masks) + self.mask_emb * time_mel_masks 
        mel_input_coarse = self.mel_encoder(mel_input_coarse) * mel_nonpadding
        mel_out_coarse, enc_dec_attn = self.decoder_coarse(x=mel_input_coarse, encoder_out=encoder_out)
        mel_out_coarse = mel_out_coarse * mel_nonpadding
        mel_out_coarse = self.mel_out_coarse(mel_out_coarse) * mel_nonpadding
        # fine decoder, without grad backward
        mel_coarse = mels * (1-time_mel_masks) + mel_out_coarse * time_mel_masks
        mel_input_fine = self.mel_encoder(mel_coarse) * mel_nonpadding
        mel_out_fine = self.decoder_fine(x=mel_input_fine)
        mel_out_fine = mel_out_fine * mel_nonpadding
        mel_out_fine = self.mel_out_fine(mel_out_fine) * mel_nonpadding
        mel_out_fine = mel_coarse + mel_out_fine * time_mel_masks
        return mel_out_coarse, mel_out_fine, enc_dec_attn