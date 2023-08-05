import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear

from modules.commons.layers import Embedding
from modules.speech_editing.a3t.a3t_conformer import ConformerEncoder, ConformerDecoder
from modules.speech_editing.a3t.a3t_postnet import Postnet
from modules.tts.fs import FastSpeech
from modules.commons.nar_tts_modules import DurationPredictor


class A3T(FastSpeech):
    def __init__(self, ph_dict_size, word_dict_size, hparams, out_dims=None):
        super().__init__(ph_dict_size, hparams, out_dims)
        self.encoder = ConformerEncoder(
            ph_dict_size, self.hidden_size, num_layers=4, kernel_size=9)
        # build linguistic encoder
        predictor_hidden = hparams['predictor_hidden'] if hparams['predictor_hidden'] > 0 else self.hidden_size
        self.dur_predictor = DurationPredictor(
            self.hidden_size,
            n_chans=predictor_hidden,
            n_layers=hparams['dur_predictor_layers'],
            dropout_rate=hparams['predictor_dropout'],
            kernel_size=hparams['dur_predictor_kernel'])
        self.a3t_decoder = ConformerDecoder(
            self.hidden_size, num_layers=4, kernel_size=31)
        self.a3t_postnet = Postnet(idim=self.hidden_size, odim=hparams['audio_num_mel_bins'])
        self.mel_out_decoder = Linear(self.hidden_size, self.out_dims, bias=True)
        del self.decoder
        del self.dur_predictor
        del self.length_regulator

    def build_embedding(self, dictionary, embed_dim):
        num_embeddings = len(dictionary)
        emb = Embedding(num_embeddings, embed_dim, self.padding_idx)
        return emb

    def forward(self, txt_tokens, mel2ph,
                spk_embed=None, spk_id=None, mels=None, stutter_mel_masks=None, time_mel_masks=None, 
                infer=False, tgt_mels=None, global_step=None, *args, **kwargs):     
        ret = {}
        style_embed = spk_embed
        encoder_out, pos_emb, encoder_padding_mask = self.run_encoders(
            txt_tokens, mel2ph, mels, style_embed, stutter_mel_masks, time_mel_masks, ret)
        ret['mel_out_decoder'], ret['mel_out_postnet'] = self.run_decoder(encoder_out, pos_emb, encoder_padding_mask, mels, time_mel_masks, global_step)
        return ret

    def run_encoders(self, txt_tokens, mel2ph, mels, style_embed, stutter_mel_masks, time_mel_masks, ret):
        txt_nonpadding = (txt_tokens > 0).float()[:, :, None]
        encoder_out, pos_emb, encoder_padding_mask = self.encoder(txt_tokens, txt_nonpadding, mels, mel2ph, time_mel_masks)
        encoder_out = encoder_out * encoder_padding_mask
        return encoder_out, pos_emb, encoder_padding_mask

    def run_decoder(self, encoder_out, pos_emb, encoder_padding_mask, mels, time_mel_masks, global_step=0):
        mel_nonpadding = (mels.abs().sum(-1) > 0).float()[:, :, None]
        decoder_out = self.a3t_decoder(encoder_out, pos_emb, encoder_padding_mask)[:,:mel_nonpadding.shape[1],:] * mel_nonpadding
        mel_out_decoder = self.mel_out_decoder(decoder_out) * mel_nonpadding

        mel_decoder = mels * (1-time_mel_masks) + mel_out_decoder * time_mel_masks
        mel_input_postnet = self.encoder.mel_embed(mel_decoder) * mel_nonpadding
        mel_out_postnet = self.a3t_postnet(mel_input_postnet) * mel_nonpadding
        mel_out_postnet = mel_decoder + mel_out_postnet * time_mel_masks
        return mel_out_decoder, mel_out_postnet