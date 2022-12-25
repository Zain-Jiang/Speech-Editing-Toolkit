import math
import random
from functools import partial
from modules.speech_editing.spec_denoiser.diffusion_utils import *
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from modules.tts.fs import FastSpeech
from modules.commons.transformer import SinusoidalPositionalEmbedding, DEFAULT_MAX_TARGET_POSITIONS
from modules.speech_editing.editspeech.lstm import LSTM_Seq2Seq
from utils.commons.hparams import hparams


class EditSpeech(nn.Module):
    def __init__(self, phone_encoder, out_dims):
        super().__init__()
        self.fs = FastSpeech(len(phone_encoder), hparams)
        self.padding_idx = 0
        self.embed_positions = SinusoidalPositionalEmbedding(
            self.fs.hidden_size, self.padding_idx, init_size=DEFAULT_MAX_TARGET_POSITIONS,
        )
        self.decoder = LSTM_Seq2Seq(prenet_hidden_size=self.fs.hidden_size,
                                    hidden_size=1024, 
                                    output_dim=hparams['audio_num_mel_bins'])
        self.fs.decoder = None

    def forward(self, txt_tokens, time_mel_masks, mel2ph=None, spk_embed=None,
                ref_mels=None, f0=None, uv=None, energy=None, infer=False):
        b, *_, device = *txt_tokens.shape, txt_tokens.device
        ret = {}
        ret = self.fs(txt_tokens, mel2ph, spk_embed, f0, uv, energy,
                       skip_decoder=True, infer=infer)
        decoder_inp = ret['decoder_inp']
        tgt_nonpadding = (mel2ph > 0).float()[:, :, None]

        # positional embedding
        mel_pos_emb = self.embed_positions(ref_mels[..., 0])
        decoder_inp = decoder_inp + mel_pos_emb
        
        # forward lstm decoder
        framelevel_hidden = decoder_inp.transpose(0, 1)
        ref_mels = ref_mels.transpose(0, 1)
        forward_outputs, backward_outputs = self.decoder(framelevel_hidden, ref_mels, framelevel_hidden.shape[0], time_mel_masks, infer=infer)
        ret['forward_outputs'] = forward_outputs.transpose(0, 1)
        ret['backward_outputs'] = backward_outputs.transpose(0, 1)
        return ret