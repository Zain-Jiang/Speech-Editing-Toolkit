import math
import torch
from torch import nn
import torch.nn.functional as F
from modules.speech_editing.commons.conformer_layers import Swish, ConvolutionModule, EncoderLayer, MultiLayeredConv1d
from modules.commons.layers import Embedding
from modules.speech_editing.commons.transformer import SinusoidalPositionalEmbedding, MultiheadAttention
from modules.speech_editing.commons.mel_encoder import MelEncoder

DEFAULT_MAX_SOURCE_POSITIONS = 4000
DEFAULT_MAX_TARGET_POSITIONS = 4000


class ConformerLayers(nn.Module):
    def __init__(self, hidden_size, num_layers, kernel_size=9, dropout=0.2, attn_dropout=0.2, num_heads=4,
                 use_last_norm=True, save_hidden=False):
        super().__init__()
        self.use_last_norm = use_last_norm
        self.layers = nn.ModuleList()
        positionwise_layer = MultiLayeredConv1d
        positionwise_layer_args = (hidden_size, hidden_size * 4, 1, dropout)
        self.encoder_layers = nn.ModuleList([EncoderLayer(
            hidden_size,
            MultiheadAttention(hidden_size, num_heads, self_attention=True, dropout=attn_dropout, bias=False),
            positionwise_layer(*positionwise_layer_args),
            positionwise_layer(*positionwise_layer_args),
            ConvolutionModule(hidden_size, kernel_size, Swish()),
            dropout,
        ) for _ in range(num_layers)])
        if self.use_last_norm:
            self.layer_norm = nn.LayerNorm(hidden_size)
        else:
            self.layer_norm = nn.Linear(hidden_size, hidden_size)
        self.save_hidden = save_hidden
        if save_hidden:
            self.hiddens = []

    def forward(self, x, padding_mask=None):
        """

        :param x: [B, T, H]
        :param padding_mask: [B, T]
        :return: [B, T, H]
        """
        self.hiddens = []
        padding_mask = x.abs().sum(-1).eq(0)
        nonpadding_mask = (1-padding_mask.float())[:, :, None]
        for l in self.encoder_layers:
            x, _ = l(x, padding_mask.transpose(0,1))
            if self.save_hidden:
                self.hiddens.append(x[0])
        x = x[0]
        x = self.layer_norm(x) * nonpadding_mask
        return x


class ConformerEncoder(ConformerLayers):
    def __init__(self, dict_size, hidden_size, num_layers=None, kernel_size=9):
        super().__init__(hidden_size, num_layers, kernel_size=kernel_size)
        self.padding_idx = 0
        self.embed_scale = math.sqrt(hidden_size)
        self.txt_embed = Embedding(dict_size, hidden_size, padding_idx=0)
        self.mel_embed = MelEncoder(hidden_size=hidden_size)
        self.pos_embed = SinusoidalPositionalEmbedding(
            hidden_size, self.padding_idx, init_size=DEFAULT_MAX_TARGET_POSITIONS,
        )
        self.seg_embed = Embedding(2000, hidden_size, padding_idx=0)
        self.dropout = 0.2
        self.embed_scale = math.sqrt(hidden_size)

    def forward(self, txt_tokens, txt_nonpadding, mels, mel2ph, time_mel_masks):
        """

        :param src_tokens: [B, T]
        :return: [B x T x C]
        """
        txt_nonpadding = txt_nonpadding
        mel_nonpadding = (mel2ph > 0).float()[:, :, None]
        encoder_padding_mask = torch.cat([mel_nonpadding, txt_nonpadding], dim=1)

        text_embed = self.forward_txt_embedding(txt_tokens, txt_nonpadding) # [B, T_t, H]
        mel_embed = self.forward_mel_embedding(mels, mel_nonpadding, mel2ph, time_mel_masks) # [B, T_t, H]
        x = torch.cat([mel_embed, text_embed], dim=1) * encoder_padding_mask # [B, T_t + T_m, H]
        x = super(ConformerEncoder, self).forward(x)
        return x, encoder_padding_mask

    def forward_txt_embedding(self, txt_tokens, txt_nonpadding):
        ph2ph = torch.arange(txt_tokens.shape[1])[None, :].to(txt_tokens.device) + 1
        # text embedding
        txt_feat = self.embed_scale * self.txt_embed(txt_tokens)
        # positional embedding
        txt_pos_emb = self.pos_embed(txt_tokens)
        txt_feat = txt_feat + txt_pos_emb
        # segment embedding
        txt_seg_emb = self.seg_embed(ph2ph) 
        txt_feat = txt_feat + txt_seg_emb
        # padding
        txt_feat = txt_feat * txt_nonpadding
        txt_feat = F.dropout(txt_feat, p=self.dropout, training=self.training)
        return txt_feat
    
    def forward_mel_embedding(self, mels, mel_nonpadding, mel2ph, time_mel_masks):
        # mel embedding
        mels_masked = mels * (1-time_mel_masks)
        mel_feat = self.mel_embed(mels_masked) * mel_nonpadding
        # positional embedding
        mel_pos_emb = self.pos_embed(mels[..., 0])
        mel_feat = mel_feat + mel_pos_emb
        # segment embedding
        mel_seg_emb = self.seg_embed(mel2ph)
        mel_feat = mel_feat + mel_seg_emb
        # padding mask
        mel_feat = mel_feat * mel_nonpadding
        mel_feat = F.dropout(mel_feat, p=self.dropout, training=self.training)
        return mel_feat


class ConformerDecoder(ConformerLayers):
    def __init__(self, hidden_size, num_layers=None, kernel_size=9):
        super().__init__(hidden_size, num_layers, kernel_size=kernel_size)

    def forward(self, encoder_out, encoder_padding_mask):
        """

        :param src_tokens: [B, T]
        :return: [B x T x C]
        """
        x = super(ConformerDecoder, self).forward(encoder_out, encoder_padding_mask)
        return x