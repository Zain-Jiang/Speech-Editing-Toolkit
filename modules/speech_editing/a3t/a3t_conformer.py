import math
import torch
from torch import nn
import torch.nn.functional as F
from modules.speech_editing.a3t.conformer_layers import Swish, ConvolutionModule, EncoderLayer, MultiLayeredConv1d
from modules.commons.layers import Embedding
from modules.commons.conformer.espnet_positional_embedding import RelPositionalEncoding
from modules.commons.conformer.espnet_transformer_attn import RelPositionMultiHeadedAttention
from modules.speech_editing.commons.mel_encoder import MelEncoder

DEFAULT_MAX_SOURCE_POSITIONS = 4000
DEFAULT_MAX_TARGET_POSITIONS = 4000


class ConformerLayers(nn.Module):
    def __init__(self, hidden_size, num_layers, kernel_size=9, dropout=0.0, num_heads=4,
                 use_last_norm=True):
        super().__init__()
        self.use_last_norm = use_last_norm
        self.layers = nn.ModuleList()
        positionwise_layer = MultiLayeredConv1d
        positionwise_layer_args = (hidden_size, hidden_size * 4, 1, dropout)
        self.encoder_layers = nn.ModuleList([EncoderLayer(
            hidden_size,
            RelPositionMultiHeadedAttention(num_heads, hidden_size, 0.0),
            positionwise_layer(*positionwise_layer_args),
            positionwise_layer(*positionwise_layer_args),
            ConvolutionModule(hidden_size, kernel_size, Swish()),
            dropout,
        ) for _ in range(num_layers)])
        if self.use_last_norm:
            self.layer_norm = nn.LayerNorm(hidden_size)
        else:
            self.layer_norm = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, pos_emb, padding_mask=None):
        """

        :param x: [B, T, H]
        :param padding_mask: [B, T]
        :return: [B, T, H]
        """
        nonpadding_mask = x.abs().sum(-1) > 0
        for l in self.encoder_layers:
            x, mask = l(x, pos_emb, nonpadding_mask[:, None, :])
        x = self.layer_norm(x) * nonpadding_mask.float()[:, :, None]
        return x


class ConformerEncoder(ConformerLayers):
    def __init__(self, dict_size, hidden_size, num_layers=None, kernel_size=9):
        super().__init__(hidden_size, num_layers, kernel_size=kernel_size)
        self.padding_idx = 0
        self.dropout = 0.2
        self.embed_scale = math.sqrt(hidden_size)
        self.txt_embed = Embedding(dict_size, hidden_size, padding_idx=0)
        self.mel_embed = MelEncoder(hidden_size=hidden_size)
        # positional encoding
        self.pos_embed = RelPositionalEncoding(hidden_size, self.dropout)
        # segment embedding
        self.seg_embed = Embedding(2000, hidden_size, padding_idx=0)
        self.embed_scale = math.sqrt(hidden_size)

    def forward(self, txt_tokens, txt_nonpadding, mels, mel2ph, time_mel_masks):
        """

        :param src_tokens: [B, T]
        :return: [B x T x C]
        """
        txt_nonpadding = txt_nonpadding
        mel_nonpadding = (mel2ph > 0).float()[:, :, None]
        encoder_padding_mask = torch.cat([mel_nonpadding, txt_nonpadding], dim=1)

        text_embed, text_pos = self.forward_txt_embedding(txt_tokens, txt_nonpadding) # [B, T_t, H]
        mel_embed, mel_pos = self.forward_mel_embedding(mels, mel_nonpadding, mel2ph, time_mel_masks) # [B, T_t, H]
        x = torch.cat([mel_embed, text_embed], dim=1) * encoder_padding_mask # [B, T_t + T_m, H]
        pos_emb = torch.cat([mel_pos, text_pos], dim=1) * encoder_padding_mask # [B, T_t + T_m, H]
        x = super(ConformerEncoder, self).forward(x, pos_emb)
        return x, pos_emb, encoder_padding_mask

    def forward_txt_embedding(self, txt_tokens, txt_nonpadding):
        ph2ph = torch.arange(txt_tokens.shape[1])[None, :].to(txt_tokens.device) + 1
        # text embedding
        txt_feat = self.txt_embed(txt_tokens) * txt_nonpadding
        # positional embedding
        txt_feat, txt_pos = self.pos_embed(txt_feat)
        # segment embedding
        txt_seg_emb = self.seg_embed(ph2ph) 
        txt_feat = txt_feat + txt_seg_emb
        return txt_feat, txt_pos
    
    def forward_mel_embedding(self, mels, mel_nonpadding, mel2ph, time_mel_masks):
        # mel embedding
        mels_masked = mels * (1-time_mel_masks)
        mel_feat = self.mel_embed(mels_masked) * mel_nonpadding
        # positional embedding
        mel_feat, mel_pos = self.pos_embed(mel_feat)
        # segment embedding
        mel_seg_emb = self.seg_embed(mel2ph)
        mel_feat = mel_feat + mel_seg_emb
        return mel_feat, mel_pos


class ConformerDecoder(ConformerLayers):
    def __init__(self, hidden_size, num_layers=None, kernel_size=9):
        super().__init__(hidden_size, num_layers, kernel_size=kernel_size)

    def forward(self, encoder_out, pos_emb, encoder_padding_mask):
        """

        :param src_tokens: [B, T]
        :return: [B x T x C]
        """
        x = super(ConformerDecoder, self).forward(encoder_out, pos_emb, encoder_padding_mask)
        return x