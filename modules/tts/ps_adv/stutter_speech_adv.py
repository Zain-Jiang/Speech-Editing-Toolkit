import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear

from modules.commons.conv import ConvBlocks, ConditionalConvBlocks
from modules.commons.layers import Embedding
from modules.commons.rel_transformer import RelTransformerEncoder
from modules.commons.transformer import MultiheadAttention, FFTBlocks
from modules.tts.ps_adv.transformer_decoder import TransformerEncoder, TransformerDecoder, MelEncoder
from modules.tts.commons.align_ops import clip_mel2token_to_multiple, build_word_mask, expand_states, mel2ph_to_mel2word
from modules.tts.fs import FS_DECODERS, FastSpeech
from modules.tts.portaspeech.fvae import FVAE
from utils.commons.meters import Timer
from utils.nn.seq_utils import group_hidden_by_segs
from modules.commons.nar_tts_modules import DurationPredictor


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """

        :param x: [B, T]
        :return: [B, T, H]
        """
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, :, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.05, max_len=2000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + torch.autograd.Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class StutterSpeech_adv(FastSpeech):
    def __init__(self, ph_dict_size, word_dict_size, hparams, out_dims=None):
        super().__init__(ph_dict_size, hparams, out_dims)
        self.encoder = TransformerEncoder(
            ph_dict_size, self.hidden_size, num_layers=3,
            kernel_size=hparams['dec_ffn_kernel_size'], num_heads=2)
        # build linguistic encoder
        if hparams['num_spk'] > 1:
            if self.hparams['use_spk_embed']:
                self.spk_embed_proj = nn.Linear(256, self.hidden_size)
            elif self.hparams['use_spk_id']:
                self.spk_embed_proj = Embedding(hparams['num_spk'], self.hidden_size)
        if hparams['use_word_encoder']:
            self.word_encoder = RelTransformerEncoder(
                word_dict_size, self.hidden_size, self.hidden_size, self.hidden_size, 2,
                hparams['word_enc_layers'], hparams['enc_ffn_kernel_size'])
        if hparams['dur_level'] == 'word':
            if hparams['word_encoder_type'] == 'rel_fft':
                self.ph2word_encoder = RelTransformerEncoder(
                    0, self.hidden_size, self.hidden_size, self.hidden_size, 2,
                    hparams['word_enc_layers'], hparams['enc_ffn_kernel_size'])
            if hparams['word_encoder_type'] == 'fft':
                self.ph2word_encoder = FFTBlocks(
                    self.hidden_size, hparams['word_enc_layers'], 1, num_heads=hparams['num_heads'])
            self.sin_pos = SinusoidalPosEmb(self.hidden_size)
            self.enc_pos_proj = nn.Linear(self.hidden_size, self.hidden_size)
            self.dec_query_proj = nn.Linear(self.hidden_size, self.hidden_size)
            self.dec_res_proj = nn.Linear(self.hidden_size, self.hidden_size)
            self.attn = MultiheadAttention(self.hidden_size, 1, encoder_decoder_attention=True, bias=False)
            self.attn.enable_torch_version = False
            # if hparams['text_encoder_postnet']:
            #     self.text_encoder_postnet = ConvBlocks(
            #         self.hidden_size, self.hidden_size, [1] * 3, 5, layers_in_block=2)
        else:
            self.sin_pos = SinusoidalPosEmb(self.hidden_size)
        self.mel_encoder = MelEncoder()
        self.txt_pos_emb = PositionalEncoding(d_model=self.hidden_size)
        self.mel_pos_emb = PositionalEncoding(d_model=self.hidden_size)
        predictor_hidden = hparams['predictor_hidden'] if hparams['predictor_hidden'] > 0 else self.hidden_size
        self.dur_predictor = DurationPredictor(
            self.hidden_size,
            n_chans=predictor_hidden,
            n_layers=hparams['dur_predictor_layers'],
            dropout_rate=hparams['predictor_dropout'],
            kernel_size=hparams['dur_predictor_kernel'])
        # build VAE decoder
        if hparams['use_fvae']:
            del self.decoder
            del self.mel_out
            self.fvae = FVAE(
                c_in_out=self.out_dims,
                hidden_size=hparams['fvae_enc_dec_hidden'], c_latent=hparams['latent_size'],
                kernel_size=hparams['fvae_kernel_size'],
                enc_n_layers=hparams['fvae_enc_n_layers'],
                dec_n_layers=hparams['fvae_dec_n_layers'],
                c_cond=self.hidden_size,
                use_prior_flow=hparams['use_prior_flow'],
                flow_hidden=hparams['prior_flow_hidden'],
                flow_kernel_size=hparams['prior_flow_kernel_size'],
                flow_n_steps=hparams['prior_flow_n_blocks'],
                strides=[hparams['fvae_strides']],
                encoder_type=hparams['fvae_encoder_type'],
                decoder_type=hparams['fvae_decoder_type'],
            )
        else:
            # self.decoder = FS_DECODERS[hparams['decoder_type']](hparams)
            self.decoder = TransformerDecoder(
                self.hidden_size, num_layers=6, 
                ffn_kernel_size=hparams['dec_ffn_kernel_size'], num_heads=2)
            self.mel_out = Linear(self.hidden_size, self.out_dims, bias=True)
        if hparams['use_pitch_embed']:
            self.pitch_embed = Embedding(300, self.hidden_size, 0)
        if self.hparams['add_word_pos']:
            self.word_pos_proj = Linear(self.hidden_size, self.hidden_size)
        self.mask_emb = torch.nn.Parameter(torch.zeros(1, 1, 80), requires_grad=True)

    def build_embedding(self, dictionary, embed_dim):
        num_embeddings = len(dictionary)
        emb = Embedding(num_embeddings, embed_dim, self.padding_idx)
        return emb

    def forward(self, txt_tokens, word_tokens, ph2word, word_len,
                spk_embed=None, spk_id=None, mels=None, stutter_mel_masks=None, time_mel_masks=None, 
                pitch=None, infer=False, tgt_mels=None, global_step=None, *args, **kwargs):     
        # if self.hparams['use_spk_embed']:
        #     spk_embed = self.spk_embed_proj(spk_embed[:, None, :])
        # elif self.hparams['use_spk_id']:
        #     spk_embed = self.spk_embed_proj(spk_id)[:, None, :]  
        # else:
        #     spk_embed = 0

        ret = {}
        style_embed = spk_embed
        # style_embed = self.forward_style_embed(spk_embed, spk_id) # speaker embedding, [B, 1, C]
        x, src_nonpadding, mel_encoder_out, tgt_nonpadding = self.run_encoders(
            txt_tokens, word_tokens, ph2word, word_len, mels, style_embed, stutter_mel_masks, time_mel_masks, ret)
        # x = x + style_embed # it maybe necessary to achieve multi-speaker
        # x = x * src_nonpadding
        ret['nonpadding'] = tgt_nonpadding
        if self.hparams['use_pitch_embed']:
            x = x + self.pitch_embed(pitch)
        ret['decoder_inp'] = x
        attn_mask = time_mel_masks.repeat(2,1,x.shape[1]) * -1e9 # attention mask for masked prediction
        ret['mel_out'], ret['attn'] = self.run_decoder(x, mel_encoder_out, attn_mask, tgt_nonpadding, ret, infer, tgt_mels, global_step)
        return ret

    def run_encoders(self, txt_tokens, word_tokens, ph2word, word_len, mels, style_embed, stutter_mel_masks, time_mel_masks, ret):
        word2word = torch.arange(word_len)[None, :].to(ph2word.device) + 1  # [B, T_mel, T_word]
        src_nonpadding = (txt_tokens > 0).float()[:, :, None]
        ph_encoder_out = self.encoder(txt_tokens) * src_nonpadding
        # if self.hparams['use_word_encoder']:
        #     word_encoder_out = self.word_encoder(word_tokens) + style_embed
        #     text_encoder_out = ph_encoder_out + expand_states(word_encoder_out, ph2word)
        x = ph_encoder_out * src_nonpadding

        mel_nonpaddings = (mels.abs().sum(-1) > 0).float()[:, :, None]
        mels_masked = mels * (1-time_mel_masks) + time_mel_masks * self.mask_emb
        mel_encoder_out = self.mel_encoder(mels_masked) * mel_nonpaddings

        # ret['attn'] = weight
        # dur_input = ph_encoder_out * src_nonpadding
        # if self.hparams['dur_level'] == 'word':
        #     word_encoder_out = 0
        #     h_ph_gb_word = group_hidden_by_segs(ph_encoder_out, ph2word, word_len)[0]
        #     word_encoder_out = word_encoder_out + self.ph2word_encoder(h_ph_gb_word)
        #     if self.hparams['use_word_encoder']:
        #         word_encoder_out = word_encoder_out + self.word_encoder(word_tokens)
        #     mel2word = self.forward_dur(dur_input, mel2word, ret, ph2word=ph2word, word_len=word_len)
        #     mel2word = clip_mel2token_to_multiple(mel2word, self.hparams['frames_multiple'])
        #     ret['mel2word'] = mel2word
        #     tgt_nonpadding = (mel2word > 0).float()[:, :, None]
        #     enc_pos = self.get_pos_embed(word2word, ph2word)  # [B, T_ph, H]
        #     dec_pos = self.get_pos_embed(word2word, mel2word)  # [B, T_mel, H]
        #     dec_word_mask = build_word_mask(mel2word, ph2word)  # [B, T_mel, T_ph]
        #     x, weight = self.attention(ph_encoder_out, enc_pos, word_encoder_out, dec_pos, mel2word, dec_word_mask)
        #     if self.hparams['add_word_pos']:
        #         x = x + self.word_pos_proj(dec_pos)
        #     ret['attn'] = weight
        # else:
        #     mel2ph = self.forward_dur(dur_input, mel2ph, ret)
        #     mel2ph = clip_mel2token_to_multiple(mel2ph, self.hparams['frames_multiple'])
        #     mel2word = mel2ph_to_mel2word(mel2ph, ph2word)
        #     x = expand_states(ph_encoder_out, mel2ph)
        #     if self.hparams['add_word_pos']:
        #         dec_pos = self.get_pos_embed(word2word, mel2word)  # [B, T_mel, H]
        #         x = x + self.word_pos_proj(dec_pos)
        #     tgt_nonpadding = (mel2ph > 0).float()[:, :, None]
        # return x, mel_nonpaddings
        return x, src_nonpadding, mel_encoder_out, mel_nonpaddings

    def attention(self, text_encoder_out, mel_encoder_out, mel_encoder_out_masked, time_mel_masks):
        q = self.dec_query_proj(self.mel_pos_emb(mel_encoder_out_masked))
        x_res = self.dec_res_proj(self.mel_pos_emb(mel_encoder_out_masked))
        kv = self.enc_pos_proj(self.txt_pos_emb(text_encoder_out))
        q, kv = q.transpose(0, 1), kv.transpose(0, 1)
        x, (weight, _) = self.attn(q, kv, kv, attn_mask=time_mel_masks[:,:,None].repeat(1,1,kv.shape[0]) * -1e9)
        x = x.transpose(0, 1)
        x = x + x_res
        return x, weight

    def run_decoder(self, encoder_out, tgt, attn_mask, tgt_nonpadding, ret, infer, tgt_mels=None, global_step=0):
        if not self.hparams['use_fvae']:
            x, enc_dec_attn = self.decoder(x=tgt, encoder_out=encoder_out, attn_mask=None)
            x = self.mel_out(x)
            ret['kl'] = 0
            return x * tgt_nonpadding, enc_dec_attn
        else:
            # x is the phoneme encoding
            x = x.transpose(1, 2)  # [B, H, T]
            tgt_nonpadding_BHT = tgt_nonpadding.transpose(1, 2)  # [B, H, T]
            if infer:
                z = self.fvae(cond=x, infer=True)
            else:
                tgt_mels = tgt_mels.transpose(1, 2)  # [B, 80, T]
                z, ret['kl'], ret['z_p'], ret['m_q'], ret['logs_q'] = self.fvae(
                    tgt_mels, tgt_nonpadding_BHT, cond=x)
                if global_step < self.hparams['posterior_start_steps']:
                    z = torch.randn_like(z)
            x_recon = self.fvae.decoder(z, nonpadding=tgt_nonpadding_BHT, cond=x).transpose(1, 2)
            ret['pre_mel_out'] = x_recon
            return x_recon

    def forward_dur(self, dur_input, mel2word, ret, **kwargs):
        """

        :param dur_input: [B, T_txt, H]
        :param mel2ph: [B, T_mel]
        :param txt_tokens: [B, T_txt]
        :param ret:
        :return:
        """
        word_len = kwargs['word_len']
        ph2word = kwargs['ph2word']
        src_padding = dur_input.data.abs().sum(-1) == 0
        dur_input = dur_input.detach() + self.hparams['predictor_grad'] * (dur_input - dur_input.detach())
        dur = self.dur_predictor(dur_input, src_padding)

        B, T_ph = ph2word.shape
        dur = torch.zeros([B, word_len.max() + 1]).to(ph2word.device).scatter_add(1, ph2word, dur)
        dur = dur[:, 1:]
        ret['dur'] = dur
        if mel2word is None:
            mel2word = self.length_regulator(dur).detach()
        return mel2word

    def get_pos_embed(self, word2word, x2word):
        x_pos = build_word_mask(word2word, x2word).float()  # [B, T_word, T_ph]
        x_pos = (x_pos.cumsum(-1) / x_pos.sum(-1).clamp(min=1)[..., None] * x_pos).sum(1)
        x_pos = self.sin_pos(x_pos.float())  # [B, T_ph, H]
        return x_pos

    def store_inverse_all(self):
        def remove_weight_norm(m):
            try:
                if hasattr(m, 'store_inverse'):
                    m.store_inverse()
                nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(remove_weight_norm)
