import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm
import math

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PAD_IDX = 2

class Seq2SeqTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, d_ff, seq_len, src_vocab_size, tgt_vocab_size, dropout = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.seq_len = seq_len

        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        self.encoder_layers = nn.ModuleList([TransformerEncoderLayer(d_model, d_ff, nhead, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([TransformerDecoderLayer(d_model, d_ff, nhead, dropout) for _ in range(num_layers)])
        self.output_linear = nn.Linear(d_model, tgt_vocab_size)
        
    def forward(self, src, tgt):
        tgt_len = tgt.size(1)
        src_padding_mask, tgt_padding_mask, causal_mask = create_masks(src, tgt, tgt_len)
        src_embed = self.src_embedding(src) * math.sqrt(self.d_model)
        src_embed = self.positional_encoding(src_embed)
        enc_output = src_embed
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_padding_mask)
        tgt_embed = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_embed = self.positional_encoding(tgt_embed)
        dec_output = tgt_embed
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_output, src_padding_mask, tgt_padding_mask, causal_mask)
        output = self.output_linear(dec_output)
        return output
    
    def reset_cache(self):
        for layer in self.decoder_layers:
            layer.reset_cache()


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, nhead, dropout= 0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.nhead = nhead

        self.mha = MultiHeadAttention(d_model, nhead, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
        
    def forward(self, x, padding_mask):
        norm1_x = self.norm1(x)
        attn_out = self.mha(norm1_x, norm1_x, norm1_x, padding_mask)
        x = x + self.dropout1(attn_out)
        norm2_x = self.norm2(x)
        ffn_out = self.ffn(norm2_x)
        x = x + self.dropout2(ffn_out)
        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, nhead, dropout= 0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.nhead = nhead

        self.self_mha = MultiHeadAttention(d_model, nhead, dropout)
        self.cross_mha = MultiHeadAttention(d_model, nhead, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)

    def forward(self, y, enc_out, src_padding_mask, tgt_padding_mask, tgt_mask, kvcache=True):
        norm1_y = self.norm1(y)
        if kvcache:
            self_attn_out = self.self_mha(norm1_y, norm1_y, norm1_y, tgt_padding_mask, tgt_mask, kvcache="self")
            y = y + self.dropout1(self_attn_out)
            norm2_y = self.norm2(y)
            cross_attn_out = self.cross_mha(norm2_y, enc_out, enc_out, src_padding_mask, kvcache="cross")
        else:
            self_attn_out = self.self_mha(norm1_y, norm1_y, norm1_y, tgt_padding_mask, tgt_mask)
            y = y + self.dropout1(self_attn_out)
            norm2_y = self.norm2(y)
            cross_attn_out = self.cross_mha(norm2_y, enc_out, enc_out, src_padding_mask)
        y = y + self.dropout2(cross_attn_out)
        norm3_y = self.norm3(y)
        ffn_out = self.ffn(norm3_y)
        y = y + self.dropout3(ffn_out)
        return y
    
    def reset_cache(self):
        self.self_mha.reset_cache()
        self.cross_mha.reset_cache()

class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, embedding, start_pos=0):
        return self.dropout(embedding + self.pos_embedding[:, start_pos:start_pos + embedding.size(1), :]) # start_pos is necessary when using kvcache. Otherwise it will return the same positional id for every token.

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout = 0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        assert d_model % nhead == 0
        self.d_k = d_model // nhead
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)

    def forward(self, query_x, key_x, value_x, padding_mask, tgt_mask=None, kvcache=None):
        Q = self.W_q(query_x)
        Q = rearrange(Q, 'b n (h d_k) -> b h n d_k', h=self.nhead)

        if kvcache == "self":
            K_new = self.W_k(key_x)
            V_new = self.W_v(value_x)
            K_new = rearrange(K_new, 'b n (h d_k) -> b h n d_k', h=self.nhead)
            V_new = rearrange(V_new, 'b n (h d_k) -> b h n d_k', h=self.nhead)
            if self.cache_k is None:
                self.cache_k, self.cache_v = K_new, V_new
            else:
                self.cache_k = torch.cat([self.cache_k, K_new], dim=2)
                self.cache_v = torch.cat([self.cache_v, V_new], dim=2)
            K, V = self.cache_k, self.cache_v
        elif kvcache == "cross":
            if self.cache_k is None:
                K_new = self.W_k(key_x)
                V_new = self.W_v(value_x)
                self.cache_k = rearrange(K_new, 'b n (h d_k) -> b h n d_k', h=self.nhead)
                self.cache_v = rearrange(V_new, 'b n (h d_k) -> b h n d_k', h=self.nhead)
            K, V = self.cache_k, self.cache_v
        else:
            K = self.W_k(key_x)
            V = self.W_v(value_x)
            K = rearrange(K, 'b n (h d_k) -> b h n d_k', h=self.nhead)
            V = rearrange(V, 'b n (h d_k) -> b h n d_k', h=self.nhead)

        attention = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        if kvcache is None or kvcache == "cross":
            attention = attention + padding_mask
        if tgt_mask is not None and kvcache is None:
            attention = attention + tgt_mask
        attention = torch.nn.functional.softmax(attention, dim = -1)
        attention = self.dropout(attention)
        output = torch.matmul(attention, V)
        output = rearrange(output, 'b h n d_k -> b n (h d_k)', h = self.nhead)
        output = self.W_o(output)
        return output
    
    def reset_cache(self):
        self.cache_k, self.cache_v = None, None

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout = 0.1):
        super(PositionWiseFeedForward, self).__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

def create_masks(src, tgt, tgt_len):
    src_padding_mask = (src != PAD_IDX).unsqueeze(1).unsqueeze(2)
    tgt_padding_mask = (tgt != PAD_IDX).unsqueeze(1).unsqueeze(2)
    causal_mask = torch.tril(torch.ones((tgt_len, tgt_len), device = DEVICE)).bool().unsqueeze(0).unsqueeze(1)
    def to_additive(mask):
        return mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)

    return to_additive(src_padding_mask), to_additive(tgt_padding_mask), to_additive(causal_mask)
    