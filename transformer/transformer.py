import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask

class ScaledDotProductAttention(nn.Module):

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2,3))

        if mask is not None:
            attn = torch.masked_fill(mask == 0, -1e9)
        
        attn = self.dropout(F.softmax(attn, dim=-1))                    # dim=-1 설정해 마지막 차원을 기준으로 적용하라는 뜻
        out = torch.matmul(attn, v)

        return out, attn
    
class MultiHeadAttention(nn.Module):
    
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        """
        n_head : head 수
        d_model: mode dimension
        d_k , d_v : key, value dimenstion
        """

        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v

        # Linear projection of q, k, v
        self.linear_q = nn.Linear(d_model, n_head * d_k, bias=False)
        self.linear_k = nn.Linear(d_model, n_head * d_k, bias=False)
        self.linear_v = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k * 0.5)       # sqrut(v_k) 로 scailing

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        
        residual = q

        q = self.linear_q(q)
        k = self.linear_k(k)
        v = self.linear_v(v)

        batch_size, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        # tensor를 새로운 차원으로 재구성 각 어텐션 헤드가 독립적으로 계산할 수 있도록
        # 기존 Tensor : b x lq x (n x d_q)
        # 변화 : b x lq x n x d_q
        q = q.view(batch_size, len_q, self.n_head, self.d_k)            
        k = k.view(batch_size, len_k, self.n_head, self.d_k)
        v = v.view(batch_size, len_v, self.n_head, self.d_v)

        # dot product 위해 변환 => b x n x lq x d_q
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        out, attn = self.attention(q, k, v, mask=mask)

        out = out.transpose(1, 2).contiquous().view(batch_size, len_q, -1)
        out = self.dropout(self.fc(out))
        out += residual

        out = self.layer_norm(out)

        return out, attn
    
class FeedForward(nn.Module):

    def __init__(self, in_channels, hidden_channels, dropout=0.1):
        super().__init__()

        self.layer1 = nn.Linear(in_channels, hidden_channels)
        self.layer2 = nn.Linear(hidden_channels, in_channels)
        self.layer_norm = nn.LayerNorm(in_channels, eps=1e-16)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        
        residual = x

        x = self.layer2(F.relu(self.layer1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm

        return x
    
class PositionalEncoding(nn.Module):

    def __init__(self):
        super().__init__()

    def _get_sinusoid_encoding_table(self, n_position, d_model):

        def get_position_angle_Vec(position):
            return [position / np.power(10000, 2*(i)/d_model) for i in range(d_model)]
        
        sinusoid_table = np.array([get_position_angle_Vec(j) for j in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, hidden_channels, n_head, d_k, d_v, dropout=0.1):
        super().__init__()

        self.mhattn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.ffn = FeedForward(d_model, hidden_channels, dropout=dropout)

    def forward(self, input, mask=None):
        out, attn = self.mhattn(input, input, input, mask=mask)
        out = self.ffn(out)

        return out, attn

class DecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, hidden_channels, d_k, d_v, dropout=0.1):
        super().__init__()

        self.mhattn1 = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.mhattn2 = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.ffn = FeedForward(d_model, hidden_channels, dropout=dropout)

    def forward(self, input, enc_out, mask1=None, mask2=None):
        out, attn1 = self.mhattn1(input, input, input, mask=mask1)
        out, attn2 = self.mhattn2(out, enc_out, enc_out, mask=mask2)
        out = self.ffn(out)
        
        return out, attn1, attn2
        
class Encoder(nn.Module):
    def __init__(self, n_src_vocab, d_word_vec, n_layer, n_head, d_k, d_v,
                 d_model, hidden_channels, pad_idx, dropout=0.1, n_position=200, scale_emb=False):
        super().__init__()

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(n_position, d_word_vec)
        self.dropout = nn.Dropout(dropout)
        self.layer_stack = nn.ModuleList([EncoderLayer(d_model, hidden_channels, n_head, d_k, d_v, dropout=dropout) for _ in range(n_layer)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, src_seq, src_mask, return_attns=False):

        attn_List = []

        out = self.src_word_emb(src_seq)
        if self.scale_emb:
            out *= self.d_model ** 0.5
        out = self.dropout(self.pos_enc(out))
        out = self.layer_norm(out)

        for layer in self.layer_stack:
            out, attn = layer(out, mask=src_mask)
            attn_List += [attn] if return_attns else []
        
        if return_attns:
            return out, attn_List
        return out
    
class Decoder(nn.Module):
    def __init__(self, n_trg_vocab, d_word_vec, n_layer, n_head, d_k, d_v,
                 d_model, hidden_channels, pad_idx, n_position=200, dropout=0.1, scale_emb=False):
        super().__init__()

        self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(n_position, d_word_vec)
        self.dropout = nn.Dropout(dropout)
        self.layer_stack = nn.ModuleList([DecoderLayer(n_head, d_model, hidden_channels, d_k, d_v, dropout=dropout) for _ in range(n_layer)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, trg_seq, enc_out, trg_mask, src_mask, return_attns=False):
        
        attn_list1, attn_list2 = [], []

        out = self.trg_word_emb(trg_seq)
        if self.scale_emb:
            out += self.d_model ** 0.5
        out = self.pos_enc(out)
        out = self.layer_norm(out)

        for layer in self.layer_stack:
            out, attn1, attn2 = layer(out, enc_out, mask1=trg_mask, mask2=src_mask)
            attn_list1 += [attn1] if return_attns else []
            attn_list2 += [attn2] if return_attns else []

        if return_attns:
            return out, attn_list1, attn_list2
        return out

class Transformer(nn.Module):
    def __init__(self, n_src_vocab, n_trg_vocab, src_pad_idx, trg_pad_idx,
                 d_word_vec=512, d_model=512, hidden_channels=2048, n_layer=6, 
                 n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=200,
                 trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=True,
                 scale_emb_or_prj='prj'):
        super().__init__()

        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx

        assert scale_emb_or_prj in ['emb', 'prj', 'none']

        scale_emb = (scale_emb_or_prj == 'emb') if trg_emb_prj_weight_sharing else False
        self.scale_prj = (scale_emb_or_prj == 'prj') if trg_emb_prj_weight_sharing else False

        self.d_model = d_model

        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, hidden_channels=hidden_channels,
            n_layer=n_layer, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=src_pad_idx, dropout=dropout, scale_emb=scale_emb)

        self.decoder = Decoder(
            n_trg_vocab=n_trg_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, hidden_channels=hidden_channels,
            n_layer=n_layer, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=trg_pad_idx, dropout=dropout, scale_emb=scale_emb)

        self.trg_word_prj = nn.Linear(d_model, n_trg_vocab, bias=False)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        if trg_emb_prj_weight_sharing:
            # Share the weight between target word embedding & last dense layer
            self.trg_word_prj.weight = self.decoder.trg_word_emb.weight

        if emb_src_trg_weight_sharing:
            self.encoder.src_word_emb.weight = self.decoder.trg_word_emb.weight

        def forward(self, src_seq, trg_seq):

            src_mask = get_pad_mask(src_seq, self.src_pad_idx)
            trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)

            enc_out, *_ = self.encoder(src_seq, src_mask)
            dec_out, *_ = self.decoder(trg_mask, trg_mask)
            seq_logit = self.trg_word_prj(out)

            if self.scale_prj:
                seq_logit += self.d_model ** -0.5
            
            return seq_logit.view(-1, seq_logit.size(2))