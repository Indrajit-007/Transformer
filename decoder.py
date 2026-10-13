import torch
from torch import nn
from util.embedder import Embedder
from util.positional_encoder import Positional_Encoder
from util.norm import Norm
from util.feed_forward import FeedForward
from util.multi_head_attention import Multi_Head_Attention
import copy

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Decoder_Layer(nn.Module):
    def __init__(self, d_model, heads, dropout = 0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)
        self.attn = Multi_Head_Attention(heads, d_model, dropout)
        self.msk_attn = Multi_Head_Attention(heads, d_model, dropout)
        self.feedf = FeedForward(d_model).cuda()
        self.dropout_1 = nn.Dropout(dropout).cuda()
        self.dropout_2 = nn.Dropout(dropout).cuda()
        self.dropout_3 = nn.Dropout(dropout).cuda()

    def forward(self, x, e_outputs, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.msk_attn(x2,x2,x2,mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn(x2,e_outputs,e_outputs,mask=None))
        x2 = self.norm_3(x)
        x = x +self.dropout_3(self.feedf(x2))
        return x

class Decoder(nn.Module):
    def __init__(self, d_model, N, heads, dropout = 0.1):
        super().__init__()
        self.N = N
        self.embed = Embedder(d_model)
        self.pe = Positional_Encoder(d_model,1)
        self.layers = get_clones(Decoder_Layer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, trg, e_outputs, mask = None):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x,e_outputs,mask)
        return self.norm(x)
