# reference: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
class TransformerModel(nn.Module):
    def __init__(self, ntoken, embsize, nhead, nlayers, nhid, dropout = 0.1):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(embsize, dropout)
        self.encoder_layer = TransformerEncoderLayer(embsize,  nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, nlayers)
        self.encoder = nn.Embedding(ntoken, embsize)
        self.embsize = embsize
        self.decoder = nn.Linear(embsize, ntoken)
        # different from the original code
        #self.softmax = nn.Softmax(dim=2)
        self.init_weights()
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
    def forward(self, src):
        # src size [seq_len,batch_size]
        # return: output size [seq_len, batch_size, ntoken]
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask
    
        src = self.encoder(src) * math.sqrt(self.embsize)
        #print(src.size())
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        #output = self.softmax(output)
        return output
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        #print(self.pe[:x.size(0), :].size())
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)