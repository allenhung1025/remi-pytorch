from module import TransformerModel
from utils import *
from glob import glob
import torch
preprocess = datapreprocessing(512, 5, './dictionary.pkl', './checkpoint_path')
midi_paths = glob('./data/train/*00.midi')
segments = preprocess.prepare_data(midi_paths)
ntoken = preprocess.return_token()
#print(segments.shape)
T = TransformerModel(ntoken, embsize = 300, nhead = 4, nlayers = 12, nhid = 512)
segments = torch.tensor(segments)
src = segments[:, :, 0, :]
target = segments[:, :, 1, :]
#print(src.size(), target.size())
src = src.view(-1, src.size(2)) # src size [batchsize, seq_len]
target = target.view(-1,  target.size(2))  # src target [batchsize, seq_len]
#print(src[0])
#print(target[0])
# src [batchsize, seqlen] -> [seqlen, batchsize]
src = src.transpose(1, 0)
target = target.transpose(1, 0)

output = T(src)