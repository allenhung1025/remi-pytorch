from module import TransformerModel
from utils import *
from glob import glob
import torch
import torch.nn as nn
import torch.optim as  optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
device = torch.device('cuda:1')
class midi_dataset(Dataset):
    def __init__(self, data_src, data_target):
        self.data_src = data_src #[seqlen, bs]
        self.data_target = data_target #[seqlen, bs]
    def __getitem__(self, idx):
        return self.data_src[:, idx], self.data_target[:, idx]
    def __len__(self):
        return self.data_src.size(1)
    
class training_class(object):
    def __init__(self, x_len, group_size,pickle_path, checkpoint_path, epoch, batchsize, \
                                        embsize, nhead, nlayers, nhid):
        self.x_len = x_len
        self.group_size = group_size
        self.pickle_path = pickle_path
        self.checkpoint_path = checkpoint_path
        self.epoch = epoch
        self.batchsize = batchsize
        self.embsize = embsize
        self.nhead = nhead
        self.nlayers = nlayers
        self.nhid = nhid
    def get_training_data(self):
        preprocess = datapreprocessing(self.x_len, self.group_size, self.pickle_path, self.checkpoint_path)
        midi_paths = glob('./data/train/*00.midi')
        segments = preprocess.prepare_data(midi_paths)
        self.ntoken = preprocess.return_token()
        segments = torch.tensor(segments)
        src = segments[:, :, 0, :]
        target = segments[:, :, 1, :]   
        # [batchsize, seq_len]
        src = src.view(-1, src.size(2))  
        target = target.view(-1,  target.size(2))  
        # [batchsize, seqlen] -> [seqlen, batchsize]
        src = src.transpose(1, 0)
        target = target.transpose(1, 0)
        return src, target
    def return_dataloader(self):
        src, target = self.get_training_data()
        midi_dataset_ = midi_dataset(src, target)
        midi_dataloader_ = DataLoader(midi_dataset_, batch_size = self.batchsize, shuffle = True, num_workers=10, drop_last=False)
        return midi_dataloader_
    def train(self):
        # prepare the dataloader
        midi_dataloader = self.return_dataloader()
        # prepare the transformer model
        T = TransformerModel(self.ntoken, self.embsize, self.nhead, self.nlayers, self.nhid).to(device)
        T.train()
        # define cross entropy loss as the loss function
        CE_loss = nn.CrossEntropyLoss()
        lr = 0.001
        optimizer = optim.Adam(T.parameters(), lr = lr)
        for i, (src, target) in enumerate(tqdm(midi_dataloader)):
            optimizer.zero_grad()
            
            src, target =  src.to(device), target.to(device)
            src = src.transpose(1, 0)
            
            output = T(src)
            output = output.view(-1, output.size(2))
            target = target.view(-1)
            loss = CE_loss(output, target) / self.batchsize
            print(loss)
            #print(src.size(), target.size())
if __name__ == '__main__':
    training_class = training_class(512, 5, './dictionary.pkl', './checkpoint', 20, 4, 300, 4, 12, 512)
    training_class.train()