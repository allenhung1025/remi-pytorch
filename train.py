from module import TransformerModel
from utils import *
from glob import glob
import torch
import torch.nn as nn
import torch.optim as  optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from tensorboardX import  SummaryWriter
import os
####################################################################################################
#######       reference: https://github.com/soobinseo/Transformer-TTS/blob/master/train_transformer.py        #########
####################################################################################################
# adjust_learning_rate is very effectivein training transformer. The training loss is decreasing stably
def adjust_learning_rate(optimizer, step_num, lr, warmup_step=4000):
    lr = lr * warmup_step**0.5 * min(step_num * warmup_step**-1.5, step_num**-0.5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
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
        midi_paths = glob('./data/train/*.midi')
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
        # prepare summary writer
        writer = SummaryWriter('run/exp-adjusting_learing_rate_')
        # prepare the dataloader
        midi_dataloader = self.return_dataloader()
        # prepare the transformer model
        T = TransformerModel(self.ntoken, self.embsize, self.nhead, self.nlayers, self.nhid).to(device)
        T.train()
        # define cross entropy loss as the loss function
        CE_loss = nn.CrossEntropyLoss()
        lr = 0.001
        optimizer = optim.Adam(T.parameters(), lr = lr)
        
        global_step = 0
        for iteration in range(self.epoch):
            #all_loss = 0.0
            for i, (src, target) in enumerate(midi_dataloader):
                global_step += 1
                # adjust learning rate
                if global_step < 40000:
                    adjust_learning_rate(optimizer, global_step, lr)
                
                optimizer.zero_grad()

                src, target =  src.to(device), target.to(device)
                # debugging
                #print(src)
                #print(target)
                src = src.transpose(1, 0)
                
                output = T(src)
                output = output.permute(1, 0, 2)
                output = output.reshape(-1, output.size(2))
                #import pdb; pdb.set_trace()
                #print(output.size())
                #print(output.size())
                #print(torch.sum(output[0, 0, :]), output[0, 0, :])
                target = target.view(-1)
                loss = CE_loss(output, target)
                #print(loss.item())
                #print(loss.item() / target.size(0))
                #all_loss += loss
                #print(loss.item())
                print('epoch: {:03d}, step: {:03d}, loss: {:4f}'.format(iteration, i, loss))
                
                # write loss to the tensorboardx
                writer.add_scalar('loss', loss.item(), iteration * len(midi_dataloader) + i)

                loss.backward()
                optimizer.step()
            
            #all_loss /= len(midi_dataloader)
            # save the model
            if (iteration + 1) % 100 == 0 or iteration == 0: 
                torch.save({
                    'epoch': self.epoch,
                    'model_state_dict': T.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                     }, os.path.join(self.checkpoint_path, 'epoch-' + str(iteration + 1) + '.pkl'))
if __name__ == '__main__':
    training_class = training_class(512, 5, './dictionary.pkl', './checkpoint/', 500, 16, 300, 4, 12, 512)
    training_class.train()
    #for src, target in x:
    #   print(src, target)