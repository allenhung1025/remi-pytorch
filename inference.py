import torch
import torch.nn as nn
from utils import *
import numpy as np
from module import TransformerModel
device = torch.device("cuda:1")
########################################
# temperature sampling
########################################
def temperature_sampling(logits, temperature, topk):
    probs = nn.Softmax(dim = 0)(logits / temperature)
    if topk == 1:
        prediction = torch.argmax(probs)
    else:
        sorted_index = torch.argsort(probs)
        sorted_index = sorted_index.numpy()[::-1]
        candi_index = sorted_index[:topk]
        candi_probs = [probs[i].numpy() for i in candi_index]
        # normalize probs
        candi_probs /= sum(candi_probs)
        # choose by predicted probs
        prediction = np.random.choice(candi_index, size=1, p=candi_probs)[0]
    return prediction
#logits = torch.tensor([1.0,2.0, 3.0, 4.0])
#temperature_sampling(logits, 0.5, 2)
########################################
# generate
########################################
def generate(n_target_bar, output_path, prompt=None):
    ## Initialize the prompt
    checkpoint_path = './checkpoint/epoch-500.pkl'
    preprocess = datapreprocessing(512, 5, './dictionary.pkl', checkpoint_path)
    if prompt:
        events = preprocess.extract_events(prompt)
        words = [[preprocess.event2word['{}_{}'.format(e.name, e.value)] for e in events]]
        words[0].append(preprocess.event2word['Bar_None'])
    else:
        words = []
        for _ in range(1):
            ws = [preprocess.event2word['Bar_None']]
            if 'chord' in checkpoint_path:
                tempo_classes = [v for k, v in preprocess.event2word.items() if 'Tempo Class' in k]
                tempo_values = [v for k, v in preprocess.event2word.items() if 'Tempo Value' in k]
                chords = [v for k, v in preprocess.event2word.items() if 'Chord' in k]
                ws.append(preprocess.event2word['Position_1/16'])
                ws.append(np.random.choice(chords))
                ws.append(preprocess.event2word['Position_1/16'])
                ws.append(np.random.choice(tempo_classes))
                ws.append(np.random.choice(tempo_values))
            else:
                tempo_classes = [v for k, v in preprocess.event2word.items() if 'Tempo Class' in k]
                tempo_values = [v for k, v in preprocess.event2word.items() if 'Tempo Value' in k]
                ws.append(preprocess.event2word['Position_1/16'])
                ws.append(np.random.choice(tempo_classes))
                ws.append(np.random.choice(tempo_values))
            words.append(ws)
        # initialize the model
        ntoken = len(preprocess.event2word)
        embsize = 300
        nhead = 4
        nlayers = 12 
        nhid = 512
        model = TransformerModel(ntoken, embsize, nhead, nlayers, nhid).to(device)
        # load the model
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        # inference the remaing sequences from the prompt and the model
        original_length = len(words[0])
        initial_flag = 1
        current_generated_bar = 0
        #words = torch.LongTensor(words).to(device)
        #words = words.transpose(0, 1) # [[seqlen, 1]], the first Bar
        #output = model(words)
        #next_token = output[-1, :]
        #next_token = torch.argmax(next_token).item()
        temperature = 1.2
        topk = 5
        with torch.no_grad():
            words = torch.LongTensor(words).to(device)
            words = words.transpose(0, 1) # [[seqlen, 1]], the first Bar
            while current_generated_bar < n_target_bar:
                output = model(words)
                logits = output[-1, :].detach().cpu()
                logits  = logits.squeeze(0)
                next_token = temperature_sampling(
                    logits  = logits,
                    temperature = temperature,
                    topk = topk
                )
                next_token = torch.tensor([[next_token]]).to(device)
                words = torch.cat((words, next_token), dim=0)
                if next_token == preprocess.event2word['Bar_None']:
                    current_generated_bar += 1
        word_list = words.squeeze(1).tolist()
        # write
        if prompt:
            write_midi(
                words=word_list[original_length:],
                word2event=preprocess.word2event,
                output_path=output_path,
                prompt_path=prompt)
        else:
            write_midi(
                words=word_list,
                word2event=preprocess.word2event,
                output_path=output_path,
                prompt_path=None)
        #import pdb; pdb.set_trace()
generate(16,  './hello.midi', None)