#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np
import random
from random import randint
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import json
import torch.nn.functional as F
import sys


# In[2]:


if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")


# In[11]:


def generateX(path, a, timestep):
    # prepare data with right shape from initial csv file
    data = pd.read_csv(path)
    print(len(data))
    data = data[:a]
    X = data.iloc[:,:]
    X = np.array(X,dtype = np.float32)
    blocks = int(len(X)/timestep)
    X = np.array(np.split(X,blocks))
    return X


# In[3]:


# generate dataset
class generatedataset(Dataset):
    
    def __init__ (self, x_body, x_face, y):
        self.x_body = x_body
        self.x_face = x_face
        self.y = y
        
    def __len__(self):
        length = len(self.y)
        return length
    
    def __getitem__(self, index):
        x_body = self.x_body[index]
        x_face = self.x_face[index]
        y_data = self.y[index]
        x_body = torch.from_numpy(x_body)
        x_face = torch.from_numpy(x_face)
        y_data = torch.from_numpy(y_data)
        return x_body, x_face, y_data       


# In[21]:


class model(nn.Module):
    
    def __init__ (self):
        super().__init__()
        self.rnn1 = nn.LSTM(input_size = 24, hidden_size = 512, num_layers = 2, batch_first = True)
        self.rnn2 = nn.LSTM(input_size = 40, hidden_size = 512, num_layers = 2, batch_first = True)
        self.fc1 = nn.Linear(8*512,128)
        self.fc2 = nn.Linear(8*512,128)
        self.fc3 = nn.Linear(128,3)
        self.dropout = nn.Dropout(p = 0.2)
    def forward (self,x_body,x_face):
        output_body, (_,_) = self.rnn1(x_body)
        output_body = output_body.reshape(-1, 8*512)
        output_body = self.fc1(output_body)
        #print(output_body)
        output_face, (_,_) = self.rnn2(x_face)
        output_face = output_face.reshape(-1,8*512)
        output_face = self.fc2(output_face)
        #print(output_face)
        output = output_body + output_face
        #print(output)
        output = self.dropout(output)
        output = self.fc3(output)
        #print(output)
        return output


# In[15]:


def writetext(x,name):
    f = open(name,'w+')
    f.write(str(x))
    f.close      
def generate_sample_video_graph(model, body_path, face_path, length, timestep,timelength, n_frames):
    body = generateX(body_path, length, timestep)
    face = generateX(face_path, length, timestep)
    y = np.ones(body.shape[0], dtype = np.int32)
    y = y.reshape(y.shape[0], 1)
    dataset = generatedataset(body, face, y)
    sample_loader = DataLoader(dataset, batch_size= 1, shuffle = False)
    result = []
    i = 0
    n = []
    for body, face, _ in sample_loader:
        body = body.to(device)
        face = face.to(device)
        output = model(body, face)
        result.append(output)
        n.append(i)
        i = i + 1
    dic = {}
    def characterization(x):
        indice = torch.argmax(x)
        if indice == 0:
            indice_label = 'talk to people'
            indice_ = 1
        elif indice == 1:
            indice_label = 'talk on phone'
            indice_ = -1
        else:
            indice_label = 'other'
            indice_ = 0
        return '{}, {}'.format(x, indice_label), indice_
    indice = []
    for i in n:
        dic[i], single_indice = characterization(result[i])
        indice.append(single_indice)
    writetext(dic,'timeLabel.json')
    print(indice)
    font = {
      'family' : 'Bitstream Vera Sans',
      'weight' : 'bold',
      'size'   : 18}
    matplotlib.rc('font', **font)
    width = 12
    height = 12
    f = plt.figure(figsize=(width, height))
    x_axis = np.array(range(1, len(result)+1, 1))
    plt.plot(x_axis*(timelength/n_frames)*8, np.array(indice), "r-.", linewidth=1.0, label="0: talktopeople, 1: talkonphone, 2: other")
    plt.title("labels over time")
    plt.ylabel('label')
    plt.xlabel('time')
    plt.show()
    f.savefig('timeLabel.png')
    return dic, x_axis*(timelength/n_frames)*8, np.array(indice)


# In[14]:


def generate_comparative_video_graph(x1,y1,x2,y2, figurename):
    font = {
      'family' : 'Bitstream Vera Sans',
      'weight' : 'bold',
      'size'   : 18}
    matplotlib.rc('font', **font)
    width = 12
    height = 12
    f = plt.figure(figsize=(width, height))
    plt.plot(x1, y1, "r-.", linewidth=1.5, label="0: talktopeople, 1: talkonphone, 2: other")
    plt.plot(x2, y2, "b-", linewidth=1.0, label="0: talktopeople, 1: talkonphone, 2: other")
    plt.title("labels over time")
    plt.ylabel('label')
    plt.xlabel('time')
    plt.show()
    f.savefig(figurename + '.png')


# In[13]:


def get_parameters(body_path):
    data = pd.read_csv(body_path)
    length = len(data)
    input_length = int(length/8)*8
    return length, input_length


# In[22]:


if __name__== "__main__":
    #learning rate change 
    model = model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay = 0.05)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[250, 1000,1250, 1500], gamma=0.5)
    loss_fn = nn.CrossEntropyLoss()
    state_dict = torch.load('checkpoint_add_MDHB_2nd.pth')
    model.load_state_dict(state_dict)
    length, input_length = get_parameters(sys.argv[1])
    result, x, y = generate_sample_video_graph(model, sys.argv[1], sys.argv[2], input_length, 8, int(sys.argv[3]), length)


# In[14]:


#length, input_length = get_parameters('sample_body.csv')


# In[16]:


#result, x, y = generate_sample_video_graph(model, 'sample_body.csv', 'sample_face.csv', input_length, 8, 140, length)

