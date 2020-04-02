#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import json


# In[2]:


#model design
class model (nn.Module):
    
    def __init__ (self):
        super().__init__()
        self.hidden_layer = nn.LSTM(input_size = 24, hidden_size = 48, num_layers = 2, batch_first = True)
        self.linear = nn.Linear(8*48,3)
        self.softmax = nn.Softmax(dim=1)
    def forward (self,X):
        #print(X.shape)
        output, (_,_) = self.hidden_layer(X)
        #print(output.shape)
        output = output.reshape(-1, 8*48)
        output = self.linear(output)
        output_ = self.softmax(output)
        #print(output.shape)
        return output, output_


# In[3]:


model = model()
state_dict = torch.load('new_checkpoint_step8.pth')
model.load_state_dict(state_dict)


# In[4]:


def generateX(path, a, timestep):
    # prepare data with right shape from initial csv file
    data = pd.read_csv(path)
    print(len(data))
    data = data[:a]
    X = data.iloc[:,:24]
    X = np.array(X,dtype = np.float32)
    blocks = int(len(X)/timestep)
    X = np.array(np.split(X,blocks))
    return X


# In[5]:


# generate dataset
class generatedataset(Dataset):
    
    def __init__ (self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        length = len(self.X)
        return length
    
    def __getitem__(self, index):
        X_data = self.X[index]
        y_data = self.y[index]
        X_data = torch.from_numpy(X_data)
        y_data = torch.from_numpy(y_data)
        return X_data, y_data        


# In[6]:


#drawing loss curve
def drawlosscurve(epoch, loss, loss_name):
    font = {
      'family' : 'Bitstream Vera Sans',
      'weight' : 'bold',
      'size'   : 18}
    matplotlib.rc('font', **font)
    width = 12
    height = 12
    f = plt.figure(figsize=(width, height))
    indep_train_axis = np.array(range(1, n_epoch+1, 1))
    plt.plot(indep_train_axis, np.array(loss), "r-.", linewidth=2.0, label="loss")
    print(len(loss))
    plt.title("Loss over epochs")
    plt.legend(loc='upper right', shadow=True)
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.show()
    f.savefig(loss_name + '.png')


# In[7]:


def generate_sample_video_graph(path, length, timestep):
    x = generateX(path, length, timestep)
    y = np.ones(x.shape[0], dtype = np.int32)
    y = y.reshape(y.shape[0], 1)
    dataset = generatedataset(x, y)
    sample_loader = DataLoader(dataset, batch_size= 1, shuffle = False)
    result = []
    i = 0
    n = []
    for X, _ in sample_loader:
        output, output_ = model(X)
        result.append(output_)
        n.append(i)
        i = i + 1
    dic = {}
    def characterization(x):
        indice = torch.argmax(x)
        if indice == 0:
            indice = 'talk to people'
        elif indice == 1:
            indice = 'talk on phone'
        else:
            indice = 'other'
        return '{}, {}'.format(x, indice)
    for i in n:
        dic[i] = characterization(result[i])
    return dic


# In[9]:


result = generate_sample_video_graph('test_5_2.csv',256, 8)
result

