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


# In[35]:


if torch.cuda.is_available():
    device = torch.device("cuda:1")  # you can continue going on here, like cuda:1 cuda:2....etc. 
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")


# In[2]:


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


# In[42]:


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


# In[47]:


#model design
class model (nn.Module):
    
    def __init__ (self):
        super().__init__()
        self.hidden_layer = nn.LSTM(input_size = 64, hidden_size = 1024, num_layers = 3, batch_first = True)
        self.linear = nn.Linear(8*1024,3)
        self.softmax = nn.Softmax(dim=1)
    def forward (self,X):
        #print(X.shape)
        output, (_,_) = self.hidden_layer(X)
        #print(output.shape)
        output = output.reshape(-1, 8*1024)
        output = self.linear(output)
        output_ = self.softmax(output)
        #print(output.shape)
        return output, output_


# In[48]:


model = model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001, weight_decay = 0.005)
loss_fn = nn.CrossEntropyLoss()


# In[51]:


n_epoch = 1000
def training_loop(n_epochs, optimizer, model, loss_fn, train_loader, val_loader):
    highest = 0
    train_acc = []
    val_acc = []
    loss_train_plot = []
    loss_val_plot = []
    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        total_train = len(train_loader)*1353
        corr_train = 0
        for X_train, y in train_loader:
            X_train = X_train.to(device)
            y = y.to(device)
            outputs, output_ = model(X_train)
            y = y.type(torch.LongTensor)
            y = y.squeeze()
            loss = loss_fn(outputs, y.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            indices = torch.argmax(output_, dim=1)            
            for i in range(len(indices)):
                if y[i] == indices[i]:
                    corr_train += 1
        with torch.no_grad():
                total_val = 1353
                corr_val = 0
                y_pred, y_pred_ = model(example_data)
                y_val = example_targets.type(torch.LongTensor)
                y_val = y_val.squeeze()
                loss_val = loss_fn(y_pred, y_val.to(device))
                indices_val = torch.argmax(y_pred_, dim=1)  
                for i in range(len(y_val)):
                    if y_val[i] == indices_val[i]:
                        corr_val += 1
        acc_train = corr_train/total_train
        acc_val = corr_val/total_val
        if acc_val > highest:
            highest = acc_val
            torch.save(model.state_dict(), 'checkpoint_alldata_8timestep_optimizedata.pth')
        train_acc.append(acc_train)
        val_acc.append(acc_val)
        loss_train_plot.append(loss_train)
        loss_val_plot.append(loss_val)
        print('Epoch {}, Training loss {}, Validation loss {}, Training_accuracy {}, Validation_accuracy {}, highest accuracy {}'.format(epoch, float(loss_train),float(loss_val), acc_train, acc_val, highest))
    font = {
      'family' : 'Bitstream Vera Sans',
      'weight' : 'bold',
      'size'   : 18}
    matplotlib.rc('font', **font)
    width = 12
    height = 12
    f = plt.figure(figsize=(width, height))
    indep_train_axis = np.array(range(1, n_epoch+1, 1))
    plt.plot(indep_train_axis, np.array(train_acc), "g--", label="Train accuracies")
    plt.plot(indep_train_axis, np.array(val_acc), "b-", linewidth=2.0, label="Validation accuracies")
    print(len(val_acc))
    print(len(train_acc))
    print(len(loss_train_plot))
    plt.title("Training session's Accuracy over epochs")
    plt.legend(loc='lower right', shadow=True)
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.show()
    f.savefig('accuracy_8_optimizedata.png')
    return loss_train_plot, loss_val_plot


# In[53]:


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


# In[55]:


state_dict = torch.load('checkpoint_alldata_8timestep_optimizedata.pth')
model.load_state_dict(state_dict)


# In[56]:


#test set result
def test_score(test_loader):
    with torch.no_grad():
        test_total = 1351
        test_correct = 0
        for X_test,y in test_loader:
            X_test = X_test.to(device)
            output, output_ = model(X_test)
            y = y.type(torch.LongTensor)
            y = y.squeeze()
            indices_test = torch.argmax(output_, dim=1)  
            for i in range(len(y)):
                    if y[i] == indices_test[i]:
                      test_correct += 1
        test_acc = test_correct/test_total
    return test_acc


# In[58]:


def writetext(x,name):
    f = open(name,'w+')
    f.write(str(x))
    f.close       


# In[59]:


def generate_sample_video_graph(path, length, timestep, name):
    x = generateX(path, length, timestep)
    y = np.ones(x.shape[0], dtype = np.int32)
    y = y.reshape(y.shape[0], 1)
    dataset = generatedataset(x, y)
    sample_loader = DataLoader(dataset, batch_size= 1, shuffle = False)
    result = []
    i = 0
    n = []
    for X, _ in sample_loader:
        X = X.to(device)
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
    writetext(dic,name)
    return dic


# In[67]:


result = generate_sample_video_graph('test_csv/test_5_2_bodyface.csv',256, 8, 'test_5_2_alldata_8timestep_optimize.txt')
result

