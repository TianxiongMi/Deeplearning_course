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


# In[183]:


#generate input X
X_1 = generateX('C:/Users/mickey/Desktop/deeplearning_homework/talktopeople_modify_2.csv',67456, 8)
X_2 = generateX('C:/Users/mickey/Desktop/deeplearning_homework/talkbyphone_modify_2.csv',39712,8)
X_3 = generateX('C:/Users/mickey/Desktop/deeplearning_homework/others_2.csv',42112,8)
X = np.concatenate((X_1,X_2,X_3),axis = 0)
print(X.shape)


# In[184]:


def generatey(x,i):
    y = np.ones(x.shape[0],dtype = np.int32)
    Y = np.zeros((y.shape[0],3))
    Y[:,i] = y
    return Y


# In[185]:


#generate labels
Y_1 = np.zeros(X_1.shape[0],dtype = np.int32)
Y_2 = np.ones(X_2.shape[0],dtype = np.int32)
Y_3 = 2*(np.ones(X_3.shape[0], dtype = np.int32))
print(Y_1)
print(Y_2)
print(Y_3)
Y_1 = Y_1.reshape(Y_1.shape[0],1)
Y_2 = Y_2.reshape(Y_2.shape[0],1)
Y_3 = Y_3.reshape(Y_3.shape[0],1)
Y = np.concatenate((Y_1,Y_2,Y_3), axis = 0)
Y


# In[187]:


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


# In[188]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=1/10)


# In[189]:


dataset = generatedataset(X_train, y_train)
# training set and validation set seperation
trainset, valset = random_split(dataset, [14928, 1866])


# In[190]:


train_loader = DataLoader(trainset, batch_size=1866, shuffle=True)
val_loader = DataLoader(valset, batch_size =1866, shuffle = False)


# In[191]:


#generate validation data
examples = enumerate(val_loader)
batch_idx, (example_data, example_targets) = next(examples)


# In[192]:


type(example_targets)


# In[193]:


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


# In[194]:


model = model()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0025, weight_decay = 0.0015)
loss_fn = nn.CrossEntropyLoss()


# In[195]:


n_epoch = 200
def training_loop(n_epochs, optimizer, model, loss_fn, train_loader, val_loader):
    train_acc = []
    val_acc = []
    loss_train_plot = []
    loss_val_plot = []
    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        total_train = len(train_loader)*1866
        corr_train = 0
        for X_train, y in train_loader:
            outputs, output_ = model(X_train)
            y = y.type(torch.LongTensor)
            y = y.squeeze()
            loss = loss_fn(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            indices = torch.argmax(output_, dim=1)            
            for i in range(len(indices)):
                if y[i] == indices[i]:
                    corr_train += 1
        with torch.no_grad():
                total_val = 1866
                corr_val = 0
                y_pred, y_pred_ = model(example_data)
                y_val = example_targets.type(torch.LongTensor)
                y_val = y_val.squeeze()
                loss_val = loss_fn(y_pred, y_val)
                indices_val = torch.argmax(y_pred_, dim=1)  
                for i in range(len(y_val)):
                    if y_val[i] == indices_val[i]:
                        corr_val += 1
        acc_train = corr_train/total_train
        acc_val = corr_val/total_val
        train_acc.append(acc_train)
        val_acc.append(acc_val)
        loss_train_plot.append(loss_train)
        loss_val_plot.append(loss_val)
        print('Epoch {}, Training loss {}, Validation loss {}, Training_accuracy {}, Validation_accuracy {}'.format(epoch, float(loss_train),float(loss_val), acc_train, acc_val))
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
    f.savefig('accuracy_16.png')
    return loss_train_plot, loss_val_plot


# In[196]:


trainingloss, valloss = training_loop(n_epoch, optimizer, model, loss_fn, train_loader, val_loader)


# In[197]:


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


# In[198]:


drawlosscurve(n_epoch,trainingloss, "training_loss_2")
drawlosscurve(n_epoch,valloss,"Validation_loss_2")


# In[199]:


#test set result
def test_score(test_loader):
    with torch.no_grad():
        test_total = 933*2
        test_correct = 0
        for X_test,y in test_loader:
            output, output_ = model(X_test)
            y = y.type(torch.LongTensor)
            y = y.squeeze()
            indices_test = torch.argmax(output_, dim=1)  
            for i in range(len(y)):
                    if y[i] == indices_test[i]:
                      test_correct += 1
        test_acc = test_correct/test_total
    return test_acc


# In[200]:


test_dataset = generatedataset(X_test,y_test)
test_loader = DataLoader(test_dataset, batch_size=933, shuffle=True)
test_score(test_loader)


# In[201]:


def generate_sample_video_graph(path, length, timestep,name):
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


# In[212]:


result = generate_sample_video_graph('test_5_2.csv',256, 8, 'test_1.png')
result


# In[211]:


torch.save(model.state_dict(), 'new_checkpoint_step8.pth')

