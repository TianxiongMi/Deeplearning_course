#!/usr/bin/env python
# coding: utf-8

# In[97]:


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


# In[288]:


def generateX(path, a, timestep):
    data = pd.read_csv(path)
    data = data[:a]
    X = data.iloc[:,:24]
    X = np.array(X,dtype = np.float32)
    blocks = int(len(X)/timestep)
    X = np.array(np.split(X,blocks))
    return X


# In[289]:


X_1 = generateX('talktopeople_2.csv',72448, 16)
X_2 = generateX('talkbyphone_2.csv',41408,16)
X = np.concatenate((X_1,X_2),axis = 0)


# In[290]:


Y_1 = np.ones(X_1.shape[0],dtype = np.int32)
Y_2 = np.zeros(X_2.shape[0],dtype = np.int32)
Y_1 = Y_1.reshape(4528,1)
Y_2 = Y_2.reshape(2588,1)
Y = np.concatenate((Y_1,Y_2), axis = 0)


# In[291]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=1/6)


# In[106]:


type(X_train[0][0][0])


# In[125]:


y_train.shape


# In[292]:


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


# In[293]:


dataset = generatedataset(X_train, y_train)
trainset, valset = random_split(dataset, [4744, 1186])


# In[294]:


train_loader = DataLoader(trainset, batch_size=1186, shuffle=True)
val_loader = DataLoader(valset, batch_size =1186, shuffle = False)


# In[295]:


examples = enumerate(val_loader)
batch_idx, (example_data, example_targets) = next(examples)
example_targets.type()


# In[313]:


class model (nn.Module):
    
    def __init__ (self):
        super().__init__()
        self.hidden_layer = nn.LSTM(input_size = 24, hidden_size = 48, num_layers = 2, batch_first = True)
        self.linear = nn.Linear(16*48,1)
        self.sigmoid = nn.Sigmoid()
    def forward (self,X):
        output, (_,_) = self.hidden_layer(X)
        #print(output.shape)
        output = output.reshape(-1, 16*48)
        output = self.linear(output)
        output = self.sigmoid(output)
        #print(output.shape)
        return output


# In[314]:


model = model()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0025, weight_decay = 0.0015)
loss_fn = nn.BCELoss()


# In[315]:


n_epoch = 300
def training_loop(n_epochs, optimizer, model, loss_fn, train_loader, val_loader):
    train_acc = []
    val_acc = []
    loss_train_plot = []
    loss_val_plot = []
    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        for X_train, y in train_loader:
            outputs = model(X_train)
            y = y.type(torch.FloatTensor)
            y = y.squeeze()
            loss = loss_fn(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            y_pred_ = []
            outputs = outputs.detach().numpy()
            for j in range(len(outputs)):
                if outputs[j] > 0.5:
                    y_pred_.append(1)
                else:
                    y_pred_.append(0)
            #print(y_pred_)
            whole = len(outputs)
            y_pred_ = np.array(y_pred_)
            correct_ = 0
            true = y.numpy()
            #print(true)
            for j in range(len(outputs)):
                k = np.array_equal(y_pred_[j],true[j])
                #print(y_pred_[j],true[j])
                if k == True:
                    correct_ +=1
            ri = correct_/whole
        with torch.no_grad():
                y_pred = model(example_data)
                y_p = []
                loss_val = loss_fn(y_pred, example_targets.type(torch.FloatTensor))
                for i in range(len(y_pred)):
                    if y_pred[i] > 0.5:
                        y_p.append(1)
                    else:
                        y_p.append(0)
                total = len(y_p)
                y_p = np.array(y_p)
                correct = 0
                true = example_targets.numpy()
                #print(len(y_pred),len(y_p),len(example_targets))
                for i in range(len(y_p)):
                    #print(i)
                    #print(y_p[i],true[i])
                    z = np.array_equal([y_p[i]],true[i])
                    if z == True:
                        correct +=1
                r = correct/total
        train_acc.append(ri)
        val_acc.append(r)
        loss_train_plot.append(loss_train)
        loss_val_plot.append(loss_val)
        print('Epoch {}, Training loss {}, Validation loss {}, Training_accuracy {}, Validation_accuracy {}'.format(epoch, float(loss_train),float(loss_val), ri, r))
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


# In[316]:


trainingloss, valloss = training_loop(n_epoch, optimizer, model, loss_fn, train_loader, val_loader)


# In[311]:


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


# In[312]:


drawlosscurve(n_epoch,trainingloss, "training_loss_2")
drawlosscurve(n_epoch,valloss,"Validation_loss_2")


# In[302]:


def test_score(test_loader):
    with torch.no_grad():
        for X_test,y in test_loader:
            output = model(X_test)
            y_test_pre = []
            for i in range(len(output)):
                if output[i] > 0.5:
                    y_test_pre.append(1)
                else:
                    y_test_pre.append(0)
            test_total = len(output)
            y_test_pre = np.array(y_test_pre)
            test_correct = 0
            test_true = y.numpy()
                #print(len(y_pred),len(y_p),len(example_targets))
            for i in range(len(y_test_pre)):
                    #print(i)
                    #print(y_p[i],true[i])
                d = np.array_equal([y_test_pre[i]],test_true[i])
                if d == True:
                    test_correct +=1
                r_test = test_correct/test_total
    return r_test


# In[306]:


test_dataset = generatedataset(X_test,y_test)
test_loader = DataLoader(test_dataset, batch_size=1186, shuffle=True)
test_score(test_loader)

