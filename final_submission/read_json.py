#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import json
import numpy as np
import sys


# In[2]:


body = []
face = []


# In[5]:


k = os.listdir(sys.argv[1])
k.sort()
n_b = []
n_f = []
for i in range(len(k)-1):
    print(k[i])
    f = open(sys.argv[1]+ '/'+str(k[i]),'r')
    j = json.load(f)
    #print(j)
    y = j['people']
    x = pd.DataFrame.from_dict(y)
    #print(x)
    if x.empty != True:
        
        z_1 = x.iloc[:,1]
        #print(z_1)
        z_1 = list(z_1)
        #print(z_1)
        z_1 = np.array(z_1)
        #print(z_1)
        #print(z_1)
        z_2 = x.iloc[:,2]
        #print(z_2)
        z_2 = list(z_2)
        #print(z_2)
        z_2 = np.array(z_2)
        #print(z_2)
        #print(z)
        p = len(z_1)
        #print(p)
        if p == 1:
            a = []
            b = []
            for i in range(len(z_2[0])):
                if (i+1)%3 != 0:
                    b.append(z_2[0][i])
            for i in range(len(z_1[0])):
                if (i+1)%3 != 0:
                    a.append(z_1[0][i])
            n_b.append(a)
            n_f.append(b)
        #else:
            #for i in range(p):
                #a = []
                #for o in range(len(z[i])):
                    #if (o+1)%3 != 0:
                        #a.append(z[i][o])
                #n.append(a)
body += n_b
face += n_f


# In[6]:


l_body = np.array(body)
df = pd.DataFrame(l_body)
df_1 = df[df.columns[0:16]]
df_2 = df[df.columns[30:38]]
df_body = pd.concat([df_1,df_2],axis = 1,ignore_index = True)
l_face = np.array(face)
df_face = pd.DataFrame(l_face)
df_face = pd.DataFrame(df_face.iloc[:,96:136])
print(df_body.shape,df_face.shape, df_body,df_face)


# In[7]:


df_body.to_csv('body.csv', index = False)
df_face.to_csv('face.csv', index = False)

