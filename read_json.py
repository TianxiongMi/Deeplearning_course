#!/usr/bin/env python
# coding: utf-8

# In[113]:


import os
import pandas as pd
import json
import numpy as np


# In[129]:


k = os.listdir('C:/Users/mickey/Desktop/both/content/openpose/output_json_5')
n = []
for i in range(len(k)-1):
    print(k[i])
    f = open('C:/Users/mickey/Desktop/both/content/openpose/output_json_5/'+str(k[i]),'r')
    j = json.load(f)
    y = j['people']
    x = pd.DataFrame.from_dict(y)
    if x.empty != True:
        z = x.iloc[:,7]
        z = list(z)
        z = np.array(z)
        p = len(z)
        if p == 1:
            a = []
            for i in range(len(z[0])):
                if (i+1)%3 != 0:
                    a.append(z[0][i])
            n.append(a)
        else:
            for i in range(p):
                a = []
                for o in range(len(z[i])):
                    if (o+1)%3 != 0:
                        a.append(z[i][o])
                n.append(a)
l = np.array(n)
type(l)


# In[130]:


df = pd.DataFrame(l)
df_1 = df[df.columns[0:16]]
df_2 = df[df.columns[30:38]]
df_3 = pd.concat([df_1,df_2],axis = 1,ignore_index = True)
df_3


# In[131]:


s = np.ones(594)
s = np.expand_dims(s,axis = 1)
k = np.zeros(594)
k = np.expand_dims(k,axis = 1)
s = pd.DataFrame(s)
k = pd.DataFrame(k)
final = pd.concat([df_3,s,k], axis = 1, ignore_index = True)
final.to_csv('both.csv')

