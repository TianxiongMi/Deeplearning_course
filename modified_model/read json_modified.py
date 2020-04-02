#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import json
import numpy as np


# In[27]:


s = []


# In[28]:


k = os.listdir('C:/Users/mickey/Desktop/deeplearning_homework/both/content/openpose/output_json_5')
n_1 = []
for i in range(len(k)-1):
    print(k[i])
    f = open('C:/Users/mickey/Desktop/deeplearning_homework/both/content/openpose/output_json_5/'+str(k[i]),'r')
    j = json.load(f)
    y = j['people']
    x = pd.DataFrame.from_dict(y)
    if x.empty != True:
        z = x.iloc[:,7]
        z = list(z)
        z = np.array(z)
        #print(z)
        p = len(z)
        #print(p)
        if p == 2:
            a = []
            for i in range(len(z[1])):
                if (i+1)%3 != 0:
                    a.append(z[1][i])
            n_1.append(a)
        #else:
            #for i in range(p):
                #a = []
                #for o in range(len(z[i])):
                    #if (o+1)%3 != 0:
                        #a.append(z[i][o])
                #n.append(a)
s +=n_1


# In[29]:


l = np.array(s)
df = pd.DataFrame(l)
df_1 = df[df.columns[0:16]]
df_2 = df[df.columns[30:38]]
df_3 = pd.concat([df_1,df_2],axis = 1,ignore_index = True)
df_3


# In[30]:


df_3.to_csv('test_5_2.csv')

