#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import json
import numpy as np


# In[33]:


body = []
face = []


# In[34]:


k = os.listdir('C:/Users/mickey/Desktop/deeplearning_homework/new_test/test_5/content/openpose/test_5')
n_b = []
n_f = []
for i in range(len(k)-1):
    print(k[i])
    f = open('C:/Users/mickey/Desktop/deeplearning_homework/new_test/test_5/content/openpose/test_5/'+str(k[i]),'r')
    j = json.load(f)
    y = j['people']
    x = pd.DataFrame.from_dict(y)
    if x.empty != True:
        z_1 = x.iloc[:,7]
        z_1 = list(z_1)
        z_1 = np.array(z_1)
        z_2 = x.iloc[:,0]
        z_2 = list(z_2)
        z_2 = np.array(z_2)
        #print(z)
        p = len(z_1)
        #print(p)
        if p == 2:
            a = []
            b = []
            for i in range(len(z_2[1])):
                if (i+1)%3 != 0:
                    b.append(z_2[1][i])
            for i in range(len(z_1[1])):
                if (i+1)%3 != 0:
                    a.append(z_1[1][i])
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


# In[35]:


l_body = np.array(body)
df = pd.DataFrame(l_body)
df_1 = df[df.columns[0:16]]
df_2 = df[df.columns[30:38]]
df_body = pd.concat([df_1,df_2],axis = 1,ignore_index = True)
l_face = np.array(face)
df_face = pd.DataFrame(l_face)
df_face = pd.DataFrame(df_face.iloc[:,96:136])
print(df_body.shape,df_face.shape, df_body,df_face)


# In[99]:


l_body = np.array(body)
df = pd.DataFrame(l_body)
df_1 = df[df.columns[0:16]]
df_2 = df[df.columns[30:38]]
df_body = pd.concat([df_1,df_2],axis = 1,ignore_index = True)
l_face = np.array(face)
df_face = pd.DataFrame(l_face)
print(df_body.shape,df_face.shape, df_body,df_face)


# In[36]:


df_total = pd.concat([df_body,df_face],axis = 1,ignore_index = True)
df_total.shape


# In[37]:


df_total.to_csv('test_5_2_bodyface.csv')

