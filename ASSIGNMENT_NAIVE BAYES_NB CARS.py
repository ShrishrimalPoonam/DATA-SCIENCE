#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv(r"E:\360DIGITMG\ASSIGNMENT\ASSIGNMENT12\NB_Car_Ad.csv")


# In[3]:


df.head()


# In[4]:


df.isnull().sum()


# In[8]:


df.drop(['User ID'], axis = 1, inplace = True)


# In[9]:


df.head()


# In[11]:


target = df.Purchased
inputs = df.drop('Purchased', axis=1)


# In[12]:


dummies = pd.get_dummies(inputs.Gender)
dummies.head()


# In[14]:


inputs = pd.concat([inputs, dummies], axis = 1)


# In[15]:


inputs.head()


# In[16]:


inputs.drop(['Gender'], axis=1, inplace=True)


# In[17]:


inputs.head()


# In[20]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(inputs, target, test_size=0.2)


# In[27]:


from sklearn.naive_bayes import GaussianNB
model = GaussianNB()


# In[28]:


model.fit(x_train, y_train)


# In[29]:


model.score(x_test, y_test)


# In[30]:


model.predict(x_test[:10])


# In[32]:


#lets compare with actual values
y_test[:10]


# In[ ]:


# we see it predicted the last one wrong. # based on our accuracy score it is acceptable


# In[34]:


model.predict_proba(x_test[:10])


# In[ ]:




