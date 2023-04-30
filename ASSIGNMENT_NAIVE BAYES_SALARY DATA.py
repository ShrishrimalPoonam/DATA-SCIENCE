#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


test= pd.read_csv(r"E:\360DIGITMG\ASSIGNMENT\ASSIGNMENT12\SalaryData_Test.csv")


# In[3]:


train= pd.read_csv(r"E:\360DIGITMG\ASSIGNMENT\ASSIGNMENT12\SalaryData_Train.csv")


# In[4]:


train['Salary'] = train['Salary'].replace({' <=50K':0, ' >50K':1})


# In[5]:


test['Salary'] = test['Salary'].replace({' <=50K':0, ' >50K':1})


# In[6]:


test['sex'] = test['sex'].replace({'Male':0, 'Female':1})


# In[7]:


train['sex'] = train['sex'].replace({' Male':0, ' Female':1})


# In[8]:


test['sex'] = test['sex'].replace({' Male':0, ' Female':1})


# In[9]:


train.drop(['relationship','educationno'], axis=1, inplace=True)
test.drop(['relationship','educationno'], axis=1, inplace=True)


# In[10]:


# Label Encoder
from sklearn.preprocessing import LabelEncoder
# creating instance of labelencoder
labelencoder = LabelEncoder()


# In[11]:


#Label encode train and test data
train['workclass']= labelencoder.fit_transform(train['workclass'])
train['race'] = labelencoder.fit_transform(train['race'])
train['native'] = labelencoder.fit_transform(train['native'])
train['maritalstatus'] = labelencoder.fit_transform(train['maritalstatus'])
train['occupation'] = labelencoder.fit_transform(train['occupation'])
train['education'] = labelencoder.fit_transform(train['education'])
train.head()


# In[12]:


#Label encode train and test data
test['workclass']= labelencoder.fit_transform(test['workclass'])
test['race'] = labelencoder.fit_transform(test['race'])
test['native'] = labelencoder.fit_transform(test['native'])
test['maritalstatus'] = labelencoder.fit_transform(test['maritalstatus'])
test['occupation'] = labelencoder.fit_transform(test['occupation'])
test['education'] = labelencoder.fit_transform(test['education'])
test.head()


# In[13]:


# Data Split into Input and Output variables
x_train = train.iloc[:, 0:11]
y_train= train['Salary']


# In[14]:


# Data Split into Input and Output variables
x_test = test.iloc[:, 0:11]
y_test= test['Salary']


# In[15]:


from sklearn.naive_bayes import GaussianNB
model = GaussianNB()


# In[16]:


model.fit(x_train, y_train)


# In[17]:


model.score(x_test, y_test)


# In[18]:


model.predict(x_test[:10])


# In[19]:


#lets compare with actual values
y_test[:10]


# In[20]:


# we see it predicted the index 3 value as wrong. # based on our accuracy score it is acceptable

