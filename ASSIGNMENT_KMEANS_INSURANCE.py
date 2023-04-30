#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pylab as plt


# In[2]:


df=pd.read_csv(r"E:\360DIGITMG\ASSIGNMENT\ASSIGNMENT5\Insurance Dataset.csv")


# In[3]:


df.head()


# In[4]:


df.isnull().sum()


# In[5]:


df.dtypes


# In[6]:


# Normalization function 
def norm_func(i):
    x = (i - i.min())	/ (i.max() - i.min())
    return (x)


# In[8]:


# Normalized data frame (considering the numerical part of data)
df_scaled = norm_func(df)


# In[9]:


TWSS = []             #INITIALIZING TOTAL WITHIN SUM OF SQUARE
k = list(range(2, 9)) # WE DONT TAKE 1 TO 2 AS THERE EXISTS MASSIVE DECREASE ...WE LEARNT IN LEC.


# In[12]:


from sklearn.cluster import	KMeans
# from scipy.spatial.distance import cdist

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_scaled)
    TWSS.append(kmeans.inertia_)


# In[13]:


TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")


# In[14]:


# Selecting  clusters from the above scree plot which is the optimum number of clusters . 
model = KMeans(n_clusters = 4)
model.fit(df_scaled)


# In[15]:


model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
df['clust'] = mb # creating a  new column and assigning it to new column


# In[16]:


df.head()


# In[17]:


# Aggregate mean of each cluster
df.groupby(df.clust).mean()  #grouping by cluster and taking mean


# In[18]:


# creating a csv file 
df.to_csv(r"E:\360DIGITMG\ASSIGNMENT\ASSIGNMENT5\CreatedInsurance.csv", encoding = "utf-8")


# In[ ]:




