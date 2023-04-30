#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pylab as plt


# In[2]:


df=pd.read_csv(r"E:\360DIGITMG\ASSIGNMENT\ASSIGNMENT4\crime_data.csv")


# In[3]:


df.head()


# In[4]:


df.isna().sum()


# In[5]:


df.rename(columns={'Unnamed: 0':'State'}, inplace=True)


# In[6]:


df.head()


# In[7]:


df.dtypes


# In[8]:


df.Murder = df.Murder.astype('int64') 
df.Rape = df.Rape.astype('int64') 
df.dtypes


# In[9]:


df1 = df.drop(["State"], axis=1)


# In[10]:


#standardize the data to normal distribution
from sklearn import preprocessing
from sklearn.preprocessing import normalize
df1_scaled=normalize(df1)
df1_scaled = pd.DataFrame(df1_scaled, columns = df1.columns)
df1_scaled.head()


# In[11]:


# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 


# In[12]:


# Dendrogram
plt.figure(figsize = (10,7))
plt.title("Dendrograms")
plt.xlabel('Index');plt.ylabel('Distance')
dend = sch.dendrogram(sch.linkage(df1_scaled, method='complete'))


# In[13]:


# Now applying AgglomerativeClustering choosing 5 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering
df1_cluster = AgglomerativeClustering(n_clusters = 4, linkage = 'complete', affinity = "euclidean")
df1_cluster.fit(df1_scaled)


# In[14]:


cluster_labels = pd.Series(df1_cluster.labels_)


# In[15]:


df['clust'] = cluster_labels # creating a new column and assigning it to new column1


# In[16]:


df.head()


# In[17]:


# Aggregate mean of each cluster
df.groupby(df.clust).mean()  #grouping by cluster and taking mean


# In[18]:


# creating a csv file 
df.to_csv(r"E:\360DIGITMG\ASSIGNMENT\ASSIGNMENT4\CreatedCrimeData.csv", encoding = "utf-8")


# In[ ]:




