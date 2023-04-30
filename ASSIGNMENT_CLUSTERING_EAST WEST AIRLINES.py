#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pylab as plt
import numpy as np


# In[2]:


df1= pd.read_excel(r"E:\360DIGITMG\ASSIGNMENT\ASSIGNMENT4\EastWestAirlines.xlsx")


# In[3]:


df1.head()


# In[4]:


df1 = df1.drop(["ID#","Award?"], axis=1)


# In[5]:


#converting categorical miles data into numerical data by taking the average of the range

df1['cc1_miles'] = df1['cc1_miles'].replace({1:2500, 2:5000, 3:17500, 4:32500, 5:50000})
df1['cc2_miles'] = df1['cc2_miles'].replace({1:2500, 2:5000, 3:17500, 4:32500, 5:50000})
df1['cc3_miles'] = df1['cc3_miles'].replace({1:2500, 2:5000, 3:17500, 4:32500, 5:50000})
df1.head()


# In[6]:


df1.dtypes


# In[7]:


#standardize the data to normal distribution
from sklearn import preprocessing
from sklearn.preprocessing import normalize
df1_scaled=normalize(df1)
df1_scaled = pd.DataFrame(df1_scaled, columns = df1.columns)
df1_scaled.head()


# In[8]:


# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 


# In[9]:


# Dendrogram
plt.figure(figsize = (10,7))
plt.title("Dendrograms")
plt.xlabel('Index');plt.ylabel('Distance')
dend = sch.dendrogram(sch.linkage(df1_scaled, method='ward'))


# In[10]:


# Now applying AgglomerativeClustering choosing 5 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering
df1_cluster = AgglomerativeClustering(n_clusters = 3, linkage = 'ward', affinity = "euclidean")
df1_cluster.fit(df1_scaled)


# In[11]:


cluster_labels = pd.Series(df1_cluster.labels_)


# In[12]:


df1['clust'] = cluster_labels # creating a new column and assigning it to new column


# In[13]:


df1.head(10)


# In[14]:


# Aggregate mean of each cluster
df1.iloc[:, 0:10].groupby(df1.clust).mean()  #grouping by cluster and taking mean


# In[15]:


# creating a csv file 
df1.to_csv(r"E:\360DIGITMG\ASSIGNMENT\ASSIGNMENT4\CreatedEastWestAirlines.csv", encoding = "utf-8")


# In[ ]:




