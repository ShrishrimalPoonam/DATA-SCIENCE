#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pylab as plt


# In[2]:


df=pd.read_csv(r"E:\360DIGITMG\ASSIGNMENT\ASSIGNMENT4\AutoInsurance.csv")


# In[3]:


df.head()


# In[4]:


df1 = df.drop(["Customer","State","Response","Coverage","Education","Effective To Date","EmploymentStatus","Gender","Marital Status","Location Code","Policy Type","Policy","Renew Offer Type","Sales Channel","Vehicle Class","Vehicle Size"], axis=1)


# In[5]:


df1.head()


# In[6]:


#standardize the data to normal distribution
from sklearn import preprocessing
from sklearn.preprocessing import normalize
df1_scaled=normalize(df1)
df1_scaled = pd.DataFrame(df1_scaled, columns = df1.columns)
df1_scaled.head()


# In[7]:


# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 


# In[8]:


# Dendrogram
plt.figure(figsize = (10,7))
plt.title("Dendrograms")
plt.xlabel('Index');plt.ylabel('Distance')
dend = sch.dendrogram(sch.linkage(df1_scaled, method='ward'))


# In[15]:


# Now applying AgglomerativeClustering choosing 5 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering
df1_cluster = AgglomerativeClustering(n_clusters = 5, linkage = 'ward', affinity = "euclidean")
df1_cluster.fit(df1_scaled)


# In[16]:


cluster_labels = pd.Series(df1_cluster.labels_)


# In[17]:


df['clust'] = cluster_labels # creating a new column and assigning it to new column


# In[18]:


df.head()


# In[19]:


# Aggregate mean of each cluster
df.groupby(df.clust).mean()  #grouping by cluster and taking mean


# In[20]:


# creating a csv file 
df.to_csv(r"E:\360DIGITMG\ASSIGNMENT\ASSIGNMENT4\CreatedAutoInsurance.csv", encoding = "utf-8")


# In[ ]:




