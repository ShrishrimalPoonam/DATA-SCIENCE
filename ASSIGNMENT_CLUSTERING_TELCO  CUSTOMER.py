#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns


# In[2]:


df=pd.read_excel(r"E:\360DIGITMG\ASSIGNMENT\ASSIGNMENT4\Telco_customer_churn.xlsx")


# In[3]:


df.head(2)


# In[4]:


df1 = df.drop(["Customer ID","Quarter","Internet Service","Streaming TV","Device Protection Plan","Premium Tech Support","Online Security","Online Backup","Streaming Movies","Streaming Music","Count","Referred a Friend","Phone Service","Multiple Lines","Unlimited Data","Paperless Billing","Offer","Internet Type","Contract","Payment Method"], axis=1)


# In[5]:


df1.head()


# In[6]:


df.describe()


# In[7]:


df1.dtypes


# In[8]:


df1.isna().sum()


# In[9]:


#standardize the data to normal distribution
from sklearn import preprocessing
from sklearn.preprocessing import normalize
df1_scaled=normalize(df1)
df1_scaled = pd.DataFrame(df1_scaled, columns = df1.columns)
df1_scaled.head()


# In[10]:


# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 


# In[11]:


# Dendrogram
plt.figure(figsize = (10,7))
plt.title("Dendrograms")
plt.xlabel('Index');plt.ylabel('Distance')
dend = sch.dendrogram(sch.linkage(df1_scaled, method='ward'))


# In[12]:


# Now applying AgglomerativeClustering choosing 5 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering
df1_cluster = AgglomerativeClustering(n_clusters = 4, linkage = 'ward', affinity = "euclidean")
df1_cluster.fit(df1_scaled)


# In[13]:


cluster_labels = pd.Series(df1_cluster.labels_)


# In[14]:


df.Count = df.Count.astype('object') 


# In[15]:


df['clust'] = cluster_labels # creating a new column and assigning it to new column


# In[16]:


df.head()


# In[17]:


# Aggregate mean of each cluster
df.groupby(df.clust).mean()  #grouping by cluster and taking mean


# In[18]:


# creating a csv file 
df.to_csv(r"E:\360DIGITMG\ASSIGNMENT\ASSIGNMENT4\CreatedTelcoCustomer.csv", encoding = "utf-8")


# In[ ]:




