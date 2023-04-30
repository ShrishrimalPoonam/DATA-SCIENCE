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


# Normalization function 
def norm_func(i):
    x = (i - i.min())	/ (i.max() - i.min())
    return (x)


# In[11]:


# Normalized data frame (considering the numerical part of data)
df1_scaled = norm_func(df1)


# In[12]:


TWSS = []             #INITIALIZING TOTAL WITHIN SUM OF SQUARE
k = list(range(2, 9)) # WE DONT TAKE 1 TO 2 AS THERE EXISTS MASSIVE DECREASE ...WE LEARNT IN LEC.


# In[13]:


from sklearn.cluster import	KMeans
# from scipy.spatial.distance import cdist

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df1_scaled)
    TWSS.append(kmeans.inertia_)


# In[14]:


TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")


# In[15]:


# Selecting  clusters from the above scree plot which is the optimum number of clusters . 
model = KMeans(n_clusters = 4)
model.fit(df1_scaled)


# In[16]:


model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
df['clust'] = mb # creating a  new column and assigning it to new column


# In[17]:


df.head(10)


# In[19]:


# Aggregate mean of each cluster
df.groupby(df.clust).mean()  #grouping by cluster and taking 


# In[20]:


# creating a csv file 
df.to_csv(r"E:\360DIGITMG\ASSIGNMENT\ASSIGNMENT5\CreatedCrimeData.csv", encoding = "utf-8")


# In[ ]:




