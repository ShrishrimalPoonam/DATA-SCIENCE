#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np


# In[2]:


df=pd.read_csv(r"E:\360DIGITMG\ASSIGNMENT\ASSIGNMENT6\wine.csv")


# In[3]:


df.head()


# In[4]:


df1 = df.drop(['Type'], axis = 1)


# In[5]:


# Normalization function 
def norm_func(i):
    x = (i - i.min())	/ (i.max() - i.min())
    return (x)


# In[6]:


# Normalized data frame (considering the numerical part of data)
df1_scaled = norm_func(df1)


# In[7]:


TWSS = []             #INITIALIZING TOTAL WITHIN SUM OF SQUARE
k = list(range(2, 9)) # WE DONT TAKE 1 TO 2 AS THERE EXISTS MASSIVE DECREASE ...WE LEARNT IN LEC.


# In[8]:


from sklearn.cluster import	KMeans
# from scipy.spatial.distance import cdist

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df1_scaled)
    TWSS.append(kmeans.inertia_)


# In[9]:


import matplotlib.pyplot as plt
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")


# In[10]:


# Selecting  clusters from the above scree plot which is the optimum number of clusters . 
model = KMeans(n_clusters = 4)
model.fit(df1_scaled)


# In[11]:


model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
df1['clust'] = mb # creating a  new column and assigning it to new column


# In[12]:


df1.head()


# In[13]:


# Aggregate mean of each cluster
df1.groupby(df1.clust).mean()  #grouping by cluster and taking 


# In[14]:


plt.scatter(x = df1.Alcohol, y = df1.Malic, c=df.Type)


# In[15]:


df2 = df1.drop(["clust"], axis = 1)


# In[16]:


from sklearn.preprocessing import StandardScaler


# In[17]:


scaler =StandardScaler()
scaler.fit(df1)


# In[18]:


scaled_data=scaler.transform(df1)


# In[19]:


scaled_data


# In[20]:


from sklearn.decomposition import PCA


# In[21]:


pca=PCA(n_components = 13)


# In[22]:


pca.fit(scaled_data)


# In[23]:


x_pca=pca.transform(scaled_data)


# In[24]:


scaled_data


# In[25]:


x_pca


# In[26]:


var = pca.explained_variance_ratio_
var


# In[27]:


pca.components_
pca.components_[0]   #0 means first column
# Cumulative variance 


# In[28]:


var1 = np.cumsum(np.round(var, decimals = 2) * 100)
var1


# In[29]:


# Variance plot for PCA components obtained 
import matplotlib.pyplot as plt
plt.plot(var1, color = "red")


# In[30]:


df2 = df1.drop(["clust"], axis = 1)


# In[31]:


pca_data = pd.DataFrame(x_pca)
pca_data.columns = "comp0", "comp1", "comp2", "comp3", "comp4", "comp5","comp6","comp7","comp8","comp9","comp10","comp11","comp12"
final = pd.concat([df.Type, pca_data.iloc[:, 0:8]], axis = 1)


# In[32]:


final.head()


# In[33]:


# Scatter diagram
import matplotlib.pylab as plt
plt.scatter(x = final.comp0, y = final.comp1, c=df.Type)


# In[34]:


# Normalization function 
def norm_func(i):
    x = (i - i.min())	/ (i.max() - i.min())
    return (x)


# In[35]:


# Normalized data frame (considering the numerical part of data)
final_scaled = norm_func(final)


# In[36]:


TWSS = []             #INITIALIZING TOTAL WITHIN SUM OF SQUARE
k = list(range(2, 9)) # WE DONT TAKE 1 TO 2 AS THERE EXISTS MASSIVE DECREASE ...WE LEARNT IN LEC.


# In[37]:


from sklearn.cluster import	KMeans
# from scipy.spatial.distance import cdist

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(final_scaled)
    TWSS.append(kmeans.inertia_)


# In[38]:


TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")


# In[39]:


# Selecting  clusters from the above scree plot which is the optimum number of clusters . 
model = KMeans(n_clusters = 4)
model.fit(final_scaled)


# In[40]:


model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
final['clust'] = mb # creating a  new column and assigning it to new column


# In[41]:


final.head()


# In[46]:


# Aggregate mean of each cluster
final.groupby(final.clust).mean()  #grouping by cluster and taking


# In[44]:


plt.scatter(x = final.comp0, y = final.comp1, c=df.Type)


# In[ ]:




