#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pylab as plt


# In[2]:


df=pd.read_csv(r"E:\360DIGITMG\ASSIGNMENT\ASSIGNMENT6\heart disease.csv")


# In[3]:


df.head()


# In[4]:


df1 = df.drop(["sex","cp",'fbs',"restecg",'exang',"slope","ca","thal",'target'], axis = 1)


# In[5]:


df1.head()


# In[6]:


# Normalization function 
def norm_func(i):
    x = (i - i.min())	/ (i.max() - i.min())
    return (x)


# In[7]:


# Normalized data frame (considering the numerical part of data)
df1_scaled = norm_func(df1)


# In[8]:


TWSS = []             #INITIALIZING TOTAL WITHIN SUM OF SQUARE
k = list(range(2, 9)) # WE DONT TAKE 1 TO 2 AS THERE EXISTS MASSIVE DECREASE ...WE LEARNT IN LEC.


# In[9]:


from sklearn.cluster import	KMeans
# from scipy.spatial.distance import cdist

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df1_scaled)
    TWSS.append(kmeans.inertia_)


# In[10]:


TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")


# In[11]:


# Selecting  clusters from the above scree plot which is the optimum number of clusters . 
model = KMeans(n_clusters = 4)
model.fit(df1_scaled)


# In[12]:


model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
df1['clust'] = mb # creating a  new column and assigning it to new column


# In[13]:


df1.head()


# In[14]:


# Aggregate mean of each cluster
df1.groupby(df1.clust).mean()  #grouping by cluster and taking 


# In[15]:


plt.scatter(x = df1.age, y = df1.trestbps, c=df.target)


# In[16]:


df2 = df1.drop(["clust"], axis = 1)


# In[17]:


from sklearn.preprocessing import StandardScaler


# In[18]:


scaler =StandardScaler()
scaler.fit(df2)


# In[19]:


scaled_data=scaler.transform(df2)


# In[20]:


scaled_data


# In[21]:


from sklearn.decomposition import PCA


# In[22]:


pca=PCA(n_components = 5)


# In[23]:


pca.fit(scaled_data)


# In[24]:


x_pca=pca.transform(scaled_data)


# In[25]:


scaled_data.shape


# In[26]:


x_pca.shape


# In[27]:


scaled_data


# In[28]:


x_pca


# In[29]:


var = pca.explained_variance_ratio_
var


# In[30]:


pca.components_
pca.components_[0]   #0 means first column
# Cumulative variance 


# In[31]:


var1 = np.cumsum(np.round(var, decimals = 2) * 100)
var1


# In[32]:


# Variance plot for PCA components obtained 
plt.plot(var1, color = "red")


# In[33]:


# PCA scores
x_pca


# In[34]:


pca_data = pd.DataFrame(x_pca)
pca_data.columns = "comp0", "comp1","comp2","comp3","comp4"
final = pd.concat([pca_data, df.target], axis = 1)


# In[35]:


final.head()


# In[36]:


# Scatter diagram
import matplotlib.pylab as plt
plt.scatter(x = final.comp0, y = final.comp1,c=df.target)


# In[37]:


# Normalization function 
def norm_func(i):
    x = (i - i.min())	/ (i.max() - i.min())
    return (x)


# In[38]:


# Normalized data frame (considering the numerical part of data)
final_scaled = norm_func(final)


# In[39]:


TWSS = []             #INITIALIZING TOTAL WITHIN SUM OF SQUARE
k = list(range(2, 9)) # WE DONT TAKE 1 TO 2 AS THERE EXISTS MASSIVE DECREASE ...WE LEARNT IN LEC.


# In[40]:


from sklearn.cluster import	KMeans
# from scipy.spatial.distance import cdist

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(final_scaled)
    TWSS.append(kmeans.inertia_)


# In[41]:


TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")


# In[42]:


# Selecting  clusters from the above scree plot which is the optimum number of clusters . 
model = KMeans(n_clusters = 4)
model.fit(final_scaled)


# In[43]:


model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
final['clust'] = mb # creating a  new column and assigning it to new column


# In[44]:


final.head()


# In[45]:


# Aggregate mean of each cluster
final.iloc[:, 0:5].groupby(final.clust).mean()  #grouping by cluster and taking


# In[46]:


plt.scatter(x = final.comp0, y = final.comp1, c=df.target)


# In[ ]:




