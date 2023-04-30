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


# Selecting 5 clusters from the above scree plot which is the optimum number of clusters . 
#You can aslo choose 3 as there is max. drop there and than its smooth i.e, similat drop
model = KMeans(n_clusters = 4)
model.fit(df1_scaled)


# In[12]:


model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
df['clust'] = mb # creating a  new column and assigning it to new column


# In[13]:


df.head()


# In[14]:


# Aggregate mean of each cluster
df.groupby(df.clust).mean()  #grouping by cluster and taking mean


# In[15]:


# creating a csv file 
df.to_csv(r"E:\360DIGITMG\ASSIGNMENT\ASSIGNMENT5\CreatedKmeansAutoInsurance.csv", encoding = "utf-8")


# In[ ]:




