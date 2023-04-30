#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import networkx as nx 
import matplotlib.pyplot as plt


# In[3]:


G = pd.read_csv(r"E:\360DIGITMG\ASSIGNMENT\ASSIGNMENT9\connecting_routes.csv")
G = G.iloc[:, 1:10]


# In[3]:


G.head()


# In[4]:


g = nx.Graph() #THAT IS WE ARE DEFINING G WILL CONTAIN GRAPH DATA ONLY i.e, we can say type casting 


# In[5]:


g = nx.from_pandas_edgelist(G, source = 'Source Airport', target = 'Destination Airport')


# In[6]:


print(nx.info(g))


# In[7]:


b = nx.degree_centrality(g)  # Degree Centrality
print(b)


# In[8]:


pos = nx.spring_layout(g, k = 0.15)


# In[9]:


nx.draw_networkx(g, pos, node_size = 25, node_color = 'blue')


# In[10]:


# closeness centrality
closeness = nx.closeness_centrality(g)
print(closeness)


# In[15]:


## Betweeness Centrality 
b = nx.betweenness_centrality(g) # Betweeness_Centrality
print(b)


# In[16]:


## Eigen-Vector Centrality
evg = nx.eigenvector_centrality(g) # Eigen vector centrality
print(evg)


# In[17]:


# cluster coefficient
cluster_coeff = nx.clustering(g)
print(cluster_coeff)


# In[14]:


# Average clustering
cc = nx.average_clustering(g) 
print(cc)

