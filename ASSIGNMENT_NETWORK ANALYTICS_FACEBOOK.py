#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
import numpy as np
import networkx as nx 
import matplotlib.pyplot as plt


# In[28]:


df = pd.read_csv(r"E:\360DIGITMG\ASSIGNMENT\ASSIGNMENT9\facebook.csv")


# In[29]:


df


# In[47]:


df_symmetric = nx.Graph()
df_symmetric.add_edge('0', '2')
df_symmetric.add_edge('0', '9')
df_symmetric.add_edge('1', '1')
df_symmetric.add_edge('1', '3')
df_symmetric.add_edge('2', '2')
df_symmetric.add_edge('2', '4')
df_symmetric.add_edge('3', '3')
df_symmetric.add_edge('3', '5')
df_symmetric.add_edge('4', '4')
df_symmetric.add_edge('4', '6')
df_symmetric.add_edge('5', '5')
df_symmetric.add_edge('5', '7')
df_symmetric.add_edge('6', '6')
df_symmetric.add_edge('6', '8')
df_symmetric.add_edge('7', '7')
df_symmetric.add_edge('7', '9')
df_symmetric.add_edge('8', '8')
df_symmetric.add_edge('8', '1')


# In[48]:


nx.draw_networkx(df_symmetric)


# In[49]:


# Degree Centrality 
nx.degree_centrality(df_symmetric)


# In[52]:


# closeness centrality
nx.closeness_centrality(df_symmetric)


# In[53]:


#Betweeness Centrality 
nx.betweenness_centrality(df_symmetric)


# In[54]:


#Eigen-Vector Centrality
nx.eigenvector_centrality(df_symmetric)


# In[55]:


# cluster coefficient
nx.clustering(df_symmetric)


# In[ ]:




