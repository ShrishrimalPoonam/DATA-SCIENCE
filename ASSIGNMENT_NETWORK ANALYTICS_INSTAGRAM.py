#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import networkx as nx 
import matplotlib.pyplot as plt


# In[2]:


# Degree Centrality
df = pd.read_csv(r"E:\360DIGITMG\ASSIGNMENT\ASSIGNMENT9\instagram.csv")


# In[3]:


df


# In[4]:


df_symmetric = nx.Graph()
df_symmetric.add_edge('0', '2')
df_symmetric.add_edge('0', '3')
df_symmetric.add_edge('0', '4')
df_symmetric.add_edge('0', '5')
df_symmetric.add_edge('0', '6')
df_symmetric.add_edge('0', '7')
df_symmetric.add_edge('0', '8')
df_symmetric.add_edge('1', '1')
df_symmetric.add_edge('2', '1')
df_symmetric.add_edge('3', '1')
df_symmetric.add_edge('4', '1')
df_symmetric.add_edge('5', '1')
df_symmetric.add_edge('6', '1')
df_symmetric.add_edge('7', '1')


# In[5]:


nx.draw_networkx(df_symmetric)


# In[6]:


# Degree Centrality 
nx.degree_centrality(df_symmetric) 


# In[7]:


# closeness centrality
nx.closeness_centrality(df_symmetric)


# In[8]:


#Betweeness Centrality 
nx.betweenness_centrality(df_symmetric)


# In[9]:


# cluster coefficient
nx.clustering(df_symmetric)


# In[10]:


# Average clustering
nx.average_clustering(df_symmetric)


# In[ ]:




