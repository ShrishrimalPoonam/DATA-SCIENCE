#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import networkx as nx 
import matplotlib.pyplot as plt


# In[2]:


# Degree Centrality
df = pd.read_csv(r"E:\360DIGITMG\ASSIGNMENT\ASSIGNMENT9\linkedin.csv")


# In[3]:


df


# In[4]:


df_symmetric = nx.Graph()
df_symmetric.add_edge('0', '2')
df_symmetric.add_edge('0', '3')
df_symmetric.add_edge('1', '1')
df_symmetric.add_edge('1', '3')
df_symmetric.add_edge('1', '4')
df_symmetric.add_edge('2', '1')
df_symmetric.add_edge('2', '2')
df_symmetric.add_edge('2', '4')
df_symmetric.add_edge('3', '2')
df_symmetric.add_edge('3', '3')
df_symmetric.add_edge('3', '13')
df_symmetric.add_edge('4', '6')
df_symmetric.add_edge('4', '7')
df_symmetric.add_edge('5', '5')
df_symmetric.add_edge('5', '7')
df_symmetric.add_edge('5', '8')
df_symmetric.add_edge('6', '5')
df_symmetric.add_edge('6', '6')
df_symmetric.add_edge('6', '8')
df_symmetric.add_edge('7', '6')
df_symmetric.add_edge('7', '7')
df_symmetric.add_edge('7', '13')
df_symmetric.add_edge('8', '10')
df_symmetric.add_edge('8', '11')
df_symmetric.add_edge('9', '9')
df_symmetric.add_edge('9', '11')
df_symmetric.add_edge('9', '12')
df_symmetric.add_edge('10', '9')
df_symmetric.add_edge('10', '10')
df_symmetric.add_edge('10', '12')
df_symmetric.add_edge('11', '10')
df_symmetric.add_edge('11', '11')
df_symmetric.add_edge('11', '13')
df_symmetric.add_edge('12', '4')
df_symmetric.add_edge('12', '8')
df_symmetric.add_edge('12', '12')


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




