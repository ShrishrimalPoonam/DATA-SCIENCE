#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


# In[2]:


df=pd.read_csv(r"E:\360DIGITMG\ASSIGNMENT\ASSIGNMENT7\my_movies.csv")


# In[3]:


df.head()


# In[4]:


df=df.drop(["V1","V2","V3","V4","V5"], axis=1)


# In[5]:


frequent_movies = apriori(df, min_support = 0.0075, max_len = 4, use_colnames = True)


# In[6]:


frequent_movies


# In[7]:


#Most Frequent item sets based on support 
frequent_movies.sort_values('support', ascending = False, inplace = True)


# In[8]:


import matplotlib.pyplot as plt
plt.bar(x = list(range(0, 9)), height = frequent_movies.support[0:9], color ='rgmyk') 
# rgmyk is color seq. red, green....you can try any
plt.xticks(list(range(0, 9)), frequent_movies.itemsets[0:9], rotation=90)
plt.xlabel('movies')
plt.ylabel('support')
plt.show()


# In[9]:


rules = association_rules(frequent_movies, metric = "lift", min_threshold = 1)
rules.head(20)
rules.sort_values('lift', ascending = False).head(10)


# In[ ]:




