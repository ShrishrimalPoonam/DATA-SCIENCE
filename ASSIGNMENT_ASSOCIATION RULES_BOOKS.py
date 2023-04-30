#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


# In[2]:


df=pd.read_csv(r"E:\360DIGITMG\ASSIGNMENT\ASSIGNMENT7\book.csv")


# In[3]:


df.head()


# In[4]:


df1 = apriori(df, min_support = 0.0075, max_len = 4, use_colnames = True)


# In[5]:


df1


# In[6]:


#Most Frequent item sets based on support 
df1.sort_values('support', ascending = False, inplace = True)


# In[7]:


import matplotlib.pyplot as plt
plt.bar(x = list(range(0, 11)), height = df1.support[0:11], color ='rgmyk') 
# rgmyk is color seq. red, green....you can try any
plt.xticks(list(range(0, 11)), df1.itemsets[0:11], rotation=90)
plt.xlabel('item-sets')
plt.ylabel('support')
plt.show()


# In[14]:


rules = association_rules(df1, metric = "lift", min_threshold = 1)
rules.head()
rules.sort_values('lift',ascending = False).head()


# In[ ]:




