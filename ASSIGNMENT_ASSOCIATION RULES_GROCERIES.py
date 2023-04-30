#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Implementing Apriori algorithm from mlxtend
#conda install mlxtend
# or
#pip install mlxtend


# In[33]:


pip install matrix


# In[35]:


import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import numpy as np


# In[3]:


groceries = []
with open(r"E:\360DIGITMG\ASSIGNMENT\ASSIGNMENT7\groceries.csv") as f:
    groceries = f.read()
    
#if we would have loaded as pd.read we would have get many nan values as in many cases there is a single item purchase also
#to avoid that we are using or importing the file as above


# In[4]:


# splitting the data into separate transactions using separator as "\n"
groceries = groceries.split("\n")


# In[5]:


groceries


# In[6]:


groceries_list = []
for i in groceries:
    groceries_list.append(i.split(",")) #seperating every item and appending or adding it to groceries list


# In[7]:


groceries_list


# In[8]:


all_groceries_list = [i for item in groceries_list for i in item]


# In[9]:


all_groceries_list


# In[10]:


from collections import Counter # ,OrderedDict


# In[11]:


item_frequencies = Counter(all_groceries_list)  #counting each item i.e, number of times it is purchased. See below


# In[12]:


item_frequencies


# In[13]:


# after sorting
item_frequencies = sorted(item_frequencies.items(), key = lambda x:x[1])


# In[14]:


item_frequencies


# In[15]:


# Storing frequencies and items in separate variables 
frequencies = list(reversed([i[1] for i in item_frequencies])) # 1 is second item we are taking i.e, frequency of occurence
items = list(reversed([i[0] for i in item_frequencies]))# 0 is first item we are taking i.e, item name


# In[16]:


frequencies


# In[17]:


items


# In[18]:


# barplot of top 10 
import matplotlib.pyplot as plt


# In[19]:


plt.bar(height = frequencies[0:11], x = list(range(0, 11)), color = 'rgbkymc')
plt.xticks(list(range(0, 11), ), items[0:11], rotation=90)
plt.xlabel("items")
plt.ylabel("Count")
plt.show()


# In[20]:


# Creating Data Frame for the transactions data
groceries_series = pd.DataFrame(pd.Series(groceries_list))
groceries_series = groceries_series.iloc[:9835, :] # removing the last empty transaction


# In[21]:


groceries_series


# In[22]:


groceries_series.columns = ["transactions"] # assignning column name as transactions


# In[23]:


groceries_series.columns


# In[24]:


# creating a dummy columns for the each item in each transactions ... Using column names as item name
X = groceries_series['transactions'].str.join(sep = '*').str.get_dummies(sep = '*')


# In[25]:


X


# In[26]:


frequent_itemsets = apriori(X, min_support = 0.0075, max_len = 4, use_colnames = True)
#0.0075 means that atleast the item is bought 0.75% times. This value we have used based on previous exp. 
#say if we use 30% here than chances of 30% ppl has bought that item will be quite less so we use it minimum and so 0.0075 
# 4 i.e, we don't want group more than 4


# In[27]:


frequent_itemsets


# In[28]:


# Most Frequent item sets based on support 
frequent_itemsets.sort_values('support', ascending = False, inplace = True)


# In[29]:


plt.bar(x = list(range(0, 11)), height = frequent_itemsets.support[0:11], color ='rgmyk') 
# rgmyk is color seq. red, green....you can try any
plt.xticks(list(range(0, 11)), frequent_itemsets.itemsets[0:11], rotation=90)
plt.xlabel('item-sets')
plt.ylabel('support')
plt.show()


# In[30]:


rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1)
rules.head()
rules.sort_values('lift', ascending = False).head()


# In[ ]:




