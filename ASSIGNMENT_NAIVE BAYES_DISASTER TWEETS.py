#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer


# In[2]:


df= pd.read_csv(r"E:\360DIGITMG\ASSIGNMENT\ASSIGNMENT12\Disaster_tweets_NB.csv")


# In[3]:


df.head()


# In[4]:


df = df.drop (['id','keyword','location'], axis=1)


# In[5]:


df.head()


# In[6]:


# cleaning data 
import re
stop_words = []
# Load the custom built Stopwords
with open(r"E:\360DIGITMG\DOWNLOAD CODES\stopwords_en.txt","r") as sw:
    stop_words = sw.read()


# In[7]:


stop_words = stop_words.split("\n")


# In[8]:


def cleaning_text(i):
    i = re.sub("[^A-Za-z" "]+"," ",i).lower()
    i = re.sub("[0-9" "]+"," ",i)
    w = []
    for word in i.split(" "):
        if len(word)>3:
            w.append(word)
    return (" ".join(w))               

#we are pulling words greater than 3


# In[9]:


df.text = df.text.apply(cleaning_text)

# removing empty rows
df = df.loc[df.text != " ",:]


# In[10]:


# CountVectorizer
# Convert a collection of text documents to a matrix of token counts


# In[11]:


# splitting data into train and test data sets 
from sklearn.model_selection import train_test_split


# In[12]:


df_train, df_test = train_test_split(df, test_size = 0.2)


# In[13]:


# creating a matrix of token counts for the entire text document 
def split_into_words(i):
    return [word for word in i.split(" ")]


# In[14]:


# Defining the preparation of email texts into word count matrix format - Bag of Words
df_bow = CountVectorizer(analyzer = split_into_words).fit(df.text)


# In[15]:


# Defining BOW for all messages
all_df_matrix = df_bow.transform(df.text)


# In[16]:


# For training messages
train_df_matrix = df_bow.transform(df_train.text)


# In[17]:


# For testing messages
test_df_matrix = df_bow.transform(df_test.text)


# In[18]:


# Learning Term weighting and normalizing on entire emails
tfidf_transformer = TfidfTransformer().fit(all_df_matrix)


# In[19]:


# Preparing TFIDF for train emails
train_tfidf = tfidf_transformer.transform(train_df_matrix)
train_tfidf.shape # (row, column)        


# In[20]:


# Preparing TFIDF for test emails
test_tfidf = tfidf_transformer.transform(test_df_matrix)
test_tfidf.shape #  (row, column)


# In[21]:


# Preparing a naive bayes model on training data set 


# In[22]:


from sklearn.naive_bayes import MultinomialNB as MB


# In[25]:


# Multinomial Naive Bayes
classifier_mb = MB()
classifier_mb.fit(train_tfidf, df_train.target)


# In[28]:


# Evaluation on Test Data
test_pred_m = classifier_mb.predict(test_tfidf)
accuracy_test_m = np.mean(test_pred_m == df_test.target)
accuracy_test_m


# In[29]:


from sklearn.metrics import accuracy_score
accuracy_score(test_pred_m,df_test.target)


# In[30]:


pd.crosstab(test_pred_m, df_test.target)


# In[31]:


# Training Data accuracy
train_pred_m = classifier_mb.predict(train_tfidf)
accuracy_train_m = np.mean(train_pred_m == df_train.target)
accuracy_train_m


# In[32]:


# Multinomial Naive Bayes changing default alpha for laplace smoothing
# if alpha = 0 then no smoothing is applied and the default alpha parameter is 1
# the smoothing process mainly solves the emergence of zero probability problem in the dataset.


# In[33]:


classifier_mb_lap = MB(alpha = 3)
classifier_mb_lap.fit(train_tfidf, df_train.target)


# In[34]:


# Evaluation on Test Data after applying laplace
test_pred_lap = classifier_mb_lap.predict(test_tfidf)
accuracy_test_lap = np.mean(test_pred_lap == df_test.target)
accuracy_test_lap


# In[35]:


from sklearn.metrics import accuracy_score
accuracy_score(test_pred_lap, df_test.target) 


# In[36]:


pd.crosstab(test_pred_lap,df_test.target) 

# i.e, all 823  were identified correctly and 388  were wrongly identified


# In[37]:


# Training Data accuracy
train_pred_lap = classifier_mb_lap.predict(train_tfidf)
accuracy_train_lap = np.mean(train_pred_lap == df_train.target)
accuracy_train_lap


# In[ ]:




