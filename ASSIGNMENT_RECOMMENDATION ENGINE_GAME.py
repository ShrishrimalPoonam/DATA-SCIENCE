#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv(r"E:\360DIGITMG\ASSIGNMENT\ASSIGNMENT8\game.csv")


# In[3]:


df.head()


# In[4]:


df.rating.unique()


# In[5]:


df.head()


# In[6]:


from sklearn.feature_extraction.text import TfidfVectorizer 
tfidf = TfidfVectorizer(stop_words = "english")


# In[7]:


# Preparing the Tfidf matrix by fitting and transforming
tfidf_matrix = tfidf.fit_transform(df.game)   #Transform a count matrix to a normalized tf or tf-idf representation


# In[8]:


tfidf_matrix.shape


# In[9]:


from sklearn.metrics.pairwise import linear_kernel


# In[10]:


# Computing the cosine similarity on Tfidf matrix
cosine_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)


# In[11]:


# creating a mapping of df name to index number 
df_index = pd.Series(df.index, index = df['userId']).drop_duplicates()


# In[15]:


def get_recommendations(userId, topN):    
    # topN = 10
    # Getting the movie index using its title 
    df_id = df_index[userId]
    
    # Getting the pair wise similarity score for all the anime's with that 
    # anime
    cosine_scores = list(enumerate(cosine_sim_matrix[df_id]))
    
    # Sorting the cosine_similarity scores based on scores 
    cosine_scores = sorted(cosine_scores, key=lambda x:x[1], reverse = True)
    
    # Get the scores of top N most similar movies 
    cosine_scores_N = cosine_scores[0: topN+1]
    
    # Getting the movie index 
    df_idx  =  [i[0] for i in cosine_scores_N]
    df_scores =  [i[1] for i in cosine_scores_N]
    
    # Similar movies and scores
    df_similar_show = pd.DataFrame(columns=["game", "scores"])
    df_similar_show["game"] = df.loc[df_idx, "game"]
    df_similar_show["scores"] = df_scores
    df_similar_show.reset_index(inplace = True)  
    # df_similar_show.drop(["index"], axis=1, inplace=True)
    print (df_similar_show)
    # return (df_similar_show)


# In[19]:


# Enter your userId and number of games to be recommended 
get_recommendations(10, topN = 10)


# In[ ]:




