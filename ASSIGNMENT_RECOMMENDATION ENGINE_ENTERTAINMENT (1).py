#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv(r"E:\360DIGITMG\ASSIGNMENT\ASSIGNMENT8\Entertainment.csv")


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


from sklearn.feature_extraction.text import TfidfVectorizer 
#term frequencey- inverse document frequncy is a numerical statistic that is 
#intended to reflect how important a word is to document in a collecion or corpus


# In[6]:


# replacing the NaN values in overview column with empty string
df["Category"].isnull().sum() 


# In[7]:


tfidf = TfidfVectorizer(stop_words = "english")


# In[8]:


# Preparing the Tfidf matrix by fitting and transforming
tfidf_matrix = tfidf.fit_transform(df.Category)   #Transform a count matrix to a normalized tf or tf-idf representation


# In[9]:


tfidf_matrix.shape


# In[10]:


from sklearn.metrics.pairwise import linear_kernel


# In[11]:


# Computing the cosine similarity on Tfidf matrix
cosine_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)


# In[12]:


# creating a mapping of anime name to index number 
df_index = pd.Series(df.index, index = df['Titles']).drop_duplicates()


# In[13]:


df_id = df_index["Toy Story (1995)"]
df_id


# In[14]:


def get_recommendations(Titles, topN):    
    # topN = 10
    # Getting the movie index using its title 
    df_id = df_index[Titles]
    
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
    df_similar_show = pd.DataFrame(columns=["Titles", "Score"])
    df_similar_show["Titles"] = df.loc[df_idx, "Titles"]
    df_similar_show["Score"] = df_scores
    df_similar_show.reset_index(inplace = True)  
    # df_similar_show.drop(["index"], axis=1, inplace=True)
    print (df_similar_show)
    # return (df_similar_show)


# In[15]:


# Enter your anime and number of anime's to be recommended 
get_recommendations("Toy Story (1995)", topN = 10)


# In[ ]:




