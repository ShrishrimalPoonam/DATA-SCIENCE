#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv(r"E:\360DIGITMG\ASSIGNMENT\ASSIGNMENT11\Data.csv")


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


tweets=df.drop(['tweet_id','tweet_created','tweet_location','user_timezone'], axis=1)


# In[6]:


tweets.head()


# In[7]:


#lDA


# In[8]:


pip install pyLDAvis


# In[9]:


import pyLDAvis


# In[10]:


#Dependencies
import pandas as pd
import gensim #the library for Topic modelling
from gensim.models.ldamulticore import LdaMulticore
from gensim import corpora, models
import pyLDAvis.gensim_models #LDA visualization library

from nltk.corpus import stopwords
import string
from nltk.stem.wordnet import WordNetLemmatizer

import warnings
warnings.simplefilter('ignore')
from itertools import chain


# In[11]:


#clean the data
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

def clean(text):
    stop_free = ' '.join([word for word in text.lower().split() if word not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = ' '.join([lemma.lemmatize(word) for word in punc_free.split()])
    return normalized.split()


# In[12]:


tweets['text_clean']=tweets['text'].apply(clean)


# In[13]:


tweets


# In[14]:


#create dictionary
dictionary = corpora.Dictionary(tweets['text_clean'])
#Total number of non-zeroes in the BOW matrix (sum of the number of unique words per document over the entire corpus).
print(dictionary.num_nnz)


# In[15]:


#create document term matrix
doc_term_matrix = [dictionary.doc2bow(doc) for doc in tweets['text_clean'] ]
print(len(doc_term_matrix))


# In[16]:


lda = gensim.models.ldamodel.LdaModel


# In[17]:


num_topics=3
get_ipython().run_line_magic('time', 'ldamodel = lda(doc_term_matrix,num_topics=num_topics,id2word=dictionary,passes=50,minimum_probability=0)')


# In[18]:


ldamodel.print_topics(num_topics=num_topics)


# #Unable to do display
# import pyLDAvis.gensim_models as gensimvis
# pyLDAvis.enable_notebook()
# vis = gensimvis.prepare(ldamodel, corpus, dictionary)
# vis

# In[19]:


# Assigns the topics to the documents in corpus
lda_corpus = ldamodel[doc_term_matrix]


# In[20]:


[doc for doc in lda_corpus]


# In[21]:


scores = list(chain(*[[score for topic_id,score in topic]                       for topic in [doc for doc in lda_corpus]]))

threshold = sum(scores)/len(scores)
print(threshold)


# In[22]:


cluster1 = [j for i,j in zip(lda_corpus,df.index) if i[0][1] > threshold]
cluster2 = [j for i,j in zip(lda_corpus,df.index) if i[1][1] > threshold]
cluster3 = [j for i,j in zip(lda_corpus,df.index) if i[2][1] > threshold]


print(len(cluster1))
print(len(cluster2))
print(len(cluster3))


# In[23]:


tweets.iloc[cluster1]


# In[24]:


tweets.iloc[cluster2]


# In[25]:


tweets.iloc[cluster3]


# In[26]:


#LSA


# In[36]:


df1=pd.read_csv(r"E:\360DIGITMG\ASSIGNMENT\ASSIGNMENT11\Data.csv")


# In[37]:


df1.head()


# In[38]:


tweets=df1.drop(['tweet_id','tweet_created','tweet_location','user_timezone'], axis=1)


# In[42]:


tweets['sentiment'] = tweets['sentiment'].replace({'negative':0,'positive':1, 'neutral':2})


# In[43]:


tweets.head(2)


# In[44]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf=TfidfVectorizer()
tfidf.fit(tweets['text'])


# In[45]:


X =tfidf.transform (tweets['text'])


# In[46]:


tweets['text'][1]


# In[47]:


#check out tfid score for a few words say in sentence one i.e, above sentence


# In[48]:


print([X[1, tfidf.vocabulary_['tacky']]])


# In[49]:


print([X[1, tfidf.vocabulary_['commercials']]])


# In[50]:


#sentiment classification
tweets.sentiment.unique()


# In[51]:


import numpy as np


# In[55]:


tweets=tweets[tweets['sentiment']!=2]


# In[56]:


tweets.head(2)


# In[57]:


from sklearn.model_selection import train_test_split


# In[148]:


x=tweets.text
y=tweets.sentiment


# In[149]:


x_train,x_test,y_train,y_test=train_test_split(x,y, random_state=0)


# In[150]:


print("Test set has total {0} entries with {1:.2f}% negative,{2:.2f}% positive".format(len(x_test),
      (len(x_test[y_test==0])/(len(x_test)*1.))*100,
      (len(x_test[y_test==1])/(len(x_test)*1.))*100))


# In[151]:


print("Test set has total {0} entries with {1:.2f}% negative,{2:.2f}% positive".format(len(x_train),
      (len(x_train[y_train==0])/(len(x_train)*1.))*100,
      (len(x_train[y_train==1])/(len(x_train)*1.))*100))


# In[152]:


#similarly we can see for train
# we see an imbalance ratio here so will make use of Random Forest here


# In[153]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


# In[154]:


def accuracy_summary(pipeline, x_train, y_train, x_test, y_test):
    sentiment_fit= pipeline.fit(x_train,y_train)
    y_pred=sentiment_fit.predict(x_test)
    accuracy=accuracy_score(y_test, y_pred)
    print("accuracy score: {0:.2f}%".format(accuracy*100))
    return accuracy


# In[155]:


#To have efficient sentiment analysis or solving nlp problem we need a lot of features.
#Its not easy to figure out the exact number of features needed .
#We are going to try 10,000 to 30,000 and print accuracy scores associated with number of features


# In[156]:


cv = CountVectorizer()
rf=RandomForestClassifier(class_weight="balanced")
n_features=np.arange(1000,5001,1000)


# In[168]:


def nfeature_accuracy_checker (vectorizer=cv, n_features=n_features, stop_words=None, ngram_range=(1,1), classifier=rf):
    result=[]
    print(classifier)
    print("\n")
    for n in n_features:
        vectorizer.set_params(stop_words=stop_words, max_features=n, ngram_range=ngram_range)
        checker_pipeline=Pipeline([('vectorizer', vectorizer), ('classifier', classifier)])
        print("Test result for {} features".format(n))
        nfeature_accuracy=accuracy_summary(checker_pipeline, x_train, y_train, x_test, y_test)
        result.append((n,nfeature_accuracy))
    return result


# In[169]:


tfidf=TfidfVectorizer()
print("Result for trigram with stop words(Tfidf)\n")
feature_result_tgt=nfeature_accuracy_checker(vectorizer=tfidf,ngram_range=(1,3))


# In[171]:


#classification report
from sklearn.metrics import classification_report
cv=CountVectorizer(max_features=5000, ngram_range=(1,3))
pipeline=Pipeline([('vectorizer',cv),('classifier',rf)])
sentiment_fit=pipeline.fit(x_train,y_train)
y_pred=sentiment_fit.predict(x_test)
print(classification_report(y_test, y_pred,target_names=['negative','positive']))


# In[ ]:





# In[ ]:




