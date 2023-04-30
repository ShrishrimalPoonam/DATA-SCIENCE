#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import requests
from bs4 import BeautifulSoup
import numpy as np


# In[2]:


url = "https://www.sitejabber.com/reviews/360digitmg.com"


# In[3]:


soup = BeautifulSoup(requests.get(url).content,'html.parser')


# In[4]:


Reviews =soup.find_all(class_='review__text')


# In[5]:


Reviews


# In[6]:


Review360= []
for i in range (0,len(Reviews)):
    Review360.append(Reviews[i].get_text().strip())
Review360


# In[7]:


df = pd.DataFrame()


# In[8]:


df['reviews']= Review360


# In[9]:


df.head(20)


# In[10]:


from nltk.corpus import stopwords
from textblob import TextBlob
from textblob import Word


# In[11]:


#lower casing and removing punctuations

df['reviews'] = df['reviews'].apply(lambda x:" ".join(x.lower() for x in x.split()))


# In[12]:


df['reviews'] = df['reviews'].str.replace('[^\w\s]',"")


# In[13]:


df['reviews'].head()


# In[14]:


#removing stopwords
stop = stopwords.words()


# In[15]:


#spelling correction
df['reviews'] =df['reviews'].apply(lambda x : str(TextBlob(x).correct()))


# In[16]:


import seaborn as sns
import re
import os
import sys
import ast
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
#function for gettinf the sentiment
cp = sns.color_palette()
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()


# In[17]:


#Generating Sentiment for all the sentence present in the dataset

emptyline=[] 
for row in df['reviews']:
    
    vs = analyzer.polarity_scores(row)
    emptyline.append(vs)
#creating new dataframe with sentiments
df_sentiments=pd.DataFrame(emptyline)
df_sentiments.head()


# In[18]:


#Merging the sentiments back to our df dataframe

df_c = pd.concat([df.reset_index(drop=True), df_sentiments], axis=1)
df_c


# In[19]:


import numpy as np
df_c['sentiment'] = np.where(df_c['compound']>=0, 'Positive', 'Negative')
df_c


# In[20]:


result =df_c['sentiment'].value_counts()
result.plot(kind='bar', rot=0, color=['Green','Red']);


# In[21]:


df1=df["reviews"]


# In[22]:


df1.to_csv(r"E:\360DIGITMG\ASSIGNMENT\ASSIGNMENT10\360Reviews.csv",index=False, header=False)


# In[24]:


with open(r"E:\360DIGITMG\ASSIGNMENT\ASSIGNMENT10\360Reviews.txt","r") as rc:
    review = rc.read()


# In[25]:


review = review.split("\n")


# In[26]:


review_string = " ".join(review)


# In[27]:


from wordcloud import WordCloud
from wordcloud import STOPWORDS
import matplotlib.pyplot as plt


# In[28]:


wordcloud = WordCloud(background_color ='white').generate(review_string)
plt.figure(figsize=(10,10))
plt.imshow (wordcloud)
plt.axis("off")
plt.show()


# In[29]:


watch_reviews_words = review_string.split(" ")


# In[30]:


# positive words # Choose the path for +ve words stored in system
with open(r"E:\360DIGITMG\DOWNLOAD CODES\positive-words.txt","r") as pos:
  poswords = pos.read().split("\n")


# In[31]:


# Positive word cloud
# Choosing the only words which are present in positive words
watch_pos_in_pos = " ".join ([w for w in watch_reviews_words if w in poswords])

wordcloud_pos_in_pos = WordCloud(
                      background_color='White',
                      width=1800,
                      height=1400
                     ).generate(watch_pos_in_pos)
plt.figure(2)
plt.axis("off")
plt.imshow(wordcloud_pos_in_pos)


# In[32]:


# negative words # Choose the path for +ve words stored in system
with open(r"E:\360DIGITMG\DOWNLOAD CODES\negative-words.txt","r") as neg:
  negwords = neg.read().split("\n")


# In[33]:


# negative word cloud
# Choosing the only words which are present in negwords
watch_neg_in_neg = " ".join ([w for w in watch_reviews_words if w in negwords])

wordcloud_neg_in_neg = WordCloud(
                      background_color='black',
                      width=2000,
                      height=1400
                     ).generate(watch_neg_in_neg)
plt.figure(3)
plt.axis("off")
plt.imshow(wordcloud_neg_in_neg)


# In[34]:


# wordcloud with bigram


# In[35]:


import nltk


# In[36]:


text = review_string


# In[37]:


# Remove single quote early since it causes problems with the tokenizer.
text = text.replace("'", "")


# In[38]:


tokens = nltk.word_tokenize(text)
text1 = nltk.Text(tokens)


# In[39]:


# Remove extra chars and remove stop words.
text_content = [''.join(re.split("[ .,;:!?‘’``''@#$%^_&*()<>{}~\n\t\\\-]", word)) for word in text1]


# In[40]:


# Create a set of stopwords
stopwords_wc = set(STOPWORDS)


# In[41]:


# Remove stop words
text_content = [word for word in text_content if word not in stopwords_wc]


# In[42]:


# Take only non-empty entries
text_content = [s for s in text_content if len(s) != 0]


# In[43]:


WNL = nltk.WordNetLemmatizer()
# Best to get the lemmas of each word to reduce the number of similar words
text_content = [WNL.lemmatize(t) for t in text_content]


# In[44]:


nltk_tokens = nltk.word_tokenize(text)  
bigrams_list = list(nltk.bigrams(text_content))
print(bigrams_list)


# In[45]:


dictionary2 = [' '.join(tup) for tup in bigrams_list]
print (dictionary2)


# In[46]:


# Using count vectoriser to view the frequency of bigrams
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(ngram_range=(2, 2))
bag_of_words = vectorizer.fit_transform(dictionary2)
vectorizer.vocabulary_


# In[47]:


sum_words = bag_of_words.sum(axis=0)
words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
print(words_freq[:100])


# In[48]:


# Generating wordcloud
words_dict = dict(words_freq)
WC_height = 1000
WC_width = 1500
WC_max_words = 200
wordCloud = WordCloud(max_words=WC_max_words, height=WC_height, width=WC_width)
wordCloud.generate_from_frequencies(words_dict)
plt.figure(4)
plt.title('Most frequently occurring bigrams connected by same colour and font size')
plt.imshow(wordCloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[ ]:




