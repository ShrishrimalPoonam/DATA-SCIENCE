#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import requests
from bs4 import BeautifulSoup
import numpy as np
import time


# In[2]:


url = "https://www.imdb.com/title/tt0347304/reviews?ref_=tt_urv"


# In[3]:


headers = {"user-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"}


# In[4]:


page= requests.get(url, headers=headers)


# In[5]:


soup = BeautifulSoup(page.content,'html.parser')


# In[6]:


print(soup.prettify())


# In[7]:


review = soup.find_all('a', class_='title')


# In[8]:


review


# In[9]:


review_title = []
for i in range (0,len(review)):
    review_title.append(review[i].get_text().strip())
review_title


# In[10]:


content= soup.find_all('div', class_='text show-more__control')


# In[11]:


review_content = []
for i in range (0,len(content)):
    review_content.append(content[i].get_text().strip())
review_content


# In[12]:


df = pd.DataFrame()


# In[13]:


df['review_title']= review_title


# In[14]:


df['review_content']= review_content


# In[15]:


df.head()


# In[16]:


from nltk.corpus import stopwords
from textblob import TextBlob
from textblob import Word


# In[17]:


#lower casing and removing punctuations

df['review_title'] = df['review_title'].apply(lambda x:" ".join(x.lower() for x in x.split()))


# In[18]:


df['review_title'] = df['review_title'].str.replace('[^\w\s]',"")


# In[19]:


df['review_title'].head()


# In[20]:


#removing stopwords

stop = stopwords.words()


# In[21]:


#spelling correction
df['review_title'] =df['review_title'].apply(lambda x : str(TextBlob(x).correct()))


# In[22]:


df['review_title']


# In[23]:


#Lemmatization

df['review_title'] = df['review_title'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))


# In[24]:


df['review_title']


# In[25]:


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


# In[26]:


#Generating Sentiment for all the sentence present in the dataset

emptyline=[] 
for row in df['review_title']:
    
    vs = analyzer.polarity_scores(row)
    emptyline.append(vs)
#creating new dataframe with sentiments
df_sentiments=pd.DataFrame(emptyline)
df_sentiments.head()


# In[27]:


#Merging the sentiments back to our df dataframe

df_c = pd.concat([df.reset_index(drop=True), df_sentiments], axis=1)
df_c


# In[28]:


import numpy as np
df_c['sentiment'] = np.where(df_c['compound']>=0, 'Positive', 'Negative')
df_c


# In[29]:


result =df_c['sentiment'].value_counts()
result.plot(kind='bar', rot=0, color=['Green','Red']);


# In[30]:


# Similarly we can do for review content


# In[31]:


#lower casing and removing punctuations

df['review_content'] = df['review_content'].apply(lambda x:" ".join(x.lower() for x in x.split()))


# In[32]:


df['review_content'] = df['review_content'].str.replace('[^\w\s]',"")


# In[33]:


df['review_content'].head()


# In[34]:


#removing stopwords

stop = stopwords.words()


# In[35]:


#spelling correction
df['review_content'] =df['review_content'].apply(lambda x : str(TextBlob(x).correct()))


# In[36]:


#Lemmatization

df['review_content'] = df['review_content'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))


# In[37]:


df['review_content']


# In[40]:


#Generating Sentiment for all the sentence present in the dataset

emptyline1=[] 
for row in df['review_content']:
    
    vs = analyzer.polarity_scores(row)
    emptyline1.append(vs)
#creating new dataframe with sentiments
df_content_sentiments=pd.DataFrame(emptyline1)
df_content_sentiments.head()


# In[41]:


df_c = pd.concat([df.reset_index(drop=True), df_content_sentiments], axis=1)
df_c


# In[42]:


import numpy as np
df_c['content sentiment'] = np.where(df_c['compound']>=0, 'Positive', 'Negative')
df_c


# In[43]:


result =df_c['content sentiment'].value_counts()
result.plot(kind='bar', rot=0, color=['Green','Red']);


# In[45]:


df1=df["review_title"]


# In[46]:


df1.to_csv(r"E:\360DIGITMG\ASSIGNMENT\ASSIGNMENT10\IMDBReview_content.csv",index=False, header=False)


# In[47]:


with open(r"E:\360DIGITMG\ASSIGNMENT\ASSIGNMENT10\IMDBReview_content.txt","r") as rc:
    review = rc.read()


# In[48]:


review = review.split("\n")


# In[49]:


review_string = " ".join(review)


# In[50]:


from wordcloud import WordCloud
from wordcloud import STOPWORDS
import matplotlib.pyplot as plt


# In[51]:


wordcloud = WordCloud(background_color ='white').generate(review_string)
plt.figure(figsize=(10,10))
plt.imshow (wordcloud)
plt.axis("off")
plt.show()


# In[52]:


watch_reviews_words = review_string.split(" ")


# In[53]:


# positive words # Choose the path for +ve words stored in system
with open(r"E:\360DIGITMG\DOWNLOAD CODES\positive-words.txt","r") as pos:
  poswords = pos.read().split("\n")


# In[54]:


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


# In[55]:


# negative words # Choose the path for +ve words stored in system
with open(r"E:\360DIGITMG\DOWNLOAD CODES\negative-words.txt","r") as neg:
  negwords = neg.read().split("\n")


# In[56]:


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


# In[57]:


# wordcloud with bigram


# In[58]:


import nltk


# In[59]:


text = review_string


# In[60]:


# Remove single quote early since it causes problems with the tokenizer.
text = text.replace("'", "")


# In[61]:


tokens = nltk.word_tokenize(text)
text1 = nltk.Text(tokens)


# In[62]:


# Remove extra chars and remove stop words.
text_content = [''.join(re.split("[ .,;:!?‘’``''@#$%^_&*()<>{}~\n\t\\\-]", word)) for word in text1]


# In[63]:


# Create a set of stopwords
stopwords_wc = set(STOPWORDS)


# In[64]:


# Remove stop words
text_content = [word for word in text_content if word not in stopwords_wc]


# In[65]:


# Take only non-empty entries
text_content = [s for s in text_content if len(s) != 0]


# In[66]:


WNL = nltk.WordNetLemmatizer()
# Best to get the lemmas of each word to reduce the number of similar words
text_content = [WNL.lemmatize(t) for t in text_content]


# In[67]:


nltk_tokens = nltk.word_tokenize(text)  
bigrams_list = list(nltk.bigrams(text_content))
print(bigrams_list)


# In[68]:


dictionary2 = [' '.join(tup) for tup in bigrams_list]
print (dictionary2)


# In[69]:


# Using count vectoriser to view the frequency of bigrams
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(ngram_range=(2, 2))
bag_of_words = vectorizer.fit_transform(dictionary2)
vectorizer.vocabulary_


# In[70]:


sum_words = bag_of_words.sum(axis=0)
words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
print(words_freq[:100])


# In[71]:


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





# In[ ]:




