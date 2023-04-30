#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests


# In[2]:


from bs4 import BeautifulSoup

https://www.amazon.in/Nautica-Cruise-Nac-10-NAPNAI801/product-reviews/B07ZPJCHLS/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews
# In[3]:


url="https://www.amazon.in/Nautica-Cruise-Nac-10-NAPNAI801/product-reviews/B07ZPJCHLS/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews"


# In[4]:


headers = {"user-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"}


# In[5]:


page= requests.get(url, headers=headers)


# In[6]:


page


# In[7]:


page.content


# In[8]:


soup = BeautifulSoup(page.content,'html.parser')


# In[9]:


print(soup.prettify())


# In[10]:


names = soup.find_all('span', class_='a-profile-name')


# In[11]:


names


# In[12]:


customer_name = []
for i in range (0,len(names)):
    customer_name.append(names[i].get_text())
customer_name


# In[13]:


title=soup.find_all(class_='review-title-content')


# In[14]:


title


# In[15]:


review_title = []
for i in range (0,len(title)):
    review_title.append(title[i].get_text().strip())
review_title


# In[16]:


review = soup.find_all('span', class_='a-size-base review-text review-text-content')


# In[17]:


review


# In[18]:


review_content = []
for i in range (0,len(review)):
    review_content.append(review[i].get_text().strip())
review_content


# In[19]:


rating=soup.find_all(class_='review-rating')


# In[20]:


rating


# In[21]:


customer_rating = []
for i in range (0,len(rating)):
    customer_rating.append(rating[i].get_text().strip())
customer_rating


# In[22]:


import pandas as pd


# In[23]:


customer_name
customer_rating
review_content
review_title


# In[24]:


df = pd.DataFrame()


# In[25]:


df['customer_name'] = customer_name


# In[26]:


df


# In[27]:


df['customer_rating'] = customer_rating


# In[28]:


df['review_title'] = review_title


# In[29]:


df['review_content'] = review_content


# In[30]:


df


# In[31]:


df.to_csv(r"E:\360DIGITMG\ASSIGNMENT\ASSIGNMENT10\AmazonReviewScraping.csv",index=False)


# In[32]:


# SENTIMENT ANALYSIS


# In[33]:


get_ipython().system(' pip install textblob')


# In[34]:


from nltk.corpus import stopwords
from textblob import TextBlob
from textblob import Word


# In[35]:


#lower casing and removing punctuations

df['review_content'] = df['review_content'].apply(lambda x:" ".join(x.lower() for x in x.split()))


# In[36]:


df['review_content'] = df['review_content'].str.replace('[^\w\s]',"")


# In[37]:


df['review_content'].head()


# In[38]:


#removing stopwords

stop = stopwords.words()


# In[39]:


df['review_content'] =df['review_content'] .apply(lambda x: " ".join(x for x in x.split() if x not in stop))
df['review_content'].head()


# In[40]:


#spelling correction
df['review_content'] =df['review_content'].apply(lambda x : str(TextBlob(x).correct()))


# In[41]:


df['review_content']


# In[42]:


#Lemmatization

df['review_content'] = df['review_content'].apply(lambda x:" ".join([Word(word).lemmatize() for word in x.split()]))


# In[43]:


df['review_content']


# In[44]:


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


# In[45]:


#Generating Sentiment for all the sentence present in the dataset

emptyline=[] 
for row in df['review_content']:
    
    vs = analyzer.polarity_scores(row)
    emptyline.append(vs)
#creating new dataframe with sentiments
df_sentiments=pd.DataFrame(emptyline)
df_sentiments.head()


# In[46]:


#Merging the sentiments back to our df dataframe

df_c = pd.concat([df.reset_index(drop=True), df_sentiments], axis=1)
df_c


# In[47]:


import numpy as np
df_c['sentiment'] = np.where(df_c['compound']>=0, 'Positive', 'Negative')
df_c


# In[48]:


result =df_c['sentiment'].value_counts()
result.plot(kind='bar', rot=0, color=['plum','cyan']);


# In[49]:


df1=df['review_content']


# In[50]:


df1.to_csv(r"E:\360DIGITMG\ASSIGNMENT\ASSIGNMENT10\AmazonReview_content.csv",index=False, header=False)


# In[51]:


with open(r"E:\360DIGITMG\ASSIGNMENT\ASSIGNMENT10\AmazonReview_content.txt","r") as rc:
    review_content = rc.read()


# In[52]:


review_content = review_content.split("\n")


# In[53]:


review_content


# In[54]:


review_string = " ".join(review_content)


# In[55]:


from wordcloud import WordCloud
from wordcloud import STOPWORDS
import matplotlib.pyplot as plt


# In[56]:


wordcloud = WordCloud(background_color ='white').generate(review_string)
plt.figure(figsize=(10,10))
plt.imshow (wordcloud)
plt.axis("off")
plt.show()


# In[57]:


watch_reviews_words = review_string.split(" ")


# In[58]:


# positive words # Choose the path for +ve words stored in system
with open(r"E:\360DIGITMG\DOWNLOAD CODES\positive-words.txt","r") as pos:
  poswords = pos.read().split("\n")


# In[74]:


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


# In[60]:


# negative words # Choose the path for +ve words stored in system
with open(r"E:\360DIGITMG\DOWNLOAD CODES\negative-words.txt","r") as neg:
  negwords = neg.read().split("\n")


# In[75]:


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


# In[62]:


# wordcloud with bigram


# In[63]:


import nltk


# In[64]:


text = review_string


# In[65]:


# Remove single quote early since it causes problems with the tokenizer.
text = text.replace("'", "")


# In[66]:


tokens = nltk.word_tokenize(text)
text1 = nltk.Text(tokens)


# In[67]:


# Remove extra chars and remove stop words.
text_content = [''.join(re.split("[ .,;:!?‘’``''@#$%^_&*()<>{}~\n\t\\\-]", word)) for word in text1]


# In[68]:


nltk_tokens = nltk.word_tokenize(text)  
bigrams_list = list(nltk.bigrams(text_content))
print(bigrams_list)


# In[69]:


dictionary2 = [' '.join(tup) for tup in bigrams_list]
print (dictionary2)


# In[70]:


# Using count vectoriser to view the frequency of bigrams
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(ngram_range=(2, 2))
bag_of_words = vectorizer.fit_transform(dictionary2)
vectorizer.vocabulary_


# In[71]:


sum_words = bag_of_words.sum(axis=0)
words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
print(words_freq[:100])


# In[72]:


# Generating wordcloud
words_dict = dict(words_freq)
WC_height = 1000
WC_width = 1500
WC_max_words = 200
wordCloud = WordCloud(max_words=WC_max_words, height=WC_height, width=WC_width)
wordCloud.generate_from_frequencies(words_dict)


# In[73]:


plt.figure(4)
plt.title('Most frequently occurring bigrams connected by same colour and font size')
plt.imshow(wordCloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[ ]:




