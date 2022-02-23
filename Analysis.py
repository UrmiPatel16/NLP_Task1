#!/usr/bin/env python
# coding: utf-8

# In[1]:


# read csv file
import pandas as pd
df = pd.read_csv('/Users/urmi/Documents/NLP/Assignment_1/pmc-data-all.csv')


# In[2]:


# import necessary lib
import numpy as np
import re
import nltk
#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger') 
#import spacy
import string
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings 
warnings.filterwarnings('ignore')
from nltk.tokenize import RegexpTokenizer #will use this to remove puntuation and tokenize the text


# In[3]:


df.info()


# In[4]:


df.head()


# In[5]:


# print out word cloud for article-title column
from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)

def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='black',
        stopwords=stopwords,
        max_words=500,
        max_font_size=40, 
        scale=3,
        random_state=42 # chosen at random by flipping a coin; it was heads
).generate(str(data))

    fig = plt.figure(1, figsize=(15, 15))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=22)
        fig.subplots_adjust(top=5)

    plt.imshow(wordcloud)
    plt.show()

show_wordcloud(df['Article_Title'])


# In[6]:


text_a = df['Abstract'].str.cat(sep = ' ').lower()
tokenizer = RegexpTokenizer(r'\w+') 
tokens_a = tokenizer.tokenize(text_a)
pos_list_a = nltk.pos_tag(tokens_a)


# In[19]:


# function to test what text data is an adjective or not !
def is_adj(pos):
    result = False
    if pos in ('JJ','JJR','JJS'):
        result = True
    return result
adj_a = [word for word, pos in pos_list_a if is_adj(pos) and len(word) > 1]
#print(adj_a)
freq_a = nltk.FreqDist(adj_a)
print(freq_a.most_common(25))


# In[8]:


ax = plt.subplots(figsize=(10, 6))
freq_a.plot(25,cumulative=False,title='Top 25 adjectives for Abstract')


# In[9]:


# function to test what text data is a verb or not !
def is_verb(pos):
    result = False
    if pos in ('VB','VBD','VBG','VBN','VBP','VBZ'):
        result = True
    return result
verb_a = [word for word, pos in pos_list_a if is_verb(pos) and len(word) > 1] 
freq_a_verb = nltk.FreqDist(verb_a)
print(freq_a_verb.most_common(20)) 


# In[10]:


ax = plt.subplots(figsize=(10, 6))
freq_a_verb.plot(25,cumulative=False,title='Top 25 verbs for Abstract',color='red')


# In[11]:


# function to test what text data is a noun or not !
def is_noun(pos):
    result = False
    if pos in ('NN','NNP','NNPS','NNS'):
        result = True
    return result

noun_a = [word for word, pos in pos_list_a if is_noun(pos) and len(word) > 1]
freq_a_noun = nltk.FreqDist(noun_a) 
print(freq_a_noun.most_common(20))


# In[12]:


ax = plt.subplots(figsize=(10, 6))
freq_a_noun.plot(25,cumulative=False,title='Top 25 nouns for Abstract',color='green')


# In[13]:


stopwords = nltk.corpus.stopwords.words('english')

stop_a = [t for t in tokens_a if t in stopwords]

freq_a_stop = nltk.FreqDist(stop_a) 
ax = plt.subplots(figsize=(10, 6))
freq_a_stop.plot(25,cumulative=False,title='Top 25 stop words for Abstract',color ='orange')


# In[14]:


# how many % percentage papers published by any particular pulisher
count=df['Publisher_Name'].value_counts(normalize=True).sort_values(ascending=False)[:10]
c_df=pd.DataFrame(count*100)
c_df


# In[15]:


df2 = pd.read_csv('/Users/urmi/Documents/NLP/Assignment_1/new.csv')


# In[16]:


df2.info()


# In[17]:


import time
#df["Stop word Tokenized abstract"] = df["Stop word Tokenized abstract"].astype(str)

def applyWordStemmer(data):
    stemmer = nltk.stem.PorterStemmer()
    return [stemmer.stem(word) for word in data]

start = time.time()
df2['Stemmed abstract time'] = df2['Stop word Tokenized abstract'].apply(applyWordStemmer)
end = time.time()
print(end - start)


# In[18]:


def applyWordLemmatizer(data):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in data]

start = time.time()
#Word lemmatization
df2['Lemmatized abstract'] = df2['Stop word Tokenized abstract'].apply(applyWordLemmatizer)
end = time.time()
print(end - start)


# In[ ]:


df["abstract_lower"] = df["Abstract"].str.lower()
df["title_lower"] = df["Article_Title"].str.lower()


# In[ ]:


# find specific word from the article title data
c=0
for i in df['title_lower']:
   # print(str(i)+'\n')
    
    if("cancer" in str(i)):
       c=c+1
c  

