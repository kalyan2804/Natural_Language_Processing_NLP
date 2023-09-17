#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# # Extracting the live Tweets from Twitter API
# 

# Importing necessary packages

# In[1]:


import os
import tweepy as tw
import pandas as pd
import nltk


# In[2]:


#twitter API connection
access_token= '1228242149247860742-LvB1ZifktBaTciJKmLcZIsomsHIaZY'
access_token_secret= 'gvKiUsttlEYazbPf7pUs2FzhISvJreOv6KhzLeNS9isP9'
consumer_key= 'GGP3RRXNZEr4xeQYuEXMsiLH1'
consumer_secret= '1nBtRP9LMUFEGv0eO0SvVSox1ak2rLchz55rQtE2SPF3hog72P'


# In[3]:


auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth)


# Give the MOVIE NAME(you want to check)

# In[10]:


movie_name=input("Enter the Movie name ")
movie_name_with_hashtag='#'+movie_name


# In[11]:


search_words = movie_name_with_hashtag
date_since = "2019-04-28"


# In[12]:


#Extracting the 1000 tweets of the entered movie name
tweets = tw.Cursor(api.search,
              q=search_words,
              lang="en",
              since=date_since).items(1000)


# In[13]:


users_locs = [[tweet.user.id, tweet.user.screen_name, tweet.user.location, tweet.user.url, tweet.user.description, tweet.user.protected, tweet.user.verified, tweet.user.followers_count, tweet.user.favourites_count, tweet.user.friends_count,tweet.user.statuses_count] for tweet in tweets]


# In[19]:


#conveting the extracted tweeets in to data frame using pandas
df= pd.DataFrame(data=users_locs, 
                    columns=['id','user', 'location', 'url', 'description', 'protected', 'verified', 'followers_count', 'favourites_count', 'friends_count','total_tweets_count'])
df.head()


# In[20]:


#the data set shape
df.shape


# In[21]:


#removing the null descriptive rows if any
df=df[df['description'].isna()==False]
df.shape


# # preprocessing the data

# (1)Removing the unnecessary characters
# 

# In[23]:


#importing the necessary packages
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
import re


# In[24]:


clean_1=[]
for each_row in range(df.shape[0]):
    text=df['description'].values[each_row]
    tempo=str(text)
    tempe=re.sub('[^A-Za-z0-9!?]', ' ', tempo)
    clean_1.append(tempe)
df["cleaned_1"]=clean_1


# In[25]:


df[['cleaned_1']]


# (2).Converting the cleaned_1 description of tweets in to lower case

# In[26]:


clean_2=[]
for each_row in range(df.shape[0]):
    text=df['cleaned_1'].values[each_row]
    tempo=str(text)
    tempe=tempo.lower()
    clean_2.append(tempe)
df["cleaned_2"]=clean_2


# In[27]:


df[['cleaned_2']]


# (3).Removing the stop words 

# In[28]:


#importing necessary packages
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize
stopwords_list=stopwords.words('english')


# In[29]:


#tokenizing and remocving stop words
mlist=[]
for each_row in range(df.shape[0]):
    text=df['cleaned_2'].values[each_row]
    tokenized_words=word_tokenize(text)
    clean_words=[]
    for each_word in tokenized_words:
        if not each_word in stopwords_list:
            clean_words.append(each_word)
    dre=''
    for each_data in clean_words:
        dre=dre+str(each_data)+' '
    mlist.append(dre)
df['cleaned_3']=mlist
        


# In[30]:


df[['cleaned_3']]


# (4).converting all the words in to their root words

# In[32]:


#importing the necessary_packages
from nltk.stem import WordNetLemmatizer
lm=WordNetLemmatizer()


# In[33]:


#tokenizing and converting to lem words
clean_4=[]
for each_row in range(df.shape[0]):
    text=df['cleaned_3'].values[each_row]
    tokenized_words=word_tokenize(text)
    clean_words=[]
    for each_word in tokenized_words:
        clean_words.append(lm.lemmatize(each_word))
    dre=''
    for each_data in clean_words:
        dre=dre+str(each_data)+' '
    clean_4.append(dre)
df['cleaned_4']=clean_4
        


# In[34]:


df[['cleaned_4']]


# (5).Removing words that are not there in english dictionary

# In[37]:


#importing the necessary packages
from nltk.corpus import words
word_list = words.words()


# In[38]:


#tokenizing and removing the words not there in dictionary
clean_5=[]
for each_row in range(df.shape[0]):
    text=df['cleaned_4'].values[each_row]
    tokenized_words=word_tokenize(text)
    clean_words=[]
    for each_word in tokenized_words:
        if each_word in word_list:
            clean_words.append(each_word)
        
    dre=''
    for each_data in clean_words:
        dre=dre+str(each_data)+' '
    clean_5.append(dre)
df['cleaned_5']=clean_5
        


# In[39]:


df[['cleaned_5']]


# # Cleaning the Data Set

# Removing the null value cleaned description rows

# In[47]:


df=df[df['cleaned_5']!='']


# In[50]:


#checking shape after removing
df.shape


# using VADER for pre-analysis

# In[53]:


#import necessaty packages
from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()


# In[54]:


tra_list=[]
for each_row in range(df.shape[0]):
    text=df['cleaned_5'].values[each_row]
    Ana_dict=sia.polarity_scores(text)
    neg_amount=Ana_dict['neg']
    neu_amount=Ana_dict['neu']
    pos_amount=Ana_dict['pos']
    if (max(neg_amount,neu_amount,pos_amount)==neg_amount):
        val='negative'
    elif (max(neg_amount,neu_amount,pos_amount)==pos_amount):
        val='positive'
    else:
        val='neutral'
    tra_list.append(val)
df['sentiment']=tra_list


# In[58]:


df['sentiment']


# In[62]:


df[df['sentiment']=='negative'].shape


# Converting the data set in to almost equal distribution Set for making good ML model prediction

# In[63]:


kn=df[df['sentiment']=='neutral'][:160]


# In[64]:


df=df[df['sentiment']!='neutral']


# In[65]:


df=df.append(kn)


# Removing the Unnecessary rows

# In[69]:


df.columns


# In[70]:


df=df[['user','cleaned_5','sentiment']]


# In[71]:


df['cleaned_description']=df['cleaned_5']


# In[72]:


df.columns


# In[73]:


df=df[['user', 'sentiment', 'cleaned_description']]


# In[74]:


df.head()


# In[75]:


#shuffling the data frame for more accurate training
df = df.sample(frac = 1)


# In[76]:


df.head()


# # Converting the data in to data for training and data for Sentiment Prediction 

# Converting cleaned_description into Sparse_matrix(BAG OF WORDS FORM)

# In[78]:


#importing the necessary packages
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()


# In[86]:


sparse_matrix=cv.fit_transform(df['cleaned_description'])
print(sparse_matrix.shape)


# Encoding the Sentiment in to int format 

# In[87]:


#importing the necessary packages
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[90]:


df['sentiment']=le.fit_transform(df['sentiment'])
df['sentiment'].head()


# Splitting the data for modelling

# In[91]:


#importing the necessary packages
from sklearn.model_selection import train_test_split


# In[92]:


x_train,x_test,y_train,y_test=train_test_split(sparse_matrix,df['sentiment'])


# In[93]:


x_train.shape, y_train.shape, x_test.shape, y_test.shape


# # Training Data using Naive Bayes

# In[94]:


#importing the necessary packages
from sklearn.naive_bayes import MultinomialNB
nb=MultinomialNB()


# In[95]:


nb.fit(x_train,y_train)


# #  Training Data using Support Vector Clustering

# In[99]:


#importing the necessary packages
from sklearn.svm import SVC
svc=SVC()


# In[100]:


svc.fit(x_train,y_train)


# #   Training Data using Random Forest Classifier

# In[102]:


#importing the necessary packages
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()


# In[104]:


rfc.fit(x_train,y_train)


# # Predicting the Accuracy of each Algorithm 

# In[106]:


#importing the necessary packages
from sklearn.metrics import accuracy_score


# Accuracy of Naive Bayes

# In[108]:


y_pred_1=nb.predict(x_test)


# In[109]:


accuracy_score(y_pred_1,y_test)


# Accuracy of Support Vector Clustering

# In[110]:


y_pred_2=svc.predict(x_test)


# In[111]:


accuracy_score(y_pred_2,y_test)


# Accuracy of Random Forest Classifier

# In[112]:


y_pred_3=rfc.predict(x_test)


# In[113]:


accuracy_score(y_pred_3,y_test)


# # Applying Ensembling Technique

# In[122]:


list1=list(y_pred_1)
list2=list(y_pred_2)
list3=list(y_pred_3)
list4=[]


# In[119]:


#importing necessary packages
import matplotlib.pyplot as plt
import numpy as np


# LIST1

# In[123]:


count_0=0
count_1=0
count_2=0
for each in list1:
    if each==0:
        count_0=count_0+1
    elif each==1:
        count_1=count_1+1
    elif each==2:
        count_2=count_2+1
    else:
        pass
    
if max(count_1,count_2,count_0)==count_1:
    list4.append(1)
elif max(count_1,count_2,count_0)==count_2:
    list4.append(2)
if max(count_1,count_2,count_0)==count_0:
    list4.append(0)
y_y = np.array([count_0,count_1,count_2])
mylabels = ["negative", "neutral", "positive"]
plt.pie(y_y, labels = mylabels)


# LIST2

# In[124]:


count_0=0
count_1=0
count_2=0
for each in list2:
    if each==0:
        count_0=count_0+1
    elif each==1:
        count_1=count_1+1
    elif each==2:
        count_2=count_2+1
    else:
        pass
    
if max(count_1,count_2,count_0)==count_1:
    list4.append(1)
elif max(count_1,count_2,count_0)==count_2:
    list4.append(2)
if max(count_1,count_2,count_0)==count_0:
    list4.append(0)
y_y = np.array([count_0,count_1,count_2])
mylabels = ["negative", "neutral", "positive"]
plt.pie(y_y, labels = mylabels)


# LIST3

# In[125]:


count_0=0
count_1=0
count_2=0
for each in list3:
    if each==0:
        count_0=count_0+1
    elif each==1:
        count_1=count_1+1
    elif each==2:
        count_2=count_2+1
    else:
        pass
    
if max(count_1,count_2,count_0)==count_1:
    list4.append(1)
elif max(count_1,count_2,count_0)==count_2:
    list4.append(2)
if max(count_1,count_2,count_0)==count_0:
    list4.append(0)
y_y = np.array([count_0,count_1,count_2])
mylabels = ["negative", "neutral", "positive"]
plt.pie(y_y, labels = mylabels)


# Deciding

# In[128]:


count_0=0
count_1=0
count_2=0
for each in list4:
    if each==0:
        count_0=count_0+1
    elif each==1:
        count_1=count_1+1
    elif each==2:
        count_2=count_2+1
    else:
        pass


y_y = np.array([count_0,count_1,count_2])
mylabels = ["negative", "neutral", "positive"]
plt.pie(y_y, labels = mylabels)

if max(count_1,count_2,count_0)==count_1:
    print('neutral')
elif max(count_1,count_2,count_0)==count_2:
    print('positive')
if max(count_1,count_2,count_0)==count_0:
    print('negative')


# In[ ]:




