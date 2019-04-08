#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import requests
import time
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

from tokenizer import token_func


# In[ ]:


headers = {'User-agent': 'Grace'}

def get_posts():
    onion_posts = []
    after = None
    for i in range(40):
        if after == None:
            params = {}
        else:
            params = {'after': after}
        url = 'https://www.reddit.com/r/theonion.json'
        res = requests.get(url, params=params, headers=headers)
        if res.status_code == 200:
            the_json = res.json()
            onion_posts.extend(the_json['data']['children'])
            after = the_json['data']['after']
        else:
            print(res.status_code)
            break
        time.sleep(2)

    titles = []
    for i in range(len(onion_posts)):
        titles.append(onion_posts[i]['data']['title'])

    onion_titles = list((set(titles)))

    news_posts = []
    after = None
    for i in range(40):
        if after == None:
            params = {}
        else:
            params = {'after': after}
        url = 'https://www.reddit.com/r/worldnews.json'
        res = requests.get(url, params=params, headers=headers)
        if res.status_code == 200:
            the_json = res.json()
            news_posts.extend(the_json['data']['children'])
            after = the_json['data']['after']
        else:
            print(res.status_code)
            break
        time.sleep(2)

    titles = []
    for i in range(len(news_posts)):
        titles.append(news_posts[i]['data']['title'])

    news_titles = list(set(titles))

    onion = pd.DataFrame(onion_titles)
    onion['is_onion'] = 1

    news = pd.DataFrame(news_titles)
    news['is_onion'] = 0

    titles = news.append(onion, ignore_index=True)
    titles.rename({0: 'title'}, axis=1, inplace=True)
    
    return titles


# In[ ]:


def naive_bayes(X_test, y_test):
    df = pd.read_csv('./materials/titles.csv')

    X_train = df['title']
    y_train = df['is_onion']
    
    cvec = CountVectorizer(tokenizer=token_func, max_features=X_train.shape[0], min_df=1, max_df=0.9)
    cvec.fit(X_train)
    
    X_train_cvec = pd.DataFrame(cvec.transform(X_train).todense(), columns=cvec.get_feature_names())
    X_test_cvec  = pd.DataFrame(cvec.transform(X_test).todense(), columns=cvec.get_feature_names())
    
    mnb = MultinomialNB(alpha=1)
    mnb.fit(X_train_cvec, y_train)
    print(f'Accuracy score: {mnb.score(X_test_cvec, y_test)}')
    
    y_pred = mnb.predict(X_test_cvec)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    print(f'Sensitivity: {tp/(tp+fn)}')
    print(f'Specificity: {tn/(tn+fp)}')


# In[ ]:


def knn(X_test, y_test):
    df = pd.read_csv('./materials/titles.csv')

    X_train = df['title']
    y_train = df['is_onion']
    
    tvec = TfidfVectorizer(tokenizer=token_func, max_features=X_train.shape[0], min_df=1, max_df=0.9)
    tvec.fit(X_train)
    
    X_train_tvec = pd.DataFrame(tvec.transform(X_train).todense(), columns=tvec.get_feature_names())
    X_test_tvec  = pd.DataFrame(tvec.transform(X_test).todense(), columns=tvec.get_feature_names())
    
    knn = KNeighborsClassifier(n_neighbors=15, weights='distance')
    knn.fit(X_train_tvec, y_train)
    print(f'Accuracy score: {knn.score(X_test_tvec, y_test)}')
    
    y_pred = knn.predict(X_test_tvec)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    print(f'Sensitivity: {tp/(tp+fn)}')
    print(f'Specificity: {tn/(tn+fp)}')


# In[ ]:


def svc(X_test, y_test):
    df = pd.read_csv('./materials/titles.csv')

    X_train = df['title']
    y_train = df['is_onion']
    
    tvec = TfidfVectorizer(tokenizer=token_func, max_features=X_train.shape[0], min_df=2, max_df=0.9)
    tvec.fit(X_train)
    
    X_train_tvec = pd.DataFrame(tvec.transform(X_train).todense(), columns=tvec.get_feature_names())
    X_test_tvec  = pd.DataFrame(tvec.transform(X_test).todense(), columns=tvec.get_feature_names())
    
    svc = SVC(kernel='rbf', C=10, gamma='scale')
    svc.fit(X_train_tvec, y_train)
    print(f'Accuracy score: {svc.score(X_test_tvec, y_test)}')
    
    y_pred = svc.predict(X_test_tvec)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    print(f'Sensitivity: {tp/(tp+fn)}')
    print(f'Specificity: {tn/(tn+fp)}')

