#!/usr/bin/env python
# coding: utf-8

# In[1]:


from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

# To return a list of lemmatized tokens
def token_func(document):
    tokenizer = RegexpTokenizer('[a-zA-Z]+')
    lemmatizer = WordNetLemmatizer()
    tokens = tokenizer.tokenize(document.lower())
    words = [w for w in tokens if w not in stopwords.words('english')]
    tagged = pos_tag(words)
    lemmas = []
    for word, tag in tagged:
        wntag = get_wordnet_pos(tag)
        if wntag is None:
            lemma = lemmatizer.lemmatize(word)
            lemmas.append(lemma)
        else:
            lemma = lemmatizer.lemmatize(word, pos=wntag)
            lemmas.append(lemma)
    return lemmas

