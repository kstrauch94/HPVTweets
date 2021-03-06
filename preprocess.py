#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 17:26:22 2018

@author: Samuele Garda
"""

# spell checking -> store in a separate file ?

import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

STOPWORDS = set(stopwords.words('english'))
# TODO add stemming and check what happens if word is mispelled 


# regexp to find multiple character occurrencies
REDUCE_LEN = re.compile(r"(.)\1{2,}")
STEMMER = PorterStemmer()

def stem_word(w,stem):
  """
  Stem single word:
  
  :params:
    w (str) : word
    stem (bool) : stem word
  :return:
    w (str) : word
  """
  w = STEMMER.stem(w) if stem else w
  
  return w
  
  
def stem_tweet(tweet):
  
  return [(stem_word(w[0],stem = True),w[1]) for w in tweet]

def rm_stopwords(tweet):
  """
  Remove stopwords from tweet
  
  :params:
    tweet (list) : list of tuple (word,pos)
  """
  
  return [w for w in tweet if not w[0] in STOPWORDS]

def remove_url(tweet):
  """
  Remove urls from tweet.
  
  :params:
    tweet (list) : list of tuple (word,pos)
    
  """
  return [w for w in tweet if not w[1] == 'U']


def reduce_length(tweet):
  """
  Redule words length :  fantaaaastic -> faantastic.
  
  :params:
    tweet (list) : list of tuple (word,pos)
  """
  
  return [(REDUCE_LEN.sub(r"\1\1", w[0]),w[1]) for w in tweet]


def _lowercase_word(w):
  """
  Lowercase single word (word,postag)
  
  :params:
    w (tuple) : tuple (word,pos)
  """
  
  w = (w[0].lower(),w[1]) if not w[1] == 'E' else w
  
  return w


def lowercase_tweet(tweet):
  """
  Lowercase tweet (everything but emoticons)
  
  :params:
    tweet (list) : list of tuple (word,pos)
  
  """
  return [_lowercase_word(w) for w in tweet]


def _url_to_string(w):     
  """
  Replace url string with `url`
   
  :params:
    w (tuple) : tuple (word,pos)
   
  """
  w = w if not w[1] == 'U' else ('url',w[1])
   
  return w

def replace_url(tweet):
  """
  Replace urls with string `url` in a tweet
  
  :params:
    tweet (list) : list of tuple (word,pos)
  """
  
  return [_url_to_string(w) for w in tweet]


def preprocessing(tweet, rm_url = True, red_len = True, lower = True, rm_sw = True, rm_tags_mentions = False, stem = False, out_pos = False):
  """
  Apply preprocessing to tweet.
  
  :params:
    rm_url (bool) : remove urls 
    red_len (bool) : reduce words length 
    lower (bool) : lowercase words
    repl_url (bool) : replace urls with string `urls`
    rm_sw (bool) : remove stopwords
    
  :return:
    
    preprocessed tweet (list of tokens)
  """
  if rm_url:
    tweet = remove_url(tweet)
  else:
    tweet = replace_url(tweet)
  if red_len:
    tweet = reduce_length(tweet)
  if lower:
    tweet = lowercase_tweet(tweet)
  if rm_sw:
    tweet = rm_stopwords(tweet)
  if rm_tags_mentions:
    tweet = delete_hashtags_mentions(tweet)
  if stem:
    tweet = stem_tweet(tweet)

  if out_pos:
    
    return ['\t'.join(w) for w in tweet]

  else:
    
    return [w[0] for w in tweet]
 
  
def delete_hashtags_mentions(tweet):
  """
  Remove hashtags and mentions from the tweet
  
  :params:
    tweet (list) : list of tuple (word,pos)
  """
  
  return [w for w in tweet if not w[0].startswith("#") and not w[0].startswith("@")]
    

if __name__ == "__main__":
  
  from load import load_data
  
  TWEET_FILE = './data/dataset/tweet_for_dp.txt.predict'
  ANNOS = './data/dataset/TweetsAnnotation.txt'
  
  
  df = load_data(dep_file = TWEET_FILE, annotations = ANNOS)
  
  df['toks'] = df['toks_pos'].apply(preprocessing,rm_url = True, red_len = True, lower = True, rm_sw = False, spell = True) 
  
  print(df.head())
  

    
  