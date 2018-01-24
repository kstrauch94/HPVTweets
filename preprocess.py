#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 17:26:22 2018

@author: Samuele Garda
"""

# spell checking -> store in a separate file ?
# stemming -> what happens if word is mispelled 
# remove stopword

import re

# regexp to find multiple character occurrencies
REDUCE_LEN = re.compile(r"(.)\1{2,}")

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


def preprocessing(tweet, rm_url = True, red_len = True, lower = True, out_pos = True):
  """
  Apply preprocessing to tweet.
  
  :params:
    rm_url (bool) : remove urls 
    red_len (bool) : reduce words length 
    lower (bool) : lowercase words
    repl_url (bool) : replace urls with string `urls`
    out_pos (bool) : output form. If `True` is list of postagged tokens `['word\tpos','word\tpos',...]` else list of tokens ['word','word',...]
      
  """
  
  if rm_url:
    tweet = remove_url(tweet)
  else:
    tweet = replace_url(tweet)
  if red_len:
    tweet = reduce_length(tweet)
  if lower:
    tweet = lowercase_tweet(tweet)
 
  if out_pos:
    return ['\t'.join(w) for w in tweet]
  else:
    return [w[0] for w in tweet]
    

#if __name__ == "__main__":
#  
#  from load import load_data_complete
#  
#  DATA_PATH = './data/dataset/tweet_data_complete.2tsv'
#  
#  data = load_data_complete(DATA_PATH)
#    
#  
#  preprop_data = data.tok_pos.apply(preprocessing ,  rm_url = True, red_len = True, lower= True, out_pos = True)
#  
#  print(preprop_data.head())
#  
#  
    
  