#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 12:12:59 2017

@author: Samuele Garda
"""


import re
import hunspell


class TweetsPreprocessor(object):
  """
  Preprocessor class
  """
  
  def __init__(self,hunspell_dics,reduce_len = True, lowercase = False, spell = False):
    """
    Instantiate preprocessor. All tweet MUST be in the form [(word,postag),...]
    
    :params:
      hunspell_dics (list) : list of dictionaries for hunspell (e.g. en_US.dic,en_US.aff)
      reduce_len (bool) : redule length fantaaaastic -> fantastc
      lowercase (bool) : lowercase tweet (everything but emoticons)
      spell (bool) : apply spell checking
      
    """
    self.reduce_len = reduce_len
    self.lowercase = lowercase
    self.spell = spell 
    self.speller = hunspell.HunSpell(*hunspell_dics)
    self._not_to_spell = ['^','Z','L','M','!','Y','#','@','~','U','E',',','G','S']
    self.re_reduce_len = re.compile(r"(.)\1{2,}")
  

  def rm_url(self,tweet):
    """
    Remove urls.
    """
    return [w for w in tweet if not w[1] == 'U']
  
  def reduce_length(self,tweet):
    """
    edule length fantaaaastic -> fantastc
    """
    
    return [(self.re_reduce_len.sub(r"\1", w[0]),w[1]) for w in tweet]
  
  def lowercase_tweet(self,tweet):
    """
    owercase tweet (everything but emoticons)
    """
    return [self._lowercase_word(w) for w in tweet]
  
  def correct_mispelling(self,tweet):
    """
    spell check for tweet
    """
    return [self._hunspell(w) for w in tweet]
 
  def _hunspell(self,w):
    """
    spell check word (word,postag)
    """
    if w[1] not in self._not_to_spell:
      try:
        w = w if self.speller.spell(w[0]) else (self.speller.suggest(w[0])[0],w[1])
      except (IndexError,UnicodeDecodeError,SystemError):
        pass
    else:
      pass
      
    return w
  
  def _lowercase_word(self,w):
    """
    lowercase single word (word,postag)
    """
    
    w = (w[0].lower(),w[1]) if not w[1] == 'E' else w
    
    return w
  
  def replace_url(self,tweet):
    """
    Replace urls with `url`
    """
    
    def url_to_string(w):      
      w = w if not w[1] == 'U' else ('url',w[1])
      return w
      
    return [url_to_string(w) for w in tweet]

  
  def apply_processing(self,tweet):
    """
    Apply preprocessing as in initaliazation parameters
    """
    
    tweet = self.rm_url(tweet)
    
    if self.reduce_len:
      tweet = self.reduce_length(tweet)
    if self.lowercase:
      tweet = self.lowercase_tweet(tweet)
    if self.spell:
      tweet = self.correct_mispelling(tweet)
      
    return tweet
  
  
  def ref_processing(self,tweet):
    """
    Apply preprocessing as defined by `Du et al. 2017`
    INDEPENDENT FROM INITIALIZATION PARAMETERS
    """

    tweet = self.lowercase_tweet(tweet)
    tweet = [w for w in tweet if not w[1] in ['#','@']]
    tweet = self.reduce_length(tweet)
    tweet = self.replace_url(tweet)
    
    return tweet
    
  
  def tokenize(self,tweet):
    """
    Return list of words of preprocessed tweet
    """
    
    return [w[0] for w in tweet]
  
  def pos_tag(self,tweet):
    """
    Return list of `word\tpos` of preprocessed tweet
    """
    
    return ['\t'.join(w) for w in tweet]