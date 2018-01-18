#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 09:47:07 2017

@author: Samuele Garda
"""

import logging
import pandas
import subprocess
from utils import timeit

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(module)s: %(message)s', level = 'INFO')

class TweetsLoader(object):
  """
  Load tweet from tweet file and annotation file
  """
  
  def __init__(self,ark_parser = ['./ark-tweet-nlp-0.3.2/runTagger.sh'], text_to_discard = ['\n', 'Twitter / Account gesperr\n']):
#    self.SentTweet = namedtuple('SentTweet','id text tag')
    self.ark_parser = ark_parser
    self.ark_pos_arguments = ['--output-format', 'conll', '--no-confidence']
    self.text_to_discard = text_to_discard
    
    
  def _replace_newline(self,line):
    """
    Replace empty tweet
    """
    if line in self.text_to_discard:
      line = 'NOTFOUND'
    else:
      pass
    return line
  
  def load_tweets(self,tweet_data):
    """
    Load tweet from file. Apply tokenization and pos tagging via TweetNLP.
    """
    with open(tweet_data, encoding = "ISO-8859-1") as infile:
      raw_tweets = [self._replace_newline(line.split('\t')[1]) for line in infile]
      
    p = subprocess.Popen( self.ark_parser + self.ark_pos_arguments, 
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines = True)
      
    tweets_cleaned = '\n'.join([tw.replace('\n', ' ') for tw in raw_tweets])
    result = p.communicate(tweets_cleaned)[0]
    pos_tagged_tweets = result.strip('\n\n').split('\n\n')
        
    return pos_tagged_tweets
    
  def load_annotations(self,annotations):
    """
    Load tweet labels
    """
    with open(annotations) as infile:
      anns = [(line.split('\t')[0],line.split('\t')[2].strip()) for line in infile]
      return anns
  
  @timeit    
  def load(self,tweet_data, annotations):
    """
    Load pos tagged tweet with label. Store as a pandas dataframe
    """
    
    logger.info("Loading data from `{}` - Annotations stored in `{}`".format(tweet_data,annotations))
    
    pos_tweets = self.load_tweets(tweet_data)
    annots = self.load_annotations(annotations)
      
    assert len(pos_tweets) == len(annots), "# of tweet and # of annotations must be the same!"
    
    tweets = {}
    
    for idx,tweet in enumerate(pos_tweets):
      if tweet.startswith('NOTFOUND'):
        pass
      else:
        word_pos = tweet.split('\n')
        tweet_id = annots[idx][0] 
        tweets[tweet_id] = {}
        tweets[tweet_id]['words'] = [tuple(w.split('\t')) for w in word_pos]  
        tweets[tweet_id]['label'] = annots[idx][1]
        
    
    df = pandas.DataFrame.from_dict(tweets, orient='index')
    df.info()
    
    return df
    
    
    
  
  
  
    
  
