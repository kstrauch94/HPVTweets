#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 17:23:37 2018

@author: Samuele Garda
"""

import pandas


LABEL = "label"
TWEET = "tweet"
NORMALIZED_LABEL = "normalized_label"


def _load_annotations(annotations):
  """
  Load tweet labels.
  
  :params: 
    annotations (str) : annotation file with tweet id and label
  :return:
    anns (list) : list of tuple (id,label)
    
  """
  with open(annotations) as infile:
    anns = [(line.split('\t')[0],line.split('\t')[2].strip()) for line in infile]
    return anns
  
  
def _load_tweets_pos_dependecy(dep_file):
  """
  Load tweets from dependency parser output file. One word per line. Each tweet separated by new line.
  
  :params:
    dep_file (str) : output of TweeboParser
  :return:
    tweet_data (list) : list of tweets to per parsed
  """
  # please leave the encoding parameter. It doesnt work on a windows system without it.
  tweet_data = open(dep_file, encoding="utf-8").read().split('\n\n')
  tweet_data.pop()
  
  return tweet_data


def load_data(dep_file,annotations):
  """
  Create pandas Dataframe for storing data
  
  :params:
    dep_file (str) : output of TweeboParser
    annotations (str) : annotation file with tweet id and label
  :return:
   df (pandas.DataFreme) : dataframe containing list of tweets, pos tags, 
   dependency parse tags and tuple (pos tag, tweet tokens) (only for preprocessing)
  """
  
  tweet_data = _load_tweets_pos_dependecy(dep_file)
  anns = _load_annotations(annotations)
  
  
  tok_idx = 1
  pos_idx = 3
  dep_idx = 6
  
  assert len(tweet_data) == len(anns), "# of tweet and # of annotations must be the same!"
  
  data = {}
  
  for tweet,anns in zip(tweet_data,anns):
    # REMOVING NOT FOUND TWEETS
    if tweet.split('\n')[0].split('\t')[tok_idx] == 'NOTFOUND':
      pass
    else:
      _id = anns[0]
      data[_id] = {}
      data[_id]['label'] = anns[1]
      data[_id]['toks'] = [w.split('\t')[tok_idx] for w in tweet.split('\n')]
      data[_id]['pos'] = [w.split('\t')[pos_idx] for w in tweet.split('\n')]
      data[_id]['dep'] = [int(w.split('\t')[dep_idx]) for w in tweet.split('\n')]
      
      # ONLY FOR PREPROCESSING
      data[_id]['toks_pos'] = [(w.split('\t')[tok_idx],w.split('\t')[pos_idx]) for w in tweet.split('\n')]
      
  
  df = pandas.DataFrame.from_dict(data, orient='index')
  
  return df

if __name__ == "__main__":
  
  ANNOTATIONS = './data/dataset/TweetsAnnotation.txt'
  TWEET_DP = './data/dataset/tweet_for_dp.txt.predict'
  
  df = load_data(TWEET_DP,ANNOTATIONS)
  
  tweets = list(df['toks'])
  