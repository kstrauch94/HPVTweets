#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 16:28:51 2018

@author: Samuele Garda
"""

import logging
import argparse

DISCARD = ['\n', 'Twitter / Account gesperr\n']

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(module)s: %(message)s', level = 'INFO')

def parse_arguments():
  parser = argparse.ArgumentParser(description='Utility for pos tagging tweets via TweetNLP ark-tweet-nlp without using java. ONLY LINUX SUPPORTED. \
                                   Strings discarded when loading tweets: `\n`,`Twitter / Account gesperr\n`')
  parser.add_argument('-t' , '--tweets', required = True, help = 'Path where to file where tweets are stored')
  parser.add_argument('-o' , '--output', default = 'tweet_data_for_dp.txt', help = 'Out path to write to')
  
  return parser.parse_args()
  
def _replace_newline(line, text_to_discard):
  """
  Replace a line to be discarded with `ǸOTFOUND`
  
  :params:
    line (str) : line to be processed
    text_to_discard (list) : strings to be disarded
  :return:
    line if line not in found
  """
  if line in text_to_discard:
    line = 'NOTFOUND\n'
  else:
    pass
  return line
        
def write_tweets(tweet_file,text_to_discard,out_file):
  """
  Rewrite only tweets to a file from id\ttweet file. Input for dependecy parser.
  
  :params:
    tweet_file (str) : path to original downloaded tweet file
    text_to_discard (list) : list of strings to be replaced. E.g. `\n`
    out_file (str) : path where to write new file
  """
  
  out_file = open(out_file, 'w+')
  
  with open(tweet_file) as infile:
    for line in infile:
      _id,tweet = line.split('\t')
      out_file.write('{}'.format(_replace_newline(tweet,text_to_discard)))
 
  
if __name__ == "__main__":
  
  args = parse_arguments()

  write_tweets(tweet_file = args.tweets, text_to_discard = DISCARD, out_file = args.output)
  
  
  
