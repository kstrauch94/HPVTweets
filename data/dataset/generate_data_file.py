#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 16:28:51 2018

@author: Samuele Garda
"""

import logging
import argparse
from hunspell import HunSpell


DISCARD = ['\n', 'Twitter / Account gesperr\n']
DICS = ['./en_US.dic', './en_US.aff']
SPELLER = HunSpell(*DICS)
NO_SPELL = ['^','Z','L','M','!','Y','#','@','~','U','E',',','G','S']

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(module)s: %(message)s', level = 'INFO')


def parse_arguments():
  parser = argparse.ArgumentParser(description='Utility for pos tagging tweets via TweetNLP ark-tweet-nlp without using java. ONLY LINUX SUPPORTED. \
                                   Strings discarded when loading tweets: `\n`,`Twitter / Account gesperr\n`')
  parser.add_argument('-r' , '--raw-tweets', help = 'Path where to file where raw tweets are stored')
  parser.add_argument('-p' , '--parsed-tweets', help = 'Path where to file where parsed tweets are stored')
  parser.add_argument('-o' , '--output', default = 'tweet_data_for_dp.txt', help = 'Out path to write to')
  
  return parser.parse_args()


def spell_word(w):
  """
  Correct mispelling of single word
  """
  
  try:
    if SPELLER.spell(w):
      pass
    else:
      w =  SPELLER.suggest(w)[0]
  except (IndexError,UnicodeDecodeError,SystemError) as e:
    pass
  
  return w
  
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


def write_spelled_parsed_tweets(tweet_file,out_file):
  
  logger.info("Correcting mispelling in parsed file...")
  
  out_file = open(out_file, 'w+')
  
  raw = open(tweet_file,encoding = "ISO-8859-1").read().split('\n\n')
  raw.pop()
  
  tok_idx = 1
  pos_idx = 3
  
  tweets = [tweet.split('\n') for tweet in raw]
  for tweet in tweets:
    by_word = [word.split('\t') for word in tweet]
    for w in by_word:
      w[tok_idx] = spell_word(w[tok_idx])  if not w[pos_idx] in NO_SPELL else w[tok_idx]
    spelled = ['\t'.join(w) for w in by_word]
    out_file.write('{}\n'.format('\n'.join(spelled)))
    out_file.write('\n')
        
def write_raw_tweets(tweet_file,text_to_discard,out_file):
  """
  Rewrite only tweets to a file from id\ttweet file. Input for dependecy parser.
  
  :params:
    tweet_file (str) : path to original downloaded tweet file
    text_to_discard (list) : list of strings to be replaced. E.g. `\n`
    out_file (str) : path where to write new file
  """
  
  logger.info("Creating new data set file (`\n` and expired account correction)...")
  
  out_file = open(out_file, 'w+')
  
  with open(tweet_file) as infile:
    for line in infile:
      _id,tweet = line.split('\t')
      out_file.write('{}'.format(_replace_newline(tweet,text_to_discard)))
 
  
if __name__ == "__main__":
  
  args = parse_arguments()
  
  try:
    if args.raw_tweets:
      write_raw_tweets(tweet_file = args.raw_tweets, text_to_discard = DISCARD, out_file = args.output)
  except AttributeError:
    pass
  
  try:
    if args.parsed_tweets:
      write_spelled_parsed_tweets(tweet_file = args.parsed_tweets,out_file = args.output)
  except AttributeError:
    pass
  
  
  
  
