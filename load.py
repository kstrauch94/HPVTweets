#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 17:23:37 2018

@author: Samuele Garda
"""

import pandas

def load_data_complete(data_file):
  """
  Create pandas Dataframe from dataset file created via `generate_data_file.py`
  :params:
    data_file (str) : path to file
  
  :return:
    df (pandas.DataFrame) : dataframe storing tweet id, label, postagged texts,  tokens and pos representation of tweets
  """
  
  # read data from file containg output of ARKParser
  f = open(data_file, encoding="utf-8").read().split('\n')
  #remove last new line
  f.pop()
  
  data = {}
  
  for line in f:
    _id,label,words = line.split('\t\t')
    data[_id] = {}
    data[_id]['label'] = label
    
    pos_words = words.split(' ')
#    data[_id]['tok_pos'] = words.split(' ')
    data[_id]['tok'] = [w.split('\t')[0] for w in pos_words]
    data[_id]['pos'] = [w.split('\t')[1] for w in pos_words]
    data[_id]['tok_pos'] = [tuple(w.split('\t')) for w in pos_words]
      
    
  df = pandas.DataFrame.from_dict(data, orient='index')
  df.info()
  
  return df
