#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 17:38:34 2017

@author: Samuele Garda
"""

import scipy
import logging
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from collections import defaultdict


logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(module)s: %(message)s', level = 'INFO')  



class FeatureExtractor:
  """
  Class for feature extraction. All inputs must be a list of tokens.
  """
  
  def ngramsfeatures(texts, ngram_range):
    """
    Compute ngrams feature matrix. 
    
    :params:
      texts (list) : list of list of tokens
      ngram_range  (tuple) : unigram,bigram,...
      
    :return:
      ngram features (scipy.sparse.csr_matrix)
    """
    
    vect = CountVectorizer(preprocessor = ' '.join, tokenizer = str.split, ngram_range = ngram_range)
    
    ngrams = vect.fit_transform(texts)
    
    logger.info("Computed ngrams feature matrix with range {} - Resulting shape : {}".format(ngram_range,ngrams.shape))
    
    return ngrams
  
  
  def lsafeatures(texts, ngram_range, k):
    """
    Apply TFIDF weighting and TruncatedSVD
    
    :params:
      texts (list) : list of list of tokens
      ngram_range (tuple): unigram,bigram,...
      k (int) : reduced dimensionality
      
    :return:
      lsa features (scipy.sparse.csr_matrix)
    """
    
    vect = TfidfVectorizer(preprocessor = ' '.join, tokenizer = str.split, ngram_range = ngram_range)
    
    ngrams = vect.fit_transform(texts)
     
    svd = TruncatedSVD(n_components = k, random_state = 7)
     
    lsafeatures = svd.fit_transform(ngrams)
    
    logger.info("Computed Truncated SVD with k = {} - Resulting shape : {}".format(k,lsafeatures.shape))
    
    return lsafeatures
  
  
  def tweetclusterfeatures(texts, path_to_clusters):
    """
    Compute cluster feature matrix with frequencies of clusters for each tweet.
    
    :params:
      texts (list) : list of lists of tokens
      path_to_clusters (str) : cluster file
      
    :returns:
      
      cluster features (scipy.sparse.csr_matrix)
      
    """
    
    vec = CountVectorizer(preprocessor = ' '.join, tokenizer = str.split)
    
    clusters = defaultdict(set)
    
    with open(path_to_clusters) as infile:
      for line in infile:
        c,w,i = line.split('\t')
        clusters[c].add(w)
    
    mapped_texts = []
    clusters_in_data = set()
    
    for text in texts:
      mapped_text = []
      for w in text:
        for k,v in clusters.items():
          if w in v:
            mapped_text.append(k)
            clusters_in_data.add(k)
      mapped_texts.append(mapped_text)
    
    cluster_features = vec.fit_transform(mapped_texts)
    
    logger.info("Computed Twitter Clusters matrix - Resulting shape : {}".format(cluster_features.shape))
    
    return cluster_features
    
  
  def posfeatures(texts):
    """
    Compute pos feature matrix with frequencies of pos for each tweet.
    
    params:
      texts (list) : list of list of postagged tokens (word\tpos)
    
    :returns:
      pos features (scipy.sparse.csr_matrix)
 
    """
    
    vec = CountVectorizer(preprocessor = ' '.join, tokenizer = str.split)
    
    pos_texts = []
    
    for text in texts:
      pos_text = []
      for w in text:
        pos_text.append(w.split('\t')[1])
      pos_texts.append(pos_text)
        
    pos_features = vec.fit_transform(pos_texts)
    
    logger.info("Computed PoS matrix - Resulting shape : {}".format(pos_features.shape))
    
    return pos_features
  
  def getsentimentfeatures(texts,pos_file,neg_file):
    """
    Return sentiment words feature matrix with frequencies of positive and negative words for each tweet.
    
    :param:
      texts (list) : list of lists of tokens
      pos_file (str) : Opinion Lexicon. https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html
      neg_file (str) : Opinion Lexicon. https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html
    
    :return : 
      sentiment word features (scipy.sparse.csr_matrix)
    """
        
    n = open(pos_file).readlines()
    p = open(neg_file).readlines()
    
    neg_vocab = [w.strip() for w in n if not w.startswith(';')]
    pos_vocab = [w.strip() for w in p if not w.startswith(';')]
    
    logger.info("Performing search trough sentiment lexicon for each tweet. This might take a while...")
    
    pos_neg_tweets = []
    for tweet in texts:
      pos_neg_tweet = []
      for tok in tweet:
        if tok in pos_vocab or tok in neg_vocab:
          pos_neg_tweet.append(tok)
      pos_neg_tweets.append(pos_neg_tweet)
          
    vec = CountVectorizer(preprocessor = ' '.join, tokenizer = str.split)
      
    p_n_features = vec.fit_transform(pos_neg_tweets)
    
    logger.info("Computed bing liu sentiment word feature matrix - Resulting shape : {}".format(p_n_features.shape))
        
    return p_n_features
  
  
  def getnegatedwordfeatures(texts):
    """
    Return negated words feature matrix. 
    Negated words are: words ending in `n`t` and `not`,`no`,`nobody`,`nothing`,`none`,`nowhere`,`neither`
    
    :params:
      texts (list) : list of lists of tokens
      
    :return:
      sentiment word features (scipy.sparse.csr_matrix)
    """
          
    neg_tweets = []
    neg_list = ['not', 'no', 'never', 'nobody', 'nothing', 'none', 'nowhere', 'neither']  
      
    for tweet in texts:
      neg_tweet = []
      for idx,tok in enumerate(tweet):
        if tok.endswith("n't") or tok in neg_list:
          next_idx = idx+1
          prev_idx = idx-1
          if next_idx < len(tweet):
            next_neg = 'not_'+tweet[next_idx]
            neg_tweet.append(next_neg)
          if prev_idx > 0:
            prev_neg = 'not_'+tweet[prev_idx]
            neg_tweet.append(prev_neg)
      neg_tweets.append(neg_tweet)
      
    
    vec = CountVectorizer(preprocessor = ' '.join, tokenizer = str.split)
      
    neg_features = vec.fit_transform(neg_tweets)
    
    logger.info("Computed negated word feature matrix - Resulting shape : {}".format(neg_features.shape))
    
    return neg_features
    
  def concat_feat(*features):
    """
    Concatenate feature vectors
    """
    
    if all([isinstance(f, scipy.sparse.csr_matrix) for f in features]):
      all_feat = scipy.sparse.hstack(features).tocsr()
    elif all([isinstance(f,np.ndarray) for f in features]):
      all_feat = np.concatenate(features, axis = 1)
    else:
      logger.warning("Cannot combine different type of matrices! Convert them before!")
      
    logger.info("Concatenated features - Resulting shape : {}".format(all_feat.shape))
    
    return all_feat
  
    
    
    
    
    
     
    
    