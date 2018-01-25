from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, ClusterMixin

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

import numpy as np

from nltk.corpus import sentiwordnet as swn

from preprocess import preprocessing

import io
import random
import os
import copy

UNRELATED = "Unrelated"
NEG = "Neg"

LABEL = "label"
TWEET = "tweet"
NORMALIZED_LABEL = "normalized label"

C = "C"
GAMMA = "gamma"

def read_cluster_file(path):
    
    lookup = {}

    with io.open(path, "r", encoding="utf-8") as clusters:
        for line in clusters.readlines():
            id, token, count = line.split("\t")
            
            lookup[token] = id
            
    return lookup

def process_subjectivity_file(filename):

    scores = {}
    
    with open(filename, "r") as f:
        for line in f.readlines():
            line = line.split(" ")
            word = line[2].split("=")[1]
            score = line[-1].split("=")[1].strip()
            if score == "negative":
                score = -1
            elif score == "positive":
                score = 1
            else:
                score = 0
            
            scores[word] = score
    
    return scores

def bigrams(tokens):
    return [bg for bg in zip(tokens[:-1], tokens[1:])]
    
def clusters(tokens, cluster_lookup):

    clusters = []
    
    for token in tokens:
        if token in cluster_lookup:
            clusters.append(cluster_lookup[token])
            
    return clusters
    
def tweet_wordnet(tokens):
    
    words = []
    
    for word in tokens:
        for net in list(swn.senti_synsets(word)):
            words.append(net.synset.lemmas()[0].name())
            
    return words
    
def tweet_net_sentiment(tweet):
    pos = 0
    neg = 0
    obj = 0
    
    for word in tweet:
        for net in list(swn.senti_synsets(word)):
            pos += net.pos_score()
            neg += net.neg_score()
            obj += net.obj_score()
            
    return [pos, neg, obj]
    
def net_sentiment(pos_tweet):
    return np.array([tweet_net_sentiment(tw[1]) for tw in pos_tweet])
      

def tweet_sub_score(tweet, score_lookup):
    
    score = 0
    
    for word in tweet:
        if word in score_lookup:
            score += score_lookup[word]
            
    return score
    
def sub_score(pos_tweet):
    
    subj_score_file = "Data" + os.sep + "subjectivity_score.txt"

    score_lookup = process_subjectivity_file(subj_score_file)
    
    return np.array([tweet_sub_score(tw[1], score_lookup) for tw in pos_tweet]).reshape(-1, 1)

def tweets_length(pos_tweet):
    return np.array([len(t[1]) for t in pos_tweet]).reshape(-1, 1)
    
def build_tokenizer(do_bigrams=False, do_clusters=False, do_postags=False, do_postags_bg=False, do_sentiwords=False, cluster_lookup_file=None):
    if do_bigrams:
        get_bigrams = bigrams
    else:
        get_bigrams = lambda tokens: []
        
    if do_clusters:
        cluster_lookup = read_cluster_file(cluster_lookup_file)
        get_clusters = lambda tokens: clusters(tokens, cluster_lookup)
    else:
        get_clusters = lambda tokens: []
        
    if do_postags:
        get_tags = lambda tags: tags
    else:
        get_tags = lambda tags: []
        
    if do_postags_bg:
        get_tags_bg = lambda tags: bigrams(tags)
    else:
        get_tags_bg = lambda tags: []
        
    if do_sentiwords:
        get_sentiwords = tweet_wordnet
    else:
        get_sentiwords = lambda tokens: []

    
    def tokenizer(pos_tweet):
        words = pos_tweet[1]
        pos_tags = pos_tweet[0]
        tokens = words + get_bigrams(words) + get_clusters(words) + get_tags(pos_tags) + get_tags_bg(pos_tags) + get_sentiwords(words)
        return [str(t) for t in tokens]
            
    return tokenizer 

def build_pipeline_steps(tokenizer=None, preprocess=None, do_length=False, do_sentnet=False, do_subjscore=False, do_tfidf=False, dim_reduction=None):

    # if using the same steps on multiple locations simultaniously, deep copy before using!

    pipeline_steps = []
    features = []
    
    features.append(("cv", CountVectorizer(tokenizer=tokenizer, preprocessor=preprocess)))
    if do_length:
        print("adding length feature")
        features.append( ("length", FunctionTransformer(tweets_length, validate=False)) )
    
    if do_sentnet:
        print("adding sentinet feature")
        features.append( ("sentinet", FunctionTransformer(net_sentiment, validate=False)) )
        
    if do_subjscore:
        print("adding subj score feature")
        features.append( ("subj-score", FunctionTransformer(sub_score, validate=False)) )
        
        
    pipeline_steps = [ ("features", FeatureUnion(features)) ]
    
    if do_tfidf:
        print("Applying tfidf")
        pipeline_steps.append( ('tfidf', TfidfTransformer()) )
        
    if type(dim_reduction) == int and dim_reduction > 0:
            pipeline_steps.append( ("dim-reduction", TruncatedSVD(n_components=self.dim_reduction, random_state=42)) )
        
    
    return pipeline_steps
