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

def process_tweets(tweets,labels):
  """
  Create dictionaries containing data for each level of classification
  
  :params:
    tweets (list) : list of list of tokens
    labels (list) : list of strings 
  
  :returns:
    _all (dict) : all tweets (labels : `Unrelated`,'Related') 
    related (dict) : positive,negative,neutral tweets (labels : `Positive`,`Negative`,`Neutral`)
    negative (dict) : negative tweets (labels : `NegOthers`,`NegResistant`...)
    
    Dictionary keys are `tweet`,`label`,`normalized_label` (by level)
  """

# use this function to process list of (id, tweet) tuples

  UNRELATED = "Unrelated"
  NEG = "Neg"
  NEG_LABEL = "Negative"
  
  _all = {TWEET: [], LABEL: [], NORMALIZED_LABEL: []}
  
  related = {TWEET: [], LABEL: [], NORMALIZED_LABEL: []}
  
  negative = {TWEET: [], LABEL: [], NORMALIZED_LABEL: []}
  
  for tweet,label in zip(tweets,labels):
      # add tweet to all dict
      _all[TWEET].append(tweet)
      _all[LABEL].append(label)
      all[NORMALIZED_LABEL].append(label if label == UNRELATED else "Related")
      
      # add tweet to related if applicable
      if label != UNRELATED:
          related[TWEET].append(tweet)
          related[LABEL].append(label)
          related[NORMALIZED_LABEL].append(label if NEG not in label else NEG_LABEL)
          
      # add tweet to negative if applicable
      if NEG in label:
          negative[TWEET].append(tweet)
          negative[LABEL].append(label)
          negative[NORMALIZED_LABEL].append(label)
                 
  return _all, related, negative



class BaseClf(BaseEstimator, ClusterMixin):

    def __init__(self, pipeline_steps=None):
        
        self.pipeline_steps = pipeline_steps
        
    def fit(self, tweets, labels):
    
        #HAVE TO DEEPCOPY HERE. This is because when building the pipeline we already give made objects to it.
        # So, when passing the pipeline to multiple classifiers, the objects inside the pipeline are NOT copied
        # This means that if in any other place, a pipeline is created and then fitted(like in the hierarchical CLF) it will override the objects
        # in all other pipelines using this "pipeline_steps" list
        self._pipeline_steps = copy.deepcopy(self.pipeline_steps)    
    
        self._vectorizer = Pipeline(self._pipeline_steps)
        
        X = self._vectorizer.fit_transform(tweets)
        print(X.shape)
        
        self._clf = self.get_clf()
        
        self._clf.fit(X, labels)
        
    def predict(self, tweets):
    
        X = self._vectorizer.transform(tweets)     
            
        return self._clf.predict(X)
        
    def score(self, tweets, labels):
        X = self._vectorizer.transform(tweets)   
            
        return self._clf.score(X, labels)
        
    def get_clf(self):
        raise ValueError("get_clf function must be implemented!")
        
class TweetClassifierKNN(BaseClf):

    def __init__(self, pipeline_steps=None, neighbors=10):
        super(TweetClassifierKNN, self).__init__(pipeline_steps)
          
        self.neighbors=neighbors
        
    def get_clf(self):
        return KNeighborsClassifier(self.neighbors)
        
class TweetClassifierLR(BaseClf):

    def __init__(self, pipeline_steps=None, C=1.0, tol=1e-4):
        super(TweetClassifierLR, self).__init__(pipeline_steps)
                  
        self.C = C
        self.tol=tol
    def get_clf(self):
        return LogisticRegression(C=self.C, tol=self.tol)
        
class TweetClassifierRF(BaseClf):

    #def __init__(self, pipeline_steps=None):
    #    super(TweetClassifierLR, self).__init__(pipeline_steps)
                  
    def get_clf(self):
        return RandomForestClassifier(n_estimators=50)
        
class TweetClassifierBaseSVM(BaseClf):

    def __init__(self, pipeline_steps=None, C=256, GAMMA=0.00002):
        super(TweetClassifierBaseSVM, self).__init__(pipeline_steps)
        
        self.C = C
        self.GAMMA = GAMMA
        
    def get_clf(self):
        return svm.SVC(C=self.C, gamma=self.GAMMA)


class TweetClassifierH(BaseEstimator, ClassifierMixin):

    def __init__(self, get_clf=None, kwargs=None):
        """
        params:
            get_clf: a function that return some classifier
            kwargs: a dict contiainung the kwargs of the 3 differente classifier versions
                    {1: kwargs for first classifier, 2: kwargs for second classifier,, 3: kwargs for third classifier}
        """
        self.get_clf = get_clf

        self.kwargs = kwargs           

    def fit(self, pos_tweet, labels):
        
        if self.get_clf is None:
            raise ValueError("A classifier is needed")
        
        tweet_list = zip(labels, pos_tweet)
        all, related, negative = process_tweets(tweet_list)
        
        
        self._all_clasifier = self.get_clf(1)(**self.kwargs[1])
        self._related_clasifier = self.get_clf(2)(**self.kwargs[2])
        self._negative_clasifier = self.get_clf(3)(**self.kwargs[3])
        
        print("fittin data...")

        print("fittin all...")
        self._all_clasifier.fit(all[TWEET], all[NORMALIZED_LABEL])
        
        print("fitting related...")
        self._related_clasifier.fit(related[TWEET], related[NORMALIZED_LABEL])
        
        print("fitting negative...")
        self._negative_clasifier.fit(negative[TWEET], negative[NORMALIZED_LABEL])
        
        print("finished fitting data!")
        
        return self
       
     
    def predict(self, tweets):
    
        print("predicting...")
        
        predictions = []
        
        for tweet in tweets:
            # predict with all to get related or unrelated
            pred = self._all_clasifier.predict([tweet])[0] # the zero is because the classifier return a list with 1 element
            if pred == UNRELATED:
                # if its unrelated we are done with this tweet
                predictions.append(pred)
                continue
                
            # if its related try to get next label
            pred = self._related_clasifier.predict([tweet])[0]
            if NEG not in pred:
                # if its not negative we are done with this tweet
                predictions.append(pred)
                continue
                
            # if its negative try to get specific label
            pred = self._negative_clasifier.predict([tweet])[0]
            predictions.append(pred)
            
        return predictions
    
    def score(self, tweets, labels):
        preds = self.predict(tweets)

        return accuracy_score(labels, preds)

