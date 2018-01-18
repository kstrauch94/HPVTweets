#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 10:47:46 2017

@author: Samuele Garda
"""

import logging
import pandas
import numpy as np
from collections import defaultdict
from sklearn.base import clone
from sklearn.base import BaseEstimator, ClassifierMixin
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder,MultiLabelBinarizer #,LabelBinarizer
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support as prfs
from collections import OrderedDict


logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(module)s: %(message)s', level = 'INFO')

class EncodeLables:
  """
  Encoding labels for different classification approaches
  """
  
  @staticmethod
  def multiclass_labels(y):
    """
    Standard multilabel
    """
    
    logger.info("Encoding labels for multiclass task...")
    
    encoder = LabelEncoder()
    encoded_classes = encoder.fit_transform(y)
    
    return encoder,encoded_classes
  
  @staticmethod  
  def multilabel_binirizer(y):
    """
    Multilabel binariazer for Multilabel CNN
    """
    
    logger.info("Encoding labels for multilabel task")
    
    encoder = MultiLabelBinarizer()
    encoded_classes = encoder.fit_transform(y)
    
    return encoder,encoded_classes  
  
  @staticmethod
  def get_hierarchical_names(y) :
    """
    Hierachical labels
    """
    
    logger.info("Encoding labels for hierarchical task...")
    
    new_y = []
    for i in y:
      if i == 'Positive':
         i = 'Related-Positive'
      elif i == 'Neutral':
         i = 'Related-Neutral'
      elif i.startswith('Neg'):
         i = 'Related-Negative-{}'.format(i.strip('Neg'))
      elif i == 'Unrelated':
         pass
      new_y.append(i)
      
    return np.array(new_y)
  
  @staticmethod
  def categorical_labels(y):
    """
    One hot vectors for CNN
    """
    
    logger.info("Encoding labels as one-hot vectors for CNN...")

    encoded_classes = keras.utils.to_categorical(y)
    
    return encoded_classes
  
class HierarchicalClassifier(BaseEstimator,ClassifierMixin):
  
  def __init__(self, clfs,n_levels):
    self.clfs = clfs
    self.n_levels = n_levels
    
  def fit_classifier_by_level(self,clf,X,y,level):
    y = self.get_labels_by_level(y,level)
    split_indices = self.get_split_indices(y)
    X_new,y_new = self.get_new_X_y(X,y,split_indices)
    
    clf.fit(X = X_new, y = y_new)

  def get_split_indices(self,y):
    if not isinstance(y, np.ndarray):
      y = np.array(y)
    index_sets = {i : np.where(y==i)[0] for i in np.unique(y)}
    
    return index_sets
    
    
  def get_labels_by_level(self,y,level):
    
    by_level = []
    for item in y:
      try:
        new_name = item.split('-')
        if len(new_name) > level+1:
          new_name = new_name[level]+'-TOT'
        else:
          new_name = new_name[level]
      except IndexError:
        new_name = 'dummy'
      
      by_level.append(new_name)
  
    by_level_array = np.asarray(by_level)
      
    return by_level_array
  
  def get_new_X_y(self,X,y,split_index):
    
    if not isinstance(y,np.ndarray):
      y = np.asarray(y)
    
    new_split_index = list(np.concatenate([v for k,v in split_index.items() if not k == 'dummy']))
    
    X_new = X[new_split_index,:]
    y_new = y[new_split_index]
    
    return X_new,y_new

  def fit(self,X,y):
    
    for i in range(self.n_levels):
      self.fit_classifier_by_level(self.clfs[i],X,y,i)
      
    return self
      
  def predict(self,X):
    predictions = []
    for i in range(X.shape[0]):
      if isinstance(X, np.ndarray):
        sample = X[i,:].reshape(1,X.shape[1])
      else:
        sample = X[i,:]
      compl_pred = ""
      for j in range(self.n_levels):
        pred = self.clfs[j].predict(sample)[0]
        if len(pred.split('-')) > 1:
          compl_pred += pred.strip('TOT')
          continue
        else:
          compl_pred += pred
          predictions.append(compl_pred)
          break
        predictions.append(compl_pred)
        
    pred_array = np.asarray(predictions)
    
    return pred_array
  
  def score(self,X,y):
    
    pred = self.predict(X)
    y = np.asarray(y)
    
    score = np.sum(y==pred)/len(y)
    
    return score
  
def kfold_cross_validation(clf,X,y,k,sorted_labels_name,estimator = 'keras',verbose = False):
  """
  Compute k fold CV for CNN. Keras API for scikit learn does not allow non Sequential models
  """
  
  kf = KFold(n_splits= k, shuffle = True, random_state = 7)
  
  unique_labels = np.unique(y,axis = 0).tolist()
  
  fold = 0
  
  by_class_precision_scores = []
  by_class_recall_scores = []
  by_class_f1_scores = []
  scores_micro_avg = []
  scores_macro_avg = []
  
  logger.info("Starting Cross Validation process...")
  
  for train_index, test_index in kf.split(X):
    train_id = list(train_index)
    test_id = list(test_index)
    X_train, X_test = X[train_id,:], X[test_id,:]
    y_train, y_test = y[train_index], y[test_index]
    
    if verbose:
      logger.info("Fold {}\n".format(fold))
      
    if estimator == 'keras':
      clf_fold = clf.compile_model(X,y)
    elif estimator == 'scikit-learn':
      clf_fold = clone(clf)
          
    clf_fold.fit(X_train,y_train)
    
    y_pred = clf_fold.predict(X_test)
    
      
    by_class_scores = prfs(y_pred,y_test, labels = unique_labels if estimator == 'scikit-learn' else None)
    
    scores_micro_avg.append(np.expand_dims(prfs(y_pred,y_test,average = 'micro')[:-1],0))
    
    scores_macro_avg.append(np.expand_dims(prfs(y_pred,y_test,average = 'macro')[:-1],0))
  
    by_class_precision_scores.append(np.expand_dims(by_class_scores[0],0))
                
    by_class_recall_scores.append(np.expand_dims(by_class_scores[1],0))
        
    by_class_f1_scores.append(np.expand_dims(by_class_scores[2],0))
            
    fold += 1
    
  if verbose :
    
    by_class_scores_dict = defaultdict(OrderedDict)
    
    mean_precision_by_class = np.mean(np.concatenate(by_class_precision_scores,0),0)
    mean_recall_by_class = np.mean(np.concatenate(by_class_recall_scores,0),0)
    mean_f1_by_class = np.mean(np.concatenate(by_class_f1_scores,0),0)
    
    by_class_scores_dict['P'] = {sorted_labels_name[i] : mean_precision_by_class[i] for i in range(len(sorted_labels_name))}
    by_class_scores_dict['R'] = {sorted_labels_name[i] : mean_recall_by_class[i] for i in range(len(sorted_labels_name))}
    by_class_scores_dict['F1'] = {sorted_labels_name[i] : mean_f1_by_class[i] for i in range(len(sorted_labels_name))}
    
    visualize_results = pandas.DataFrame.from_dict(by_class_scores_dict, orient = 'columns')
    logger.info("By class mean scores  :\n {}".format(visualize_results))
      
  m_micro_avg = np.mean(np.concatenate(scores_micro_avg,0),0)
  m_macro_avg = np.mean(np.concatenate(scores_macro_avg,0),0)
    
  logger.info("{0} fold CV micro average - P : {1} - R : {2} - F1 : {3}".format(k,m_micro_avg[0],m_micro_avg[1],m_micro_avg[2]))
  logger.info("{0} fold CV macro average - P : {1} - R : {2} - F1 : {3}".format(k,m_macro_avg[0],m_macro_avg[1],m_macro_avg[2]))
    
    
    
    
    
    
    
    
  
    
  
    
    
  
    
  
    
    
    
    
      
    
    
    
 
