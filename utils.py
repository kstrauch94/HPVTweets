#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 15:04:21 2018

@author: Samuele Garda
"""

import os
import matplotlib.pyplot as plt
import itertools
import numpy as np
from tweets_classification import HierarchicalClassifier
from sklearn.linear_model import LogisticRegression


from sklearn import svm

# TAKEN FROM : 
#  http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    else:
      pass

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



def store_hyperparameters(clf,text):
  """
  Add to list of info to be printed hyperparameters of classifiers.
  
  :params:
    clf (classifier) : classifier 
    text (list) : list to which append infos
  """
  
  def base_clf_hp(clf,text):

    if isinstance(clf,svm.LinearSVC) or isinstance(clf,LogisticRegression):
      text.append("Hyperparameters: C = {}\n".format(clf.C))
    
    elif isinstance(clf,svm.SVC):
      text.append("Hyperparameters: C = {}, gamma = {}\n".format(clf.C,clf.gamma))
      
  if isinstance(clf,HierarchicalClassifier):
    for clf in clf.clfs:
      base_clf_hp(clf,text)
  
  else:

    base_clf_hp(clf,text)
    
    
def by_class_error_analysis(df,y_true,y_pred,limit,error,out_path):
  """
  False Positive, False Negative estimation in a one-vs-rest way.
  """
  
  if error == 'FP':
    out_file = open(os.path.join(out_path, 'error.FP'),'w+') 
  else :
    out_file = open(os.path.join(out_path, 'error.FN'),'w+')
  
  unique_labels = np.unique(y_true)
  
  y_true = np.asarray(y_true)
  
  for label in unique_labels:
    out_file.write("{}\n".format(str(label).upper()))
    
    if error == 'FP':
      error_idx = np.where((y_true!=label) & (y_pred==label))[0] #take indices
    else:
      error_idx = np.where((y_true==label) & (y_pred!=label))[0] #take indices
            
    if len(error_idx) < 1:
      out_file.write("No {}\n".format(error))
      
    else:
      
      error_idx = error_idx if len(error_idx) <= limit else np.random.choice(error_idx, size = limit)
    
      for e_idx in error_idx:
        out_file.write("Tweet : `{}` - true : `{}` - pred : `{}`\n".format(' '.join(df.toks[e_idx]),y_true[e_idx],y_pred[e_idx]))
        
  out_file.close()
  