#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 16:38:28 2017

@author: Samuele Garda
"""
import logging
import itertools
from utils import timeit
import numpy as np
from tensorflow import keras
from keras import backend as K
from keras.preprocessing import sequence
from keras.layers.core import Dense,Dropout
from keras.layers.convolutional import Conv1D,MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.layers import Flatten, Input
from keras.models import Model
from keras.layers.merge import Concatenate
from keras import regularizers

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(module)s: %(message)s', level = 'INFO')


def kurata_initializer(shape,ys,dtype = None):
  
  UP_glorot = np.sqrt(6 / (shape[0] + shape[1]))
  
  unique_ys = np.unique(ys,axis = 0)
  
  unique_idxs = [np.nonzero(u)[0].tolist() for u in unique_ys]

  products = set(c for t in [list(itertools.product(l,repeat = 2)) for l in unique_idxs] for c in t)
  
  weights = np.random.uniform(low = -0.01,high = 0.01, size = shape)
  
  for (i,j) in products:
    weights[i,j] = UP_glorot
    
  return K.tf.Variable(weights, K.tf.float32)
  

class SentenceCNN(object):
  
  def __init__(self,ngrams,num_filters,mode,batch,epochs):
      
    self.ngrams = ngrams
    self.num_filters = num_filters
    self.mode = mode
    self.batch = batch
    self.epochs = epochs
    self.model = None
    self.vocab  = set()
    self.maxlen = None
    self.word2index = None
    self.embed_dim = None
    self.embedding_matrix = None   
    self.multilabel_init = None
  
  @timeit
  def init_model(self,texts,we_file):
    """
    Creates word embeddings dictionary extracted from file.
    
    :param we_file: embeddings file in word2vec format  
    """
    
    with open(we_file) as we:
      embeddings_dict = {line.split()[0] : np.asarray(line.split()[1:], dtype = 'float32') for line in we}
      
    self.embed_dim = len(next(iter(embeddings_dict.values())))
    
    self.maxlen = max([len(words) for words in texts])
    
    for words in texts:
      self.vocab.update(words)
    
    self.word2index = {word : wid+1 for wid,word in enumerate(self.vocab)}
    
    print("Created word2index")
#    index2word = {v : k for k,v in word2index.items()}
    
    xs = [[self.word2index[word] for word in tweet] for tweet in texts]
    
    X = sequence.pad_sequences(xs, maxlen = self.maxlen)
    
    print("Padded sentences")
    
    found = 0
    self.embedding_matrix = np.random.uniform(low = 0.01, high = 0.01, size = [len(self.vocab)+1, self.embed_dim])
    for word, i in self.word2index.items():
      embedding_vector = embeddings_dict.get(word)
      if embedding_vector is not None:
          found += 1
          # words not found in embedding index will be all-zeros.
          self.embedding_matrix[i] = embedding_vector
      else:
        pass
      
    print("Created embedding matrix")
    
    return X
    
  def compile_model(self,X,y):
    
    print("Define architecture")
  
    main_input = Input(shape=(X.shape[1],), dtype='float32', name='main_input')
    
    
    embed = Embedding(len(self.vocab)+1, self.embed_dim, 
                                  input_length= self.maxlen, weights=[self.embedding_matrix],
                                  name = "embedding_layer",
                                  trainable = True)(main_input)
    
    
    convolutional_filters = []
    for filter_size in self.ngrams:
      
      conv = Conv1D(filters= self.num_filters, 
                    padding = "valid",
                    kernel_size= filter_size, activation="relu",
                    name = "convolution_layer_{}".format(filter_size))(embed)
      
        
      max_pool = MaxPooling1D(pool_size = 2, name = "max_pooling_layer_{}".format(filter_size))(conv)
      flatten_max  = Flatten(name = "flatten_max_layer_{}".format(filter_size))(max_pool)
      convolutional_filters.append(flatten_max)
      
    
    merged = Concatenate()(convolutional_filters)
    
    dropout = Dropout(0.5)(merged)
    
    dense_pre_out = Dense(200, activation = "relu", name = "dense_layer_pre_out")(dropout)
    
    if self.mode == "plain":
  
      final = Dense(y.shape[1], activation="softmax", kernel_regularizer= regularizers.l2(0.01), name = "softmax_layer")(dense_pre_out)
    
    elif self.mode == "multilabel":
      
      shape = [dropout._shape_as_list(),y.shape[1]]
      
      kernel_init = kurata_initializer(shape = shape, ys = y) if self.multilabel_init else 'glorot_uniform' 
      
      final = Dense(y.shape[1], activation="sigmoid", kernel_regularizer= regularizers.l2(0.01),
                    kernel_initializer = kernel_init, name = "sigmoid_layer")(dropout)
  

    self.model = Model(inputs = main_input, outputs = final)
    
    loss = 'categorical_crossentropy' if self.mode == "plain" else 'binary_crossentropy'
    
    self.model.compile(loss = loss ,optimizer = 'adam',metrics = ['accuracy'])
    
#    print(self.model.summary())
    
    return self
    
  @timeit 
  def fit(self,X,y):
    
    self.model.fit(X, y, batch_size= self.batch,epochs= self.epochs, verbose = 0)

  
  def predict(self,xs):
    
    probs = self.model.predict(xs)
    
    if self.mode == 'plain':
      
      preds = keras.utils.to_categorical(probs.argmax(axis = -1))
      
    elif self.mode == 'multilabel':
  
      probs[probs>=0.5] = 1
      
      probs[probs<0.5] = 0
      
      preds  = probs
    
    return preds
    
    
  def init_from_model(self,model):
    
    self.vocab  = model.vocab
    self.maxlen = model.maxlen
    self.word2index = model.word2index
    self.embedding_matrix = model.embedding_matrix
    self.embed_dim = model.embed_dim


    
  
  
    




