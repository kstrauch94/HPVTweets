#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 16:25:28 2017

@author: Samuele Garda
"""

import argparse
import logging
import numpy 
from load import TweetsLoader
from preprocess import TweetsPreprocessor
from feature_extraction import FeatureExtractor as fe
from classification import EncodeLables,HierarchicalClassifier,kfold_cross_validation
from CNN import SentenceCNN
from sklearn.svm import SVC,LinearSVC


logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(module)s: %(message)s', level = 'INFO')


def reference_paper_texts_pos(data,preprocessor):
  

  logger.info("Apply preprocessing to data as in `Du et al. 2017`...")
  
  reference_data = data.words.apply(preprocessor.ref_processing)
  
  pos_texts = list(reference_data.apply(preprocessor.pos_tag))
  
  reference_texts = list(reference_data.apply(preprocessor.tokenize))
  
  return reference_texts,pos_texts


def spell_checked_texts(data,spell_check_preprocessor):
    
  logger.info("Apply preprocessing with Spell Checker - This might take a while...")
  
  preprocess_data = data.words.apply(spell_check_preprocessor.apply_processing)
  
  spell_check_texts = list(preprocess_data.apply(spell_check_preprocessor.tokenize))
  
  return spell_check_texts


def experiment_1(X_ref,X_ref_spellcheck,labels,sorted_label_names,n_folds):
    
  encoder_mc,ys = EncodeLables.multiclass_labels(labels)
  
  
  logger.info("{} CV - Plain SVM RBF kernel - reference paper feature".format(n_folds))
  
  refernce_plain = SVC(C = 256, gamma = 2e-5)
    
  logger.info("{} CV - Plain SVM RBF kernel - reference paper feature".format(n_folds))
  kfold_cross_validation(clf = refernce_plain,X = X_ref,y = ys,k = n_folds,
                         sorted_labels_name = sorted_label_names, estimator = 'scikit-learn',verbose = True)
  
  logger.info("{} CV - Plain SVM RBF kernel - reference paper feature + spelling correction".format(n_folds))
  kfold_cross_validation(clf = refernce_plain, X = X_ref_spellcheck,y = ys,k = n_folds,
                         sorted_labels_name = sorted_label_names, estimator = 'scikit-learn',verbose = True)
  

def my_exp_preprocessed_texts(data,standard_preprocessor):
    
  logger.info("Apply preprocessing without Spell Checker...")
  
  preprocess_data = data.words.apply(standard_preprocessor.apply_processing)
  
  tokenized_texts = list(preprocess_data.apply(standard_preprocessor.tokenize))
  
  postagged_texts = list(preprocess_data.apply(standard_preprocessor.pos_tag))
  
  return tokenized_texts,postagged_texts
  
def experiment_2(X_ref,X_lsa,labels,sorted_label_names,n_folds = 10):
  
  
  encoder_mc,ys = EncodeLables.multiclass_labels(labels)
  
  hier_ys = EncodeLables.get_hierarchical_names(labels)
  
  linear_plain_lsa = LinearSVC()
  reference_hier = HierarchicalClassifier([SVC(C = 256, gamma = 2e-5),SVC(C = 256, gamma = 2e-4),SVC(C = 256, gamma = 2e-4)],3)
  hier_linear_svm = HierarchicalClassifier([LinearSVC(),LinearSVC(),LinearSVC()],3)
  
  
  logger.info("{} CV - Plain SVM linear kernel - LSA features...")
  
  kfold_cross_validation(clf = linear_plain_lsa,X = X_lsa, y = ys,
        sorted_labels_name = sorted_label_names,k = n_folds , estimator = 'scikit-learn',verbose = True)
  
  logger.info("{} CV - Hierarchical SVM RBF kernel - reference features".format(n_folds))
  
  kfold_cross_validation(clf = reference_hier,X = X_ref,y = hier_ys,
                         sorted_labels_name = sorted_label_names,k = n_folds , estimator = 'scikit-learn',verbose = True)
  
  logger.info("{} CV - Hierarchical SVMs linear kernel - LSA features".format(n_folds))
  
  kfold_cross_validation(clf = hier_linear_svm, X = X_lsa,y = hier_ys,
                         sorted_labels_name = sorted_label_names,k = n_folds , estimator = 'scikit-learn',verbose = True)
  

def experiment_3(texts,labels,we_file,sorted_label_names,n_folds = 10):

  
  encoder_mc,ys = EncodeLables.multiclass_labels(labels)
  
  categorical_ys = EncodeLables.categorical_labels(ys)
  
  hier_ys = EncodeLables.get_hierarchical_names(labels)
  encoder_ml , multilable_ys = EncodeLables.multilabel_binirizer([y.split('-') for y in hier_ys])
  
  logger.info("Initialiazing CNN architecture for multiclass classification task...")
  
  cnn_plain = SentenceCNN(ngrams = [3,4,5],num_filters = 12 ,mode = 'plain',batch = 32 ,epochs = 20)
  X_cnn = cnn_plain.init_model(texts,we_file)
  cnn_plain.compile_model(X_cnn, categorical_ys)
  
  logger.info("Initialiazing CNN architecture for multilabel classification task...")
  
  cnn_multilabel = SentenceCNN(ngrams = [3,4,5],num_filters = 12 ,mode = 'multilabel',batch = 32 ,epochs = 20)
  cnn_multilabel.init_from_model(cnn_plain)
  cnn_multilabel.compile_model(X_cnn,multilable_ys)
      
  logger.info("{} CV - Multiclass CNN".format(n_folds))
  
  kfold_cross_validation(clf = cnn_plain, X = X_cnn, y = categorical_ys,
        sorted_labels_name = sorted_label_names,k = n_folds , estimator = 'keras',verbose = True)
  
  logger.info("{} CV - Multilabel CNN".format(n_folds))
  
  kfold_cross_validation(clf = cnn_multilabel,X = X_cnn,y = multilable_ys,
                         sorted_labels_name = sorted_label_names,k=n_folds,verbose = True)
  
  
def experiment_4(X,labels,sorted_label_names,n_folds = 10):
  
  hier_ys = EncodeLables.get_hierarchical_names(labels)
  
  reference_hier = HierarchicalClassifier([SVC(C = 256, gamma = 2e-5),SVC(C = 256, gamma = 2e-4),SVC(C = 256, gamma = 2e-4)],3)
  
  kfold_cross_validation(clf = reference_hier,X = X, y = hier_ys,
        sorted_labels_name = sorted_label_names,k = n_folds , estimator = 'scikit-learn',verbose = True)
  

def parse_arguments():
  
  parser = argparse.ArgumentParser(description='Experiments on twitter dataset for opinion mining on HPV vaccines')
  parser.add_argument('-e', '--exp', choices = (1,2,3,4),type = int, required=True, help='Experiments.')
  args = parser.parse_args()
  return args

if __name__ == "__main__":
  
  numpy.random.seed(7)

  TWEET_DATA = './data/dataset/P3_vaccine-twitter-data.tsv'
  TWEET_ANNOTATIONS = './data/dataset/TweetsAnnotation.txt' 
  ARK_PARSER = ['./data/dataset/ark-tweet-nlp-0.3.2/runTagger.sh']
  HUNSPELL = ['/usr/share/hunspell/en_US.dic', '/usr/share/hunspell/en_US.aff']
   
  TWITTER_SSWE = 'data/we/sswe-u_tang.txt'
  TWITTER_WE = 'data/we/glove.twitter.27B.50d.txt'
  TWEET_CLUSTERS = './data/50mpaths2'
  
  BINGLIU_POS = './data/bingliu_lexicons/bingliuposs.txt'
  BINGLIU_NEG = './data/bingliu_lexicons/bingliunegs.txt'
  
  SORTED_LABELS_NAME = ['NegCost', 'NegEfficacy', 'NegOthers', 'NegResistant', 'NegSafety','Neutral', 'Positive', 'Unrelated']
  
  FOLDS = 10
  
  args = parse_arguments()
    
  data = TweetsLoader(ark_parser = ARK_PARSER).load(tweet_data= TWEET_DATA, annotations=TWEET_ANNOTATIONS)
  
  ref_preprocessor = TweetsPreprocessor(hunspell_dics=HUNSPELL)
    
  spell_chek_preprocessor = TweetsPreprocessor(reduce_len = True, lowercase = True, spell = True, hunspell_dics = HUNSPELL)
  
  standard_preprocessor = TweetsPreprocessor(reduce_len = True, lowercase = True, spell = False, hunspell_dics = HUNSPELL)
  
  labels = list(data.label)
  
  if args.exp == 1:
    
    ref_texts, postexts = reference_paper_texts_pos(data,ref_preprocessor)
    
    sc_texts = spell_checked_texts(data, spell_chek_preprocessor)
    
    pos_feat = fe.posfeatures(postexts)
    
    cluster_feat = fe.tweetclusterfeatures(ref_texts,TWEET_CLUSTERS )   
    
    ngrams_feat = fe.ngramsfeatures(ref_texts, ngram_range = (1,2))
  
    ngrams_feat_sc = fe.ngramsfeatures(sc_texts, ngram_range = (1,2))
    
    X_ref = fe.concat_feat(ngrams_feat,pos_feat,cluster_feat)
    
    X_sc = fe.concat_feat(ngrams_feat_sc,pos_feat,cluster_feat)
    
    experiment_1(X_ref=X_ref,X_ref_spellcheck=X_sc,labels=labels,sorted_label_names=SORTED_LABELS_NAME,n_folds=FOLDS)
        
  
  elif args.exp == 2:
    
    ref_texts, pos = reference_paper_texts_pos(data,ref_preprocessor)
    
    standard_preprocess_texts,postagged_texts = my_exp_preprocessed_texts(data,standard_preprocessor)
    
    pos_feat = fe.posfeatures(pos)
    
    cluster_feat = fe.tweetclusterfeatures(ref_texts,TWEET_CLUSTERS )
    
    ngrams_feat = fe.ngramsfeatures(ref_texts, ngram_range = (1,2))
    
    X_ref = fe.concat_feat(ngrams_feat,pos_feat,cluster_feat)
    
    X_lsa = fe.lsafeatures(postagged_texts, ngram_range = (1,3), k = 300)
    
    experiment_2(X_ref,X_lsa,labels,SORTED_LABELS_NAME,FOLDS)
    
    
  elif args.exp == 3:
    
    standard_preprocess_texts,postagged_texts = my_exp_preprocessed_texts(data,standard_preprocessor)
    
    experiment_3(texts = standard_preprocess_texts,
                               labels = labels,
                               sorted_label_names = SORTED_LABELS_NAME,
                               we_file = TWITTER_WE,
                               n_folds = FOLDS)
    
  elif args.exp == 4:
    
    standard_preprocess_texts,postagged_texts = my_exp_preprocessed_texts(data,standard_preprocessor)
    
    ngrams_feat = fe.ngramsfeatures(standard_preprocess_texts, ngram_range = (1,2))
    
    bingliufeatures = fe.getsentimentfeatures(standard_preprocess_texts,pos_file= BINGLIU_POS,neg_file = BINGLIU_NEG)
    
    negatedwordfeatures = fe.getnegatedwordfeatures(standard_preprocess_texts)
    
    X = fe.concat_feat(ngrams_feat,bingliufeatures,negatedwordfeatures)
    
    experiment_4(X, labels,SORTED_LABELS_NAME,FOLDS )
    
    
    
    
    
    
    
    
    
    
    
  
  
    
