from load import load_data
from preprocess import preprocessing
from tweets_feature_extractor import build_pipeline_steps
from tweets_classification import optimize_hp,HierarchicalClassifier

from sklearn import svm
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_validate, cross_val_predict

from sklearn.metrics import classification_report

import numpy as np
import argparse
import copy
import sys
import os

SVM_B = "svm_b"
LR_B  = "lr_b"
SVM_H = "svm_h"
LR_H  = "lr_h"


SVM_PARAMS = [{'kernel' : ['linear','rbf']},{'C' : [1,50,100,150,200]}]
LR_PARAMS = [{'C' : [1,10,50,100]},{"tol": [1e-3, 1e-4, 1e-5, 1e-6]}]

CV = "cv"

clf_choices = [SVM_B,LR_B,SVM_H,LR_H]

C = "C"
GAMMA = "GAMMA"

NEIGHBORS = "neighbors"

def parse_arguments():
  """
  Parse arguments via argparse.
  
  :return:
    parsed arguments
  """
  
  parser = argparse.ArgumentParser(description='Tweet classifier architecture')
  
  
  preprcessing_group = parser.add_argument_group('preprocessing')
  
  preprcessing_group.add_argument("--rm-url", action="store_false", help="Remove urls from tweet. If not urls will be converted to `url` string")
  preprcessing_group.add_argument("--red-len", action="store_false", help="Reduce words length. E.g. faaantastic -> fantaastic")
  preprcessing_group.add_argument("--lower", action="store_false", help="Lowercase all words but emoticons")
  preprcessing_group.add_argument("--rm-sw", action="store_false", help="Remove stopwords")
  preprcessing_group.add_argument("--rm-tagsmen", action="store_true", help="Remove tags and mentions from tweet")
  
  classifier_group = parser.add_argument_group('classifier')
  
  classifier_group.add_argument("-c", "--classifier", choices=clf_choices, default=SVM_B, help="Choose classifier")
  classifier_group.add_argument("--class-weights", action = "store_true", help="Apply class weights to classifier (inversely proportional to class frequencies in the input)")
  classifier_group.add_argument("--optim-single",action = "store_true", help="Optimize hyperparameters for single classifier")
  
  preprocessing_group = parser.add_argument_group('preprocessing')
  
#  preprocessing_group.add_argument("--alt-preprocess", action="store_true", help="Use alternate preprocessing. This replaces mentions by @ and hashtags by # instead of deleting them.")
  preprocessing_group.add_argument("--ngram-range",type = int, default = 2, help="Max value for building ngram matrix. E.g. `2` : bigrams")
  preprocessing_group.add_argument("--tfidf", action="store_true", help="apply tfidf to feature vector")
  preprocessing_group.add_argument("--tsvd", type=int, default=-1, help="apply dimensionality reduction to specified value with singular value decomposition (SVD)")
  preprocessing_group.add_argument("--clusters", action="store_true", help="tokenizer function will build clusters")
  preprocessing_group.add_argument("--postags", action="store_true", help="tokenizer function will build postags")
  preprocessing_group.add_argument("--sentnet", action="store_true", help="Senti net feature will be added to the vector")
  preprocessing_group.add_argument("--sentiwords", action="store_true", help="tokenizer function will build senti net words")
  preprocessing_group.add_argument("--subjscore", action="store_true", help="Subjectivity score feature will be added to the vector")
  preprocessing_group.add_argument("--bingliusent", action="store_true", help="Positive/Negative words features from Bing Liu")
  preprocessing_group.add_argument("--depsent", action="store_true", help="Dependencies feature for Positive/Negative words from Bing Liu")
  preprocessing_group.add_argument("--negwords", action="store_true", help="Negated words features")
  
  files_group = parser.add_argument_group('files')
  
  files_group.add_argument("--tweets-file", required = True, 
                              help="path to file containing tweet (dependecy parsing applied)")
  
  files_group.add_argument("--annotations", required = True,
                              help="path to file containing tweet annotations")
  
  files_group.add_argument("--clusters-file", default="{}".format(os.path.join("data","50mpaths2")), 
                              help="path to file containing word clusters. Default = data\\50mpaths2")
  
  files_group.add_argument("--subjscore-file", default = '{}'.format(os.path.join("data","subj_score.txt")),
                              help="path to file subjectivity scores file. Default = data\\subj_score.txt")
  
  files_group.add_argument("--bingliu-pos", default = "{}".format(os.path.join("data","bingliuposs.txt")),
                              help="path to file containing bing liu positive words. Default = data\\bingliuposs.txt")
  
  files_group.add_argument("--bingliu-neg",default = "{}".format(os.path.join("data","bingliunegs.txt")),
                              help="path to file containing bing liu negative words. Default = data\\bingliunegs.txt")
  
                     
  record_group = parser.add_argument_group('record')
  
  record_group.add_argument("--save", action="store_true", help="If true it writes a file with information about the test, else it just prints it")
  
  return parser.parse_args()

if __name__ == "__main__":
  
    args = parse_arguments()
    
    df = load_data(dep_file = args.tweets_file, annotations = args.annotations)
    
    # replace column of tokens with preprocessed ones 
    df['toks'] = df['toks_pos'].apply(preprocessing,rm_url = args.rm_url, red_len = args.red_len,lower = args.lower,
      rm_sw = args.rm_sw, rm_tags_mentions = args.rm_tagsmen) 
    # still dataframe with all columns
    
    print("Shuffling data")
    df = df.reindex(np.random.permutation(df.index))
    
    tweets = list(df['toks'])
    labels = list(df['label'])
    pos = list(df['pos'])
    deps = list(df['dep'])
    
    cl_weight = 'balanced' if args.class_weights else None
    
    
    pipeline_steps =  build_pipeline_steps(ngram_range = args.ngram_range,
                            do_tfidf = args.tfidf,
                             do_tsvd = args.tsvd,
                             do_neg_words = args.negwords,
                             do_bingliu = args.bingliusent,
                             do_clusters = args.clusters,
                             do_postags = args.postags,
                             do_sentnet = args.sentnet,
                             do_subjscore = args.subjscore,
                             do_dep_sent = args.depsent,
                             do_sentiwords = args.sentiwords,
                             deps = deps, 
                             bingliu_pos_path = args.bingliu_pos,
                             bingliu_neg_path = args.bingliu_neg,
                             clusters_path = args.clusters_file,
                             pos_tokens = pos,
                             subj_score_file = args.subjscore_file)
    
    

    X = pipeline_steps.fit_transform(tweets)
    
    print("Resulting feature matrix shape {}".format(X.shape))
    
    
    if args.optim_single:
      
      print("Optimizing hyperparameters of : {} with 3-fold CV".format(args.classifier))

        
    if args.classifier == SVM_B:
        
      clf = svm.SVC(class_weight = cl_weight)
      
      clf = clf if not args.optim_single else optimize_hp(clf,X,labels,SVM_PARAMS)

    elif args.classifier == LR_B:
        
      clf = LogisticRegression(class_weight = cl_weight )
      
      clf = clf if not args.optim_single else optimize_hp(clf,X,labels,LR_PARAMS)
      
      
    elif args.classifier == SVM_H:
            
      clf = HierarchicalClassifier(clfs = [svm.SVC(class_weight = cl_weight) for _ in range(3)],
                                   params = [SVM_PARAMS] * 3)
      
      if args.optim_single:
        clf.optimize_classifiers(X,labels)
      
    elif args.classifier == LR_H:
      clf = HierarchicalClassifier(clfs = [LogisticRegression(class_weight = cl_weight) for _ in range(3)],
                                   params = [LR_PARAMS] * 3)
      
      if args.optim_single:
        clf.optimize_classifiers(X,labels)
      

      
    print("Evaluating {} performances with 10 fold cross validation".format(args.classifier))  
        
    scoring = ["f1_micro", "f1_macro", "precision_micro", "precision_macro", "recall_micro", "recall_macro"]
    
    f1_scores = cross_validate(clf, X, labels, cv=10, scoring=scoring, return_train_score=False)

    y_pred = cross_val_predict(clf, X, labels, cv=10)
    
    report = classification_report(labels, y_pred)
    
    text = []
    text.append("classifier: {}\n".format(args.classifier))
    text.append("class weights: {}\n".format(args.class_weights))
    
    if args.classifier == SVM_B:
      text.append("Hyperparameters: kernel = {}. C = {}, gamma = {}\n".format(clf.kernel,clf.C,clf.gamma))
      
    elif args.classifier == LR_B:
      text.append("Hyperparameters: C = {}, tol = {}\n".format(clf.C,clf.tol))
      
    elif args.classifier == SVM_H:
      for idx,clf in enumerate(clf.clfs):
        text.append("Hyperparameters {}: kernel = {}. C = {}, gamma = {}\n".format(idx,clf.kernel,clf.C,clf.gamma))
        
    elif args.classifier == LR_H:
      for idx,clf in enumerate(clf.clfs):
        text.append("Hyperparameters {}: C = {}, tol = {}\n".format(idx,clf.C,clf.tol))
    
    text.append("10 fold cross validation")
    
    text.append("preprocessing\n")
    text.append("remove url : {}\n".format(args.rm_url))
    text.append("reduce length : {}\n".format(args.red_len))
    text.append("lowercase : {}\n".format(args.lower))
    text.append("remove stopwords : {}\n".format(args.rm_sw))
    text.append("remove tags and mentions : {}\n".format(args.rm_tagsmen))
    
    text.append("features\n")
    text.append("ngram_range: {}\n".format(args.ngram_range))
    text.append("tfidf: {}\n".format(args.tfidf))
    text.append("tsvd : {}\n\n".format(args.tsvd))
    text.append("cluster: {}\n".format(args.clusters))
    text.append("postags: {}\n".format(args.postags))
    text.append("senti net: {}\n".format(args.sentnet))
    text.append("senti words: {}\n".format(args.sentiwords))
    text.append("subjective score: {}\n".format(args.subjscore))
    text.append("bing liu sent words: {}\n".format(args.bingliusent))
    text.append("dependency sent words: {}\n".format(args.depsent))
    text.append("negated words: {}\n".format(args.negwords))
    
    text.append("Feature matrix shape: {}\n".format(X.shape))

    text.append("\n")

    for score_name, scores in f1_scores.items():
        text.append("average {} : {}\n".format(score_name,sum(scores)/len(scores)))
        
    text.append(report)
    
    for line in text:
        print(line)
     
        
    # write text to file to keep a record of stuff
    if args.save:
        features = ""
        features += "{}gram-".format(args.ngram_range)
        if args.tfidf:
            features = "tfidf-"
        if args.tsvd > 0:
            features += "tsvd-{}-".format(args.tsvd)
        if args.clusters:
            features += "clusters-"
        if args.postags:
            features += "postags-"
        if args.sentnet:
            features += "sentnet-"
        if args.sentiwords:
            features += "sentiwords-"
        if args.subjscore:
            features += "subjscore-"
        if args.bingliusent:
            features += "bingliu-"
        if args.depsent:
            features += "dep-"
        if args.negwords:
            features += "neg-"
        
        filename = "{}-{}-10cv.txt".format(args.classifier, features)
    
        with open(filename, "w") as f:
            f.writelines(text)
    
    
    
