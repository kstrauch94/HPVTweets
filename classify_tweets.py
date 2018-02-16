from load import load_data
from preprocess import preprocessing
from tweets_feature_extractor import build_pipeline_steps
from tweets_classification import optimize_hp,HierarchicalClassifier

from sklearn import svm
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_validate, cross_val_predict

from sklearn.metrics import classification_report, confusion_matrix

import numpy as np
import argparse
import os

from utils import plot_confusion_matrix,store_hyperparameters,by_class_error_analysis
import matplotlib.pyplot as plt

L_SVM_B = "l_svm_b"
RBF_SVM_B = "rbf_svm_b"
L_SVM_H = "l_svm_h"
RBF_SVM_H = "rbf_svm_h"
LR_B  = "lr_b"
LR_H  = "lr_h"

L_SVM_PARAMS = [{'C' : [1,64, 128, 256, 512, 1024]}]
RBF_SVM_PARAMS = [{'gamma' : [2e-7, 2e-6, 2e-5, 2e-4, 2e-3]},{'C' : [1,64, 128, 256, 512, 1024]}]
LR_PARAMS = [{'C' : [1,64, 128, 256, 512, 1024]}]


clf_choices = [L_SVM_B,RBF_SVM_B,L_SVM_H,RBF_SVM_H,LR_B,LR_H]


def parse_arguments():
  """
  Parse arguments via argparse.
  
  :return:
    parsed arguments
  """
  
  parser = argparse.ArgumentParser(description='Tweet classifier architecture')
  
  
  preprcessing_group = parser.add_argument_group('preprocessing')
  
  preprcessing_group.add_argument("--rm-url", action="store_true", help="Remove urls from tweet. If not urls will be converted to `url` string")
  preprcessing_group.add_argument("--red-len", action="store_false", help="Reduce words length. E.g. faaantastic -> fantaastic")
  preprcessing_group.add_argument("--lower", action="store_false", help="Lowercase all words but emoticons")
  preprcessing_group.add_argument("--rm-sw", action="store_true", help="Remove stopwords")
  preprcessing_group.add_argument("--rm-tagsmen", action="store_true", help="Remove tags and mentions from tweet")
  preprcessing_group.add_argument("--stem", action="store_true", help="Stem words")
  
  classifier_group = parser.add_argument_group('classifier')
  
  classifier_group.add_argument("-c", "--classifier", choices=clf_choices, default=L_SVM_B, help="Choose classifier")
  classifier_group.add_argument("--class-weights", action = "store_true", help="Apply class weights to classifier (inversely proportional to class frequencies in the input)")
  classifier_group.add_argument("--optim-single",action = "store_true", help="Optimize hyperparameters for single classifier")
  
  
  features_group = parser.add_argument_group('preprocessing')
  
  features_group.add_argument("--ngram-range",type = int, default = 2, help="Max value for building ngram matrix. E.g. `2` : bigrams")
  features_group.add_argument("--tfidf", action="store_true", help="apply tfidf to feature vector")
  features_group.add_argument("--tsvd", type=int, default=-1, help="apply dimensionality reduction to specified value with singular value decomposition (SVD)")
  features_group.add_argument("--clusters", action="store_true", help="tokenizer function will build clusters")
  features_group.add_argument("--postags", action="store_true", help="tokenizer function will build postags")
  features_group.add_argument("--sentnet", action="store_true", help="Senti net feature will be added to the vector")
  features_group.add_argument("--sentiwords", action="store_true", help="tokenizer function will build senti net words")
  features_group.add_argument("--subjscore", action="store_true", help="Subjectivity score feature will be added to the vector")
  features_group.add_argument("--bingliusent", action="store_true", help="Positive/Negative words features from Bing Liu")
  features_group.add_argument("--depsent", action="store_true", help="Dependencies feature for Positive/Negative words from Bing Liu")
  features_group.add_argument("--negwords", action="store_true", help="Negated words features")
  features_group.add_argument("--scale", action="store_true", help="Scale feature matrix")
  features_group.add_argument("--bigramsent", action="store_true", help="Bigram sentiment score feature will be added to the vector")
  features_group.add_argument("--unigramsent", action="store_true", help="Unigram sentiment score feature will be added to the vector")
  features_group.add_argument("--argscores", action="store_true", help="Argument lexicon score features will be added to the vector")


  
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
                              
  files_group.add_argument("--bigram-sent-file",default = "{}".format(os.path.join("data","hash-sentiments","bigrams-pmilexicon.txt")),
                              help="path to file containing bigrams hashtag sentiment scores. Default = data\\hash-sentiments\\bigrams-pmilexicon.txt")  

  files_group.add_argument("--unigram-sent-file",default = "{}".format(os.path.join("data","hash-sentiments","unigrams-pmilexicon.txt")),
                              help="path to file containing unigram hashtag sentiment scores. Default = data\\hash-sentiments\\unigrams-pmilexicon.txt")
                        
  files_group.add_argument("--arg-lexicon-folder",default = "{}".format(os.path.join("data","arg-lexicon")),
                              help="path to folder containing argument lexicon files. (please exclude trailing \\ after folder name) Default = data\\arg-lexicon")
                              
                              
  record_group = parser.add_argument_group('record')
  
  record_group.add_argument("--save", type = str, default = False, help="If true it writes a file with information about the test, else it just prints it")
  record_group.add_argument("--confusion-matrix",action = "store_true", help="Display confusion matrix")
  record_group.add_argument("--error-analysis", type = str, default = False, help="Save to a file (path to be provided) tweets misclassified")
  
  
  return parser.parse_args()

if __name__ == "__main__":
  
  args = parse_arguments()
  
  df = load_data(dep_file = args.tweets_file, annotations = args.annotations)
  
  # replace column of tokens with preprocessed ones 
  df['proc_toks'] = df['toks_pos'].apply(preprocessing,rm_url = args.rm_url, red_len = args.red_len,lower = args.lower,
    rm_sw = args.rm_sw, rm_tags_mentions = args.rm_tagsmen, stem = args.stem) 
  # still dataframe with all columns
  
  print("Shuffling data")
  np.random.seed(42)
  df = df.reindex(np.random.permutation(df.index))
  
  tweets = list(df['proc_toks'])
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
                           do_scaling = args.scale,
                           do_bigram_sent = args.bigramsent,
                           do_unigram_sent = args.unigramsent,
                           do_argument_scores = args.argscores,
                           deps = deps, 
                           stem = args.stem,
                           bingliu_pos_path = args.bingliu_pos,
                           bingliu_neg_path = args.bingliu_neg,
                           clusters_path = args.clusters_file,
                           bigram_sent_file = args.bigram_sent_file,
                           unigram_sent_file = args.unigram_sent_file,
                           arguments_folder = args.arg_lexicon_folder,
                           pos_tokens = pos,
                           subj_score_file = args.subjscore_file)
  
  

  X = pipeline_steps.fit_transform(tweets) 
  
  print("Resulting feature matrix shape {}".format(X.shape))
  
  
  if args.optim_single:
    
    print("Optimizing hyperparameters of : {} with 3-fold CV".format(args.classifier))

      
  if args.classifier == L_SVM_B:
      
    clf = svm.LinearSVC(class_weight = cl_weight)
    
    if args.optim_single:
      
      _ , clf = optimize_hp(clf,X,labels,L_SVM_PARAMS)
      
  elif args.classifier == RBF_SVM_B:
      
    clf = svm.LinearSVC(class_weight = cl_weight)
    
    if args.optim_single:
      
      _ , clf = optimize_hp(clf,X,labels,RBF_SVM_PARAMS)
    
  elif args.classifier == LR_B:
      
    clf = LogisticRegression(class_weight = cl_weight )
    
    if args.optim_single:
    
      _ ,clf = optimize_hp(clf,X,labels,LR_PARAMS)
    
  elif args.classifier == L_SVM_H:

    clf = HierarchicalClassifier(clfs = [svm.LinearSVC(class_weight = cl_weight) for _ in range(3)],
                                 params = [L_SVM_PARAMS] * 3)
    
    if args.optim_single:
      best_f1s = clf.optimize_classifiers(X,labels)
      
  elif args.classifier == RBF_SVM_H:

    clf = HierarchicalClassifier(clfs = [svm.SVC(C = 256, class_weight = cl_weight) for _ in range(3)],
                                 params = [RBF_SVM_PARAMS] * 3)
    
    if args.optim_single:
      best_f1s = clf.optimize_classifiers(X,labels)
    
  elif args.classifier == LR_H:
    clf = HierarchicalClassifier(clfs = [LogisticRegression(class_weight = cl_weight) for _ in range(3)],
                                 params = [LR_PARAMS] * 3)
    
    if args.optim_single:
      best_f1s = clf.optimize_classifiers(X,labels)
      
  print("Evaluating {} performances with 10 fold cross validation".format(args.classifier))  
      
  scoring = ["f1_micro", "f1_macro", "precision_micro", "precision_macro", "recall_micro", "recall_macro"]
  
  f1_scores = cross_validate(clf, X, labels, cv=10, scoring=scoring, return_train_score=False)

  y_pred = cross_val_predict(clf, X, labels, cv=10)
  
  report = classification_report(labels, y_pred)
      
  text = []
  text.append("classifier: {}\n".format(args.classifier))
  text.append("class weights: {}\n".format(args.class_weights))
  
  store_hyperparameters(clf,text)
  
  text.append("\n10 fold cross validation\n")
  
  text.append("preprocessing\n")
  text.append("remove url : {}\n".format(args.rm_url))
  text.append("reduce length : {}\n".format(args.red_len))
  text.append("lowercase : {}\n".format(args.lower))
  text.append("remove stopwords : {}\n".format(args.rm_sw))
  text.append("remove tags and mentions : {}\n".format(args.rm_tagsmen))
  text.append("stem : {}\n".format(args.stem))
  
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
  text.append("scaled features: {}\n".format(args.scale))
  text.append("bigram sentiment scores: {}\n".format(args.bigramsent))
  text.append("unigram sentiment scores: {}\n".format(args.unigramsent))
  text.append("argument lexicon scores: {}\n".format(args.argscores))

  
  text.append("Feature matrix shape: {}\n".format(X.shape))

  text.append("\n")
      
  for score_name, scores in f1_scores.items():
      text.append("average {} : {}\n".format(score_name,sum(scores)/len(scores)))
      
  text.append(report)
  
  for line in text:
      print(line)
   
      
  # write text to file to keep a record of stuff
  if args.save:
    preprocess = "rm"
    if args.rm_url:
      preprocess += "-url"
    if args.rm_sw:
      preprocess += "-sw"
    if args.rm_tagsmen:
      preprocess += "-tm"
    if args.stem: 
      preprocess += "-stem"
 
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
    if args.scale:
        features += "scale-"
    if args.optim_single:
        features += "optim-"
    if args.bigramsent:
        features += "bigramsent-"
    if args.unigramsent:
        features += "unigramsent-"
    if args.argscores:
        features += "argscores-"
    
    filename = "{}_{}_{}10cv.txt".format(args.classifier,preprocess,features)
    
    if not os.path.exists(args.save):
      os.mkdir(args.save)
    
    with open(os.path.join(args.save,filename), "w") as f:
        f.writelines(text)
        
  if args.confusion_matrix:
  
    cm = confusion_matrix(labels,y_pred)
    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(cm, classes=np.unique(labels),
                      title='Confusion Matrix')
    
    
    plt.savefig('confustion_matrix.png')
    
  if args.error_analysis:
    
    if not os.path.exists(args.error_analysis):
      os.mkdir(args.error_analysis)
    
    by_class_error_analysis(df = df, y_true = labels, y_pred = y_pred, limit = 10, error = 'FP', out_path = args.error_analysis )
    by_class_error_analysis(df = df, y_true = labels, y_pred = y_pred, limit = 10, error = 'FN', out_path = args.error_analysis )
    
  
        

    
    
    
