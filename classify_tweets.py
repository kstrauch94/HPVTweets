from tweet_data_parser import read_tab_sep
from tweets_ml import build_tokenizer, preprocess
from tweets_ml import TweetClassifierH, TweetClassifierBaseSVM, TweetClassifierKNN, TweetClassifierLR
from tweets_ml import build_pipeline_steps

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate, cross_val_predict

from sklearn.metrics import classification_report

import argparse
import copy

SVM_B = "svm_b"
KNN_B = "knn_b"
LR_B  = "lr_b"
RF_B  = "rf_b"

SVM_H = "svm_h"
KNN_H = "knn_h"

KNN_2SVM = "knn_2svm"

clf_choices = [SVM_B, KNN_B, LR_B, RF_B, KNN_H, SVM_H, KNN_2SVM]

C = "C"
GAMMA = "GAMMA"

NEIGHBORS = "neighbors"

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Tweet classifier architecture')
    
    classifier_group = parser.add_argument_group('classifier')
    
    classifier_group.add_argument("-c", "--classifier", choices=clf_choices, default=SVM_H, help="Choose classifier")
    classifier_group.add_argument("--cv-k", type=int, default=3, help="choose k for a k-fold cross validation")
    classifier_group.add_argument("--dim-reduction", type=int, default=-1, help="apply dimensionality reduction to specified value with singular value decomposition (SVD)")
    classifier_group.add_argument("--tfidf", action="store_true", help="apply tfidf to feature vector")
    classifier_group.add_argument("--neighbors", type=int, default=10, help="Neighbors parameter for KNN classifier")
    
    
    preprocessing_group = parser.add_argument_group('preprocessing')
    
    preprocessing_group.add_argument("--alt-preprocess", action="store_true", help="Use alternate preprocessing. This replaces mentions by @ and hashtags by # instead of deleting them.")
    preprocessing_group.add_argument("--bigrams", action="store_true", help="tokenizer function will build bigrams")
    preprocessing_group.add_argument("--clusters", action="store_true", help="tokenizer function will build clusters")
    preprocessing_group.add_argument("--postags", action="store_true", help="tokenizer function will build postags")
    preprocessing_group.add_argument("--postags-bg", action="store_true", help="tokenizer function will build postags bigrams")
    preprocessing_group.add_argument("--length", action="store_true", help="Length feature will be added to the vector")
    preprocessing_group.add_argument("--sentnet", action="store_true", help="Senti net feature will be added to the vector")
    preprocessing_group.add_argument("--sentiwords", action="store_true", help="tokenizer function will build senti net words")
    preprocessing_group.add_argument("--subjscore", action="store_true", help="Subjectivity score feature will be added to the vector")
    

    files_group = parser.add_argument_group('files')
    
    files_group.add_argument("--tweets-file", default="Data\\merged_tweets.tsv", 
                                help="path to file containing label \\t tweet lines. Default = Data\merged_tweets.tsv") 
    files_group.add_argument("--clusters-file", default="Data\\tweet_word_clusters.txt", 
                                help="path to file containing word clusters. Default = Data\\tweet_word_clusters.txt")
    files_group.add_argument("--postags-file", default="Data\\processed_tags.txt", 
                                help="path to file containing pos tags. Default = Data\\pos_tagged_tweet.csv")                         
                                
    record_group = parser.add_argument_group('record')
    
    record_group.add_argument("--save", action="store_true", help="If true it writes a file with information about the test, else it just prints it")
    
    args = parser.parse_args()
    
    clusters_path = args.clusters_file
    tweets_path= args.tweets_file
    annotations_path = "Data\\TweetsAnnotation.txt"
    postag_file = args.postags_file

    all_svm_C = 512
    related_svm_C = 128
    negative_svm_C = 512

    all_svm_gamma = 0.00002
    related_svm_gamma = 0.0002
    negative_svm_gamma = 0.0002

    clist = [all_svm_C, related_svm_C, negative_svm_C]
    gammalist = [all_svm_gamma, related_svm_gamma, negative_svm_gamma]

    # unpack labels and repack id + tweets
    tweet_list = read_tab_sep(tweets_path)
    id, labels, tweets = zip(*tweet_list)
    tweets = zip(id, tweets)
    
    tokenizer = build_tokenizer(do_bigrams=args.bigrams, do_clusters=args.clusters, do_postags=args.postags, 
                                do_postags_bg=args.postags_bg, do_sentiwords=args.sentiwords, cluster_lookup_file=clusters_path, pos_tag_file=postag_file)
    
    pipeline_steps = build_pipeline_steps(tokenizer=tokenizer, preprocess=lambda id_text : preprocess(id_text, alt_pre=args.alt_preprocess), do_length=args.length,
                                    do_tfidf=args.tfidf, do_sentnet=args.sentnet, do_subjscore=args.subjscore, dim_reduction=args.dim_reduction)
   
    """
    #test
    tweet = ["626942369506684932", "@DewsNewz RI Mandates HPV Vaccine for 7th Graders #HPV"]
    tweet2 = ["1", "I am good and happy"]
    
    #from tweets_ml import tweets_length
    #print(tweets_length([tweet]))
    
    #from tweets_ml import net_sentiment
    #print(net_sentiment([tweet, tweet2]))
    
    #from tweets_ml import tweet_wordnet
    #print tweet_wordnet(tweet[1])
    
    from tweets_ml import sub_score
    print sub_score([tweet, tweet2])
    
    import sys
    sys.exit()   
    
    #    
    """
    kwargs_pre = {"pipeline_steps": pipeline_steps}
    kwargs = {1: copy.copy(kwargs_pre), 2: copy.copy(kwargs_pre), 3: copy.copy(kwargs_pre)}
    
    
    if args.classifier == SVM_B:
        clf = TweetClassifierBaseSVM(**kwargs_pre)
    
    elif args.classifier == KNN_B:
        kwargs_pre.update({NEIGHBORS: args.neighbors})
        clf = TweetClassifierKNN(**kwargs_pre)  

    elif args.classifier == LR_B:
        clf = TweetClassifierLR(**kwargs_pre)
        
    elif args.classifier == RF_B:
        clf = TweetClassifierRF(**kwargs_pre)          
    
    elif args.classifier == SVM_H:
        kwargs[1].update({C: all_svm_C, GAMMA: all_svm_gamma})
        kwargs[2].update({C: related_svm_C, GAMMA: related_svm_gamma})
        kwargs[3].update({C: negative_svm_C, GAMMA: negative_svm_gamma})
        clf = TweetClassifierH(lambda x: TweetClassifierBaseSVM, kwargs)
        
    elif args.classifier == KNN_H:
        kwargs[1].update({NEIGHBORS: args.neighbors})
        kwargs[2].update({NEIGHBORS: args.neighbors})
        kwargs[3].update({NEIGHBORS: args.neighbors})
        clf = TweetClassifierH(lambda x: TweetClassifierKNN, kwargs)

    elif args.classifier == KNN_2SVM:
        kwargs[1].update({NEIGHBORS: args.neighbors})
        kwargs[2].update({C: related_svm_C, GAMMA: related_svm_gamma})
        kwargs[3].update({C: negative_svm_C, GAMMA: negative_svm_gamma})
        
        def get_clf(tier):
            if tier == 1: return TweetClassifierKNN
            if tier == 2 or tier == 3: return TweetClassifierBaseSVM
                
        clf = TweetClassifierH(get_clf, kwargs)
    
    scoring = ["f1_micro", "f1_macro", "precision_micro", "precision_macro", "recall_micro", "recall_macro"]
    
    f1_scores = cross_validate(clf, tweets, labels, cv=args.cv_k, scoring=scoring, return_train_score=False)

    
    y_pred = cross_val_predict(clf, tweets, labels, cv=args.cv_k)
    
    report = classification_report(labels, y_pred)
    
    text = []
    text.append("classifier: {}\n".format(args.classifier))
    text.append("dim reduction : {}\n\n".format(args.dim_reduction))
    
    text.append("cross val: {}-fold\n".format(args.cv_k))
    
    text.append("alt preprocess: {}\n\n".format(args.alt_preprocess))


    
    text.append("features\n")
    text.append("bigrams: {}\n".format(args.bigrams))
    text.append("cluster: {}\n".format(args.clusters))
    text.append("postags: {}\n".format(args.postags))
    text.append("postags bigrams: {}\n".format(args.postags_bg))
    text.append("length: {}\n".format(args.length))
    text.append("senti net: {}\n".format(args.sentnet))
    text.append("senti words: {}\n".format(args.sentiwords))
    text.append("subjective score: {}\n".format(args.subjscore))




    text.append("\n")

    for score_name, scores in f1_scores.iteritems():
        text.append("{} : {}, average: {}\n".format(score_name, scores, sum(scores)/args.cv_k))
        
    text.append(report)
    
    for line in text:
        print(line)
     
        
    # write text to file to keep a record of stuff
    if args.save:
        features = ""
        if args.alt_preprocess:
            features += "altpre-"
        if args.bigrams:
            features += "bigrams-"
        if args.clusters:
            features += "clusters-"
        if args.postags:
            features += "postags-"
        if args.postags_bg:
            features += "postagsbg-"
        if args.length:
            features += "length-"
        if args.sentnet:
            features += "sentnet-"
        if args.sentiwords:
            features += "sentiwords-"
        if args.subjscore:
            features += "subjscore-"
               
        if args.dim_reduction > 0:
            features += "dim-{}-".format(args.dim_reduction)

        if args.tfidf:
            features = "tfidf-"
        
        cvk = "cvk-{}".format(args.cv_k)
        filename = "{}-{}{}.txt".format(args.classifier, features, cvk)
    
        with open(filename, "w") as f:
            f.writelines(text)
    
    
    
