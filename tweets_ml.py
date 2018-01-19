from tweet_data_parser import *

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

from arktwokenize import twokenize

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

def read_cluster_file(path):
    
    lookup = {}

    with io.open(path, "r", encoding="utf-8") as clusters:
        for line in clusters.readlines():
            id, token, count = line.split("\t")
            
            lookup[token] = id
            
    return lookup
    
def read_tags(path):
    tags = {}
    with open(path, "r") as f:
        for line in f.readlines():
            splits = line.split(" ")
            #                    id is here | tags are here  /\   we delete trailing newline char from last tag
            tweet_id, pos_tags = splits[0],    splits[1:-1]   +   [splits[-1].strip()]

            tags[tweet_id] = pos_tags

    return tags
    
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
    
def coalesce(token):
    new_tokens = []
    for char in token:
        if len(new_tokens) < 2 or char != new_tokens[-1] or char != new_tokens[-2]:
            new_tokens.append(char)
    return ''.join(new_tokens)
    
def preprocess(id_text, alt_pre=False):
    # id_text is a list containing id in first positions and text in second
    # text input is one string
    # output is tokenized and preprocessed(as defined below) text
    
    # lowercase
    # no hashtags or mentions unless alt_pre = True, in which case they are replace by # or @ respectively
    # any url converted to "url"
    # replace multiple repeated chars with 2 of them. eg paaaarty -> paarty
    
    id, text = id_text
    text = twokenize.tokenizeRawTweetText(text.lower())
    
    tokens = []
    for token in text:
        if token.startswith("@") or token.startswith("#"):
            if alt_pre:
                if token.startswith("@"):
                    tokens.append("@")
                elif token.startswith("#"):
                    tokens.append("#")
                    
        elif token.startswith("https://") or token.startswith("http://"):
            tokens.append(u"url")
        else:
            tokens.append(coalesce(token))
            
    return [id] + tokens

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
    
    tokens = preprocess(tweet)[1:]
    
    for word in tokens:
        for net in list(swn.senti_synsets(word)):
            pos += net.pos_score()
            neg += net.neg_score()
            obj += net.obj_score()
            
    return [pos, neg, obj]
    
def net_sentiment(tweets):
    
    return np.array([tweet_net_sentiment(tw) for tw in tweets])
      

def tweet_sub_score(tweet, score_lookup):
    
    score = 0

    tokens = preprocess(tweet)[1:]
    
    for word in tokens:
        if word in score_lookup:
            score += score_lookup[word]
            
    return score
    
def sub_score(tweets):
    
    subj_score_file = "Data" + os.sep + "subjectivity_score.txt"

    score_lookup = process_subjectivity_file(subj_score_file)
    
    return np.array([tweet_sub_score(tw, score_lookup) for tw in tweets]).reshape(-1, 1)

def tweets_length(tweets):
    return np.array([len(t[1]) for t in tweets]).reshape(-1, 1)
    
def build_tokenizer(do_bigrams=False, do_clusters=False, do_postags=False, do_postags_bg=False, do_sentiwords=False, cluster_lookup_file=None, pos_tag_file=None):
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
        tags_lookup = read_tags(pos_tag_file)
        get_tags = lambda id: tags_lookup[id]
    else:
        get_tags = lambda id: []
        
    if do_postags_bg:
        tags_lookup = read_tags(pos_tag_file)
        get_tags_bg = lambda id: bigrams(tags_lookup[id])
    else:
        get_tags_bg = lambda id: []
        
    if do_sentiwords:
        get_sentiwords = tweet_wordnet
    else:
        get_sentiwords = lambda tokens: []

    
    def tokenizer(tokens):
        id = tokens[0]
        words= tokens[1:]
        return words + get_bigrams(words) + get_clusters(words) + get_tags(id) + get_tags_bg(id) + get_sentiwords(words)
            
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

    def fit(self, tweets, labels):
        
        if self.get_clf is None:
            raise ValueError("A classifier is needed")
        
        tweet_list = zip(labels, tweets)
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



# useless since im using sklearn cross val now :v
class CrossValSVMH:

    def __init__(self, tweet_clf, tweet_list, k=10, randomize=False):
        """
        tweets clf: TweetClassifier object
        tweet_list_ list of (label, tweet) pairs
        """
    
    
        self.clf = tweet_clf
        self.tweet_list = tweet_list
        self.k = k

        self.start(randomize)
        self.print_results()
        
        
    def start(self, randomize):
    
        self.chunk(randomize)
        
        self.acc = []
        
        for k in range(self.k):
            print("Cross validating step {}".format(k))
            train, test = self._split(k)
            test_labels, test_tweets = zip(*test)
            
            all, related, negative = process_tweets(train)
            
            self.clf.fit(all, related, negative)
            preds = self.clf.predict(test_tweets)
            
            self.acc.append(accuracy_score(test_labels, preds))
            
            
    def print_results(self):
         
         print("{}-fold cross validation scores: {}\n".format(self.k, self.acc))
         print("Average score: {}\n".format(sum(self.acc)/self.k))
        
    def _split(self, test_k):
        # get all chunks except testing one
        train_chunks = [self.chunks[c] for c in range(0, self.k) if c != test_k]
        merged_train_chunks = sum(train_chunks, [])
        
        return merged_train_chunks, self.chunks[test_k]
        
    def chunk(self, randomize=False):
    
        size = len(self.tweet_list)
        chunk_size = int(size/self.k)+1
        
        if randomize:
            random.shuffle(self.tweet_list)
            
        self.chunks = [self.tweet_list[i:i + chunk_size] for i in range(0, size, chunk_size)]
                
        
        
        


