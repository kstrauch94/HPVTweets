
from preprocess import stem_word
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer,MaxAbsScaler
from sklearn.feature_selection import SelectKBest, chi2

from collections import defaultdict

from sklearn.decomposition import TruncatedSVD


import numpy as np

from nltk.corpus import sentiwordnet as swn

                                                                                                                         

  
def build_pipeline_steps(do_tfidf,do_tsvd,do_neg_words,do_bingliu,
                         do_clusters,do_postags,do_sentnet,
                         do_subjscore,do_dep_sent,do_sentiwords,
                         ngram_range,deps, bingliu_pos_path,do_scaling,
                         bingliu_neg_path,clusters_path,pos_tokens,subj_score_file,stem):
  
  """
  Create pipeline for feature extraction. All parameters `do_.*` accepts boolean to decide whether to add the correspondent set of features.
  All others are paths pointing to external resource. Except for : `ngram_range` (parameter of CountVectorizer)  and `do_tsvd` and interger for number of 
  components in the TruncatedSVD (if negative or non int is not performed).
  """
    
  features = []
  
  
  print("Adding ngram features : ngram_range {}".format(ngram_range))
  text_pipeline = ('text_pipeline', Pipeline([('ngrams',CountVectorizer(preprocessor = ' '.join, tokenizer = str.split, ngram_range = (1,ngram_range)))]))
  
  if do_tfidf:
    print("Add weighting for ngram matrix via tf-idf")
    text_pipeline[1].steps.insert(1,('tf-idf',TfidfTransformer()))
    
  if type(do_tsvd) == int and do_tsvd > 0:
    print("Add TruncatedSVD dim red - k : {} ".format(do_tsvd))
    text_pipeline[1].steps.insert(2,('tsvd_{}'.format(do_tsvd),TruncatedSVD(n_components= do_tsvd, random_state=42)))
    
  features.append(text_pipeline)
      
  if do_neg_words:
    print("Add negated words")
    neg_list = get_negation_list(stem)
    features.append(('vect_negated_words', FunctionTransformer(getnegatedwordfeatures, kw_args = {'neg_list' : neg_list} ,validate = False)))
    
  if do_bingliu:
    print("Add bingliu sentiment lexicon")
    pos_vocab,neg_vocab = get_pos_neg_words(bingliu_neg_path,bingliu_pos_path,stem) 
    features.append(('vect_bingliu_words', FunctionTransformer(getsentimentfeatures,validate = False, 
                                                              kw_args = {'pos_vocab': pos_vocab, 'neg_vocab' : neg_vocab})))
  if do_clusters:
    print("Add clusters features")
    clusters = get_clusters(clusters_path,stem)
    features.append(('vect_clusters', FunctionTransformer(tweetclusterfeatures,validate = False, 
                                                                          kw_args = {'clusters': clusters})))    
  if do_postags:
    print("Add postags")
    features.append(('vect_postags', FunctionTransformer(posfeatures,validate = False, 
                                                                          kw_args = {'pos_tokens': pos_tokens})))
  if do_sentnet:
    print("Add sentinet scores")
    features.append( ("sentinet", FunctionTransformer(net_sentiment, validate=False)) )
        
  if do_subjscore:
    print("Add subjectivity scores")
    score_lookup = process_subjectivity_file(subj_score_file,stem)
    features.append( ("subj-score", FunctionTransformer(sub_score, kw_args = {'score_lookup' : score_lookup}, validate=False)) )
    
  if do_dep_sent:
    print("Add positive/negative words dependency parse related")
    pos_vocab,neg_vocab = get_pos_neg_words(bingliu_neg_path,bingliu_pos_path,stem)
    features.append( ("dependency pos-neg", FunctionTransformer(get_sents_dependency, kw_args = {'pos_vocab': pos_vocab, 'deps' : deps, 'neg_vocab' : neg_vocab},
                                                        validate=False)) )
  if do_sentiwords:
    print("Adding sentisynset words")
    features.append( ("sentisynset_words", FunctionTransformer(tweet_wordnet,validate = False)) )
    
#  features.append(("scaling" , MaxAbsScaler()))
  
  pipeline_steps = Pipeline([ ("features", FeatureUnion(features)) ])
  
  if do_scaling:
    print("Scale feature matrix")
    pipeline_steps.steps.append(("scaler", MaxAbsScaler()))
  
#  print("Selecting K best features")  
#  pipeline_steps.steps.append(("Select-kbest", SelectKBest(chi2, k = 3000)))
  
  return pipeline_steps
  

def get_clusters(cluster_path, stem):
  """
  Creates clusters lookup table
  
  :params:
    cluster_path (str) : path to tweet clusters
    stem (bool) : stem word if true
    
  :returns:
    clusters (dict) : key = cluster id, value = set of words
  """
  
  clusters = defaultdict(set)
  
  with open(cluster_path) as infile:
    for line in infile:
      c,w,i = line.split('\t')
      w = stem_word(w,stem)
      clusters[c].add(w)
      
  return clusters
  
    
def tweetclusterfeatures(texts, clusters):
  """
  Compute cluster feature matrix with frequencies of clusters for each tweet.
  
  :params:
    texts (list) : list of lists of tokens
    path_to_clusters (str) : cluster file
    
  :returns:
    
    cluster features (scipy.sparse.csr_matrix)
    
  """
  
  vec = CountVectorizer(preprocessor = ' '.join, tokenizer = str.split)
  
  mapped_texts = []
  
  for text in texts:
    mapped_text = []
    for w in text:
      for k,v in clusters.items():
        if w in v:
          mapped_text.append(k)
    mapped_texts.append(mapped_text)
  
  cluster_features = vec.fit_transform(mapped_texts)
  
  return cluster_features


def posfeatures(texts,pos_tokens):
  """
  Compute pos feature matrix with frequencies of pos for each tweet. Fake parameter texts for compatibility
  with sklearn pipeline. All the operations will be performed on `pos_tokens`
  
  params:
    texts (list) : list of list of tokens (FAKE)
    pos_tokens (list) : list of list of pos 
  
  :returns:
    pos features (scipy.sparse.csr_matrix)
 
  """
  
  vec = CountVectorizer(preprocessor = ' '.join, tokenizer = str.split)
      
  pos_features = vec.fit_transform(pos_tokens)
    
  return pos_features

def get_pos_neg_words(pos_file,neg_file,stem):
  """
  Create set of positive and negative words
  
  :params:
    pos_file (str) : Opinion Lexicon. https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html
    neg_file (str) : Opinion Lexicon. https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html
    stem (bool) : stem word if true
    
  :return:
    (pos_vocab,neg_vocab) (set) : set of words
    
  """
  
  n = open(pos_file).readlines()
  p = open(neg_file).readlines()
  neg_vocab = set([stem_word(w.strip(),stem) for w in n if not w.startswith(';')])
  pos_vocab = set([stem_word(w.strip(),stem) for w in p if not w.startswith(';')])
  
  
  
  return pos_vocab,neg_vocab
  
def getsentimentfeatures(texts,pos_vocab,neg_vocab):
  """
  Return sentiment words feature matrix with frequencies of positive and negative words for each tweet.
  
  :param:
    texts (list) : list of lists of tokens
    pos_vocab (set) : set of positive words 
    neg_vocab (set) : set of negative words
  
  :return : 
    sentiment word features (scipy.sparse.csr_matrix)
  """
  
  pos_neg_tweets = []
  for tweet in texts:
    pos_neg_tweet = []
    for tok in tweet:
      if tok in pos_vocab or tok in neg_vocab:
        pos_neg_tweet.append(tok)
    pos_neg_tweets.append(pos_neg_tweet)
    

  vec = CountVectorizer(preprocessor = ' '.join, tokenizer = str.split)
      
  p_n_features = vec.fit_transform(pos_neg_tweets)
            
  return p_n_features  
  
  
def get_negation_list(stem):
  """
  Retrieve negation words in list. Just in case we want to add more words
  or find a 'negation lexicon'
  
  :return:
    neg_set (set) : list of common negation words
    stem (bool) : stem word if true
  """
  
  neg_set = set(['none', 'hasnt', 'couldnt', 'nowhere', 'havent', 'dont', 'cant', 'didnt', 'arent', 'never', 
                'not', 'nothing', 'nobody', 'wouldnt', 'hadnt', 'shouldnt', 'noone', 'aint', 'isnt', 'neither',
                'wont', 'doesnt', 'no'])
  
  if stem:
    neg_set = set([stem_word(w,stem) for w in neg_set])

  
  return neg_set
  
def getnegatedwordfeatures(texts,neg_list):
  """
  Return negated words feature matrix. 
  Negated words are: words ending in `n`t` and `not`,`no`,`nobody`,`nothing`,`none`,`nowhere`,`neither`
  
  :params:
    texts (list) : list of lists of tokens
    
  :return:
    sentiment word features (scipy.sparse.csr_matrix)
  """
        
  neg_tweets = []
    
  for tweet in texts:
    neg_tweet = []
    for idx,tok in enumerate(tweet):
      if tok.endswith("n't") or tok in neg_list:
        next_idx = idx+1
        prev_idx = idx-1
        if next_idx < len(tweet):
          next_neg = 'not_'+tweet[next_idx]
          neg_tweet.append(next_neg)
        if prev_idx > 0:
          prev_neg = 'not_'+tweet[prev_idx]
          neg_tweet.append(prev_neg)
    neg_tweets.append(neg_tweet)
    
  vec = CountVectorizer(preprocessor = ' '.join, tokenizer = str.split)
    
  neg_features = vec.fit_transform(neg_tweets)
    
  return neg_features

def process_subjectivity_file(filename,stem):
  """
  Load subjecitvity score lookup
  
  :params:
    filename (str) : path to file
    stem (bool) : stem word if true
  :return:
    scores (dict) : word-scores lookup
  """

  scores = {}
  
  with open(filename, "r") as f:
      for line in f.readlines():
        if line == '\n':
          pass
        else:
          line = line.split(" ")
          word = line[2].split("=")[1]
          score = line[-1].split("=")[1].strip()
          if score == "negative":
              score = -1
          elif score == "positive":
              score = 1
          else:
              score = 0
          
          scores[stem_word(word,stem)] = score
  
  return scores

# WHAT THIS FUNCTION IS DOING EXACTLY?    
def coalesce(token):
    new_tokens = []
    for char in token:
        if len(new_tokens) < 2 or char != new_tokens[-1] or char != new_tokens[-2]:
            new_tokens.append(char)
    return ''.join(new_tokens)
    
def clusters(tokens, cluster_lookup):

    clusters = []
    
    for token in tokens:
        if token in cluster_lookup:
            clusters.append(cluster_lookup[token])
            
    return clusters
    
def tweet_wordnet(tweets):
  """
  Build feauture matrix with sentinet sysnsets.
  
  :params:
    tweet (list) : list of lists of tokens
    stem (bool) : stem word if true
    
  :return:
    sentinet synsets features (scipy.sparse.csr_matrix)
  """
  
  
  vec = CountVectorizer(preprocessor = ' '.join, tokenizer = str.split)
  
  sentinet_tweets = []  
  for tweet in tweets:
    words = set()
    for word in tweet:
        for net in list(swn.senti_synsets(word)):
            words.add(net.synset.lemmas()[0].name())
    sentinet_tweets.append(list(words))
    
  sentiwordnet_feat = vec.fit_transform(sentinet_tweets)
            
  return sentiwordnet_feat
    
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
    
def net_sentiment(tweets):
    
    return np.array([tweet_net_sentiment(tw) for tw in tweets])
      

def tweet_sub_score(tweet, score_lookup):
    
    score = 0
    
    for word in tweet:
        if word in score_lookup:
            score += score_lookup[word]
            
    return score
    
def sub_score(tweets,score_lookup):
    
    return np.array([tweet_sub_score(tw, score_lookup) for tw in tweets]).reshape(-1, 1)

def tweets_length(tweets):
    return np.array([len(t[1]) for t in tweets]).reshape(-1, 1)
    


def get_sents_dependency(texts,deps,pos_vocab,neg_vocab):
  """
  Create a feature matrix with tokens connected to bing liu positive or negative tokens.
  
  :params:
    texts (list) : list of lists of tokens
    deps (list) : list of lists of dependencies
    pos_vocab (set) : set of positive words 
    neg_vocab (set) : set of negative words

    
  :return:
    sentiment word features (scipy.sparse.csr_matrix)
    
  `text` and `deps` MUST be aligned!
  """

  vec = CountVectorizer(preprocessor = ' '.join, tokenizer = str.split)
  
  # position reference
  dep_tweets = []
  for (tweet,dep) in zip(texts,deps):
    dep_tweet = []
    for (tok,d) in zip(tweet,dep):
      if tok in pos_vocab or tok in neg_vocab:
        try:
          # EXCLUDE : NOT INCLUDED WORDS, ROOT NODE
          if d == -1 or d == 0:
            pass
          else:
            dep_tweet.append('_'.join((tok,tweet[d-1])))
        # MISLABELED EXAMPLES ?    
        except IndexError:
          pass
    dep_tweets.append(dep_tweet)
    
  dep_features = vec.fit_transform(dep_tweets)
    
  return dep_features
    

