

from sklearn.pipeline import Pipeline

from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin, ClusterMixin

import copy

UNRELATED = "Unrelated"
NEG = "Neg"
NEG_LABEL = "Negative"

LABEL = "label"
TWEET = "tweet"
NORMALIZED_LABEL = "normalized label"

C = "C"
GAMMA = "gamma"


def process_tweets(tweets,labels,pipeline_steps):
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
      _all[NORMALIZED_LABEL].append(label if label == UNRELATED else "Related")
      
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
          
  _all[TWEET] = pipeline_steps.fit_transform(_all[TWEET])
  related[TWEET] = pipeline_steps.fit_transform(related[TWEET])
  negative[TWEET] = pipeline_steps. fit_transform(negative[TWEET])   
   
  return _all, related, negative


class HierarchicalClassifier(BaseEstimator, ClassifierMixin):
  """
  Classifier for hierarchical classification task. Perform by level classification with 3 different classifier.
  Use only within this contest (tweet classification with 3 levels)
  """
  
  def __init__(self,clfs,params):
    """
    Construct new classifier object.
    
    :params:
      clfs (list) : list of sklearn classifiers (initialized). E.g. [LogisticRegression(),SVC(),LinearSVC()]
      params (list) : list of parameters of each classifier. Each list in `params` should contain dictionaries as <param_name> : [<values>]. See
      `optimize_hp` for more detail.
      
    """
    self.clfs = clfs
    self.params = params
    
    
  def get_by_level_data(self,X,y,level):
    """
    For each level of classfication retrieve correspondent feature matrix and create new set of labels
    
    :params:
      X (scipy.sparse.csr) : original feature matrix
      y (list) : original labels
      level (int) : hierarchy (0,1,2)
      
    :return:
      X_new (scipy.sparse.csr) : by level feature matrix
      y_new (list) : by level labels (new) names
      
    For instance with `level = 1` only the samples with label Positive,Neutral,Neg.* are retrieved. Then the labels `Ǹeg.*`
    are replaced with `Negative`
    """
    
    if level == 0:
      y_new = [label if label == "Unrelated" else "Related" for label in y]
      X_new = X
      
    elif level == 1:
      y_new = []
      idx_X_new = []
      for idx,label in enumerate(y):
        if label != UNRELATED:
          y_new.append(label if NEG not in label else NEG_LABEL)
          idx_X_new.append(idx)
      
      X_new = X[idx_X_new]
      
    elif level == 2:
      y_new = []
      idx_X_new = []
      for idx,label in enumerate(y):
        if NEG in label:
          y_new.append(label)
          idx_X_new.append(idx)
          
      X_new = X[idx_X_new]

    return X_new,y_new
  
  
  def optimize_classifiers(self,X,y):
    """
    Optimize classifiers hyperparameters. See `optimize_hp`.
    
    :params:
      X (scipy.sparse.csr) : original feature matrix
      y (list) : original labels
      
    Replace classifiers in `self.clfs` with optimized ones
    """
    
    for idx,clf in enumerate(self.clfs):
      X_new,y_new = self.get_by_level_data(X = X,y = y,level = idx)
      self.clfs[idx] = optimize_hp(clf = self.clfs[idx],X = X_new,y = y_new, params = self.params[idx])
    
    
    
  def fit(self,X,y):
    """
    Fit data with 3 different classifiers
    """
    
    for idx,clf in enumerate(self.clfs):
      
      X_new,y_new = self.get_by_level_data(X = X,y = y,level = idx)
      
      clf.fit(X_new,y_new)
    
    return self
  
  def predict(self, X):
    """
    Make by level predictions
    """
    
    predictions = []
    
    for i in range(X.shape[0]):
      tweet = X[i,:]
      # predict with all to get related or unrelated
      pred = self.clfs[0].predict(tweet)[0] # the zero is because the classifier return a list with 1 element
      if pred == UNRELATED:
          # if its unrelated we are done with this tweet
          predictions.append(pred)
          continue
          
      # if its related try to get next label
      pred = self.clfs[1].predict(tweet)[0]
      if NEG not in pred:
          # if its not negative we are done with this tweet
          predictions.append(pred)
          continue
          
      # if its negative try to get specific label
      pred = self.clfs[2].predict(tweet)[0]
      predictions.append(pred)
      
    return predictions
    
  def score(self, tweets, labels):
      preds = self.predict(tweets)

      return accuracy_score(labels, preds)
    

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
        print("Shape of the feature matrix to be fitted : {}".format(X.shape))
        
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

def optimize_hp(clf,X,y,params):
  """
  Optimize one classifier parameter at time. For each parameter:
    - performs grid search (with train-test split for avoid training too many models), find best parameter value w.r.t. magro averaged f1 score
    - instantiate new classifier with the found best parameter
    - repeat
    
  :params:
    
    clf (sklearn classifier) : classifier to be optimized
    X (scipy.sparse.csr_matrix or np.ndarray) : feature matrix
    y ( np.ndarray) : labels vector
    params (list) : list of dictionaries containing as key a parameter name and as value a list of possible parameters values. E.g.
    params = [{'C' : [1,10,100]},{'gamma' : [2e-3,2e-4,2e-5]}]
    random_state (int) : random seed for repruducibility
    
   :returns:
     clf (sklearn classifier) : optimized classifier
  """
  
  for param in params:
    
    gs = GridSearchCV(clf,param,refit = False, scoring = 'f1_macro')  # 3 fold CV
    gs.fit(X,y)
    searched_param = list(param.keys())[0]
    best_value = gs.best_params_[searched_param]
    print("Best value for {} : {}".format(searched_param,best_value))
    print("Best F1 macro : {}".format(gs.best_score_))
    clf.set_params(**{searched_param : best_value})
    
  return clf 