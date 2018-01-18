#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 17:29:43 2017

@author: Samuele Garda
"""
import logging
import pandas
import time

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(module)s: %(message)s', level = 'INFO')

def timeit(method):
  """
  Decorator for timing functions call
  """
  def timed(*args, **kw):
      ts = time.time()
      result = method(*args, **kw)
      te = time.time()

      if 'log_time' in kw:
          name = kw.get('log_name', method.__name__.upper())
          kw['log_time'][name] = int((te - ts) * 1000)
      else:
          logger.info('Function %r took :  %2.2fs' %(method.__name__, (te - ts)))
      return result

  return timed


def visualize_scores(scores, tolatex = False):
  viz_scores = pandas.DataFrame.from_dict(scores, orient = 'index')
  print(viz_scores)
  if tolatex:
    print(viz_scores.to_latex())