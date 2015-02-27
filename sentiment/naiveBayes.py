# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 01:14:03 2015

@author: atulkumar
"""

import re, math, collections, itertools
import nltk, nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
import csv
from collections import defaultdict


def evaluate_features(feature_select):
    #reading pre-labeled input and splitting into lines
    data = defaultdict(set)
    csvfile = open('/Users/atulkumar/kaggle/sentiment/train.tsv', 'r')
    rows = csv.reader(csvfile, delimiter='\t')
    i =0
    for row in rows:
        if(i == 0):
            phrase = re.findall(r"[\w']+|[.,!?;]", row[2])
            features = phrase
            #feature_select(phrase) 
            data[row[3]].add(features)
        else:
            i=1
            
    train = defaultdict(set)
    dev = defaultdict(set)
    
    for key, values in data:
        lenData = len(values)
        print(lenData)
            

            
    