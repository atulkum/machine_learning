
# coding: utf-8

# In[2]:

import numpy as np
import pandas as pd 
import csv
import itertools

import numpy as np
from scipy import linalg

from sklearn import mixture
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import cPickle
from sklearn.preprocessing import StandardScaler
import multiprocessing

cores=multiprocessing.cpu_count()-2
data_root = '/Users/svloaner/Desktop/yelp/'

n_train=234842

X_std = pd.read_csv(data_root+ 'train_feature_pca.csv')
photo_ids = X_std['photo_id']
X_std.drop('photo_id', axis = 1, inplace=True)

print 'step 3'
from sklearn.manifold import TSNE
model = TSNE(n_components=3, init='pca', perplexity=40,  random_state=0)

print 'step 1'
X_tsne = model.fit_transform(X_std) 

print 'step 2'
X_all_tsne = pd.DataFrame(X_tsne)
X_all_tsne['photo_id'] = photo_ids

X_train_tsne = X_all_tsne[:n_train]
X_test_tsne = X_all_tsne[n_train:]
X_train_tsne.to_csv('train_feature_tsne.csv', index=False) 
X_test_tsne.to_csv('test_feature_tsne.csv', index=False)


