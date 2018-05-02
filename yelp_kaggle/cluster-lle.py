

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

train_df = pd.read_csv(data_root+"train_feature_pca500.csv")
test_df = pd.read_csv(data_root+"test_feature_pca500.csv")

photo_ids = np.vstack((train_df['photo_id'].reshape(-1,1),test_df['photo_id'].reshape(-1,1)))

train_df.drop('photo_id', axis=1, inplace=True)
test_df.drop('photo_id', axis=1, inplace=True)

X_std = np.vstack((train_df,test_df))
n_train = len(train_df)


from sklearn import manifold

X_lle, err = manifold.locally_linear_embedding(X_std, n_neighbors=10, n_components=300)

X_all = pd.DataFrame(X_lle)
X_all['photo_id'] = photo_ids

X_train = X_all[:n_train]
X_test = X_all[n_train:]
X_train.to_csv('train_feature_lle.csv', index=False) 
X_test.to_csv('test_feature_lle.csv', index=False)






