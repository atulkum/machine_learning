
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


feature_column = ['f'+str(i) for i in range(1, 2049)]


train_df = pd.read_csv(data_root+"train_features.csv", header=None)
train_df.columns = ['photo_id'] +  feature_column
train_photo_ids = train_df['photo_id'].reshape(-1,1)
train_df.drop('photo_id', axis=1, inplace=True)

std = StandardScaler()
X_train_std = std.fit_transform(train_df)


pca = PCA(n_components=500)
X_train_pca = pca.fit_transform(X_train_std)
X_train = pd.DataFrame(X_train_pca)
X_train['photo_id'] = train_photo_ids
X_train.to_csv('train_feature_pca500_only_train.csv', index=False) 



test_df = pd.read_csv(data_root+"test_features.csv", header=None)
test_df.columns = ['photo_id'] +  feature_column
test_photo_ids = test_df['photo_id'].reshape(-1,1)
test_df.drop('photo_id', axis=1, inplace=True)
X_test_std = std.transform(test_df)

X_test_pca = pca.fit_transform(X_test_std)
X_test = pd.DataFrame(X_test_pca)
X_test['photo_id'] = test_photo_ids
X_test.to_csv('test_feature_pca500_only_train.csv', index=False) 


