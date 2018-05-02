
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
from sklearn.decomposition import FastICA

cores=multiprocessing.cpu_count()-2
data_root = '/Users/svloaner/Desktop/yelp/'


feature_column = ['f'+str(i) for i in range(1, 2049)]


n_train=234842
X_std = pd.read_csv(data_root+'X_all_std.csv')
f = open('photo_ids.pkl', 'rb')
photo_ids = cPickle.load(f)
f.close()

ica = FastICA(n_components=300, max_iter =1000)
X_ica = ica.fit_transform(X_std)

X_all = pd.DataFrame(X_ica)
X_all['photo_id'] = photo_ids

X_train = X_all[:n_train]
X_test = X_all[n_train:]
X_train.to_csv('train_feature_ica.csv', index=False)
X_test.to_csv('test_feature_ica.csv', index=False)







