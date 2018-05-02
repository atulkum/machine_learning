import numpy as np
import pandas as pd 
import csv
from sklearn.metrics.pairwise import chi2_kernel
from sklearn import svm, datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import multiprocessing
from sklearn.metrics import f1_score
cores=multiprocessing.cpu_count()-2
from sklearn.naive_bayes import GaussianNB
random_state = np.random.RandomState(0)

data_root =  '/Users/svloaner/Desktop/yelp1/'


train_business_feature_ica = pd.read_csv(data_root+'pca_dataset/train_business_feature_300_ica.csv')
train_business_feature_pca = pd.read_csv(data_root+'pca_dataset/train_business_feature_300.csv')

train_label = pd.read_csv(data_root+'input_dataset/train.csv')
train_df3 = pd.merge(train_business_feature_ica, train_label, how='inner', on='business_id')
train_df3['labels'].fillna('', inplace=True)
y_train = np.array([y.split() for y in train_df3['labels']])
y_train = [map(int, y) for y in y_train]
mlb = MultiLabelBinarizer()
y_ptrain_mlb= mlb.fit_transform(y_train) 

train_business_feature_ica.drop('business_id', axis=1, inplace=True)
train_business_feature_pca.drop('business_id', axis=1, inplace=True)

train_business_feature_pca = train_business_feature_pca.as_matrix()
train_business_feature_ica = train_business_feature_ica.as_matrix()

n = len(train_business_feature_ica)
n_total = range(n)
np.random.shuffle(n_total)
n_test = n_total[:400]
n_train = n_total[400:]


X_ptrain_ica = train_business_feature_ica[n_train]
X_ptest_ica = train_business_feature_ica[n_test]
X_ptrain_pca = train_business_feature_pca[n_train]
X_ptest_pca = train_business_feature_pca[n_test]
y_ptrain = y_ptrain_mlb[n_train]
y_ptest = y_ptrain_mlb[n_test]

print X_ptrain_ica.shape, X_ptest_ica.shape, X_ptrain_pca.shape, X_ptest_pca.shape, y_ptrain.shape, y_ptest.shape

clf1 = GaussianNB()
classifier1 = OneVsRestClassifier( clf1, n_jobs=cores)

clf2 = svm.SVC(kernel='rbf', C= 2.0,gamma = np.power(2.0, -8.0), probability=True)
classifier2 = OneVsRestClassifier( clf2, n_jobs=cores)

"""
classifier1.fit(X_ptrain_ica, y_ptrain)
y_gnb = classifier1.predict(X_ptest_ica)
y_gnb1 = classifier1.predict_proba(X_ptest_ica)

classifier2.fit(X_ptrain_pca, y_ptrain)
y_svm = classifier2.predict(X_ptest_pca)
y_svm1 = classifier2.predict_proba(X_ptest_pca)

y_gnb1 = y_gnb1[np.newaxis,:,:]
y_svm1 = y_svm1[np.newaxis,:,:]

y_ppredict = np.vstack((y_gnb1, y_svm1))
y_ppredict = np.average(y_ppredict, axis = 0, weights=[0.25, 0.75])
#y_ppredict = np.max(y_ppredict, axis = 0)

f = np.vectorize((lambda x: 1 if x>= 0.5 else 0), otypes=[np.float])
y_ppredict_1 = f(y_ppredict)

print "F1 score: ", f1_score(y_ptest, y_ppredict_1, average='micro')
print "F1 score: ", f1_score(y_ptest, y_gnb, average='micro')
print "F1 score: ", f1_score(y_ptest, y_svm, average='micro')
#print "Individual Class F1 score: ", f1_score(y_ptest, y_ppredict, average=None)
"""

#########test

classifier1.fit(train_business_feature_ica, y_ptrain_mlb)
classifier2.fit(train_business_feature_pca, y_ptrain_mlb)

test_business_feature_ica = pd.read_csv(data_root+'pca_dataset/test_business_feature_300_ica.csv')
business_id = test_business_feature_ica['business_id'].reshape(-1,1)
test_business_feature_ica.drop('business_id', axis=1, inplace=True)

test_business_feature_pca = pd.read_csv(data_root+'pca_dataset/test_business_feature_300.csv')
test_business_feature_pca.drop('business_id', axis=1, inplace=True)

test_business_feature_pca = test_business_feature_pca.as_matrix()
test_business_feature_ica = test_business_feature_ica.as_matrix()

y_predict_test_ica = classifier1.predict_proba(test_business_feature_ica)
y_predict_test_pca = classifier2.predict_proba(test_business_feature_pca)

y_predict_test_ica = y_predict_test_ica[np.newaxis,:,:]
y_predict_test_pca = y_predict_test_pca[np.newaxis,:,:]

y_predict_test = np.vstack((y_predict_test_ica, y_predict_test_pca))
y_predict_test = np.average(y_predict_test, axis = 0, weights=[0.25, 0.75])

f = np.vectorize((lambda x: 1 if x>= 0.5 else 0), otypes=[np.float])
y_predict_test = f(y_predict_test)

y_predict_label = mlb.inverse_transform(y_predict_test)

df = pd.DataFrame(columns=['business_id','labels'])

for i in range(len(y_predict_label)):
    biz = business_id[i][0]
    label = y_predict_label[i]
    label = str(label)[1:-1].replace(",", " ")
    df.loc[i] = [str(biz), label]

with open(data_root+"submissions/sub_ica_pca_300.csv",'w') as f:
    df.to_csv(f, index=False) 
