#svm rbf
#256  1.0 .84976
#300  2.0 .85219
#350  3.0 .85164
#svm_c = 2.0 
#blend   0.852449888641
#svm linear 1.0 0.8047

#rf
#gini 1024 8 0.8215
#entropy 512 8  0.8202

import numpy as np
import pandas as pd 
import csv
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier

from sklearn.metrics import f1_score
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import multiprocessing
cores=multiprocessing.cpu_count()-2

random_state = np.random.RandomState(0)

data_root = '/Users/svloaner/Desktop/yelp/'

train_business_feature = pd.read_csv(data_root+'train_business_feature_pca300_latest.csv')
train_label = pd.read_csv(data_root+'train.csv')
train_df3 = pd.merge(train_business_feature, train_label, how='inner', on='business_id')
train_business_feature.drop('business_id', axis=1, inplace=True)

train_df3['labels'].fillna('', inplace=True)
y_train = np.array([y.split() for y in train_df3['labels']])
y_train = [map(int, y) for y in y_train]

mlb = MultiLabelBinarizer()
y_ptrain_mlb= mlb.fit_transform(y_train) 

X_ptrain, X_ptest, y_ptrain, y_ptest = train_test_split(train_business_feature, y_ptrain_mlb, 
                 test_size=.2,random_state=random_state)
print X_ptrain.shape, X_ptest.shape, y_ptrain.shape, y_ptest.shape
"""
clf = RandomForestClassifier(n_estimators=512,criterion='entropy',max_depth=8)
classifier = OneVsRestClassifier(clf, n_jobs=cores)
classifier.fit(X_ptrain, y_ptrain)
y_ppredict = classifier.predict(X_ptest)
print "F1 score: ", f1_score(y_ptest, y_ppredict, average='micro')
print "Individual Class F1 score: ", f1_score(y_ptest, y_ppredict, average=None)

clf = RandomForestClassifier(n_estimators=1024,criterion='gini',max_depth=8)
classifier = OneVsRestClassifier(clf, n_jobs=cores)
classifier.fit(X_ptrain, y_ptrain)
y_ppredict = classifier.predict(X_ptest)
print "F1 score: ", f1_score(y_ptest, y_ppredict, average='micro') 
print "Individual Class F1 score: ", f1_score(y_ptest, y_ppredict, average=None)

clf = svm.SVC(kernel='linear', C= 4.0, probability=True)
classifier = OneVsRestClassifier(clf, n_jobs=cores)
classifier.fit(X_ptrain, y_ptrain)
y_ppredict = classifier.predict(X_ptest)
print "F1 score: ", f1_score(y_ptest, y_ppredict, average='micro')
print "Individual Class F1 score: ", f1_score(y_ptest, y_ppredict, average=None)

clf = ExtraTreesClassifier(n_estimators=1024,criterion='entropy',max_depth=16)
classifier = OneVsRestClassifier(clf, n_jobs=cores)
classifier.fit(X_ptrain, y_ptrain)
y_ppredict = classifier.predict(X_ptest)
print "F1 score: ", f1_score(y_ptest, y_ppredict, average='micro')
print "Individual Class F1 score: ", f1_score(y_ptest, y_ppredict, average=None)

"""
clf = GradientBoostingClassifier(n_estimators=50,learning_rate=0.03 , max_depth=8)
classifier = OneVsRestClassifier(clf, n_jobs=cores)
classifier.fit(X_ptrain, y_ptrain)
y_ppredict = classifier.predict_proba(X_ptest)
print y_ppredict.shape
#print "F1 score: ", f1_score(y_ptest, y_ppredict, average='micro')
#print "Individual Class F1 score: ", f1_score(y_ptest, y_ppredict, average=None)
"""
classifier.fit(train_business_feature, y_ptrain_mlb)

test_business_feature = pd.read_csv(data_root+'test_business_feature_pca300_latest.csv')
business_id = test_business_feature['business_id'].reshape(-1,1)
test_business_feature.drop('business_id', axis=1, inplace=True)
y_predict_test = classifier.predict(test_business_feature)

y_predict_label = mlb.inverse_transform(y_predict_test)

df = pd.DataFrame(columns=['business_id','labels'])

for i in range(len(y_predict_label)):
    biz = business_id[i][0]
    label = y_predict_label[i]
    label = str(label)[1:-1].replace(",", " ")
    df.loc[i] = [str(biz), label]

with open(data_root+"sub_pca300.csv",'w') as f:
    df.to_csv(f, index=False) 

"""
