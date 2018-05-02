import numpy as np
import pandas as pd 
import csv
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from sklearn.metrics import f1_score
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import multiprocessing
cores=multiprocessing.cpu_count()-2

random_state = np.random.RandomState(0)

data_root = '/Users/atulkumar/ml/yelp/'

train_business_feature = pd.read_csv(data_root+'pca_dataset/train_business_feature_300_tsne1.csv')
train_label = pd.read_csv(data_root+'input_dataset/train.csv')
train_df3 = pd.merge(train_business_feature, train_label, how='inner', on='business_id')
train_business_feature.drop('business_id', axis=1, inplace=True)
X_train = train_business_feature.as_matrix()

test_business_feature = pd.read_csv(data_root+'pca_dataset/test_business_feature_300_tsne1.csv')
test_business_ids = test_business_feature['business_id'].reshape(-1,1)
test_business_feature.drop('business_id', axis=1, inplace=True)
X_test = test_business_feature.as_matrix()

train_df3['labels'].fillna('', inplace=True)
y_train = np.array([y.split() for y in train_df3['labels']])
y_train = [map(int, y) for y in y_train]

mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(y_train) 

n_train = len(X_train) 
n_test = len(X_test) 

kfs = list(KFold(n_train, n_folds=5))

clfs = [
	OneVsRestClassifier(RandomForestClassifier(n_estimators=512,criterion='entropy',max_depth=8), n_jobs=cores),
	OneVsRestClassifier(RandomForestClassifier(n_estimators=1024,criterion='gini',max_depth=8), n_jobs=cores),
	OneVsRestClassifier(svm.SVC(kernel='linear', C= 4.0, probability=True), n_jobs=cores),
	OneVsRestClassifier(svm.SVC(kernel='rbf', C= 2.0, gamma = np.power(2.0, -8.0),probability=True), n_jobs=cores),
	OneVsRestClassifier(ExtraTreesClassifier(n_estimators=1024,criterion='entropy',max_depth=16), n_jobs=cores),
	OneVsRestClassifier(GradientBoostingClassifier(n_estimators=1024,learning_rate=0.01 , max_depth=8), n_jobs=cores)
	]

#meta features

blend_train_X = None
blend_train_y = None
blend_test = None

for j, clf in enumerate(clfs):
	print j, clf
	blend_test_j = None
	blend_train_X_j = None

        for i, (train, test) in enumerate(kfs):
            print "Fold", i
            X_train_f = X_train[train]
            y_train_f = y_train[train]
            X_test_f = X_train[test]
            y_test_f = y_train[test]

            clf.fit(X_train_f, y_train_f)
		
            y_pred_j = 	clf.predict_proba(X_test_f)

	    if i == 0:
            	blend_train_X_j = y_pred_j
	    else:	
            	blend_train_X_j = np.vstack((blend_train_X_j, y_pred_j))

            validation_j = clf.predict_proba(X_test)
	    validation_j = validation_j[np.newaxis,:,:]
	    if i == 0:
		blend_test_j = validation_j
	    else:
		blend_test_j= np.vstack((blend_test_j, validation_j))

        if j == 0:
		blend_test =  blend_test_j.mean(0)
		blend_train_X = blend_train_X_j
	else:
		blend_test = np.hstack((blend_test,  blend_test_j.mean(0)))
		blend_train_X = np.hstack ((blend_train_X, blend_train_X_j))
		

#####prepare blend_train_y as per the order of k folds

for i, (train, test) in enumerate(kfs):
	y_test_f = y_train[test]
        if i == 0:
        	blend_train_y = y_test_f.copy()
        else:
                blend_train_y = np.vstack((blend_train_y, y_test_f.copy()))


print blend_test.shape, blend_train_X.shape, blend_train_y.shape

print "Blending."

clf =OneVsRestClassifier(LogisticRegression(), n_jobs=cores)
clf.fit(blend_train_X, blend_train_y)

print "Predicting."

y_pred = clf.predict(blend_test)


y_predict_label = mlb.inverse_transform(y_pred)

df = pd.DataFrame(columns=['business_id','labels'])

for i in range(len(y_predict_label)):
    biz = test_business_ids[i][0]
    label = y_predict_label[i]
    label = str(label)[1:-1].replace(",", " ")
    df.loc[i] = [str(biz), label]

with open(data_root+"submissions/sub_pca300_tse1.csv",'w') as f:
    df.to_csv(f, index=False)

