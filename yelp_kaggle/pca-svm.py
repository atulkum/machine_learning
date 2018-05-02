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
from sklearn.linear_model import LogisticRegression

data_root =  '/Users/svloaner/Desktop/yelp1/'

train_business_feature_pca = pd.read_csv(data_root+'pca_dataset/train_business_feature_orig.csv')

train_label = pd.read_csv(data_root+'input_dataset/train.csv')
train_df3 = pd.merge(train_business_feature_pca, train_label, how='inner', on='business_id')
train_df3['labels'].fillna('', inplace=True)
y_train = np.array([y.split() for y in train_df3['labels']])
y_train = [map(int, y) for y in y_train]
mlb = MultiLabelBinarizer()
y_ptrain_mlb= mlb.fit_transform(y_train) 

train_business_feature_pca.drop('business_id', axis=1, inplace=True)

train_business_feature_pca = train_business_feature_pca.as_matrix()

n = len(train_business_feature_pca)
n_total = range(n)
np.random.shuffle(n_total)
n_test = n_total[:400]
n_train = n_total[400:]


X_ptrain_pca = train_business_feature_pca[n_train]
X_ptest_pca = train_business_feature_pca[n_test]
y_ptrain = y_ptrain_mlb[n_train]
y_ptest = y_ptrain_mlb[n_test]

print X_ptrain_pca.shape, X_ptest_pca.shape, y_ptrain.shape, y_ptest.shape
clf2 = svm.SVC(kernel='rbf', C= 64.0, probability=True)
classifier2 = OneVsRestClassifier( clf2, n_jobs=cores)
classifier2.fit(X_ptrain_pca, y_ptrain)
y_svm = classifier2.predict(X_ptest_pca)
print "F1 score: ", f1_score(y_ptest, y_svm, average='micro')
print "Individual Class F1 score: ", f1_score(y_ptest, y_svm, average=None)
#########test
classifier2.fit(train_business_feature_pca, y_ptrain_mlb)

test_business_feature_pca = pd.read_csv(data_root+'pca_dataset/test_business_feature_orig.csv')
business_id = test_business_feature_pca['business_id'].reshape(-1,1)
test_business_feature_pca.drop('business_id', axis=1, inplace=True)

y_predict_test = classifier2.predict(test_business_feature_pca)

y_predict_label = mlb.inverse_transform(y_predict_test)

df = pd.DataFrame(columns=['business_id','labels'])

for i in range(len(y_predict_label)):
    biz = business_id[i][0]
    label = y_predict_label[i]
    label = str(label)[1:-1].replace(",", " ")
    df.loc[i] = [str(biz), label]

with open(data_root+"submissions/sub_pca_300_tsne.csv",'w') as f:
    df.to_csv(f, index=False) 
