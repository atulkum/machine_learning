
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd 
import csv
from sklearn.metrics.pairwise import chi2_kernel

feature_column = ['f'+str(i) for i in range(1, 2049)]

# In[2]:

data_root = '/Users/svloaner/Desktop/yelp/'

cluster = "pca256"

train_photo_to_biz = pd.read_csv(data_root+'train_photo_to_biz_ids.csv')

train_df = pd.read_csv(data_root+ 'train_feature_' + cluster + '.csv')
train_df1 = pd.merge(train_photo_to_biz, train_df, how='inner', on='photo_id')
train_df2 = train_df1.groupby('business_id')


# In[3]:

feature_column = train_df.columns[:-1]


out_colum_name = ['business_id']
out_colum_name.extend(['f1', 'f2', 'f3'])

def getOneRow(name, grouped_df, train=1):
    out_list = [name]
    if len(grouped_df) > 5:
	
    for f in feature_column:
        avg = np.mean(grouped_df[f])
        out_list.append(avg)
        
    return out_list



train_out_handle = open('train_business_feature'+ cluster + '.csv', "w")
train_writer = csv.writer(train_out_handle)
train_writer.writerow(out_colum_name)
counter = 0
for name, group in train_df2:
    out_row = getOneRow(name, group, train=1)
    assert len(out_row) == len(out_colum_name)
    train_writer.writerow(out_row)
    counter += 1
    if counter%10000 == 0:
        print counter
train_out_handle.close()


# In[5]:

train_business_feature = pd.read_csv(data_root+'train_business_feature' + cluster + '.csv')
train_label = pd.read_csv(data_root+'train.csv')
train_df3 = pd.merge(train_business_feature, train_label, how='inner', on='business_id')


# In[6]:

train_df3['labels'].fillna('', inplace=True)
y_train = np.array([y.split() for y in train_df3['labels']])
y_train = [map(int, y) for y in y_train]


# In[7]:

train_business_feature.drop('business_id', axis=1, inplace=True)


# In[8]:

from sklearn import svm, datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import multiprocessing

cores=multiprocessing.cpu_count()-2

mlb = MultiLabelBinarizer()
y_ptrain_mlb= mlb.fit_transform(y_train) 

random_state = np.random.RandomState(0)
X_ptrain, X_ptest, y_ptrain, y_ptest = train_test_split(train_business_feature, y_ptrain_mlb, 
                 test_size=.2,random_state=random_state)
print X_ptrain.shape, X_ptest.shape, y_ptrain.shape, y_ptest.shape


# In[9]:

classifier = OneVsRestClassifier(svm.SVC(kernel='rbf', C= 1.0, probability=True), n_jobs=cores)
classifier.fit(X_ptrain, y_ptrain)


# In[10]:

y_ppredict = classifier.predict(X_ptest)


# In[11]:

from sklearn.metrics import f1_score

print "F1 score: ", f1_score(y_ptest, y_ppredict, average='micro') 
print "Individual Class F1 score: ", f1_score(y_ptest, y_ppredict, average=None)


"""
#########test
test_photo_to_biz = pd.read_csv(data_root+'test_photo_to_biz.csv')
test_df = pd.read_csv(data_root+ 'test_feature_' + cluster + '.csv')
test_df1 = pd.merge(test_photo_to_biz, test_df, how='inner', on='photo_id')
test_df2 = test_df1.groupby('business_id')
feature_column = test_df.columns[:-1]
test_out_handle = open('test_business_feature'+ cluster + '.csv', "w")
test_writer = csv.writer(test_out_handle)
test_writer.writerow(out_colum_name)
counter = 0
for name, group in test_df2:
    out_row = getOneRow(name, group, train=0)
    assert len(out_row) == len(out_colum_name)
    test_writer.writerow(out_row)
    counter += 1
    if counter%10000 == 0:
        print counter
test_out_handle.close()

##################


# In[ ]:

classifier.fit(train_business_feature, y_ptrain_mlb)

test_business_feature = pd.read_csv(data_root+'test_business_feature'+cluster +'.csv')
business_id = test_business_feature['business_id'].reshape(-1,1)
test_business_feature.drop('business_id', axis=1, inplace=True)
y_predict_test = classifier.predict(test_business_feature)


# In[ ]:

y_predict_label = mlb.inverse_transform(y_predict_test)

df = pd.DataFrame(columns=['business_id','labels'])

for i in range(len(y_predict_label)):
    biz = business_id[i][0]
    label = y_predict_label[i]
    label = str(label)[1:-1].replace(",", " ")
    df.loc[i] = [str(biz), label]

with open(data_root+"sub_chi2.csv",'w') as f:
    df.to_csv(f, index=False) 


# In[ ]:


"""
