import numpy as np
import pandas as pd 
import csv
from sklearn.manifold import TSNE

data_root = '/Users/svloaner/Desktop/yelp1/'

feature_column = ['f'+str(i) for i in range(1, 2049)]
out_colum_name = ['business_id'] + feature_column 

def getOneRow(name, grouped_df, train=1):
    out_list = [name]
    for f in feature_column:
        avg = np.mean(grouped_df[f])
        out_list.append(avg)
    return out_list



train_df = pd.read_csv(data_root+"../yelp/train_features.csv", header=None)
train_df.columns = ['photo_id'] +  feature_column

train_photo_to_biz = pd.read_csv(data_root+'input_dataset/train_photo_to_biz_ids.csv')

photo_ids = train_df['photo_id']
train_df.drop('photo_id', axis = 1, inplace = True)
train_df['photo_id'] = photo_ids 

train_df1 = pd.merge(train_photo_to_biz, train_df, how='inner', on='photo_id')
train_df1.drop('photo_id', axis=1, inplace=True)
train_df2 = train_df1.groupby('business_id')

train_out_handle = open('pca_dataset/train_business_feature_orig.csv', "w")
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
#########test
test_photo_to_biz = pd.read_csv(data_root+'input_dataset/test_photo_to_biz.csv')
test_df = pd.read_csv(data_root+"../yelp/test_features.csv", header=None)
test_df.columns = ['photo_id'] +  feature_column

test_photo_ids = test_df['photo_id']
test_df.drop('photo_id', axis = 1, inplace = True)
test_df['photo_id'] = test_photo_ids 

test_df1 = pd.merge(test_photo_to_biz, test_df, how='inner', on='photo_id')
test_df1.drop('photo_id', axis=1, inplace=True)
test_df2 = test_df1.groupby('business_id')

test_out_handle = open('pca_dataset/test_business_feature_orig.csv', "w")
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
