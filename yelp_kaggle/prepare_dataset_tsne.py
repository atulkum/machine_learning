import numpy as np
import pandas as pd 
import csv
from sklearn.manifold import TSNE

data_root = '/Users/atulkumar/ml/yelp/'
n_col_pca = 350

feature_column = [str(i) for i in range(n_col_pca)]
out_colum_name = ['business_id'] + feature_column + ['tsne1' , 'tsne2', 'tsne3']

def getOneRow(name, grouped_df, train=1):
    out_list = [name]
    for f in feature_column:
        avg = np.mean(grouped_df[f])
        out_list.append(avg)
    try:	
    	model = TSNE(n_components=3, init='pca', perplexity=5,  random_state=0)
    	X_tsne = model.fit_transform(grouped_df)
    	X_tsne = X_tsne.mean(0).tolist()
    except:
	X_tsne = out_list[1:4]
    out_list += X_tsne
    return out_list

train_photo_to_biz = pd.read_csv(data_root+'input_dataset/train_photo_to_biz_ids.csv')
train_df = pd.read_csv(data_root+ 'pca_dataset/train_feature_pca500.csv')

photo_ids = train_df['photo_id']
train_df.drop('photo_id', axis = 1, inplace = True)
train_df = train_df.iloc[:, :n_col_pca]
train_df['photo_id'] = photo_ids 

train_df1 = pd.merge(train_photo_to_biz, train_df, how='inner', on='photo_id')
train_df1.drop('photo_id', axis=1, inplace=True)
train_df2 = train_df1.groupby('business_id')

train_out_handle = open('pca_dataset/train_business_feature_'+str(n_col_pca)+ '_tsne1.csv', "w")
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
test_df = pd.read_csv(data_root+ 'pca_dataset/test_feature_pca500.csv')

test_photo_ids = test_df['photo_id']
test_df.drop('photo_id', axis = 1, inplace = True)
test_df = test_df.iloc[:, :n_col_pca]
test_df['photo_id'] = test_photo_ids 

test_df1 = pd.merge(test_photo_to_biz, test_df, how='inner', on='photo_id')
test_df1.drop('photo_id', axis=1, inplace=True)
test_df2 = test_df1.groupby('business_id')

test_out_handle = open('pca_dataset/test_business_feature_'+ str(n_col_pca) + '_tsne1.csv', "w")
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
