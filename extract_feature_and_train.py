"""
Adapted from https://www.kaggle.com/wenxuanchen/home-depot-product-search-relevance/sklearn-random-forest
This script basically counts the occurrences of each word of search
term in different part of information of the products.
"""


import pandas as pd
from sklearn.ensemble \
    import RandomForestRegressor, BaggingRegressor
from sklearn.datasets import dump_svmlight_file
import numpy as np


def find_occurrences(str1, str2):
    """
    find in str2 occurrences of each word in str1
    example: str1 = "good job", str2 = "good good job"
    it gets 2(for good) and 1(for job), they their sum
    is 3
    :param str1:
    :param str2:
    :return:
    """
    return sum(str2.count(word) for word in str1.split())

# import the number of training tuples
df_train = pd.read_csv("train.csv", encoding="ISO-8859-1")
train_num = df_train.shape[0]
df_train = None

df_all = pd.read_pickle('df_all')

# Coalesce all information into one column so we can apply
# map to that one column
df_all['product_info'] \
    = df_all['search_term'] + "\t" + df_all['product_title']\
    + "\t" + df_all['product_description'] + "\t"\
    + df_all['attributes'] + "\t" + df_all['brand']

# map find_occurrences to the separated information.
df_all['word_in_title'] = df_all['product_info']\
    .map(lambda x: find_occurrences(x.split('\t')[0],
                                    x.split('\t')[1]))
df_all['word_in_description'] = df_all['product_info']\
    .map(lambda x: find_occurrences(x.split('\t')[0],
                                    x.split('\t')[2]))
df_all['word_in_attributes'] = df_all['product_info']\
    .map(lambda x: find_occurrences(x.split('\t')[0],
                                    x.split('\t')[3]))
df_all['word_in_brand'] = df_all['product_info']\
    .map(lambda x: find_occurrences(x.split('\t')[0],
                                    x.split('\t')[4]))
df_all = df_all.drop(['search_term', 'product_title',
                      'product_description', 'product_info',
                      'attributes', 'brand'], axis=1)

# extract train data and test data
df_train = df_all.iloc[:train_num]
df_test = df_all.iloc[train_num:]
id_test = df_test['id']

y_train = df_train['relevance'].values
X_train = df_train.drop(
        ['id', 'relevance', 'product_uid'], axis=1).values
X_test = df_test.drop(
        ['id', 'relevance', 'product_uid'], axis=1).values
X_test_len = len(X_test)

rf = RandomForestRegressor(
        n_estimators=30, max_depth=6, random_state=0)
clf = BaggingRegressor(
        rf, n_estimators=60, max_samples=0.13, random_state=25)
clf.fit(X_train, y_train)
y_prediction = clf.predict(X_test)

# Output the result
pd.DataFrame({"id": id_test, "relevance": y_prediction})\
    .to_csv('submission.csv', index=False)


# output the result to a libSVM file.
dump_svmlight_file(X_train, y_train, 'train_libSVM.dat',
                   zero_based=True, multilabel=False)
test_file_label = np.zeros(X_test_len)
dump_svmlight_file(X_test, test_file_label, 'test_libSVM.dat',
                   zero_based=True, multilabel=False)