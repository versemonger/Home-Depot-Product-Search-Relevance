"""
Adapted from https://www.kaggle.com/wenxuanchen/home-depot-product-search-relevance/sklearn-random-forest
This script basically counts the occurrences of each word of search
term in different part of information of the products.
"""

import pandas as pd

from sklearn.datasets import dump_svmlight_file
import numpy as np
from nltk.corpus import stopwords
import store_data_in_pandas as sd

stopwords_list = stopwords.words('english')
stemmed_stopwords = [sd.stem_text(stop_word)
                     for stop_word in stopwords_list]


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
    return sum([str2.count(word) for word in str1.split()
                if word not in stemmed_stopwords])


def find_occurrences_modified(str1, str2):
    """
    find how many words in str1 appear at least once in str2
    :param str1: search term
    :param str2: a column of information of product
    :return: The number of words in str1 that appear in str2
    """
    return sum([str2.find(word) for word in str1.split()
                if word not in stemmed_stopwords])


def range_filter(x):
    """
    For each feature, round to to large data to 6
    :param x: feature in one sample
    :return: x if x is small and 6 if x is larger than 6
    """
    if x > 6:
        return 6
    else:
        return x


def modify_zero_and_one(x):
    """
    If the target value is 0, add 0.01 to it.
    Otherwise if it is 1, remove 0.01 from it.
    :param x: target value in [0, 1]
    :return: modified target value in (0, 1)
    """
    if x >= 0.99:
        return 0.99
    elif x <= 0.01:
        return 0.01
    else:
        return x


def main():
    # import the number of training tuples
    df_train = pd.read_csv("train.csv", encoding="ISO-8859-1")
    train_num = df_train.shape[0]
    df_train = None

    df_all = pd.read_pickle('df_all')

    # Coalesce all information into one column so we can apply
    # map to that one column
    df_all['product_info'] \
        = df_all['search_term'] + "\t" + df_all['product_title'] \
          + "\t" + df_all['product_description'] + "\t" \
          + df_all['attributes'] + "\t" + df_all['brand']

    # Count number of characters in each column
    df_all['title_length'] \
        = df_all['product_info'] \
        .map(lambda x: len(x.split('\t')[1]))
    df_all['description_length'] \
        = df_all['product_info'] \
        .map(lambda x: len(x.split('\t')[2]))
    df_all['attributes_length'] = df_all['product_info'] \
        .map(lambda x: len(x.split('\t')[3]))
    df_all['brand_length'] = df_all['product_info'] \
        .map(lambda x: len(x.split('\t')[4]))

    # Coalesce all information into one column so we can apply
    # map to that one column
    df_all['product_info'] \
        = df_all['product_info'] + "\t" \
          + df_all['title_length'] + "\t" \
          + df_all['description_length'] + "\t" \
          + df_all['attributes_length'] + "\t" \
          + df_all['brand_length'] + "\t"

    # map find_occurrences to the separated information
    # and divide the result by length of the corresponding column
    # content
    df_all['word_in_title'] = df_all['product_info'] \
        .map(lambda x:
             find_occurrences(
                     x.split('\t')[0], x.split('\t')[1]) /
             float(x.split('\t')[5]))
    df_all['word_in_description'] = df_all['product_info'] \
        .map(lambda x:
             find_occurrences(
                     x.split('\t')[0], x.split('\t')[2]) /
             float(x.split('\t')[6])
             )
    df_all['word_in_attributes'] = df_all['product_info'] \
        .map(lambda x:
             find_occurrences(
                     x.split('\t')[0], x.split('\t')[3]) /
             float(x.split('\t')[7]))
    df_all['word_in_brand'] = df_all['product_info'] \
        .map(lambda x:
             find_occurrences(
                     x.split('\t')[0], x.split('\t')[4]) /
             float(x.split('\t')[8]))

    df_all = df_all.drop(['search_term', 'product_title',
                          'product_description', 'product_info',
                          'attributes', 'brand'], axis=1)

    # Normalize all useful data in df
    for column in ['word_in_title', 'word_in_description',
                   'word_in_attributes', 'word_in_brand',
                   'title_length', 'description_length',
                   'attributes_length', 'brand_length']:
        df_all[column] \
            = df_all[column].map(lambda x: range_filter(x))
        mean_word_in_title = df_all[column].mean()
        std_word_in_title = df_all[column].std()
        df_all[column] \
            = df_all[column] \
            .map(
            lambda x: (x - mean_word_in_title) / std_word_in_title)
    df_all['relevance'] = df_all['relevance'] \
        .map(lambda x: (x - 1) / 2.)
    df_all['relevance'] = df_all['relevance'] \
        .map(lambda x: modify_zero_and_one(x))
    # extract train data and test data
    df_train = df_all.iloc[:train_num]
    df_test = df_all.iloc[train_num:]
    id_test = df_test['id']
    id_test.to_pickle('id_test')

    # Output targets of training data, training data, testing data
    # to pd frame file
    df_train['relevance'].to_pickle('y_train')
    df_train \
        .drop(['id', 'relevance'], axis=1) \
        .to_pickle('X_train')
    df_test \
        .drop(['id', 'relevance'], axis=1) \
        .to_pickle('X_test')

    # This snippet of code drops 'product_uid' as well,
    # which may be however useful in this project because
    # the relevance may be biased to several products.
    # X_train = df_train.drop(
    #         ['id', 'relevance', 'product_uid'], axis=1).values
    # df_train\
    #     .drop(['id', 'relevance', 'product_uid'], axis=1)\
    #     .to_pickle('X_train')
    # X_test = df_test.drop(
    #         ['id', 'relevance', 'product_uid'], axis=1).values
    # df_test\
    #     .drop(['id', 'relevance', 'product_uid'], axis=1)\
    #     .to_pickle('X_test')

    # output the feature data to libSVM files.
    y_train = df_train['relevance'].values
    X_train = df_train.drop(
            ['id', 'relevance'], axis=1).values

    X_test = df_test.drop(
            ['id', 'relevance'], axis=1).values

    validation_num = train_num / 4
    dump_svmlight_file(X_train[: train_num - validation_num],
                       y_train[: train_num - validation_num],
                       'train_libSVM.dat', zero_based=True,
                       multilabel=False)
    dump_svmlight_file(X_train[train_num - validation_num:],
                       y_train[train_num - validation_num:],
                       'validate_libSVM.dat', zero_based=True,
                       multilabel=False)
    test_file_label = np.zeros(len(X_test))
    dump_svmlight_file(X_test, test_file_label, 'test_libSVM.dat',
                       zero_based=True, multilabel=False)


if __name__ == '__main__':
    main()
