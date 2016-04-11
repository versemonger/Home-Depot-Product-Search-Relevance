"""
Adapted from https://www.kaggle.com/wenxuanchen/home-depot-product-search-relevance/sklearn-random-forest
This script basically counts the occurrences of each word of search
term in different part of information of the products.
http://orion.lcg.ufrj.br/Dr.Dobbs/books/book5/chap14.htm
"""
import pandas as pd
import sys
from sklearn.datasets import dump_svmlight_file
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

stopwords_list = stopwords.words('english')
stemmed_stopwords = [stemmer.stem(stop_word)
                     for stop_word in stopwords_list]

SVD_component_num = 20
# We will be normalize data named with these features.
normalize_feature_list = []


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
                if len(word) > 2])


def find_common_word(str1, str2):
    """
    find number of common words in str1 and str2
    :param str1: search term
    :param str2: a column of information of product
    :return: The number of words in str1 that appear in str2
    """
    return sum([str2.find(word) >= 0 for word in str1.split()
                if len(word) > 2])


def range_filter(x):
    """
    For each feature, round to to large data to 6
    :param x: feature in one sample
    :return: x if x is small and 6 if x is larger than 6
    """
    if x > 8:
        return 8
    else:
        return x
    return x


def modify_zero_and_one(x):
    """
    If the target value is 0, add 0.01 to it.
    Otherwise if it is 1, remove 0.01 from it.
    :param x: target value in [0, 1]
    :return: modified target value in (0, 1)
    """
    if x >= 0.999999:
        return 0.9999999
    elif x <= 0.0000001:
        return 0.0000001
    else:
        return x


def all_SVD_transform(df):
    """

    :param df: the whole pandas data frame
    :return: a data matrix of reduced dimension
    """
    feature_list = ['search_term', 'product_title',
                    'product_description', 'attributes', 'brand']
    first_term = 'search_term'
    reduced_matrix = single_feature_SVD_transform(df, first_term)
    for feature in feature_list:
        if feature != 'search_term':
            reduced_feature \
                = single_feature_SVD_transform(df, feature)
            reduced_matrix \
                = np.append(reduced_matrix, reduced_feature,
                            axis=1)
    return reduced_matrix


def single_feature_SVD_transform(df, feature):
    """
    TfidfVectorize the feature column and reduce the dimension
    by SVD transformation
    :param feature: name of the transformed feature
    :param df: the whole data_frame
    :return: dimension reduced numpy matrix
    """
    n_gram = 2
    if feature == 'text':
        n_gram = 1
    vectorizer = TfidfVectorizer(encoding='ascii',
                                 stop_words=stemmed_stopwords,
                                 ngram_range=(1, n_gram))
    feature_matrix = vectorizer \
        .fit_transform(df[feature].apply(str))
    reducer = TruncatedSVD(n_components=SVD_component_num,
                           random_state=1992)
    result = reducer.fit_transform(feature_matrix)
    print "Finish reducing dimensionality of " + feature
    return np.array(result)


def get_saperate_LSI_score(df, feature_name):
    """
    Add LSI score to each tuple.
    :param df: The whole data frame that holds all data and
               additionally contains the column 'product_info'
    :return: LSI Score of each search_term-product pair
    """
    if feature_name == 'text':
        df['text'] = df['search_term'] + " " + df['product_title'] \
                     + " " + df['product_description'] + " " \
                     + df['attributes']
    tuple_number = len(df[feature_name])
    df_text_and_search_term \
        = pd.concat((df[feature_name], df['search_term']), axis=0,
                    ignore_index=True)
    n_gram = 2
    if feature_name == 'text':
        n_gram = 1
    vectorizer = TfidfVectorizer(encoding='ascii',
                                 stop_words=stemmed_stopwords,
                                 ngram_range=(1, n_gram))
    text_matrix \
        = vectorizer.fit_transform(df_text_and_search_term)
    print 'get ' + feature_name + ' matrix'
    lsi_transformer = TruncatedSVD(n_components=160,
                                   random_state=10)

    reduced_vector = lsi_transformer.fit_transform(text_matrix)
    print 'vector of ' + feature_name + ' reduced'
    similarity_one_dimension = np.zeros(tuple_number)
    KL_similarity = np.zeros(tuple_number)
    for i in range(tuple_number):
        v1 = np.array(reduced_vector[i])
        v2 = np.array(reduced_vector[i + tuple_number])
        x = v1.reshape(1, -1)
        y = v2.reshape(1, -1)
        similarity_one_dimension[i] \
            = cosine_similarity(x, y)
        KL_similarity[i] = entropy(np.exp(v1), np.exp(v2))
    new_feature = 'similarity in ' + feature_name
    new_feature2 = 'KL similarity ' + feature_name
    normalize_feature_list.append(new_feature)
    normalize_feature_list.append(new_feature2)
    df[new_feature] = pd.Series(similarity_one_dimension)
    df[new_feature2] = pd.Series(KL_similarity)
    print new_feature2, KL_similarity[:10]
    if feature_name == 'text':
        df.drop(['text'], axis=1, inplace=True)


def create_feature_map(features):
    """
    :param features: The columns names of data frame
    :return: a map file between feature names and feature index
    """
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i += 1
    outfile.close()


def get_last_term(x):
    """
    get last term in the search terms
    :param x: search term string
    :return: last term in the search terms
    """
    search_term = x.split('\t')[0]
    if len(search_term) > 0:
        return search_term.split()[-1]
    else:
        return ''


def main():
    # import the number of training tuples
    df_train = pd.read_csv("train.csv", encoding="ISO-8859-1")
    train_num = df_train.shape[0]
    df_train = None

    df_all = pd.read_pickle('df_all')
    # feature_list = ['product_title', 'product_description',
    #                 'attributes', 'brand', 'text']
    feature_list = ['product_title', 'product_description',
                    'attributes', 'text']

    for feature in feature_list:
        get_saperate_LSI_score(df_all, feature)
    print 'LSI_score added.'

    # Coalesce all information into one column so we can apply
    # map to that one column
    df_all['product_info'] \
        = df_all['search_term'] + "\t" + df_all['product_title'] \
          + "\t" + df_all['product_description'] + "\t" \
          + df_all['attributes'] + "\t" + df_all['brand']

    # Count number of words in each column
    df_all['title_length'] \
        = df_all['product_info'] \
        .map(lambda x: str(len(str(x.split('\t')[1]).split())))
    df_all['description_length'] \
        = df_all['product_info'] \
        .map(lambda x: str(len(str(x.split('\t')[2]).split())))
    df_all['attributes_length'] = df_all['product_info'] \
        .map(lambda x: str(len(str(x.split('\t')[3]).split())))
    df_all['brand_length'] = df_all['product_info'] \
        .map(lambda x: str(len(str(x.split('\t')[4]).split())))

    print "Number of words in each column is counted."
    # Coalesce all information into one column so we can apply
    # map to that one column
    df_all['product_info'] \
        = df_all['product_info'] + "\t" \
          + df_all['title_length'] + "\t" \
          + df_all['description_length'] + "\t" \
          + df_all['attributes_length'] \
          + "\t" + df_all['brand_length']

    # map find_occurrences to the separated information
    # and divide the result by length of the corresponding column
    # content
    df_all['word_in_title'] = df_all['product_info'] \
        .map(lambda x:
             find_occurrences(
                     x.split('\t')[0], x.split('\t')[1]) /
             (float(x.split('\t')[5]) + 0.1))
    df_all['word_in_description'] = df_all['product_info'] \
        .map(lambda x:
             find_occurrences(
                     x.split('\t')[0], x.split('\t')[2]) /
             (float(x.split('\t')[6]) + 0.1))
    df_all['word_in_attributes'] = df_all['product_info'] \
        .map(lambda x:
             find_occurrences(
                     x.split('\t')[0], x.split('\t')[3]) /
             (float(x.split('\t')[7]) + 0.1))
    df_all['word_in_brand'] = df_all['product_info'] \
        .map(lambda x:
             find_occurrences(
                     x.split('\t')[0], x.split('\t')[4]) /
             (float(x.split('\t')[8]) + 0.1))

    print 'Word occurrences in each column counted'

    # count common words in search term and each column
    df_all['common_in_title'] = df_all['product_info'] \
        .map(lambda x:
             find_common_word(
                     x.split('\t')[0], x.split('\t')[1]))
    df_all['common_in_description'] = df_all['product_info'] \
        .map(lambda x:
             find_common_word(
                     x.split('\t')[0], x.split('\t')[2]))
    df_all['common_in_attributes'] = df_all['product_info'] \
        .map(lambda x:
             find_common_word(
                     x.split('\t')[0], x.split('\t')[3]))
    df_all['common_in_brand'] = df_all['product_info'] \
        .map(lambda x:
             find_common_word(
                     x.split('\t')[0], x.split('\t')[4]))
    print 'Common word in each column counted'

    df_all['length_of_search_term'] = df_all['search_term'] \
        .map(lambda x: len(x))

    # count occurrences of last term in search query in each
    # column
    df_all['last_search_term_in_title'] = df_all['product_info'] \
        .map(lambda x:
             find_common_word(
                     get_last_term(x),
                     x.split('\t')[1]
             ))
    df_all['last_search_term_in_description'] \
        = df_all['product_info'] \
        .map(lambda x:
             find_common_word(
                     get_last_term(x),
                     x.split('\t')[2]
             ))
    df_all['last_search_term_in_attributes'] \
        = df_all['product_info'] \
        .map(lambda x:
             find_common_word(
                     get_last_term(x),
                     x.split('\t')[3]
             ))

    print 'Last search term items counted'
    # preserve the strings for svd reduction.
    df_info = df_all[['search_term', 'product_title',
                      'product_description', 'attributes',
                      'brand']]

    df_all.drop(['search_term', 'product_title',
                 'product_description', 'product_info',
                 'attributes', 'brand'], axis=1, inplace=True)

    # Normalize all useful data in df
    # A range filter is used here so that we filter too large
    # numbers
    for column in ['word_in_title', 'word_in_description',
                   'word_in_attributes', 'word_in_brand']:
        df_all[column] \
            = df_all[column].map(lambda x: range_filter(float(x)))
        mean_word_in_title = df_all[column].mean()
        std_word_in_title = df_all[column].std()
        df_all[column] \
            = df_all[column] \
            .map(
                lambda x: (x - mean_word_in_title)
                          / std_word_in_title)

    # Normalize all useful data in df
    feature_names = ['title_length', 'description_length',
                     'attributes_length', 'brand_length',
                     'common_in_title',
                     'common_in_description',
                     'common_in_attributes',
                     'common_in_brand',
                     'length_of_search_term',
                     'last_search_term_in_title',
                     'last_search_term_in_description',
                     'last_search_term_in_attributes']
    normalize_feature_list.extend(feature_names)
    for column in normalize_feature_list:
        df_all[column] \
            = df_all[column].map(lambda x: float(x))
        mean_word_in_title = df_all[column].mean()
        std_word_in_title = df_all[column].std()
       # print std_word_in_title
        df_all[column] \
            = df_all[column] \
            .map(
                lambda x: (x - mean_word_in_title)
                          / std_word_in_title)
    print "normalized"
    df_all['relevance'] = df_all['relevance'] \
        .map(lambda x: (x - 1) / 2.)
    df_all['relevance'] = df_all['relevance'] \
        .map(lambda x: modify_zero_and_one(x))
    # extract train data and test data
    df_train = df_all.iloc[:train_num]
    df_test = df_all.iloc[train_num:]
    id_test = df_test['id']
    id_test.to_pickle('id_test')

    # # Output targets of training data, training data, testing
    #  data
    # # to pd frame file
    # df_train['relevance'].to_pickle('y_train')
    # df_train \
    #     .drop(['id', 'relevance'], axis=1) \
    #     .to_pickle('X_train')
    # df_test \
    #     .drop(['id', 'relevance'], axis=1) \
    #     .to_pickle('X_test')

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

    truncate_svd_values = all_SVD_transform(df_info)

    # output the feature data to libSVM files.
    y_train = df_train['relevance'].values
    X_train = df_train.drop(
            ['id', 'relevance'], axis=1).values
    X_train = np.append(X_train,
                        truncate_svd_values[:train_num],
                        axis=1)

    X_test = df_test.drop(
            ['id', 'relevance'], axis=1).values
    X_test = np.append(X_test,
                       truncate_svd_values[train_num:],
                       axis=1)
    np.save('X_train_with_SVD', np.nan_to_num(X_train))
    np.save('X_test_with_SVD', np.nan_to_num(X_test))
    np.save('Y_train', y_train)
    print 'Data saved to npy files.'
    # output the feature data to libSVM files.
    dump_svmlight_file(X_train, y_train, 'all_train_libSVM.dat',
                       zero_based=True, multilabel=False)

    # dump libSVM training data into two files: training data and
    # validation data
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
    features = list(
            df_train.drop(['id', 'relevance'], axis=1).columns[0:])
    feature_list = ['search_term', 'product_title',
                    'product_description', 'attributes', 'brand']
    for feature in feature_list:
        for i in range(SVD_component_num):
            features.append(feature + '_' + str(i + 1))
    create_feature_map(features)


if __name__ == '__main__':
    main()
