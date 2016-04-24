"""
Adapted from https://www.kaggle.com/wenxuanchen/home-depot-product-search-relevance/sklearn-random-forest
This script basically counts the occurrences of each word of search
term in different part of information of the products.
http://orion.lcg.ufrj.br/Dr.Dobbs/books/book5/chap14.htm
"""
import pandas as pd
import sys

import pickle
from sklearn import preprocessing
from sklearn.datasets import dump_svmlight_file
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy, zscore
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

stopwords_list = stopwords.words('english')
stemmed_stopwords = [stemmer.stem(stop_word)
                     for stop_word in stopwords_list]

SVD_component_num = 12
# We will be normalize data named with these features.
normalize_feature_list = []
SVD_component_feature_list \
    = ['search_term', 'product_title', 'product_description',
       'attributes', 'brand', 'color', 'material']


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
                if len(word) >= 1])


def find_occurrences2(str1, str2):
    """
    find in str2 occurrences of each pair of adjacent words in str1
    example: str1 = "good job", str2 = "good job good job"
    its occurrence is 2.
    :param str1:
    :param str2:
    :return:
    """
    word_list = [word for word in str1.split() if len(word) >= 1]
    new_word_list = []
    for i in range(len(word_list) - 1):
        new_word_list.append(word_list[i] + ' ' + word_list[i + 1])
    return sum([str2.count(word_pair) for word_pair
                in new_word_list])


def find_common_word(str1, str2):
    """
    find number of common words in str1 and str2
    :param str1: search term
    :param str2: a column of information of product
    :return: The number of words in str1 that appear in str2
    """
    return sum([str2.find(word) >= 0 for word in str1.split()
                if len(word) >= 1])


def range_filter(x):
    """
    For each feature, round to to large data to 6
    :param x: feature in one sample, an array
    :return: x if x is small and 6 if x is larger than 6
    """
    x[x > 6] = 6
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
    first_term = 'search_term'
    reduced_matrix = single_feature_SVD_transform(df, first_term)
    for feature in SVD_component_feature_list:
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
    return np.array(zscore(result))


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
    lsi_transformer = TruncatedSVD(n_components=120,
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
    global normalize_feature_list
    normalize_feature_list.append(new_feature)
    normalize_feature_list.append(new_feature2)
    df[new_feature] = pd.Series(similarity_one_dimension)
    df[new_feature2] = pd.Series(KL_similarity)
    print new_feature2, KL_similarity[:10]
    if feature_name == 'text':
        df.drop(['text'], axis=1, inplace=True)


def create_feature_map(features):
    """
    create feature map and dict of feature and index
    for example: {'product_uid': 0,
                  'similarity in product_title': 1}
    :param features: The columns names of data frame
    :return: a map file between feature names and feature index
    """
    mapfile = open('xgb.fmap', 'w')
    i = 0
    dicfile = open('feature_index_dict', 'w')
    feature_index_dict = {}
    for feat in features:
        feature_index_dict[feat] = i
        mapfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i += 1
    mapfile.close()
    pickle.dump(feature_index_dict, dicfile)
    dicfile.close()


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


def extract_occurrence_and_ratio_short(df_all, occurr1, index,
                                       ratio_name_1, length):
    """

    :param length: name of the length column of the attribute
    :param df_all: the whole data frame
    :param occurr1: column where we want to find occurrence of
                    single search term words
    :param index: index of the column in the text separated with
                  tab
    :param ratio_name_1: occurrence of occurr1 divided by length of
                    that column

    """
    df_all[occurr1] = df_all['product_info'] \
        .map(lambda x:
             find_occurrences(
                     x.split('\t')[0], x.split('\t')[index]))

    ratio = df_all[occurr1].values / \
            df_all[length].values.astype(float)
    ratio[np.isinf(ratio)] = 0
    ratio = np.nan_to_num(ratio)
    df_all[ratio_name_1] \
        = pd.DataFrame(ratio)


def extract_occurrence_and_ratio_short_syn(df_all, occurr1, index,
                                           ratio_name_1, length):
    """

    :param length: name of the length column of the attribute
    :param df_all: the whole data frame
    :param occurr1: column where we want to find occurrence of
                    single search term words
    :param index: index of the column in the text separated with
                  tab
    :param ratio_name_1: occurrence of occurr1 divided by length of
                    that column

    """
    df_all[occurr1] = df_all['product_info'] \
        .map(lambda x:
             find_occurrences(
                     x.split('\t')[7], x.split('\t')[index]))
    ratio = df_all[occurr1].values / \
        df_all[length].values.astype(float)
    ratio[np.isinf(ratio)] = 0
    ratio = np.nan_to_num(ratio)
    df_all[ratio_name_1] \
        = pd.DataFrame(ratio)


def extract_occurrence_and_ratio(df_all, occurr1, index,
                                 ratio_name_1, length, occurr2,
                                 ratio_name_2):
    """
    :param length: name of the length column of the attribute
    :param df_all: the whole data frame
    :param occurr1: column where we want to find occurrence of
                    single search term words
    :param occurr2: column where we want to find occurrence of
                    pair of search term words
    :param index: index of the column in the text separated with
                  tab
    :param ratio_name_1: occurrence of occurr1 divided by length of
                    that column
    :param ratio_name_2: occurrence of occurr2 divided by length of
                    that column

    """
    df_all[occurr1] = df_all['product_info'] \
        .map(lambda x:
             find_occurrences(
                     x.split('\t')[0], x.split('\t')[index]))
    print occurr1 + 'info processed'
    print df_all[occurr1][0:10]

    ratio = df_all[occurr1].values / \
            df_all[length].values.astype(float)
    ratio[np.isinf(ratio)] = 0
    ratio = np.nan_to_num(ratio)
    df_all[ratio_name_1] \
        = pd.DataFrame(ratio)
    print ratio_name_1 + 'info processed'
    print df_all[ratio_name_1][0:10]

    df_all[occurr2] = df_all['product_info'] \
        .map(lambda x:
             find_occurrences2(
                     x.split('\t')[0], x.split('\t')[index]))
    print occurr2 + 'info processed'
    print df_all[occurr2][0:10]

    ratio2 = df_all[occurr2].values / \
             df_all[length].values.astype(float)
    ratio2[np.isinf(ratio2)] = 0
    ratio2 = np.nan_to_num(ratio2)
    df_all[ratio_name_2] \
        = pd.DataFrame(ratio2)
    print ratio_name_2 + 'info processed'
    print df_all[ratio_name_2][0:10]


def main():
    # import the number of training tuples
    df_train = pd.read_csv("train.csv", encoding="ISO-8859-1")
    train_num = df_train.shape[0]
    df_train = None

    df_all = pd.read_pickle('df_all')
    # feature_list = ['product_title', 'product_description',
    #                 'attributes', 'brand', 'text']
    feature_list = ['product_title', 'product_description',
                    'attributes', 'brand', 'text']

    for feature in feature_list:
        get_saperate_LSI_score(df_all, feature)
    print 'LSI_score added.'

    # Coalesce all information into one column so we can apply
    # map to that one column
    df_all['product_info'] \
        = df_all['search_term'] + "\t" + df_all['product_title'] \
          + "\t" + df_all['product_description'] + "\t" \
          + df_all['attributes'] + "\t" + df_all['brand'] + '\t' \
          + df_all['color'] + '\t' + df_all['material'] + '\t' \
          + df_all['search_term_synonym']

    # Count number of words in each column
    df_all['title_length'] \
        = df_all['product_title'] \
        .map(lambda x: len(x.split()))
    print 'title length info:'
    print df_all['title_length'][0:10]

    df_all['description_length'] \
        = df_all['product_description'] \
        .map(lambda x: len(x.split()))
    print 'description length info'
    print df_all['description_length'][0:10]

    df_all['attributes_length'] = df_all['attributes'] \
        .map(lambda x: len(x.split()))
    print 'attribute length info'
    print df_all['attributes_length'][0:10]

    df_all['brand_length'] = df_all['brand'] \
        .map(lambda x: len(x.split()))
    print 'brand length info'
    print df_all['brand_length'][0:10]

    df_all['search_term_synonym_length'] = df_all['brand'] \
        .map(lambda x: len(x.split()))
    print 'search_term_synonym length info'
    print df_all['search_term_synonym_length'][0:10]

    print "Number of words in each column is counted."

    args1 = ['title', 'description', 'attributes', 'brand']
    for index in [0, 1, 2, 3]:
        occurr1 = 'word_in_' + args1[index]
        ratio_name1 = args1[index] + '_ratio'
        length = args1[index] + '_length'
        occurr2 = 'word_pair_in_' + args1[index]
        ratio_name2 = ratio_name1 + '_pair'
        if index != 3:
            extract_occurrence_and_ratio(df_all, occurr1, index,
                                         ratio_name1, length,
                                         occurr2, ratio_name2)
        else:
            extract_occurrence_and_ratio_short(df_all, occurr1,
                                               index, ratio_name1,
                                               length)
    print 'Word occurrences in each column counted.'
    print 'Ratios calculated.'

    for index in [0, 1, 2, 3]:
        occurr1 = 'word_in_' + args1[index] + '_syn'
        ratio_name1 = args1[index] + '_ratio' + '_syn'
        length = args1[index] + '_length'
        extract_occurrence_and_ratio_short_syn(df_all, occurr1,
                                               index, ratio_name1,
                                               length)
    print 'Word occurrences of synonyms counted.'

    common_list = ['title', 'description', 'attributes', 'brand',
                   'color', 'material']
    for i in range(6):
        df_all['common_in_' + common_list[i]] = \
            df_all['product_info'] \
            .map(lambda x:
                 find_common_word(
                     x.split('\t')[0], x.split('\t')[i + 1]
                 ))
        df_all['common_in_' + common_list[i] + '_syn'] = \
            df_all['product_info'] \
            .map(lambda x:
                 find_common_word(
                     x.split('\t')[7], x.split('\t')[i + 1]
                 ))
    print 'Common words counted'

    print 'Common words in each column counted'
    # TODO: normalize it or not?
    df_all['search_term_length'] = df_all['search_term'] \
        .map(lambda x: len(x.split()))

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
                      'brand', 'color', 'material',
                      'search_term_synonym']]

    df_all.drop(['search_term', 'product_title',
                 'product_description', 'product_info',
                 'attributes', 'brand', 'color', 'material',
                 'search_term_synonym'],
                axis=1, inplace=True)

    normalize_filtered_feature = ['word_in_title',
                                  'word_in_description',
                                  'word_in_attributes',
                                  'word_pair_in_title',
                                  'word_pair_in_description',
                                  'word_pair_in_attributes',
                                  'word_in_title_syn',
                                  'word_in_description_syn'
                                  'word_in_attributes_syn']

    # Normalize a part of data in df
    for column in normalize_filtered_feature:
        df_all[column] \
            = pd.DataFrame(
                zscore(range_filter(df_all[column].values)))
        print column, ':'
        print df_all[column][0:10]
    print 'Z norm gotten.'

    global normalize_feature_list
    normalize_feature_list.extend(
            ['title_length', 'description_length',
             'attributes_length', 'common_in_title',
             'common_in_description', 'common_in_attributes',
             'length_of_search_term', 'last_search_term_in_title',
             'last_search_term_in_description',
             'last_search_term_in_attributes',
             'title_ratio', 'description_ratio',
             'attributes_ratio', 'title_ratio_pair',
             'description_ratio_pair', 'attributes_ratio_pair',
             'common_in_title_syn',
             'common_in_description_syn',
             'common_in_attributes_syn'])

    # Normalize a part of data in df
    for column in normalize_feature_list:
        df_all[column] \
            = pd.DataFrame(zscore(df_all[column].values))
        print column, ':'
        print df_all[column][0:10]
    print 'Z norm gotten.'

    # scale the features
    min_max_scaler = preprocessing.MinMaxScaler()

    # rescale a part of data in df
    for column in ['word_in_brand', 'common_in_brand',
                   'brand_length', 'brand_ratio',
                   'common_in_color', 'common_in_material',
                   'word_in_brand_syn', 'common_in_brand_syn',
                   'brand_ratio_syn', 'common_in_color_syn',
                   'common_in_material_syn']:
        df_all[column] \
            = pd.DataFrame(min_max_scaler
                           .fit_transform(df_all[column].values.
                                          reshape(-1, 1)))
        print column, ':'
        print df_all[column][0:10]
    print "rescaled"
    df_all['relevance'] = df_all['relevance'] \
        .map(lambda x: (x - 1) / 2.)
    df_all['relevance'] = df_all['relevance'] \
        .map(lambda x: modify_zero_and_one(x))
    # extract train data and test data
    df_train = df_all.iloc[:train_num]
    df_test = df_all.iloc[train_num:]
    id_test = df_test['id']
    id_test.to_pickle('id_test')

    truncate_svd_values = all_SVD_transform(df_info)
    print truncate_svd_values[0:10]

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
    for feature in SVD_component_feature_list:
        for i in range(SVD_component_num):
            features.append(feature + '_' + str(i + 1))
    create_feature_map(features)


if __name__ == '__main__':
    main()
