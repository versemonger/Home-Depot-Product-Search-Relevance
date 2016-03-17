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
from nltk.corpus import stopwords
import store_data_in_pandas as sd
import sys
import xgboost as xgb
from sklearn import grid_search
from sklearn.metrics import mean_squared_error, make_scorer

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


def fmean_squared_error(ground_truth, predictions):
    """
    Used for evaluation of predictions with mean_squared_error
    """
    fmean_squared_error_ =\
        mean_squared_error(ground_truth, predictions)**0.5
    return fmean_squared_error_

def main():
    # If the script has at least one additional argument,
    # we train the model and generate output file.
    # Otherwise we just output data to libSVM file
    train_model = False
    if len(sys.argv) >= 2:
        train_model = True

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

    # map find_occurrences to the separated information.
    df_all['word_in_title'] = df_all['product_info'] \
        .map(lambda x: find_occurrences(x.split('\t')[0],
                                        x.split('\t')[1]))
    df_all['word_in_description'] = df_all['product_info'] \
        .map(lambda x: find_occurrences(x.split('\t')[0],
                                        x.split('\t')[2]))
    df_all['word_in_attributes'] = df_all['product_info'] \
        .map(lambda x: find_occurrences(x.split('\t')[0],
                                        x.split('\t')[3]))
    df_all['word_in_brand'] = df_all['product_info'] \
        .map(lambda x: find_occurrences(x.split('\t')[0],
                                        x.split('\t')[4]))
    df_all = df_all.drop(['search_term', 'product_title',
                          'product_description', 'product_info',
                          'attributes', 'brand'], axis=1)

    # Normalize all useful data in df
    for column in ['word_in_title', 'word_in_description',
                   'word_in_attributes', 'word_in_brand']:
        df_all[column] \
            = df_all[column].map(lambda x: range_filter(x))
        mean_word_in_title = df_all[column].mean()
        std_word_in_title = df_all[column].std()
        df_all[column] \
            = df_all[column] \
            .map(
            lambda x: (x - mean_word_in_title) / std_word_in_title)
    df_all['relevance'] = df_all['relevance']\
        .map(lambda x: (x - 1) / 2.)
    # extract train data and test data
    df_train = df_all.iloc[:train_num]
    df_test = df_all.iloc[train_num:]
    id_test = df_test['id']
    id_test.to_pickle('id_test')

    y_train = df_train['relevance'].values
    X_train = df_train.drop(
            ['id', 'relevance', 'product_uid'], axis=1).values
    X_test = df_test.drop(
            ['id', 'relevance', 'product_uid'], axis=1).values
    X_test_len = len(X_test)
    if train_model:
        rf = RandomForestRegressor(
                n_estimators=30, max_depth=6, random_state=0)
        clf = BaggingRegressor(
                rf, n_estimators=60, max_samples=0.13,
                random_state=25)
        clf.fit(X_train, y_train)
        rfr_prediction = clf.predict(X_test)

        # # Output the result
        # pd.DataFrame({"id": id_test, "relevance": y_prediction}) \
        #     .to_csv('submission.csv', index=False)

    # Builder scorer which is used to do grid search for XGBoost
    RMSE = make_scorer(fmean_squared_error,
                       greater_is_better=False)
    # Set XGBRegressor with optimized parameters.
    xgb_model\
        = xgb.XGBRegressor(learning_rate=0.06, silent=True,
                           objective="reg:logistic", gamma=2.25,
                           min_child_weight=1.5, subsample=1,
                           colsample_bylevel=0.8,
                           scale_pos_weight=0.9,
                           colsample_bytree=0.9, n_estimators=84,
                           max_depth=7)
    #              'gamma': [2.20, 2.25, 2.3, 2.35],
    #          'min_child_weight': [0.3, 0.6, 1, 2]
    #          'colsample_bytree': [0.4, 0.45, 0.5, 0.55]
    # param_grid = {
    #               'learning_rate': [0.005, 0.01, 0.03, 0.06],
    #               'min_child_weight': [0.5, 1, 1.5, 2]
    #               }
    # # Do grid search with a set of parameters for XGBoost.
    # model \
    #     = grid_search\
    #     .GridSearchCV(estimator=xgb_model, param_grid=param_grid,
    #                   n_jobs=-1, cv=2, verbose=20, scoring=RMSE)
    # print 'start search'
    # model.fit(X_train, y_train)
    # print("Best parameters found by grid search:")
    # print(model.best_params_)
    # print("Best CV score:")
    # print(-model.best_score_)

    # make predictions with tuned parameters and XGBoost model
    xgb_model.fit(X_train, y_train)
    xgb_prediction = xgb_model.predict(X_test)

    # ensemble result of two models
    prediction = (xgb_prediction + rfr_prediction) / 2

    # rescale the result to
    prediction = prediction * 2 + 1
    prediction[prediction > 3] = 3
    prediction[prediction < 1] = 1

    # output the result
    pd.DataFrame({"id": id_test, "relevance": prediction}) \
        .to_csv('submission.csv', index=False)

    # output the feature data to libSVM files.
    validation_num = train_num / 4
    dump_svmlight_file(X_train[: train_num - validation_num],
                       y_train[: train_num - validation_num],
                       'train_libSVM.dat', zero_based=True,
                       multilabel=False)
    dump_svmlight_file(X_train[train_num - validation_num:],
                       y_train[train_num - validation_num:],
                       'validate_libSVM.dat', zero_based=True,
                       multilabel=False)
    test_file_label = np.zeros(X_test_len)
    dump_svmlight_file(X_test, test_file_label, 'test_libSVM.dat',
                       zero_based=True, multilabel=False)


if __name__ == '__main__':
    main()
