import xgboost as xgb
import pandas as pd
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn import grid_search, cross_validation
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import sys


def fmean_squared_error(ground_truth, predictions):
    """

    :param ground_truth:
    :param predictions:
    :return:
    """
    fmean_squared_error_ =\
        mean_squared_error(ground_truth, predictions)**0.5
    return fmean_squared_error_



def clean_result(x):
    """
    As the Kaggle requires that the result should be in range of
    [1, 3], we change the value of the numbers that fall out of
    this range.
    :param x:
    :return:
    """
    x[x >= 3] = 3
    x[x <= 1] = 1


def random_forest_regressor(X_train, y_train, cv_flag):
    """
    Perform grid search on parameters or  cross-validation
    on the random forest regressor.
    :param cv_flag: True: perform k-fold cross-validation
                    False: perform grid search for parameters
    :param X_train: feature matrix of training data
    :param y_train: target vector of training data
    """
    # rescale y_train to range of [1, 3]
    y_train = y_train * 2 + 1
    rf = RandomForestRegressor(n_jobs=-1, n_estimators=70,
                               max_depth=8)
    # optimal in the following: 70, 10, False
    param_grid = {'n_estimators': [60, 70, 80],
                  'bootstrap_features': [True, False]}

    rmse = make_scorer(fmean_squared_error,
                       greater_is_better=False)
    if cv_flag:
        scores = cross_validation.cross_val_score(
            rf, X_train, y_train, cv=5, scoring=rmse)
        print np.mean(scores)
        return
    # Do grid search with a set of parameters for XGBoost.
    rf_model \
        = grid_search\
        .GridSearchCV(estimator=rf, param_grid=param_grid,
                      n_jobs=-1, cv=3, verbose=20,
                      scoring=rmse)

    print 'start search'
    rf_model.fit(X_train, y_train)
    print("Best parameters found by grid search:")
    print(rf_model.best_params_)
    print("Best CV score:")
    print(-rf_model.best_score_)


def XGBoost_regressor1(X_train, y_train, cv_flag):
    """
    Perform grid search on parameters or  cross-validation
    on the XGBoost regressor.
    :param cv_flag: True: perform k-fold cross-validation
                    False: perform grid search for parameters
    :param X_train: feature matrix of training data
    :param y_train: target vector of training data

    """
    # Some set of parameter grids I have used
    #              'gamma': [2.20, 2.25, 2.3, 2.35],
    #          'min_child_weight': [0.3, 0.6, 1, 2]
    #          'colsample_bytree': [0.4, 0.45, 0.5, 0.55]
    #       'learning_rate': [0.005, 0.01, 0.03, 0.06],
    #              'min_child_weight': [0.5, 1, 1.5, 2]

    # Builder scorer which is used to do grid search for XGBoost
    rmse = make_scorer(fmean_squared_error,
                       greater_is_better=False)

    # Set XGBRegressor with some parameters.
    xgb_model\
        = xgb.XGBRegressor(learning_rate=0.06, silent=True,
                           objective="reg:logistic",
                           gamma=2.25, min_child_weight=5,
                           subsample=0.8, scale_pos_weight=0.9,
                           colsample_bytree=0.8,
                           n_estimators=115, max_depth=10)

    if cv_flag:
        scores = cross_validation.cross_val_score(
            xgb_model, X_train, y_train, cv=5, scoring=rmse)
        print np.mean(scores)
        return

    # optimal 10, 2.25
    param_grid = {'max_depth': [9, 10, 11, 12],
                  'gamma': [2.15, 2.25, 2.30]}

    # Do grid search with a set of parameters for XGBoost.
    model \
        = grid_search\
        .GridSearchCV(estimator=xgb_model, param_grid=param_grid,
                      n_jobs=-1, cv=2, verbose=20, scoring=rmse)
    print 'start search'
    model.fit(X_train, y_train)
    print("Best parameters found by grid search:")
    print(model.best_params_)
    print("Best CV score:")
    print(-model.best_score_)


def XGBoost_regressor2():
    """
    Train an XGBoost model with XGBoost lib.
    """
    train = xgb.DMatrix('train_libSVM.dat')
    all_train = xgb.DMatrix('all_train_libSVM.dat')
    test = xgb.DMatrix('test_libSVM.dat')
    validation = xgb.DMatrix('validate_libSVM.dat')
    param = {'max_depth': 9, 'eta': 0.025, 'silent': 1,
             'objective': 'reg:logistic', 'gamma': 2.15,
             'subsample': 0.8, 'colsample_bytree': 0.8,
             'min_child_weight': 5}

    watchlist = [(validation, 'eval'), (train, 'train')]
    num_round = 150
    xgb_model = xgb.train(param, train, num_round, watchlist)
    # xgb_model = xgb.cv(param, all_train, num_round, nfold=5,
    #                    metrics={'error'})
    # print xgb_model.head()
    # xgb_model.info()

    prediction = xgb_model.predict(test)
    importance = xgb_model.get_fscore(fmap='xgb.fmap')
    print importance
    xgb.plot_importance(xgb_model)
    test_id = pd.read_pickle('id_test')
    prediction = prediction * 2 + 1
    prediction[prediction > 3] = 3
    prediction[prediction < 1] = 1
    clean_result(prediction)
    pd.DataFrame({"id": test_id.values, "relevance": prediction})\
        .to_csv('submission.csv', index=False)


def main():
    X_train = pd.read_pickle('X_train').values
    y_train = pd.read_pickle('y_train').values
    random_forest_regressor(X_train, y_train, True)


if __name__ == '__main__':
    main()
