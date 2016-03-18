import xgboost as xgb
import pandas as pd
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn import grid_search
import sys


def fmean_squared_error(ground_truth, predictions):
    """
    Used for evaluation of predictions with mean_squared_error
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



def main():
    X_train = pd.read_pickle('X_train').values
    X_test = pd.read_pickle('X_test').values
    y_train = pd.read_pickle('y_train').values
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
                           objective="reg:logistic", gamma=2.25,
                           min_child_weight=1.5, subsample=1,
                           colsample_bylevel=0.8,
                           scale_pos_weight=0.9,
                           colsample_bytree=0.9, n_estimators=84,
                           max_depth=7)
    param_grid = {

                  }

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

    # Don't run the following code for now because I am in para
    # tuning phase.
    sys.exit()
    train = xgb.DMatrix('train_libSVM.dat')
    all_train = xgb.DMatrix('all_train_libSVM.dat')
    test = xgb.DMatrix('test_libSVM.dat')
    validation = xgb.DMatrix('validate_libSVM.dat')
    param = {'max_depth': 15, 'eta': 0.06, 'silent': 1,
             'objective': 'reg:logistic', 'gamma': 3,
             'subsample': 0.8, 'colsample_bytree': 0.8,
             'min_child_weight': 5}

    watchlist = [(validation, 'eval'), (train, 'train')]
    num_round = 130
    xgb_model = xgb.train(param, train, num_round, watchlist)
    # xgb_model = xgb.cv(param, all_train, num_round, nfold=5,
    #                    metrics={'error'})
    # print xgb_model.head()
    # xgb_model.info()

    prediction = xgb_model.predict(test)
    test_id = pd.read_pickle('id_test')
    prediction = prediction * 2 + 1
    prediction[prediction > 3] = 3
    prediction[prediction < 1] = 1
    clean_result(prediction)
    pd.DataFrame({"id": test_id.values, "relevance": prediction})\
        .to_csv('submission.csv', index=False)


if __name__ == '__main__':
    main()
