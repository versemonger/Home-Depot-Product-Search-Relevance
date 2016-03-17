from sklearn.ensemble \
    import RandomForestRegressor, BaggingRegressor
import xgboost as xgb
from sklearn import grid_search
from sklearn.metrics import mean_squared_error, make_scorer
import pandas as pd


def fmean_squared_error(ground_truth, predictions):
    """
    Used for evaluation of predictions with mean_squared_error
    """
    fmean_squared_error_ =\
        mean_squared_error(ground_truth, predictions)**0.5
    return fmean_squared_error_


def main():
    X_train = pd.read_pickle('X_train').values
    X_test = pd.read_pickle('X_test').values
    y_train = pd.read_pickle('y_train').values
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

    # rescale the result to [1,3]
    prediction = prediction * 2 + 1
    prediction[prediction > 3] = 3
    prediction[prediction < 1] = 1

    id_test = pd.read_pickle('id_test')
    # output the result
    pd.DataFrame({"id": id_test, "relevance": prediction}) \
        .to_csv('submission.csv', index=False)


if __name__ == '__main__':
    main()