from sklearn.ensemble \
    import RandomForestRegressor, BaggingRegressor
import xgboost as xgb


import pandas as pd
import sys


def main():
    X_train = pd.read_pickle('X_train').values
    X_test = pd.read_pickle('X_test').values
    y_train = pd.read_pickle('y_train').values
    rf = RandomForestRegressor(
            n_estimators=33, max_depth=8, random_state=7)
    clf = BaggingRegressor(rf, n_estimators=70, max_samples=0.1,
                           random_state=10,
                           bootstrap_features=False, n_jobs=-1)
    clf.fit(X_train, y_train)
    rfr_prediction = clf.predict(X_test)

    # # Output the result
    # pd.DataFrame({"id": id_test, "relevance": y_prediction}) \
    #     .to_csv('submission.csv', index=False)

    # Set XGBRegressor with optimized parameters.
    xgb_model\
        = xgb.XGBRegressor(learning_rate=0.06, silent=True,
                           objective="reg:logistic", gamma=2.25,
                           min_child_weight=5, subsample=0.8,
                           colsample_bytree=0.8, n_estimators=115,
                           max_depth=10)

    # make predictions with tuned parameters and XGBoost model
    xgb_model.fit(X_train, y_train)
    xgb_prediction = xgb_model.predict(X_test)
    print xgb_prediction[:20]
    print rfr_prediction[:20]
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
