import os

from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import numpy as np
import pandas as pd
import sys


def main():
    X_train = np.load('X_train_with_SVD.npy')
    X_test = np.load('X_test_with_SVD.npy')
    y_train = np.load('Y_train.npy')
    rf_enabled = False
    if rf_enabled:
        rf = RandomForestRegressor(n_estimators=850, max_depth=9,
                                   n_jobs=-1)
        print "Fit the data with Random Forest Regressor"
        rf.fit(X_train, y_train)
        rfr_prediction = rf.predict(X_test)

    # Set XGBRegressor with optimized parameters.
    xgb_model\
        = xgb.XGBRegressor(learning_rate=0.03, silent=True,
                           objective="reg:logistic", gamma=2.2,
                           min_child_weight=5, subsample=0.8,
                           colsample_bytree=0.7, n_estimators=1198,
                           scale_pos_weight=0.6, max_depth=11)
    print 'Fit the data with XGBoost'
    # make predictions with tuned parameters and XGBoost model
    xgb_model.fit(X_train, y_train)

    xgb_prediction = xgb_model.predict(X_test)

    print xgb_prediction[:20]
    if rf_enabled:
        print rfr_prediction[:20]

    # ensemble result of two models
    ################ Temporarily use only XGBoost #######
    if rf_enabled:
        prediction = xgb_prediction * 0.85 + rfr_prediction * 0.15
    else:
        prediction = xgb_prediction

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
