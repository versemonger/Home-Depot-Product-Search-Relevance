from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import numpy as np
import pandas as pd
import sys


def main():
    X_train = np.load('X_train_with_SVD.npy')
    X_test = np.load('X_test_with_SVD.npy')
    y_train = np.load('Y_train.npy')
    rf = RandomForestRegressor(n_estimators=60, max_depth=8,
                               random_state=7, n_jobs=-1)
    #################Temporarily disable RF#######
    # rf.fit(X_train, y_train)
    # Fit the data with RF
    print "Fit the data with Random Forest Regressor"
    rfr_prediction = rf.predict(X_test)

    # # Output the result
    # pd.DataFrame({"id": id_test, "relevance": y_prediction}) \
    #     .to_csv('submission.csv', index=False)

    # Set XGBRegressor with optimized parameters.
    xgb_model\
        = xgb.XGBRegressor(learning_rate=0.03, silent=True,
                           objective="reg:logistic", gamma=2.15,
                           min_child_weight=5, subsample=0.8,
                           colsample_bytree=0.8, n_estimators=900,
                           max_depth=9)

    # make predictions with tuned parameters and XGBoost model
    xgb_model.fit(X_train, y_train)
    print 'Fit the data with XGBoost'
    xgb_prediction = xgb_model.predict(X_test)

    # # Use MPLRegressor
    # nn_model = MLPRegressor(activation='logistic',
    #                         learning_rate='adaptive',
    #                         learning_rate_init=0.005,
    #                         epsilon=1e-7)
    # nn_model.fit(X_train, y_train)
    # nn_prediction = nn_model.predict(X_test)

    print xgb_prediction[:20]
    print rfr_prediction[:20]
    # ensemble result of two models
    ################ Temporarily use only XGBoost #######
    # prediction = (xgb_prediction + rfr_prediction) / 2
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
