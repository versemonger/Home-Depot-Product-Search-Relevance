import xgboost as xgb
import pandas as pd
import numpy as np


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

train = xgb.DMatrix('train_libSVM.dat')
test = xgb.DMatrix('test_libSVM.dat')
validation = xgb.DMatrix('validate_libSVM.dat')
param = {'max_depth': 15, 'eta': 0.01, 'silent': 1,
         'objective': 'reg:linear', 'gamma': -5,
         'subsample': 0.5, 'colsample_bytree': 0.8}

watchlist = [(validation, 'eval'), (train, 'train')]
num_round = 221
xgb_model = xgb.train(param, train, num_round, watchlist)
prediction = xgb_model.predict(test)
test_id = pd.read_pickle('id_test')
clean_result(prediction)
pd.DataFrame({"id": test_id.values, "relevance": prediction})\
    .to_csv('submission.csv', index=False)
