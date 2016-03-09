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

param = {'max_depth': 5, 'eta': 0.3, 'silent': 1,
         'objective': 'reg:linear', 'gamma': 0.0001,
         'subsample': 0.75, 'colsample_bytree': 0.75}
# param = {'silent': 1, 'objective': 'reg:linear',
#          'subsample': 0.75,  'booster': 'gblinear',
#          'lambda': 1, 'alpha':1,
#          'lambda_bias': 0.4, 'eval_metric': 'logloss'}
watchlist = [(train, 'eval'), (train, 'train')]
num_round = 500
xgb_model = xgb.train(param, train, num_round, watchlist)
prediction = xgb_model.predict(test)
test_id = pd.read_pickle('id_test')
clean_result(prediction)
pd.DataFrame({"id": test_id.values, "relevance": prediction})\
    .to_csv('submission.csv', index=False)
