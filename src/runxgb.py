import xgboost as xgb
import pandas as pd
import numpy as np
import math
from xgboost.sklearn import XGBClassifier,XGBRegressor
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search

import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

target = 'Score'
IDcol = 'Id'
dtrain = xgb.DMatrix('../data/train.txt.train')
dtest = xgb.DMatrix('../data/train.txt.test')
rdtest = xgb.DMatrix('../data/test.txt')

# suse different version of label
'''
Y = dtrain.get_label()
Y = [math.log(x)for x in Y]
dtrain.set_label(Y)

Y = dtest.get_label()
Y = [math.log(x)for x in Y]
dtest.set_label(Y)
'''
def write_pred(filename, res,flag):
        fo = open( filename, 'w' )
        pos = 0
        for r in res:
            if flag == True:
                fo.write(str(40000+pos)+' ')
            fo.write(str(r))
            fo.write('\n')
            pos += 1
        fo.close()

param = {'max_depth':3,
    'eta': 0.3,
    'gamma': 0.8,
    'min_child_weight': 1,
   # 'save_period': 0,
    'booster': 'gbtree',
    'slient':1,
	'base_score':5,
    'objective': 'count:poisson',
    #'subsample':1.0,
    #'colsample_bytree':1.0 ,
    'eval_metric':'rmse',
	#'alpha':0.5,
	#'lambda':1.5,
	#'normalize_type':'forest',
}

num_round = 500
watchlist = [(dtest, 'eval'), (dtrain, 'train')]
bst = xgb.train(param, dtrain, num_round, watchlist)
preds = bst.predict(dtest)
preds_true = [math.exp (x) for x in preds]
write_pred('preds.txt',preds_true,False)
preds = bst.predict(rdtest)
write_pred('predres.txt', preds, True)
#bst.dump_model('dump2.nice.txt', 'featmap.txt')
'''
def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):

    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds, show_progress=False)
        alg.set_params(n_estimators=cvresult.shape[0])

    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['Disbursed'],eval_metric='auc')

    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]

    #Print model report:
    print "\nModel Report"
    print "Accuracy : %.4g" % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions)
    print "AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob)

    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')


x = xgb.cv(param, dtrain, num_boost_round=10, nfold=3, stratified=False, folds=None, metrics=(), \
obj=None, feval=None, maximize=False, early_stopping_rounds=None, fpreproc=None, as_pandas=True, \
verbose_eval=None, show_stdv=True, seed=0, callbacks=None)
print x

train = pd.read_csv('../data/train.csv')
train = pd.get_dummies(train)
predictors = [x for x in train.columns if x not in [target, IDcol]]
param_test1 = {
 'max_depth':range(3,4,1),
 'min_child_weight':range(1,2,1)
}
gsearch1 = GridSearchCV(estimator = XGBRegressor( learning_rate =0.1, n_estimators=140, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'reg:linear', nthread=4, scale_pos_weight=1, seed=27),
 param_grid = param_test1, scoring='neg_mean_squared_error',n_jobs=4,iid=False, cv=5)
gsearch1.fit(train[predictors],train[target])
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
'''
