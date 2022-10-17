#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import make_column_selector as selector
import xgboost as xgb 
import bentoml
from sklearn.feature_extraction import DictVectorizer


df = pd.read_csv('CreditScoring.csv')

# make all columns in lower
df.columns = df.columns.str.lower()

# mapping the target
df['status'] = df['status'].map({
    1:'ok',
    2:'default',
    0:'unk'
})

# mapping categorical features
def mapping_categorical(df, cat, cat_lst):
  to_lst = df[cat].value_counts().sort_index().index.to_list()
  cat_lst = cat_lst

  df[cat] = (
      df[cat].map({
          k:v for (k,v) in zip(to_lst, cat_lst)
      })
   )

cols = ['home', 'marital', 'records', 'job']

home_lst = ['unk', 'rent', 'owner', 'private', 'ignore', 'parents', 'other']
marital_lst = ['unk', 'single', 'married', 'widow', 'separated', 'divorced']
records_lst = ['no', 'yes', 'unk']
job_lst = ['unk', 'fixed', 'partime', 'freelance', 'others']
cat_lst = [home_lst, marital_lst, records_lst, job_lst]

for col, cat in zip(cols, cat_lst):
  mapping_categorical(df, col, cat)

# fix missing values
def fix_missing_values(df, val_to_rep, rep, *f_lst):
  for f in f_lst:
    df[f] = df[f].replace(val_to_rep, rep)

fix_missing_values(df, 99999999.0, np.nan, ['income', 'assets', 'debt'])

# don't need unk in status
df = df[df.status != 'unk']

#  data preparation
data, target = df.drop(columns=['status']), df['status'].map({'ok':0, 'default':1})

def tweaking(data, target):
  numerical = selector(dtype_include=np.number)(data)
  categorical = selector(dtype_include=object)(data)

  num_imputer = SimpleImputer(missing_values=np.NaN, strategy='constant', fill_value=0)
  cat_imputer = SimpleImputer(strategy='most_frequent', fill_value='unk')
  cat_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
  
  X_full_train, X_test, y_full_train, y_test = model_selection.train_test_split(
      data,
      target,
      test_size=.2,
      random_state=11,
    )
  X_train, X_dev, y_train, y_dev = model_selection.train_test_split(
          X_full_train,
          y_full_train,
          test_size=.25,
          random_state=11,
        )


  X_train.loc[:, numerical] = num_imputer.fit_transform(X_train[numerical])
  X_train.loc[:, categorical] = cat_imputer.fit_transform(X_train[categorical])
  X_train.loc[:, categorical] = cat_encoder.fit_transform(X_train[categorical])

  X_dev.loc[:, numerical] = num_imputer.transform(X_dev[numerical])
  X_dev.loc[:, categorical] = cat_imputer.transform(X_dev[categorical])
  X_dev.loc[:, categorical] = cat_encoder.transform(X_dev[categorical])

  X_test.loc[:, numerical] = num_imputer.transform(X_test[numerical])
  X_test.loc[:, categorical] = cat_imputer.transform(X_test[categorical])
  X_test.loc[:, categorical] = cat_encoder.transform(X_test[categorical])

  return X_train, y_train, X_dev, y_dev, X_test, y_test

X_train, y_train, X_dev, y_dev, X_test, y_test = tweaking(data, target)

# # wrap data into DMatrix — a special
# # data structure for finding splits efficiently.
# dtrain = xgb.DMatrix(
#     X_train.values, 
#     label=y_train.values, 
#     feature_names=X_train.columns
# )

# # for validation
# dval = xgb.DMatrix(
#     X_dev.values,
#     label=y_dev.values,
#     feature_names=X_dev.columns
# )

# # specifying the parameters for training
# xgb_params = {
#     'eta':.3,
#     'max_depth':6,
#     'min_child_weight': 1,
#     'objective': 'binary:logistic',
#     'nthread': -1,
#     'seed': 1,
#     'silent':1
# }

# # For training an XGBoost model, we use the train function
# model = xgb.train(
#     xgb_params,
#     dtrain,
#     num_boost_round=10
# )

# y_pred = model.predict(dval)
# metrics.roc_auc_score(y_dev, y_pred)

# watchlist = [(dtrain, 'train'), (dval, 'dev')]

# xgb_params = {
#     'eta':.05,
#     'max_depth':3,
#     'min_child_weight': 30,
#     'objective': 'binary:logistic',
#     'eval_metric': 'auc',
#     'nthread': -1,
#     'seed': 1,
#     'silent':1
# }

# model = xgb.train(
#     xgb_params,
#     dtrain,
#     num_boost_round=500,
#     evals=watchlist,
#     verbose_eval=10
# )

# dtest = xgb.DMatrix(
#     X_test.values,
#     label=y_test.values, 
#     feature_names=X_test.columns
# )

# y_pred_dev = model.predict(dval)
# y_pred_test = model.predict(dtest)

# # print(f"AUC-dev: {metrics.roc_auc_score(y_dev, y_pred_dev): .3f}")
# # print(f"AUC-test: {metrics.roc_auc_score(y_test, y_pred_test): .3f}")

# last model
X_full_train = pd.concat([X_train, X_dev])
y_full_train = pd.concat([y_train, y_dev])

full_train_dict = X_full_train.to_dict(orient='records')
X_test_dict = X_test.to_dict(orient='records')
dv = DictVectorizer(sparse=False, sort=False)

X_full_train = dv.fit_transform(full_train_dict)
X_test = dv.transform(X_test_dict)

dfulltrain = xgb.DMatrix(
    X_full_train, 
    label=y_full_train.values, 
    # feature_names=dv.get_feature_names_out(),
)

dtest = xgb.DMatrix(
    X_test,
    label=y_test.values, 
    # feature_names=dv.get_feature_names_out(),
)


xgb_params = {
    'eta': 0.1, 
    'max_depth': 3,
    'min_child_weight': 1,

    'objective': 'binary:logistic',
    'eval_metric': 'auc',

    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}

last_model = xgb.train(xgb_params, dfulltrain, num_boost_round=175)

y_pred = last_model.predict(dtest)
print(f"AUC-test: {metrics.roc_auc_score(y_test, y_pred): .3f}")


bento_xgb = bentoml.xgboost.save_model(
    "credit_risk_model", 
    last_model,
    custom_objects={
        "dico": dv
    }
)
print(bento_xgb.tag)





