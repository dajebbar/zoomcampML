import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import bentoml

data = ('https://raw.githubusercontent.com/alexeygrigorev/'
        'mlbookcamp-code/master/chapter-06-trees/CreditScoring.csv'
)
df = pd.read_csv(data)

df.columns = df.columns.str.lower()

status_values = {
    1: 'ok',
    2: 'default',
    0: 'unk'
}

df.status = df.status.map(status_values)

home_values = {
    1: 'rent',
    2: 'owner',
    3: 'private',
    4: 'ignore',
    5: 'parents',
    6: 'other',
    0: 'unk'
}

df.home = df.home.map(home_values)

marital_values = {
    1: 'single',
    2: 'married',
    3: 'widow',
    4: 'separated',
    5: 'divorced',
    0: 'unk'
}

df.marital = df.marital.map(marital_values)

records_values = {
    1: 'no',
    2: 'yes',
    0: 'unk'
}

df.records = df.records.map(records_values)

job_values = {
    1: 'fixed',
    2: 'partime',
    3: 'freelance',
    4: 'others',
    0: 'unk'
}

df.job = df.job.map(job_values)

for c in ['income', 'assets', 'debt']:
    df[c] = df[c].replace(to_replace=99999999, value=np.nan)

df = df[df.status != 'unk'].reset_index(drop=True)

df_train, df_test = train_test_split(df, test_size=0.2, random_state=11)

df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = (df_train.status == 'default').astype('int').values
y_test = (df_test.status == 'default').astype('int').values

del df_train['status']
del df_test['status']

dv = DictVectorizer(sparse=False)

train_dicts = df_train.fillna(0).to_dict(orient='records')
X_train = dv.fit_transform(train_dicts)

test_dicts = df_test.fillna(0).to_dict(orient='records')
X_test = dv.transform(test_dicts)

# rf = RandomForestClassifier(n_estimators=200,
#                             max_depth=10,
#                             min_samples_leaf=3,
#                             random_state=1)
# rf.fit(X_train, y_train)
# y_pred = rf.predict(X_test)
# # print(f"Auc: {metrics.roc_auc_score(y_test, y_pred)}")

# rf_model = bentoml.sklearn.save_model(
#     "risk_model",
#     rf,
#     custom_objects={
#         "dictVectorizer": dv
#     }
# )
# print(rf_model.tag)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

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

model = xgb.train(xgb_params, dtrain, num_boost_round=175)
# y_pred = model.predict(dtest)
# print(f"AUC: {metrics.roc_auc_score(y_test, y_pred)}")

xgb_model = bentoml.xgboost.save_model(
    'z_credit_model',
    model,
    custom_objects={
        'dictVectorizer': dv
    })

print(f"model tag: {xgb_model.tag}")
