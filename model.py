import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier

def train(X_train,y_train):
  clf2 = RandomForestClassifier(n_estimators=300, max_depth=30)
  clf4 = GaussianNB()
  clf5 = XGBClassifier(use_label_encoder=False, max_depth=2)
  votingClf = VotingClassifier(estimators=[
      ('rf', clf2), ('xgb', clf5), ('gnb', clf4)
  ], voting='hard')
  votingClf.fit(X_train, y_train)
  return votingClf

dataset = pd.read_csv('training_set_with_features.csv')

y_train = dataset.iloc[:, 12].values
dataset.pop("is_summary")
X_train=dataset.iloc[:, 3:].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 1:] = sc.fit_transform(X_train[:, 1:])

model = train(X_train,y_train)

def predict(x_test):
  x_test[:, 1:] = sc.transform(x_test[:, 1:])
  return model.predict(x_test)