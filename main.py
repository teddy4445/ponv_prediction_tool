
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.under_sampling import NearMiss
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, auc
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as mp
from tpot import TPOTClassifier

import shap
     
RS = 73

df = pd.read_csv("data.csv")
print(df.shape)
cols_to_remove = ['WEIGHT', 'SURG_DUR', 'PACU_DUR', 'LOS_TOTAL', 'LOS_POST', "GEN_ANES"]
df.drop(cols_to_remove, axis=1, inplace=True)
print(df.shape)

c = []
d = []
for col in list(df):
  if df[col].nunique() > 10 and col not in ["PROCEDURE", "OPER_DEPT", "PONV_24H_DPT"]:
    c.append("{} & {:.0f} & {:.3f}-{:.3f} & {:.3f} & {:.3f} \\".format(col.replace("_", " "), df[col].count(), np.min(df[col]), np.max(df[col]), np.mean(df[col]), np.std(df[col])))
  else:
    uni = df[col].unique()
    dist = ""
    for u in uni:
      if str(u) == "nan":
        continue
      v = len([True for val in df[col] if val == u])
      dist += "{} : {} ({:.2f}%), ".format(int(u), v, 100 * v / len(df[col]))
    d.append(col.replace("_", " ") + " & " + str(df[col].count()) + " & " + "\multicolumn{3}{l}{" + str(dist) + " } \\ ")

print("feature,N,range,mean,std")
[print(val) for val in c]

print("\n\n\n")
print("feature,N,Disribution")
z = [print(val) for val in d]

df2 = df[df["PONV_24H_DPT"] != 0]
c = []
d = []
for col in ["PONV_24H_DPT"]:
  if df2[col].nunique() > 10 and col not in ["PROCEDURE", "OPER_DEPT", "PONV_24H_DPT"]:
    c.append("{} & {:.0f} & {:.3f}-{:.3f} & {:.3f} & {:.3f} \\".format(col.replace("_", " "), df2[col].count(), np.min(df2[col]), np.max(df2[col]), np.mean(df2[col]), np.std(df2[col])))
  else:
    uni = df2[col].unique()
    dist = ""
    for u in uni:
      if str(u) == "nan":
        continue
      v = len([True for val in df2[col] if val == u])
      dist += "{} : {} ({:.2f}%), ".format(int(u), v, 100 * v / len(df2[col]))
    d.append(col.replace("_", " ") + " & " + str(df2[col].count()) + " & " + "\multicolumn{3}{l}{" + str(dist) + " } \\ ")

print("feature,N,range,mean,std")
[print(val) for val in c]

print("\n\n\n")
print("feature,N,Disribution")
z = [print(val) for val in d]

plt.figure(figsize=(12,12))
c = df.corr()
sns.heatmap(c,
            vmin=-1,
            vmax=1,
            center=0,
            mask=np.triu(np.ones_like(c)).astype(bool),
            cmap="coolwarm")
plt.tight_layout()
plt.savefig("person.pdf", dpi=400)
plt.show()
plt.close()

plt.figure(figsize=(12,12))
sns.heatmap(c,
            mask=np.triu(np.ones_like(c)).astype(bool),
            cmap="coolwarm")
plt.tight_layout()
plt.savefig("person_zoom.pdf", dpi=400)
plt.show()
plt.close()
     
plt.figure(figsize=(12,12))
cd = pd.DataFrame([c.iloc[-3,:]],
                  columns=list(df))
cd.to_csv("pearson_ponv_pacu.csv", index=False)
print("Pearson")
print(list(c.iloc[-3,:]))
print(list(df))

c = df.corr(method="spearman")
cd = pd.DataFrame([c.iloc[-3,:]],
                  columns=list(df))
cd.to_csv("spearman_ponv_pacu.csv", index=False)
print("\n\nSpearman")
print(list(c.iloc[-3,:]))
print(list(df))

def print_model_results(y_train, y_test, y_train_pred, y_test_pred):
  print("ACC | train set = {:.3f}".format(accuracy_score(y_train, y_train_pred)))
  print("ACC | test set = {:.3f}".format(accuracy_score(y_test, y_test_pred)))

  print("F1 | train set = {:.3f}".format(f1_score(y_train, y_train_pred)))
  print("F1 | test set = {:.3f}".format(f1_score(y_test, y_test_pred)))

  print("Recall | train set = {:.3f}".format(recall_score(y_train, y_train_pred)))
  print("Recall | test set = {:.3f}".format(recall_score(y_test, y_test_pred)))

  print("Precision | train set = {:.3f}".format(precision_score(y_train, y_train_pred)))
  print("Precision | test set = {:.3f}".format(precision_score(y_test, y_test_pred)))
  
y_col = list(df)[-3]
y = df[y_col]
x = df.drop([y_col], axis=1)
undersample = NearMiss(version=1, n_neighbors=3)
# transform the dataset
x, y = undersample.fit_resample(x, y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=RS)

print("apfel_simplified_score")
for val in range(6):
  y_train_pred = x_train["apfel_simplified_score"].apply(lambda x: x >= val)
  y_test_pred = x_test["apfel_simplified_score"].apply(lambda x: x >= val)
  print("Test trashold = {}".format(val))
  print_model_results(y_train, y_test, y_train_pred, y_test_pred)


print("koivuranta_score")
for val in range(6):
  y_train_pred = x_train["koivuranta_score"].apply(lambda x: x >= val)
  y_test_pred = x_test["koivuranta_score"].apply(lambda x: x >= val)
  print("Test trashold = {}".format(val))
  print_model_results(y_train, y_test, y_train_pred, y_test_pred)
  
print("guidelines_risk_factors")
for val in range(6):
  y_train_pred = x_train["guidelines_risk_factors"].apply(lambda x: x >= val)
  y_test_pred = x_test["guidelines_risk_factors"].apply(lambda x: x >= val)
  print("Test trashold = {}".format(val))
  print_model_results(y_train, y_test, y_train_pred, y_test_pred)


y_col = list(df)[-2]
x = df.drop([y_col], axis=1)
y = df[y_col]
undersample = NearMiss(version=1, n_neighbors=3)
# transform the dataset
x, y = undersample.fit_resample(x, y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=RS)

print("apfel_simplified_score")
for val in range(6):
  y_train_pred = x_train["apfel_simplified_score"].apply(lambda x: x >= val)
  y_test_pred = x_test["apfel_simplified_score"].apply(lambda x: x >= val)
  print("Test trashold = {}".format(val))
  print_model_results(y_train, y_test, y_train_pred, y_test_pred)

print("koivuranta_score")
for val in range(6):
  y_train_pred = x_train["koivuranta_score"].apply(lambda x: x >= val)
  y_test_pred = x_test["koivuranta_score"].apply(lambda x: x >= val)
  print("Test trashold = {}".format(val))
  print_model_results(y_train, y_test, y_train_pred, y_test_pred)
  

print("guidelines_risk_factors")
for val in range(6):
  y_train_pred = x_train["guidelines_risk_factors"].apply(lambda x: x >= val)
  y_test_pred = x_test["guidelines_risk_factors"].apply(lambda x: x >= val)
  print("Test trashold = {}".format(val))
  print_model_results(y_train, y_test, y_train_pred, y_test_pred)
  
y_col = list(df)[-3]
x = df.drop([y_col, list(df)[-1], list(df)[-2], 'APFEL_SCORE', 'KOIV_SCORE'], axis=1)
y = df[y_col]
print("Tested y-col: {}".format(y_col))
print(y.value_counts())

undersample = NearMiss(version=1, n_neighbors=3)
# transform the dataset
x, y = undersample.fit_resample(x, y)
print(y.value_counts())

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=RS)

parameters = {'max_depth': [3, 5, 7, 9, 11, 13],
              'criterion':["gini", "entropy"],
              'min_samples_leaf': [0, 25],
              'ccp_alpha': [0, 0.01, 0.05]}
clf = GridSearchCV(RandomForestClassifier(), parameters)
clf.fit(x_train, y_train)
y_train_pred = clf.predict(x_train)
y_test_pred = clf.predict(x_test)

print("First attempt (RF + GS) - task 1")
print_model_results(y_train, y_test, y_train_pred, y_test_pred)

from sklearn import metrics
y_pred_proba = clf.predict_proba(x_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="AUC={:.3f}".format(auc))
plt.plot([0,1],[0,1],"--",color="gray")
plt.legend(loc=4)
plt.xticks([0.1*i for i in range(11)])
plt.yticks([0.1*i for i in range(11)])
plt.ylabel("True Positive Rate", fontsize=16)
plt.xlabel("False Positive Rate", fontsize=16)
plt.xlim((-0.01, 1.01))
plt.ylim((-0.01, 1.01))
ax = plt.gca()
ax.spines[['right', 'top']].set_visible(False)
plt.grid(color="black",
         alpha=0.25)
plt.savefig("roc_pacu.pdf", dpi=400)
plt.show()
plt.close()

importances = clf.best_estimator_.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.best_estimator_.estimators_], axis=0)
forest_importances = pd.Series(importances, index=list(x_train))

forest_importances = forest_importances.sort_values(ascending=False)
a = sum(forest_importances)
forest_importances = forest_importances.loc[lambda x: x > 0.01]
b = sum(forest_importances)
print(sum(forest_importances))
forest_importances = forest_importances.apply(lambda x: x/sum(forest_importances))


fig, ax = plt.subplots()
forest_importances.plot.bar(ax=ax)
ax.set_ylabel("Importance", fontsize=16)
ax.set_xlabel("Feature", fontsize=16)
plt.yticks([0.025 * index for index in range(11)])
plt.ylim((0, 0.25))
plt.grid(alpha=0.25,
         color="black",
         axis="y")
ax.spines[['right', 'top']].set_visible(False)
fig.tight_layout()
plt.savefig("FI_ponv_in_pacu_RF.pdf", dpi=400)
plt.show()
plt.close()


explainer = shap.Explainer(clf.best_estimator_.predict, x_test)
shap_values = explainer(x_test)
shap.plots.beeswarm(shap_values)