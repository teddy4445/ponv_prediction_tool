# library imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, auc
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as mp
from tpot import TPOTClassifier
import shap
     
# make sure the seed value is set to make repeated results 
RS = 73

# define a function to print the models results - just to be consistent between the different tests and models 
def print_model_results(y_train, y_test, y_train_pred, y_test_pred):
  print("ACC | train set = {:.3f}".format(accuracy_score(y_train, y_train_pred)))
  print("ACC | test set = {:.3f}".format(accuracy_score(y_test, y_test_pred)))

  print("F1 | train set = {:.3f}".format(f1_score(y_train, y_train_pred)))
  print("F1 | test set = {:.3f}".format(f1_score(y_test, y_test_pred)))

  print("Recall | train set = {:.3f}".format(recall_score(y_train, y_train_pred)))
  print("Recall | test set = {:.3f}".format(recall_score(y_test, y_test_pred)))

  print("Precision | train set = {:.3f}".format(precision_score(y_train, y_train_pred)))
  print("Precision | test set = {:.3f}".format(precision_score(y_test, y_test_pred)))

# define a function to plot the Pearson matrix
def plot_pearson(df):
  # plot the pearson corrolation matrix as presented in the paper
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


def baseline_models(df, y_index):
  
  y_col = list(df)[y_index]
  y = df[y_col]
  x = df.drop([y_col], axis=1)
  oversample = SMOTE(version=1, n_neighbors=3)
  # transform the dataset
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=RS)
  x_train, y_train = oversample.fit_resample(x_train, y_train)

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


# a single entry point for the code which runs the entire logic at once
def main():
  # load the data
  df = pd.read_csv("data.csv")

  # For the first table in the paper, print all stats we need
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

  # Same, but only for paitents with deplayed PONV
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

  # plot pearson for the paper
  plot_pearson(df)

  # calc the baseline metrices for the second table in the paper
  baseline_models(df, -1) # delayed PONV
  baseline_models(df, -2) # PONV

  # calc the models for both metrices 
  y_col_indexs = [-1, -2]:
  for y_col_index in y_col_indexs:
    print("Working on y_col_indexs={}".format(y_col_indexs))
    y_col = list(df)[y_col_index]
    x = df.drop([list(df)[-1], list(df)[-2], "guidelines_risk_factors", 'apfel_simplified_score', 'koivuranta_score'], axis=1)
    y = df[y_col]
    print("Tested y-col: {}".format(y_col))
    print(y.value_counts())

    oversample  = SMOTE(version=1, n_neighbors=3)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=RS)
    x_train, y_train = oversample .fit_resample(x_train, y_train)

    parameters = {
        'max_depth': [3, 5, 7, 9, 11, 13],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200, 300],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'gamma': [0, 0.1, 0.2],
        'reg_alpha': [0, 0.01, 0.1],
        'reg_lambda': [1, 1.5, 2]
    }
    clf = GridSearchCV(XGBClassifier(use_label_encoder=False, eval_metric="logloss"), parameters, cv=5)
    clf.fit(x_train, y_train)
    y_train_pred = clf.predict(x_train)
    y_test_pred = clf.predict(x_test)

    # print the performance of the model 
    print("XGboost model")
    print_model_results(y_train, y_test, y_train_pred, y_test_pred)

    logistic_params = {
    'penalty': ['l1', 'l2', 'elasticnet', None],
    'C': [0.1, 1, 10, 100],
    'solver': ['liblinear', 'saga', 'lbfgs'],  # try all common solvers
    'max_iter': [100, 200, 300]
    }
    clf = GridSearchCV(LogisticRegression(), logistic_params, cv=5)
    clf.fit(x_train, y_train)
    y_train_pred = clf.predict(x_train)
    y_test_pred = clf.predict(x_test)

    # print the performance of the model 
    print("Logsitic regression model")
    print_model_results(y_train, y_test, y_train_pred, y_test_pred)

    # print the ROC graph of the XGboost model    
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

    # TPOT Model
    tpot = TPOTClassifier(
        generations=25, # change to higher value if you have more time 
        population_size=40, # change to higher value if you have more time 
        verbosity=2,
        scoring='roc_auc',
        random_state=RS
    )
    tpot.fit(x_train, y_train)
    y_train_pred_tpot = tpot.predict(x_train)
    y_test_pred_tpot = tpot.predict(x_test)

    print("TPOT model")
    print_model_results(y_train, y_test, y_train_pred_tpot, y_test_pred_tpot)

    # Export the best pipeline to a Python script
    export_file_name = f"tpot_best_pipeline_y_col_{y_col_index}.py"
    tpot.export(export_file_name)
    print(f"Best pipeline exported to {export_file_name}")

    # print feature importance graph of the XGboost model, sorted from high to low and cut in 0.01
    importances = clf.best_estimator_.feature_importances_
    forest_importances = pd.Series(importances, index=list(x_train.columns))
    forest_importances = forest_importances.sort_values(ascending=False)
    forest_importances = forest_importances.loc[lambda x: x > 0.01]
    forest_importances = forest_importances.apply(lambda x: x / sum(forest_importances))
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

    # print the SHAP plot of the XGboost  model
    explainer = shap.Explainer(clf.best_estimator_.predict, x_test)
    shap_values = explainer(x_test)
    shap.plots.beeswarm(shap_values)

main()
