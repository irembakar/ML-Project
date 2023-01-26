import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from matplotlib import pyplot as plt

df = pd.read_excel('new_dataset.xlsx')
# print(df.head())
df.count()
df.shape
print(df["M6Class"].value_counts())
print(df["M5Class"].value_counts())

# df.drop_duplicates()
df.columns

# Clearing column names

df.columns = df.columns.str.strip()

# Data type check

df.dtypes

# Display missing data

df.isnull().sum()

# Visualization of data

import seaborn as sns
sns.set_theme()
sns.set(rc={"figure.dpi":300,"figure.figsize":(12,9)})
sns.heatmap(df.isnull(),cbar=False)

# Clearing missing data. Since the median is more durable than the outlier data, we use the median and choose the ones with the most missing numbers.

M1DeliveryCost_median = df["M1DeliveryCost"].median()
M4NumberOfComments_median = df["M4NumberOfComments"].median()
M4LogNumberOfComments_median= df["M4LogNumberOfComments"].median()

df["M1DeliveryCost"].fillna(M1DeliveryCost_median, inplace=True)
df["M4NumberOfComments"].fillna(M4NumberOfComments_median, inplace=True)
df["M4LogNumberOfComments"].fillna(M4LogNumberOfComments_median, inplace=True)

# We ensure that the missing data numbers are removed from the system.

df.dropna(inplace=True)
df.isnull().sum().sum()#Toplam eksik verinin görülmesi

# Display all data

df.info()

print(df["M6Class"].value_counts())
print(df["M5Class"].value_counts())

# These summary statistics show no outliers in this column

df["M1DeliveryCost"].describe()

# M1mean = df["M1DeliveryCost"].mean()
# M3mean = df["M3LogMinChargeForOrdering"].mean()
# df = df.dropna(subset=['M5Class'])
# df = df.dropna(subset=['M6Class'])

# round(M1mean)
# round(M3mean)

# df['M1DeliveryCost'] = df['M1DeliveryCost'].fillna(M1mean)
# df['M3LogMinChargeForOrdering'] = df['M3LogMinChargeForOrdering'].fillna(M3mean)


# # print(df.count())
print(df.head())
# df.count()

df["M6Class"].value_counts().plot(kind="bar", color="red")

df["M5Class"].value_counts().plot(kind="bar", color="blue")

sns.boxplot(x="M6Class", y="M1DeliveryCost", data = df)

import matplotlib.pyplot as plt
sns.countplot(y="M1DeliveryCost",data=df)
plt.title("M1DeliveryCost class with their counts")

sns.heatmap(df.corr(), annot = True, linewidths=.5, fmt=".2f")

# X = df.drop(columns=['M6Class'])
# y = df['M6Class']
# # Split the data into training and test sets, stratifying the target column
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# corr = df.corr()
# corr = corr['M6Class'].sort_values(ascending=False)
# print(X_train)

# FEATURE SELECTİON Fishers_scores M6Class

# Separate the data into two classes
class_1 = df[df['M6Class'] == 1]
class_2 = df[df['M6Class'] == 0]

# Calculate the mean and standard deviation of each feature for each class
mean_class_1 = class_1.mean(numeric_only=True)
std_class_1 = class_1.std(numeric_only=True)
mean_class_2 = class_2.mean(numeric_only=True)
std_class_2 = class_2.std(numeric_only=True)

# Calculate Fisher's Score for each feature
fishers_scores = ((mean_class_1 - mean_class_2)**2) / (std_class_1**2 + std_class_2**2)

# Create a DataFrame to store the Fisher's Scores
fishers_scores_df = pd.DataFrame({'feature': fishers_scores.index, 'fishers_score': fishers_scores.values})

# Rank the features based on their Fisher's Scores
fishers_scores_df['rank'] = fishers_scores_df['fishers_score'].rank(ascending=False)

print(fishers_scores_df['rank'])

classM5_1 = df[df['M5Class'] == 1]
classM5_2 = df[df['M5Class'] == 0]

# Calculate the mean and standard deviation of each feature for each class
mean_classM5_1 = classM5_1.mean(numeric_only=True)
std_classM5_1 = classM5_1.std(numeric_only=True)
mean_classM5_2 = classM5_2.mean(numeric_only=True)
std_classM5_2 = classM5_2.std(numeric_only=True)

# Calculate Fisher's Score for each feature
fishers_scoresM5 = ((mean_classM5_1 - mean_classM5_2)**2) / (std_classM5_1**2 + std_classM5_2**2)

# Create a DataFrame to store the Fisher's Scores
fishers_scoresM5_df = pd.DataFrame({'feature': fishers_scoresM5.index, 'fishers_score': fishers_scoresM5.values})

# Rank the features based on their Fisher's Scores
fishers_scoresM5_df['rank'] = fishers_scoresM5_df['fishers_score'].rank(ascending=False)

print(fishers_scoresM5_df['rank'])

# M6Class Fishler Scores

fishers_scores_df.plot.bar(x='feature', y='fishers_score', rot=1)

# **LightGBM** M6CLASS

import pandas as pd
from sklearn.metrics import classification_report
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import itertools
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFECV

# Load the data into a pandas dataframe


# Select the top 10 features based on the Fisher scores
top_k = np.argsort(fishers_scores_df['rank'])[::-1][:10]
print(top_k)
selected_features = []
for feature in top_k:
  selected_features.append(df.columns[feature])

#M6 CLASS -------------------------------------------- M6 CLASS ------------------------------------------------------------ M6 CLASS------------------------------------ M6 CLASS--------------------------------

# Separate the features and target variable
X = df[selected_features]
y = df['M6Class']
# Create a lightGBM model
model = lgb.LGBMClassifier()

# Use cross-validation to evaluate the model
y_pred = cross_val_predict(model, X, y, cv=7)

# Print the evaluation scores in a report format
reportLightGBM = classification_report(y, y_pred)
print("reportLightGBM")
print(reportLightGBM)

#Random Forest Classifier

model = RandomForestClassifier()

# Use cross-validation to evaluate the model
y_pred = cross_val_predict(model, X, y, cv=7)

# Print the evaluation scores in a report format
reportRanomForest = classification_report(y, y_pred)
print("reportRanomForest")
print(reportRanomForest)

#Decision Tree Classifier
model = DecisionTreeClassifier()

# Use cross-validation to evaluate the model
y_pred = cross_val_predict(model, X, y, cv=7)

# Print the evaluation scores in a report format
reportDecisionTreeClassifier = classification_report(y, y_pred)
print("DecisionTreeClassifier")
print(reportDecisionTreeClassifier)

#ADABOOST Tree Classifier
model = AdaBoostClassifier()

# Use cross-validation to evaluate the model
y_pred = cross_val_predict(model, X, y, cv=7)

# Print the evaluation scores in a report format
reportAdaBoostClassifier = classification_report(y, y_pred)
print("AdaBoostClassifier")
print(reportAdaBoostClassifier)

#M6 CLASS -------------------------------------------- M6 CLASS ------------------------------------------------------------ M6 CLASS------------------------------------ M6 CLASS--------------------------------

#M6 CLASS Exhaustive Feature Selection-------------------------------------------- M6 CLASS Exhaustive Feature Selection ------------------------------------------------------------ M6 CLASS Exhaustive Feature Selection------------------------------------ M6 CLASS--------------------------------

#M6 CLASS Exhaustive Feature Selection-------------------------------------------- M6 CLASS Exhaustive Feature Selection------------------------------------------------------------ M6 CLASS Exhaustive Feature Selection------------------------------------ M6 CLASS--------------------------------

#M5 CLASS -------------------------------------------- M5 CLASS ------------------------------------------------------------ M5 CLASS------------------------------------ M5 CLASS--------------------------------
# print("M5 Class Reports---------------------------------------------------------------------------------------------------------------------------------------------")
top_m = np.argsort(fishers_scoresM5_df['rank'])[::-1][:10]
selected_featuresM = []
for feature in top_m:
  selected_featuresM.append(df.columns[feature])

X = df[selected_featuresM]
y = df['M5Class']

# Create a lightGBM model
model = lgb.LGBMClassifier()

# Use cross-validation to evaluate the model
y_pred = cross_val_predict(model, X, y, cv=7)

# Print the evaluation scores in a report format
reportLightGBM = classification_report(y, y_pred)
print("reportLightGBM")
print(reportLightGBM)

#Random Forest Classifier

model = RandomForestClassifier()

# Cross-validation to evaluate the model
y_pred = cross_val_predict(model, X, y, cv=7)

# Print the evaluation scores in a report format
reportRanomForest = classification_report(y, y_pred)
print("reportRanomForest")
print(reportRanomForest)

#Decision Tree Classifier
model = DecisionTreeClassifier()

# Use cross-validation to evaluate the model
y_pred = cross_val_predict(model, X, y, cv=7)

# Print the evaluation scores in a report format
reportDecisionTreeClassifier = classification_report(y, y_pred)
print("DecisionTreeClassifier")
print(reportDecisionTreeClassifier)

#ADABOOST Tree Classifier
model = AdaBoostClassifier()

# Use cross-validation to evaluate the model
y_pred = cross_val_predict(model, X, y, cv=7)

# Print the evaluation scores in a report format
reportAdaBoostClassifier = classification_report(y, y_pred)
print("AdaBoostClassifier")
print(reportAdaBoostClassifier)

#M5 CLASS -------------------------------------------- M5 CLASS ------------------------------------------------------------ M5 CLASS------------------------------------ M5 CLASS--------------------------------

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
data = df
X = data.iloc[:,0:20]  #independent columns
y = data.iloc[:,-1]    #target column i.e price range
#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(10,'Score'))

#Random Forest M6CLASS ADABOOST M6CLASS Decision Tree M6CLASS **LightGBM** M5CLASS FEATURE SELECTİON M5CLASS