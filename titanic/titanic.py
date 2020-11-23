'''
0. CORRELATE - See what we got.
    pd.groupby(...)
    pd.crosstab(...)
    pd.info()
    pd.head()
    pd.sample(10)
1. COMPLETE - Look at data.
    * Have some combine = [train, test] array to easily apply operations to both datasets.
    ? Which features are categorical, numerical?
        pd.describe(include=(...))

    pd.isnull().sum()
    a. Complete or delete missing values:
        - pd.fillna(...).median() OR mode()
        - pd.dropna()
    c. Drop ID column - PassengerId.
2. CREATE - Feature Engineering.
    a. New features:
        FamilySize = SibSp + Parch
        IsAlone = FamilySize > 1
        Name split
    b. Cleanup rare values, titles.
    c. Replace continuous variables with bins ( pd.cut(...).astype(int) ):
        FareBin, AgeBin
3. CONVERT - Objects to category
    LabelEncoder().fit_transform(...)
    pd.get_dummies(...)




5. TRAIN    # TODO
6. EVALUATE # TODO
7. CHART    # TODO

'''
# %% Imports

from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import pandas as pd
import numpy as np
import random as rnd
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

%matplotlib inline

# %% Load data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

combined = [train_data, test_data]

# %% Correlate

print(train_data.describe())
print(train_data.describe(include=object))

number_of_features = len(train_data.columns.values)

print("We have {} features, where each has null's inside:".format(number_of_features))
print(train_data.isnull().sum())

# TODO drop Cabin, Id, complete OR drop null Age rows, and fill Embarked with median()

print('='*80)
column_drop = ['PassengerId', 'Cabin', 'Ticket']
column_bin = ['Fare', 'Age']
column_fill = ['Age', 'Embarked', 'Fare']
column_engineer = ['Name', 'SibSp', 'Parch']
for feature in np.setdiff1d(train_data.columns.values, column_drop+column_bin):
    if feature != 'Survived':
        print('{:-^40}'.format(feature))
        print(train_data[[feature, 'Survived']].groupby([feature]).mean())

# print(train_data[['Pclass', 'SibSp', 'Parch', 'Survived']].groupby(['Pclass']).mean())
# print(train_data[['Pclass', 'Sex', 'Survived']].groupby(['Sex']).mean())
# TODO what's surv rate of female at all Pclass?

# %% Complete
for data in combined:
    print(data.isnull().sum())
    data.drop(column_drop, axis=1, inplace=True)
    data['Age'].fillna(data['Age'].median(), inplace=True)
    data['Fare'].fillna(data['Fare'].median(), inplace=True)
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
    print(data.isnull().sum())

# %% CREATE

for data in combined:
    data['FamilySize'] = data['SibSp']+data['Parch']+1
    data['IsAlone'] = (data['FamilySize'] > 1).astype(int)
    data['Title'] = data['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
    # print(data.head())
    # print(data['Title'].value_counts())
    # print(data[['Title', 'Survived']].groupby(['Title']).mean())

    # bins
    data['FareBin'] = pd.qcut(data['Fare'], 5)
    data['AgeBin'] = pd.cut(data['Age'], 5)
    # print(data.groupby('Title').filter(lambda t: len(t) > 10))
    rare_titles = data['Title'].value_counts() < 10
    data['Title'] = data['Title'].apply(lambda t: 'Misc' if rare_titles[t] else t)
    # print(title_names)

    # print(data.info())

# %% CONVERT
label = LabelEncoder()
# ft = train_data
# print(train_data['AgeBin'].value_counts())
# train_data.describe(include=object)
for data in combined:
    data['Sex_Code'] = label.fit_transform(data['Sex'])
    data['Embarked_Code'] = label.fit_transform(data['Embarked'])
    data['Title_Code'] = label.fit_transform(data['Title'])
    data['AgeBin_Code'] = label.fit_transform(data['AgeBin'])
    data['FareBin_Code'] = label.fit_transform(data['FareBin'])
# %% TRAIN
train_target = train_data['Survived']
train_data = train_data.drop('Survived', axis=1)
col_drop = ['Name', 'Sex', 'Embarked', 'Title', 'AgeBin', 'FareBin',
            'Age', 'SibSp', 'Parch', 'Fare']
train_data = train_data.drop(col_drop, axis=1)
test_data = test_data.drop(col_drop, axis=1)
X = train_data
y = train_target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# %%

models = [LogisticRegression(),
          RandomForestClassifier(n_estimators=100),
          KNeighborsClassifier(n_neighbors=3),
          SVC(),
          LinearSVC(),
          DecisionTreeClassifier()]
scores = pd.DataFrame(columns=['model', 'score'])
for model in models:
    model_name = type(model).__name__
    model_score = cross_val_score(model, X_train, y_train).mean()
    scores = scores.append({'model': model_name, 'score': model_score}, ignore_index=True)

scores.sort_values(by='score', ascending=False)


# %%

pipe = Pipeline([
    ('preprocessing', StandardScaler()),
    ('classifier', SVC())])
param_grid = [
    # {'classifier': [SVC()],
    #     'preprocessing':[MinMaxScaler(), StandardScaler(), None],
    #     'classifier__gamma':[0.001, 0.01, 0.1, 1, 10, 100],
    #     'classifier__C': [0.001, 0.01, 1, 10, 100]},
    # {'classifier': [RandomForestClassifier()],
    #     'preprocessing': [None],
    #     'classifier__n_estimators':[1, 5, 50, 100, 300],
    #     'classifier__max_features': [*range(1, 10, 2), "auto", None]
    #  },
    # {'classifier': [KNeighborsClassifier()],
    #     'preprocessing': [None],
    #     'classifier__n_neighbors':[1, 3, 5, 10]},
    # {'classifier': [LinearSVC()],
    #     'classifier__C': [0.001, 0.01, 1, 10, 100]},
    # {'classifier': [DecisionTreeClassifier()],
    #     'preprocessing': [None],
    #     'classifier__max_depth':[None, 1, 5, 10]},
    {
        'classifier': [XGBClassifier()],
        'preprocessing':[None],
        'classifier__eta':[0.05, 0.1, 0.15, 0.2, 0.3],
        'classifier__max_depth':[*range(3, 10)],
        'classifier__gamma':[0.0, 0.1, 0.2, 0.3, 0.4]
    }
    # {'classifier': [GradientBoostingClassifier()],
    #     'preprocessing': [None],
    #     'classifier__max_depth':[None, 1, 5],
    #     'classifier__n_estimators':[1, 5, 50, 100],
    #     'classifier__max_features': [1, 3, 5, "auto", None],
    #     'classifier__learning_rate':[0.01, 0.05, 0.1]}
]
grid = GridSearchCV(pipe, param_grid)
grid.fit(X_train, y_train)
print(grid.best_score_)
print(grid.best_estimator_)

# %% Best Model
# scaler = StandardScaler()
pipe = make_pipeline(XGBClassifier(base_score=0.5, booster='gbtree',
                                   colsample_bylevel=1, colsample_bynode=1,
                                   colsample_bytree=1, eta=0.3, gamma=0.3,
                                   gpu_id=-1, importance_type='gain',
                                   interaction_constraints='',
                                   learning_rate=0.300000012, max_delta_step=0,
                                   max_depth=3, min_child_weight=1,
                                   monotone_constraints='()', n_estimators=100,
                                   n_jobs=0, num_parallel_tree=1, random_state=0,
                                   reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                                   subsample=1, tree_method='exact',
                                   validate_parameters=1, verbosity=None))
pipe.fit(X_train, y_train)
print("Pipe score: {:.3f}".format(pipe.score(X_test, y_test)))
y_pred = pipe.predict(test_data)

test_data_again = pd.read_csv('test.csv')
result = pd.DataFrame(columns=['PassengerId', 'Survived'])
result['Survived'] = y_pred
result['PassengerId'] = test_data_again['PassengerId']
result.to_csv('titanic_submission_4_xgb.csv', index=False)

# %%
