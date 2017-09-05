#!/usr/bin/python

import sys
import pickle
import numpy as np
import pandas as pd
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',
                 'salary',
                 'deferral_payments',
                 'total_payments',
                 'loan_advances',
                 'bonus',
                 'restricted_stock_deferred',
                 'deferred_income',
                 'total_stock_value',
                 'expenses',
                 'exercised_stock_options',
                 'other',
                 'long_term_incentive',
                 'restricted_stock',
                 'director_fees',
                 'to_messages',
                 'from_poi_to_this_person',
                 'from_messages',
                 'from_this_person_to_poi',
                 'shared_receipt_with_poi']
email_features = ['to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop('TOTAL')
data_dict.pop('THE TRAVEL AGENCY IN THE PARK')

data_df = pd.DataFrame.from_dict(data_dict, orient='index')

data_df.replace('NaN', np.NaN, inplace=True)
means = data_df.mean()
medians = data_df.median()

# for feature in features_list:
#     if feature not in email_features:
#         if feature != 'poi':
#             features_list.remove(feature)

for person in data_dict:
    for feature in features_list:
        if data_dict[person][feature] == 'NaN':
            if feature in email_features:
                data_dict[person][feature] = means[feature]
            else:
                data_dict[person][feature] = medians[feature]

print len(data_dict)
### Task 3: Create new feature(s)
features_list.append('fraction_from_poi')
features_list.append('fraction_to_poi')
features_list.append('fraction_restricted_stock')
for person in data_dict:
    data_dict[person]['fraction_from_poi'] = float(data_dict[person]['from_poi_to_this_person']) / data_dict[person]['from_messages']
    data_dict[person]['fraction_to_poi'] = float(data_dict[person]['from_this_person_to_poi']) / data_dict[person]['to_messages']
    data_dict[person]['fraction_restricted_stock'] = float(data_dict[person]['restricted_stock']) / data_dict[person]['total_stock_value']


### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# from sklearn.feature_selection import SelectPercentile, chi2
# selector = SelectPercentile(score_func=chi2, k=10)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier
# clf = DecisionTreeClassifier(criterion="gini",
#                              splitter="best",
#                              max_depth=2,
#                              min_samples_split=2,
#                              class_weight='balanced')

dt = DecisionTreeClassifier(random_state=42)
metrics = 'precision'
params = {
    'criterion'         : ['gini', 'entropy'],
    'splitter'          : ['best', 'random'],
    'max_depth'         : [2, 10, 20, 30, 40],
    'min_samples_split' : [2, 4, 6, 8],
    'class_weight'      : [None, 'balanced']
}
# clf = AdaBoostClassifier(class_weights='balanced')

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# clf.fit(features_train, labels_train)

grid = GridSearchCV(dt, params, scoring='f1')
grid.fit(features_train, labels_train)

clf = grid.best_estimator_
#print clf.feature_importances_

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)