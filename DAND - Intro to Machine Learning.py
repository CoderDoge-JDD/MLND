#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# Package preparation
import pandas as pd
from feature_format import featureFormat, targetFeatureSplit

### Load the dictionary containing the dataset
with open("../final_project/final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

# Parse the data
# Extract features from keys
all_features = data_dict['TOTAL'].keys()
# Remove poi at first
all_features.pop(all_features.index("poi"))
# Remove incomputable email address
all_features.pop(all_features.index("email_address"))
# Combine all the features
features_list = ["poi",] + all_features

# Overview the data
my_dataset = data_dict
data = featureFormat(my_dataset, features_list, remove_NaN = False, remove_all_zeroes = False, sort_keys = False)
df = pd.DataFrame(data, index = data_dict.keys(), columns = features_list)
# df.describe()
# df.poi[df.poi == 1]
# sorted(df.index)

# Remove TOTAL and THE TRAVEL AGENCY IN THE PARK
data_dict.pop('TOTAL')
data_dict.pop('THE TRAVEL AGENCY IN THE PARK')
df.drop(['TOTAL', 'THE TRAVEL AGENCY IN THE PARK'], inplace = True)
# df.groupby('poi').median()

# Replace NaN with zero
df_new = df.fillna(0)

# Te first feature: ambition
# bonus_loan_incentive = df_new[['bonus','long_term_incentive','loan_advances']]
# from sklearn.decomposition import PCA
# pca = PCA(n_components = 1)
# df_new['ambition'] = pca.fit_transform(bonus_loan_incentive.values, df_new.poi.values)
# pca.explained_variance_ratio_

# The second feature: to_poi_fraction ( = from thi sperson to poi / from messages)
to_poi = df_new[['from_this_person_to_poi', 'from_messages']]
df_new['to_poi_fraction'] = 0
for index, row in zip(to_poi.index, to_poi.values):
    if row[0] != 0 and row[1] != 0:
        df_new.ix[index, 'to_poi_fraction'] = row[0] / row[1] * 100

# The third feature: from_poi_fraction ( = from poi to this person / to messages)
from_poi = df_new[['from_poi_to_this_person', 'to_messages']]
df_new['from_poi_fraction'] = 0
for index, row in zip(from_poi.index, from_poi.values):
    if row[0] != 0 and row[1] != 0:
        df_new.ix[index, 'from_poi_fraction'] = row[0] / row[1] * 100

# Prepare packages
import numpy as np
from sklearn.preprocessing import normalize
from math import log10
from sklearn.feature_selection import SelectKBest

# Min max scaler function
def minmaxscaler(f_list):
        temp = df_new[f_list].copy()
        for column in f_list:
            ma = float(np.amax(temp[column].values))
            mi = float(np.amin(temp[column].values))
            temp[column] = temp[column].apply(lambda e: (e - mi) / (ma - mi))
        temp = pd.DataFrame(temp, columns = f_list, index = df_new.index)
        return temp

# k best function
# Input a feature dataframe, return a dataframe with selected features
def kbest(df, k):
    # k is the number of selected features
    slb = SelectKBest(k = k)
        
    # Get values of the best k features
    best_features = slb.fit_transform(df.values, df_new.poi.values)
        
    # Get names of the best k features
    scores = zip(df.columns, slb.scores_)
    top_scored = sorted(scores, key = lambda x: x[1], reverse = True)[:k]
    best_names = []
    for feature in df.columns:
        if feature in [x for (x, y) in top_scored]:
                best_names.append(feature)
    best = pd.DataFrame(best_features, columns = best_names, index = df_new.index)
    return best, top_scored

def preprocess(k, escape = []):
    # log transform feature except those in escape list
    df = df_new[features_list].copy()
    f_list = [x for x in features_list if x not in escape]
    if f_list:
        transformed = minmaxscaler(f_list)
        escaped = df[escape].copy()
        df = pd.DataFrame(np.concatenate((transformed, escaped), axis = 1), index = df_new.index, columns = list(transformed.columns) + list(escaped.columns))
    
    return kbest(df, k = k)

# Preapre packages
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, precision_score, f1_score, recall_score
from collections import defaultdict
from sklearn.grid_search import GridSearchCV

# Separate eco_features with email_features
eco_features_list = [ u'salary', u'deferral_payments',\
       u'total_payments', u'exercised_stock_options', u'bonus',\
       u'restricted_stock', u'restricted_stock_deferred', u'total_stock_value', u'expenses',\
       u'loan_advances',  u'other',u'director_fees', u'deferred_income',u'long_term_incentive']
email_features_list=[u'to_messages',u'shared_receipt_with_poi',u'from_messages',u'from_this_person_to_poi',\
                     u'from_poi_to_this_person',u'to_poi_fraction', u'from_poi_fraction']
features_list = eco_features_list + email_features_list
labels = df_new.poi.values
log_diary = {}



# clf score function
def clf_score(clf, n_iter = 100, log = "", f = features_list, l = labels):
        cv = StratifiedShuffleSplit(labels, n_iter = n_iter, random_state = 42)

        results = {'precision':[], 'recall':[], 'fscore':[], 'accuracy':[]}
        for traincv, testcv in cv:
                clf.fit(f[traincv], l[traincv])
                pred = clf.predict(f[testcv])
                results['precision'].append(precision_score(l[testcv], pred))
                results['recall'].append(recall_score(l[testcv], pred))
                results['fscore'].append(f1_score(l[testcv], pred))
                results['accuracy'].append(accuracy_score(l[testcv], pred))
                
        for k, v in results.iteritems(): 
            average = sum(v) / float(len(v))
            results[k] = average
    
        if log:
            log_diary[log] = results
        return results

from sklearn.naive_bayes import GaussianNB

# clf_best_params function
def clf_best_params(clf, parameters, scoring = 'recall', f = features_list, l = labels):
        gd = GridSearchCV(clf, parameters, scoring = scoring)
        gd.fit(f, l)
        return gd

# Define a function for iterative computation of different models
def models_comparison(k, escape = [], \
                      n_iter = 100, scoring = 'recall'):
    #log transform and normalize all features except those in escape
    best, top_scored = preprocess(k = k, escape = escape)
    print '========Top {} features and their scores:'.format(k)
    for feature, score in top_scored:
        print '%-30s %s'%(feature, score)
    print
    features = best.values
    features = np.array(features)
    labels = df_new.poi.values

    from sklearn.naive_bayes import GaussianNB
    nb = GaussianNB()
    clf_score(nb, log = 'GaussianNB', f = features, l = labels)

    from sklearn.svm import SVC
    sv = SVC(kernel = 'rbf', C = 1, random_state = 42)
    parameters = {'C':[1, 10, 1000, 100000]}
    sv = clf_best_params(sv, parameters, f = features, l = labels)
    clf_score(sv, log = 'rbf_SVC', f = features, l = labels)

    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(random_state = 42)
    clf_score(clf, log = 'LogitRegression', f = features, l = labels)

    from sklearn import tree
    clf = tree.DecisionTreeClassifier(random_state = 42)
    clf = clf_best_params(clf, parameters = {'min_samples_split':[5, 10, 20, 40]}, f = features, l = labels)
    clf_score(clf, log = 'DecisionTree', f = features, l = labels)

    from sklearn.ensemble import RandomForestClassifier
    rfc = RandomForestClassifier(random_state = 42)
    parameters = {'n_estimators':[1, 2, 5, 10], 'min_samples_split':[5, 10, 20]}
    rfc = clf_best_params(rfc, parameters, f = features, l = labels)
    clf_score(rfc, log = 'RandomForest', f = features, l = labels)
    
    from sklearn.ensemble import AdaBoostClassifier
    adc = AdaBoostClassifier(random_state = 42)
    parameters = {'n_estimators':[1, 2, 5, 10]}
    adc = clf_best_params(adc, parameters, f = features, l = labels)
    clf_score(adc, log = 'AdaBoost', f = features, l = labels)

    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier()
    parameters = {'n_neighbors':[1, 2, 5, 10]}
    knn = clf_best_params(knn, parameters, f = features, l = labels)
    clf_score(knn, log = "KNeighbors", f = features, l = labels)

    # Store the final outcome of classifier
    with open('my_classifier.pkl', "w") as clf_outfile:
        pickle.dump(nb, clf_outfile)

    log_sheet = pd.DataFrame.from_dict(log_diary, orient = 'index')
    return log_sheet

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

#With all features escape from all transfromation,
#more features are selected, the better performance most of models give.
#When all features are selected, the performance get boost most.
#GaussianNB gives the highest Recall.

print "\nWhen  features selected = 21, escape=features_list"
print models_comparison(21, escape = features_list)
print "\nWhen features selected = 15, escape=features_list"
print models_comparison(15, escape = features_list)
print "\nWhen features selected = 8, escape=features_list"
print models_comparison(8, escape = features_list)

#With all features log10 transformed and normalized,
#Rebundant features pose negative effect on the performance. 
#The peak appear when number of selected features equals to 4.

print "\nWhen  features selected = 10, escape=None"
print models_comparison(10)
print "\nWhen  features selected = 5, escape=None"
print models_comparison(5)
print "\nWhen  features selected = 4, escape=None (GaussianNB here gives the most satisfying performance)"
print models_comparison(4)
print "\nWhen  features selected = 3, escape=None"
print models_comparison(3)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

features_list = ['exercised_stock_options', 'total_stock_value', 'bonus', 'salary', 'to_poi_fraction', 'deferred_income', 'long_term_incentive', 'restricted_stock']

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

selected_transformed_df, top_scored = preprocess(4)

#dump the list of features used in final_models
top_features = []
for feature, score in top_scored:
    top_features.append(feature)
    
with open('my_feature_list.pkl', "w") as featurelist_outfile:
     pickle.dump(['poi'] + top_features, featurelist_outfile)

#dump the processed dataset used in final_models
selected_transformed_dataset = defaultdict(lambda:{})

for idx in selected_transformed_df.index:
    selected_transformed_dataset[idx]['poi'] = df_new.ix[idx,'poi']    
    for cln in selected_transformed_df.columns:
        selected_transformed_dataset[idx][cln] = selected_transformed_df.ix[idx, cln]

with open('my_dataset.pkl','w') as f:
    pickle.dump(dict(selected_transformed_dataset), f)

# dump_classifier_and_data(clf_special, my_dataset, features_list)