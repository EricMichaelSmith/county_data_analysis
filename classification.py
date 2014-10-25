# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 20:46:51 2014

@author: Eric

Performs classifications for the county_data_analysis project.

Suffixes at the end of variable names:
a: numpy array
b: boolean
d: dictionary
df: pandas DataFrame
l: list
s: string
t: tuple
Underscores indicate chaining: for instance, "foo_t_t" is a tuple of tuples
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.cross_validation import cross_val_score
from sklearn.lda import LDA
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.qda import QDA
from sklearn.svm import SVC

import config
reload(config)
import plotting
reload(plotting)
import regression
reload(regression)
import selecting
reload(selecting)



def main(con, cur):
    """ Run all classifcations used for the county_data_analysis project. """
    
    # Create the feature array and output classification array to be used in classification analyses
    feature_a, stand_feature_a, feature_s_l, output_class_a = create_arrays(con, cur)
         
    # Run forward stepwise selection classification models
    forward_stepwise_selection(stand_feature_a, feature_s_l, output_class_a)



def create_arrays(con, cur):
    """ Creates the feature array and output classification array to be used in classification analyses. """
    
    # Load feature data
    feature_s_l = config.feature_s_l
    feature_d = selecting.select_fields(con, cur, feature_s_l, output_type='dictionary')
    
    # Load output variable data
    output_s = regression.global_output_s
    output_d = selecting.select_fields(con, cur, [output_s], output_type='dictionary')
    
    # Create feature and output variable arrays to be used in regression models
    feature_a, feature_s_l, output_a, no_none_features_b_a = \
        regression.create_arrays(feature_d, output_d)
        
    # Standardize feature array
    stand_feature_a = StandardScaler().fit_transform(feature_a.astype(float))
        
    # Classify counties by political leaning
    output_class_a = np.array([0 if i < 0 else 1 for i in output_a])
    # 0: Democratic-leaning; 1: Republican-leaning
    
    return (feature_a, stand_feature_a, feature_s_l, output_class_a)
            
            
            
def forward_stepwise_selection(stand_feature_a, feature_s_l, output_class_a):
    """ Run forward stepwise feature selection in order to determine the number and ranking of features most predictive of a county voting Democratic or Republican in 2012. """
    
    # Define classifiers
    classifier_d = {'Logistic regression': LogisticRegression(C=1e5),
                    'LDA': LDA(), 
                    'QDA': QDA(), 
                    'Naive Bayes': GaussianNB(),
                    'Linear SVM': SVC(C=0.8, kernel='linear'), 
                    'RBF SVM': SVC(C=1e5, kernel='rbf'),
                    'k nearest neighbors': KNeighborsClassifier(n_neighbors=3)}
        
    # Create a dictionary of the error rate of each model as features are added
    accuracy_d = {}
    
    # Iterate over classifiers
    for classifier_s, classifier in classifier_d.iteritems():
        
        print('Classifier: %s' % classifier_s)
        
        # Initialize entry in the dictionary of error rates
        accuracy_d[classifier_s] = {'feature_s_l': [], 'accuracy_l': []}
        
        # Create a list of unselected features
        i_unselected_l = range(0, stand_feature_a.shape[1])
        i_selected_l = []
        
        while len(i_unselected_l):
            
            # Select the feature most correlated with the outupt feature
            accuracy_l = []
            for i_feature in i_unselected_l:
                i_model_feature_l = i_selected_l + [i_feature]
                
                explanatory_a = stand_feature_a[:, i_model_feature_l]
                all_cv_scores_l = cross_val_score(classifier, explanatory_a,
                                                  output_class_a, cv=10)
                accuracy_l.append(sum(all_cv_scores_l)/float(len(all_cv_scores_l)))
                
            i_most_correlated_feature = i_unselected_l[accuracy_l.index(max(accuracy_l))]
            
            print('    Accuracy: %0.5f (%s)' %
                  (max(accuracy_l), feature_s_l[i_most_correlated_feature]))
            i_selected_l.append(i_most_correlated_feature)
            i_unselected_l.remove(i_most_correlated_feature)
            
            # Add feature and score to the model
            accuracy_d[classifier_s]['feature_s_l'].append(feature_s_l[i_most_correlated_feature])
            accuracy_d[classifier_s]['accuracy_l'].append(max(accuracy_l))
            
            
    ## Plot results
    
    # Plot linear classifiers
    fig = plt.figure(figsize=(24, 10))
    classifier_s_l = ['Logistic regression', 'LDA', 'Linear SVM']
    for i_classifier, classifier_s in enumerate(classifier_s_l):
        ax = fig.add_axes([0.04+0.33*i_classifier, 0.45, 0.25, 0.52])
        plotting.plot_line_score_of_features(ax, classifier_d[classifier_s]['feature_s_l'],
                                             classifier_d[classifier_s]['accuracy_l'],
                                             extremum_func=max,
                                             is_backward_selection_b=False,
                                             ylabel_s=classifier_s)
    plt.savefig(os.path.join(config.output_path_s,
                             'classification__forward_stepwise_selection__linear.png'))
                                         
    # Plot nonlinear classifiers
    fig = plt.figure(figsize=(32, 10))
    classifier_s_l = ['QDA', 'Naive Bayes', 'AIC', 'BIC']
    for i_classifier, classifier_s in enumerate(classifier_s_l):
        ax = fig.add_axes([0.04+0.25*i_classifier, 0.45, 0.19, 0.52])
        plotting.plot_line_score_of_features(ax, classifier_d[classifier_s]['feature_s_l'],
                                             classifier_d[classifier_s]['accuracy_l'],
                                             extremum_func=max,
                                             is_backward_selection_b=False,
                                             ylabel_s=classifier_s)
    plt.savefig(os.path.join(config.output_path_s,
                             'classification__forward_stepwise_selection__nonlinear.png'))