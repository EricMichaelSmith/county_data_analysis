# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 08:50:58 2014

@author: Eric

Tests for confounding factors in the voting/economy shift correlation.

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
#import pandas as pd
#import pdb
from scipy import stats
from sklearn import cross_validation, feature_selection, linear_model, preprocessing
import statsmodels.api as sm

import config
reload(config)
#import election2008
#reload(election2008)
import plotting
reload(plotting)
import selecting
reload(selecting)
import utilities
reload(utilities)



def main(con, cur):
    
    # Load output variable data
    output_s = 'dem_fraction_shift'
    output_d = selecting.select_fields(con, cur, [output_s], output_type='dictionary')
    
    # Load feature data
    feature_s_l = config.feature_s_l
    feature_d = selecting.select_fields(con, cur, feature_s_l, output_type='dictionary')    
    
    # (2) Find and plot r-value of each feature with dem_fraction_shift
#    feature_by_r_value_s_l = pearsons_r_single_features(feature_d, output_d)
    
    # (4) Plot pairwise r-values of all features in a heat map
#    pearsons_r_heatmap(feature_d, feature_by_r_value_s_l)
    
    # Create feature and output variable arrays to be used in regression models
    feature_a, ordered_feature_s_l, output_a, no_none_features_b_a = \
        create_arrays(feature_d, output_d)

    # Print ordered list of features used in regression models
#    for i_feature, feature_s in enumerate(ordered_feature_s_l):
#        print('%d: %s' % (i_feature, feature_s))

    # Plot all counties whose rows remain intact because they had no Nones
#    fips_s = 'fips_fips'
#    fips_d = selecting.select_fields(con, cur, [fips_s], output_type='dictionary')
#    print([i[1] for i in enumerate(fips_d['fips_fips']) if ~no_none_features_b_a[i[0]]])
#    fips_sr = pd.Series(data=no_none_features_b_a, index=fips_d['fips_fips'])
#    shape_index_l, shape_l = election2008.read_data()[1:]
#    shape_fig = plt.figure(figsize=(11,6))
#    plotting.make_shape_plot(shape_fig, fips_sr, shape_index_l, shape_l, 'boolean',
#                             ((0, 0, 0), (0.75, 0.75, 0.75)))
#    print(feature_a.shape)

    # Run additive regression model
    additive_regression_model(feature_a, ordered_feature_s_l, output_a)

    # Run regression with regularization
#    regularized_regression(feature_a, ordered_feature_s_l, output_a)
    
    # Run recursive feature elimination with cross-validation
#    recursive_feature_elimination(feature_a, ordered_feature_s_l, output_a)



def additive_regression_model(feature_a, feature_s_l, output_a):
    """ Builds a multivariate linear regression by iteratively adding the feature that will most increase the R^2 value. """
    
    # Create a dictionary of the scores of each model as features are added
    score_d = {}
    
    
    ## Loop over all scores
    for score_s in regression_d.iterkeys():
        
        # Initialize entry in the dictionary of scores
        score_d[score_s] = {'feature_s_l': [], 'score_value_l': []}
                
        # Create a list of unselected features
        i_unselected_l = range(0, feature_a.shape[1])
        i_selected_l = []
        
        while len(i_unselected_l):
        
            # Select the feature most correlated with dem_fraction_shift
            score_l = []
            for i_feature in i_unselected_l:
                i_model_feature_l = i_selected_l + [i_feature]
                
                if score_s == 'Cross-validated R-squared':
                    explanatory_a = feature_a[:, i_model_feature_l]
                    clf = linear_model.LinearRegression()
                    all_cv_scores_l = cross_validation.cross_val_score(clf,
                                                                       explanatory_a,
                                                                       output_a,
                                                                       cv=10)
                    score_l.append(sum(all_cv_scores_l)/float(len(all_cv_scores_l)))
                else:
                    explanatory_a = \
                        sm.add_constant(feature_a[:, i_model_feature_l].astype(float))
                    model = sm.OLS(output_a, explanatory_a)
                    results = model.fit()
                    score_l.append(getattr(results, regression_d[score_s]['attribute']))
                
            i_most_correlated_feature = \
                i_unselected_l[score_l.index(regression_d[score_s]['extremum'](score_l))]
                
            print('Next most correlated feature: %s\n    %s = %0.5f' %           
              (feature_s_l[i_most_correlated_feature],
               score_s,
               regression_d[score_s]['extremum'](score_l)))
            i_selected_l.append(i_most_correlated_feature)
            i_unselected_l.remove(i_most_correlated_feature)
            
            # Add feature and score to the model
            score_d[score_s]['feature_s_l'].append(feature_s_l[i_most_correlated_feature])
            score_d[score_s]['score_value_l'].append(regression_d[score_s]['extremum'](score_l))
    
    # {{{plot results}}}



def create_arrays(feature_d, output_d):
    """ Transform the input and output data from dicts into arrays """
    
    # Transform feature dataset
    feature_s_l = feature_d.keys()
    feature_a = np.array(feature_d[feature_s_l[0]], ndmin=2).T
    ordered_feature_s_l = [feature_s_l[0]]
    for feature_s in feature_s_l[1:]:
        feature_a = np.concatenate((feature_a, np.array(feature_d[feature_s],
                                    ndmin=2).T), axis=1)
        ordered_feature_s_l.append(feature_s)
        
    # Transform output dataset
    output_s = output_d.keys()[0]
    output_a = np.array(output_d[output_s])
        
    # Select only observations without Nones
    is_none_b_a = np.equal(feature_a, None)
    no_none_features_b_a = (np.sum(is_none_b_a, axis=1) == 0)
    feature_a = feature_a[no_none_features_b_a,]
    output_a = output_a[no_none_features_b_a]
    
    # Print how many rows each feature is missing information on
#    num_none_per_feature_b_a = np.sum(is_none_b_a, axis=0)
#    for l_feature, feature_s in enumerate(ordered_feature_s_l):
#        print('%s: %d' % (feature_s, num_none_per_feature_b_a[l_feature]))
    
    return (feature_a, ordered_feature_s_l, output_a, no_none_features_b_a)
    
    
    
def pearsons_r_heatmap(feature_d, feature_by_r_value_s_l):
    """ Plots a heatmap of the pairwise correlation coefficient between all features. """

    # Create heatmap from pairwise correlations
    heat_map_a = np.ndarray((len(feature_by_r_value_s_l), len(feature_by_r_value_s_l)))
    for i_feature1, feature1_s in enumerate(feature_by_r_value_s_l):
        for i_feature2, feature2_s in enumerate(feature_by_r_value_s_l):
            is_none_b_a = np.equal(feature_d[feature1_s], None) | \
                np.equal(feature_d[feature2_s], None)
            feature1_a = np.array(feature_d[feature1_s])[~is_none_b_a]
            feature2_a = np.array(feature_d[feature2_s])[~is_none_b_a]
            heat_map_a[i_feature1, i_feature2] = \
                stats.linregress(np.array(feature1_a.tolist()),
                                 np.array(feature2_a.tolist()))[2]
    
    # Create figure and heatmap axes
    fig = plt.figure(figsize=(10, 11))
    heatmap_ax = fig.add_axes([0.43, 0.10, 0.55, 0.55])
    
    # Show image
    color_t_t = ((1, 0, 0), (1, 1, 1), (0, 1, 0))
    max_magnitude = 1
    colormap = plotting.make_colormap(color_t_t)
    heatmap_ax.imshow(heat_map_a,
                      cmap=colormap,
                      aspect='equal',
                      interpolation='none',
                      vmin=-max_magnitude,
                      vmax=max_magnitude)
    
    # Format axes
    heatmap_ax.xaxis.set_tick_params(labelbottom='off', labeltop='on')
    heatmap_ax.set_xlim([-0.5, len(feature_by_r_value_s_l)-0.5])
    heatmap_ax.set_xticks(range(len(feature_by_r_value_s_l)))
    heatmap_ax.set_xticklabels(feature_by_r_value_s_l, rotation=90)
    heatmap_ax.invert_xaxis()
    heatmap_ax.set_ylim([-0.5, len(feature_by_r_value_s_l)-0.5])
    heatmap_ax.set_yticks(range(len(feature_by_r_value_s_l)))
    heatmap_ax.set_yticklabels(feature_by_r_value_s_l)
    
    # Add colorbar
    color_ax = fig.add_axes([0.25, 0.06, 0.50, 0.02])
    color_bar_s = "Correlation strength (Pearson's r)"
    plotting.make_colorbar(color_ax, max_magnitude, color_t_t, color_bar_s)
    
    plt.savefig(os.path.join(config.output_path_s, 'pearsons_r_heatmap.png'))

    

def pearsons_r_single_features(feature_d, output_d):
    """ Find and plot r-value of each feature (in feature_d) with dem_fraction_shift (in output_d) """
    
    # Run linear regression on each feature separately
    r_value_d = {}
    r_value_5th_percentile_d = {}
    r_value_50th_percentile_d = {}
    r_value_95th_percentile_d = {}
    output_s = output_d.keys()[0]
    for key_s in feature_d:
        is_none_b_a = np.equal(feature_d[key_s], None)
        feature1_a = np.array(feature_d[key_s])[~is_none_b_a]
        feature2_a = np.array(output_d[output_s])[~is_none_b_a]
        slope, intercept, r_value_d[key_s], p_value, std_err = \
            stats.linregress(np.array(feature1_a.tolist()),
                             np.array(feature2_a.tolist()))
        print('%s: r-value = %0.2f, p-value = %0.3g' % \
              (key_s, r_value_d[key_s], p_value))
            
        # Run bootstrap to find r-value confidence interval for each feature and
        # the output variable
        (r_value_5th_percentile_d[key_s],
         r_value_50th_percentile_d[key_s],
         r_value_95th_percentile_d[key_s]) = \
        utilities.bootstrap_confidence_interval(regression_confidence_interval_wrapper,
                                                sum(~is_none_b_a),
                                                feature1_a,
                                                feature2_a,
                                                confidence_level=0.95,
                                                num_samples=1000)
#        print('s: r-value range: %0.2f, %0.2f, %0.2f' % \
#              (r_value_5th_percentile_d[key_s],
#               r_value_50th_percentile_d[key_s],
#               r_value_95th_percentile_d[key_s]))
               
    # Get list of features sorted by r-value (this will not work if r-values are not unique)
    feature_by_r_value_d = {y:x for x, y in r_value_d.iteritems()}
    sorted_r_value_l = sorted([i for i in r_value_d.itervalues()], key=abs)
    feature_by_r_value_s_l = [feature_by_r_value_d[r_value] for r_value in \
        sorted_r_value_l]
    
    
    ## Plot the r-values of all features
    num_features = len(feature_by_r_value_s_l)
    ax = plt.figure(figsize=(10, 0.5*len(feature_by_r_value_s_l))).add_subplot(1, 1, 1)
    
    # Plot the point and error bar, as well as guide lines
    ax.plot([0, 0], [0, num_features+1], c=(1, 0, 0))
    for l_feature, feature_s in enumerate(feature_by_r_value_s_l):
        
        # For unemployment_fraction_shift, highlight region in yellow
        if feature_s == 'unemployment_fraction_shift':
            plt.axhspan(l_feature+0.5, l_feature+1.5, facecolor=(1, 1, 0), alpha=0.5)
        
        ax.plot([-1, 1], [l_feature+0.5, l_feature+0.5], c=(0.75, 0.75, 0.75))
        ax.scatter([r_value_d[feature_s]], [l_feature+1], c=(0,0,0))
        width_to_left = r_value_d[feature_s] - r_value_5th_percentile_d[feature_s]
        width_to_right = r_value_95th_percentile_d[feature_s] - r_value_d[feature_s]
        ax.errorbar([r_value_d[feature_s]], [l_feature+1],
                    xerr=np.array([[width_to_left], [width_to_right]]),
                    ecolor=(0,0,0))
        if r_value_d[feature_s] >= 0:
            text_color_t = (0, 0.5, 0)
            text_s = '+%0.2f' % r_value_d[feature_s]
        else:
            text_color_t = (0.5, 0, 0)
            text_s = '%0.2f' % r_value_d[feature_s]
        ax.text(0.90, l_feature+1, text_s, color=text_color_t,
                horizontalalignment='right', verticalalignment='center')
                    
    # Configure axes
    ax.set_position([0.43, 0.05, 0.52, 0.93])
    ax.set_xlim([-1, 1])
    ax.set_xticks(np.arange(-1, 1.25, 0.25).tolist())
    ax.set_xlabel("""Correlation strength (Pearson's r) between feature and Obama vote shift""")
    ax.set_ylim([0.5, num_features+0.5])
    ax.set_yticks(np.arange(1, num_features+1, 1).tolist())
#    feature_by_r_value_s_l = [feature_s + ' (+%0.2f)' % sorted_r_value_l[i]
#                              if sorted_r_value_l[i] >= 0
#                              else feature_s + ' (%0.2f)' % sorted_r_value_l[i]
#                              for i, feature_s in enumerate(feature_by_r_value_s_l)]
    ax.set_yticklabels(feature_by_r_value_s_l)
    
    # Show and save plot
    plt.show()
    plt.savefig(os.path.join(config.output_path_s, 'pearsons_r_single_features.png'))
    
    return feature_by_r_value_s_l

    

def print_coefficients(coeff_l, ordered_feature_s_l):
    """ Prints a ranked list of all features and their coefficients. """

    coeff_magnitude_l = [abs(i) for i in coeff_l]
    i_sorted_coefficients_l = [i[0] for i in sorted(enumerate(coeff_magnitude_l),
                                                    key=lambda x:x[1])]
    print('Feature ranking:')
    for i_feature in i_sorted_coefficients_l[::-1]:
        print('    %s: %0.3g' % (ordered_feature_s_l[i_feature], coeff_l[i_feature]))



def recursive_feature_elimination(X, ordered_feature_s_l, y):
    """ Run recursive feature elimination with cross-validation {{{write this}}}. Template from http://scikit-learn.org/stable/auto_examples/plot_rfe_with_cross_validation.html. """
    
    # Create recursive feature elimination and fit to data
    regr = linear_model.LinearRegression()
    rfecv = feature_selection.RFECV(estimator=regr, step=1)
    rfecv.fit(X, y)
    print('Optimal number of features: %d' % rfecv.n_features_)
    i_selected_features_a = rfecv.get_support(indices=True)
    for i in i_selected_features_a:
        print('Selected feature: %s' % ordered_feature_s_l[i])
    print('Ranking of features:')
    print(rfecv.ranking_)
    print('Score:')
    print(rfecv.score(X, y))
    
    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()
    


def regularized_regression(feature_raw_a, ordered_feature_s_l, output_a):
    """ Runs regularized linear regressions on all features. """
    
    # Output features for reference
#    print('Features:')
#    for l_feature, feature_s in enumerate(ordered_feature_s_l):
#        print('(%d) %s' % (l_feature, feature_s))

    for standardize_b in (False, True):
        
        if standardize_b:
            feature_a = preprocessing.scale(feature_raw_a.astype(float))
            print('\nRegressors standardized.')
        else:
            feature_a = feature_raw_a.astype(float)
            print('\nRegressors not standardized.')
            
        # Ordinary least squares
        clf = linear_model.LinearRegression()
        clf.fit(feature_a, output_a)
        print('\nOrdinary least squares: R^2 = %0.2f' % clf.score(feature_a, output_a))
        print('Magnitude of R: %0.2f' % (clf.score(feature_a, output_a))**(1.0/2.0))
        print('Intercept: %0.3g' % clf.intercept_)
        print_coefficients(clf.coef_, ordered_feature_s_l)
    
        # Ridge regression with generalized cross-validation
        alpha_l = np.logspace(-15, 5, num=11).tolist()
        clf = linear_model.RidgeCV(alphas=alpha_l)
        clf.fit(feature_a, output_a)
        print('\nRidge: R^2 = %0.2f, alpha = %0.1g' % (clf.score(feature_a, output_a),
              clf.alpha_))
        print('Magnitude of R: %0.2f' % (clf.score(feature_a, output_a))**(1.0/2.0))
        print('Intercept: %0.3g' % clf.intercept_)
        print_coefficients(clf.coef_, ordered_feature_s_l)
        
        # Lasso regression with generalized cross-validation
        alpha_l = np.logspace(-15, 5, num=11).tolist()
        clf = linear_model.LassoCV(alphas=alpha_l)
        clf.fit(feature_a, output_a)
        print('\nLasso: R^2 = %0.2f, alpha = %0.1g' % (clf.score(feature_a, output_a),
              clf.alpha_))
        print('Magnitude of R: %0.2f' % (clf.score(feature_a, output_a))**(1.0/2.0))
        print('Intercept: %0.3g' % clf.intercept_)
        print_coefficients(clf.coef_, ordered_feature_s_l)
        
        # Elastic net regression with generalized cross-validation
        l1_ratio_l = [.1, .5, .7, .9, .95, .99, 1]
        alpha_l = np.logspace(-15, 5, num=11).tolist()
        clf = linear_model.ElasticNetCV(l1_ratio=l1_ratio_l, alphas=alpha_l)
        clf.fit(feature_a, output_a)
        print('\nElastic net: R^2 = %0.2f, l1_ratio = %0.2f, alpha = %0.1g' %
              (clf.score(feature_a, output_a), clf.l1_ratio_, clf.alpha_))
        print('Magnitude of R: %0.2f' % (clf.score(feature_a, output_a))**(1.0/2.0))
        print('Intercept: %0.3g' % clf.intercept_)
        print_coefficients(clf.coef_, ordered_feature_s_l)
    


def regression_confidence_interval_wrapper(index_l, feature1_a, feature2_a):
    """ Allows for bootstrapping over samples in feature1_a and feature2_a, given index of samples index_l """
    
    slope, intercept, r_value, p_value, std_err = \
        stats.linregress(feature1_a[index_l].tolist(), feature2_a[index_l].tolist())
    return r_value
    
    
    
# Dictionary of all information needed for running unregularized regression models with various scores
regression_d = {'Adjusted R-squared': {'attribute': 'rsquared_adj',
                                             'extremum': max},
                      'AIC': {'attribute': 'aic',
                              'extremum': min},
                      'BIC': {'attribute': 'bic',
                              'extremum': min},
                      'Cross-validated R-squared': {'attribute': 'rsquared',
                                                    'extremum': max},
                      'R-squared': {'attribute': 'rsquared',
                                    'extremum': max}}