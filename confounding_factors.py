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
from scipy import stats

import config
reload(config)
import selecting
reload(selecting)
import utilities
reload(utilities)



def main(con, cur):
    
    # Load explanatory variable data
    explanatory_s = 'dem_fraction_shift'
    explanatory_d = selecting.select_fields(con, cur, [explanatory_s], output_type='dictionary')
    
    # Load feature data
    feature_s_l = config.feature_s_l
    feature_d = selecting.select_fields(con, cur, feature_s_l, output_type='dictionary')
    
    # Run linear regression on each feature separately
    r_value_d = {}
    r_value_5th_percentile_d = {}
    r_value_50th_percentile_d = {}
    r_value_95th_percentile_d = {}
    for key_s in feature_d:
        is_none_b_a = np.equal(feature_d[key_s], None)
        feature1_a = np.array(feature_d[key_s])[~is_none_b_a]
        feature2_a = np.array(explanatory_d[explanatory_s])[~is_none_b_a]
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
                                                con,
                                                cur,
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
    feature_by_r_value_l = [feature_by_r_value_d[r_value] for r_value in \
        sorted(r_value_d.itervalues())]
    print(feature_by_r_value_l)
    
    # Plot the r-values of all features
    fig, axes = plt.subplots(nrows=len(feature_by_r_value_l))
    fig.subplots_adjust(top=0.97, bottom=0.03, left=0.2, right=0.99)
    axes[0].set_title('Pearson''s r of feature and Obama vote shift')
    text_pos_x = -0.25
    text_pos_y = 0.00
    for ax, feature_s in zip(axes, reversed(feature_by_r_value_l)):
        ax.set_axis_bgcolor((1.0, 0.9, 0.9))
        ax.scatter([r_value_d[feature_s]], [0], c=(0,0,0))
        width_to_left = r_value_d[feature_s] - r_value_5th_percentile_d[feature_s]
        width_to_right = r_value_95th_percentile_d[feature_s] - r_value_d[feature_s]
        ax.errorbar([r_value_d[feature_s]], [0], xerr=[width_to_left, width_to_right])
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        ax.text(text_pos_x, text_pos_y, feature_s, va='center', ha='left',
                fontsize=10)
        ax.set_axis_off()
    plt.show()
            
            
            
def regression_confidence_interval_wrapper(index_l, con, cur, feature1_a, feature2_a):
    """ {{{write this!!!}}} """
    
    slope, intercept, r_value, p_value, std_err = \
        stats.linregress(feature1_a[index_l].tolist(), feature2_a[index_l].tolist())
    return r_value