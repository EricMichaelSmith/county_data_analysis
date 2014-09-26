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

import numpy as np
from scipy import stats

import config
reload(config)
import selecting
reload(selecting)



def main(con, cur):
    
    # Load explanatory variable data
    explanatory_s = 'dem_fraction_shift'
    explanatory_d = selecting.select_fields(con, cur, [explanatory_s], output_type='dictionary')
    
    # Load feature data
    feature_s_l = config.feature_s_l
    feature_d = selecting.select_fields(con, cur, feature_s_l, output_type='dictionary')
    
    # Run linear regresson on each feature separately
    r_value_central_d = {}
    r_value_5th_percentile_d = {}
    r_value_95th_percentile_d = {}
    for key_s in feature_d:
        is_none_b_a = np.equal(feature_d[key_s], None)
        slope, intercept, r_value_central_d[key_s], p_value, std_err = \
            stats.linregress(np.array(feature_d[key_s])[~is_none_b_a].tolist(),
                             np.array(explanatory_d[explanatory_s])[~is_none_b_a].tolist())
        print('%s: r-value = %0.2f, p-value = %0.3g' % \
            (key_s, r_value_central_d[key_s], p_value))
            
        # Run bootstrap to find r-value confidence interval for each feature and
        # the output variable
        #{{{}}}
            
            
            
def regression_confidence_interval_wrapper(index_l, con, cur, feature1_a, feature2_a):
    """ {{{write this!!!}}} """
    
    slope, intercept, r_value, p_value, std_err = \
        stats.linregress(feature1_a[index_l].tolist(), feature2_a[index_l].tolist())
    return r_value