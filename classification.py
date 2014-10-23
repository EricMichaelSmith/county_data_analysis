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

import config
reload(config)
import regression
reload(regression)
import selecting
reload(selecting)



def main(con, cur):
    """ Run all classifcations used for the county_data_analysis project. """
    
    # Load feature data
    feature_s_l = config.feature_s_l
    feature_d = selecting.select_fields(con, cur, feature_s_l, output_type='dictionary')
    
    # Load output variable data
    output_s = regression.global_output_s
    output_d = selecting.select_fields(con, cur, [output_s], output_type='dictionary')
    
    # Create feature and output variable arrays to be used in regression models
    feature_a, ordered_feature_s_l, output_a, no_none_features_b_a = \
        regression.create_arrays(feature_d, output_d)
        
    # {{{}}}