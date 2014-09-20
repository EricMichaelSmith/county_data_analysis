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

from scipy import stats

import selecting
reload(selecting)



def main(con, cur):
    
    # Load data
    explanatory_s = 'dem_fraction_shift'
    explanatory_d = selecting.select_fields(con, cur, [explanatory_s], output_type='dictionary')
    feature_s_l = ['asian_not_hispanic_fraction',
                  'average_household_size',
                  'black_not_hispanic_fraction',
                  'divorced',
                  'fertility',
                  'foreign_born',
                  'high_school_graduate',
                  'hispanic_fraction',
                  'households_with_children',
                  'households_with_senior_citizens',
                  'language_other_than_english_spoken_at_home',
                  'median_age',
                  'median_household_income',
                  'never_married',
                  'obese',
                  'percent_non_senior_citizens_without_insurance',
                  'population_change_fraction',
                  'population_density',
                  'premature_death_rate',
                  'same_house_1_yr_ago',
                  'sex_ratio',
                  'smoking',
                  'teen_birth_rate',
                  'unemployment_fraction_shift',
                  'veterans',
                  'violent_crime_rate',
                  'white_not_hispanic_fraction']
    feature_d = selecting.select_fields(con, cur, feature_s_l, output_type='dictionary')
    
    # Run linear regresson on each feature separately
    for key_s in feature_d:
        slope, intercept, r_value, p_value, std_err = \
            stats.linregress(feature_d[key_s], explanatory_d[explanatory_s])
        print('%s: r-value = %0.2f, p-value = %0.3g' % (key_s, r_value, p_value))