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

def main(con, cur):
    
    factor_s_l = ['asian_not_hispanic_fraction',
                  'average_household_size',
                  'black_not_hispanic_fraction',
                  'divorced',
                  'fertility',
                  'foreign_born',
                  'high_school_graduate'
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
                  'veterans',
                  'violent_crime_rate'
                  'white_not_hispanic_fraction']