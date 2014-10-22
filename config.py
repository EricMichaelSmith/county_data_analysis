# -*- coding: utf-8 -*-
"""
Created on Sun May 18 20:42:51 2014

@author: Eric Smith

Provides for global variables throughout all of county_data_analysis

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


## Features

# This is a master list of all features we might want to explore.
feature_s_l = ['asian_not_hispanic_fraction',
              'average_household_size',
              'black_not_hispanic_fraction',
              'fertility_rate',
              'fraction_non_senior_citizens_without_insurance',
              'fraction_obese',
#              'fraction_smoking', (Too many missing counties to include)
              'hispanic_fraction',
              'median_age',
              'median_household_income',
              'percent_divorced',
              'percent_english_not_spoken_at_home',
              'percent_foreign_born',
              'percent_high_school_graduate',
              'percent_households_with_children',
              'percent_households_with_senior_citizens',
              'percent_in_poverty',
              'percent_never_married',
              'percent_same_house_1_yr_ago',
              'percent_veterans',
              'population_change_fraction',
              'population_density',
              'premature_death_rate',
              'sex_ratio',
              'teen_birth_rate',
#              'unemployment_fraction_shift', (not as relevant)
              'unemployment_rate_2012',
              'white_not_hispanic_fraction']


# Paths
database_s = 'county_data_analysis'
base_path_s = 'C:\\E\\GitHub\\Computing\\EricMichaelSmith\\county_data_analysis'
config_local_path_s = 'C:\\E\\Dropbox\\Computing\\Personal\\Code\\county_data_analysis'
GeoDaSandbox_path_s = r'C:\E\GitHub\Computing\GeoDaSandbox\sandbox'
output_path_s = 'C:\\E\\Dropbox\\Computing\\Personal\\Code\\county_data_analysis\\output'
package_path_s = 'C:\\E\\Dropbox\\Computing\\Collected\\Python\\Packages'
raw_data_path_s = 'C:\\E\\Dropbox\\Computing\\Personal\\Code\\county_data_analysis\\raw_data'