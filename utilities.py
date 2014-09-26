# -*- coding: utf-8 -*-
"""
Created on Wed Jun 04 20:44:19 2014

@author: Eric

Utility functions for county_data_analysis

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

import config
reload(config)
import selecting
reload(selecting)



def add_derived_features(con, cur):
    
    
    ## Electoral
    # election2008_percent_dem
    command_s = 'ALTER TABLE full ADD election2008_dem_fraction FLOAT(6, 5);'
    cur.execute(command_s)
    command_s = """UPDATE full
SET election2008_dem_fraction = election2008_dem / election2008_total_votes;"""
    cur.execute(command_s)

    # election2008_percent_rep
    command_s = 'ALTER TABLE full ADD election2008_rep_fraction FLOAT(6, 5);'
    cur.execute(command_s)
    command_s = """UPDATE full
SET election2008_rep_fraction = election2008_rep / election2008_total_votes;"""
    cur.execute(command_s)

    # election2012_percent_dem
    command_s = 'ALTER TABLE full ADD election2012_dem_fraction FLOAT(6, 5);'
    cur.execute(command_s)
    command_s = """UPDATE full
SET election2012_dem_fraction = election2012_dem / election2012_total_votes;"""
    cur.execute(command_s)

    # election2012_percent_rep
    command_s = 'ALTER TABLE full ADD election2012_rep_fraction FLOAT(6, 5);'
    cur.execute(command_s)
    command_s = """UPDATE full
SET election2012_rep_fraction = election2012_rep / election2012_total_votes;"""
    cur.execute(command_s)
    
    # dem_shift
    command_s = 'ALTER TABLE full ADD dem_fraction_shift FLOAT(8, 5);'
    cur.execute(command_s)
    command_s = """UPDATE full
SET dem_fraction_shift = election2012_dem_fraction - election2008_dem_fraction;"""
    cur.execute(command_s)

    # rep_shift
    command_s = 'ALTER TABLE full ADD rep_fraction_shift FLOAT(8, 5);'
    cur.execute(command_s)
    command_s = """UPDATE full
SET rep_fraction_shift = election2012_rep_fraction - election2008_rep_fraction;"""
    cur.execute(command_s)
    
    
    # Unemployment
    
    # unemployment_rate_shift
    command_s = 'ALTER TABLE full ADD unemployment_fraction_shift FLOAT(8, 5);'
    cur.execute(command_s)
    command_s = """UPDATE full
SET unemployment_fraction_shift = (unemployment_rate_2012 - unemployment_rate_2008) / 100;"""
    cur.execute(command_s)
    
    
    # Demographic
    
    # white_not_hispanic_fraction
    command_s = 'ALTER TABLE full ADD white_not_hispanic_fraction FLOAT(8, 5);'
    cur.execute(command_s)
    command_s = """UPDATE full
SET white_not_hispanic_fraction = white_not_hispanic_number / 2008_to_2012_race_and_ethnicity_total;"""
    cur.execute(command_s)

    # black_not_hispanic_fraction
    command_s = 'ALTER TABLE full ADD black_not_hispanic_fraction FLOAT(8, 5);'
    cur.execute(command_s)
    command_s = """UPDATE full
SET black_not_hispanic_fraction = black_not_hispanic_number / 2008_to_2012_race_and_ethnicity_total;"""
    cur.execute(command_s)

    # asian_not_hispanic_fraction
    command_s = 'ALTER TABLE full ADD asian_not_hispanic_fraction FLOAT(8, 5);'
    cur.execute(command_s)
    command_s = """UPDATE full
SET asian_not_hispanic_fraction = asian_not_hispanic_number / 2008_to_2012_race_and_ethnicity_total;"""
    cur.execute(command_s)

    # hispanic_fraction
    command_s = 'ALTER TABLE full ADD hispanic_fraction FLOAT(8, 5);'
    cur.execute(command_s)
    command_s = """UPDATE full
SET hispanic_fraction = hispanic_number / 2008_to_2012_race_and_ethnicity_total;"""
    cur.execute(command_s)
    
    
    # Population

    # population_change_fraction
    command_s = 'ALTER TABLE full ADD population_change_fraction FLOAT(8, 5);'
    cur.execute(command_s)
    command_s = """UPDATE full
SET population_change_fraction = (population_2013_estimate - population_2010_census) / population_2010_census;"""
    cur.execute(command_s)

    # population_density
    command_s = 'ALTER TABLE full ADD population_density FLOAT(16, 5);'
    cur.execute(command_s)
    command_s = """UPDATE full
SET population_density = ROUND(1000*population_2013_estimate/land_area)/1000;"""
    cur.execute(command_s)
    
    print('Derived features added.')
    


def bootstrap_confidence_interval(fun, data_length, confidence_level=0.95,
                                  num_samples=1000, *args):
    """ For num_samples samples, chooses with replacement from a data set of length data_length and calculates a value using function fun(). Returns the lower bound, mean, and upper bound of a confidence interval at a level of confidence_level. """
    
    rand_a = np.random.randint(0, data_length, (num_samples, data_length))
    value_l = [fun(row, *args) for row in rand_a]
    sorted_value_l = sorted(value_l)
    i_lower_bound = round(num_samples * (1/2 - confidence_level/2))
    i_mean = round(num_samples/2)
    i_upper_bound = round(num_samples * (1/2 + confidence_level/2))
    return (sorted_value_l[i_lower_bound],
            sorted_value_l[i_mean],
            sorted_value_l[i_upper_bound])


def construct_field_string(num_columns):
    """
    Constructs a string ("(@col001, @col002,...)") for use in specifying fields in SQL queries.
    """
    
    output_s = '\n('
    for l_column in range(num_columns):
        output_s += '@col%03d, ' % (l_column+1)
    output_s = output_s[:-2] + ')'
    
    return output_s
    
    
    
def find_min_and_max_values(con, cur):
    """ Prints the highest and lowest value of every feature. This is useful to make sure that there are no wacky values in the dataset. """
    
    # Load feature data
    feature_s_l = config.feature_s_l
    feature_d = selecting.select_fields(con, cur, feature_s_l, output_type='dictionary')
    
    # Print the highest and lowest values of each feature separately
    for feature_s in feature_d:
        is_none_b = np.equal(feature_d[feature_s], None)
        sorted_l = sorted(np.array(feature_d[feature_s])[~is_none_b].tolist())
        print('%s: lowest value = %0.4g, highest value = %0.4g' % \
              (feature_s, sorted_l[0], sorted_l[-1]))
    
    
    
def print_zeros(con, cur):
    """ Prints all zero values in features, in case some of them are actually NULL values in the source file."""
    feature_s_l = config.feature_s_l
    for feature_s in feature_s_l:
        print('Feature: %s' % feature_s)
        command_s = """SELECT fips_fips, {feature_s} FROM full WHERE {feature_s} = 0;"""
        command_s = command_s.format(feature_s=feature_s)
        cur.execute(command_s)
        output_d_t = cur.fetchall()
        for output_d in output_d_t:
            print(output_d)