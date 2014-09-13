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



def construct_field_string(num_columns):
    """
    Constructs a string ("(@col001, @col002,...)") for use in specifying fields in SQL queries
    """
    
    output_s = '\n('
    for l_column in range(num_columns):
        output_s += '@col%03d, ' % (l_column+1)
    output_s = output_s[:-2] + ')'
    
    return output_s