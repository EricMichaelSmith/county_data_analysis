# -*- coding: utf-8 -*-
"""
Created on Wed Jun 04 20:44:19 2014

@author: Eric

Utility functions for county_data_analysis

Suffixes at the end of variable names:
A: numpy array
B: boolean
D: dictionary
L: list
S: string
T: tuple
Underscores indicate chaining: for instance, "fooT_T" is a tuple of tuples
"""



def construct_field_string(numColumns):
    """
    Constructs a string ("(@col001, @col002,...)") for use in specifying fields in SQL queries
    """
    
    outputS = '('
    for lColumn in range(numColumns):
        outputS += '@col%03d, ' % (lColumn+1)
    outputS = outputS[:-2] + ')'
    
    return outputS