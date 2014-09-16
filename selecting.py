# -*- coding: utf-8 -*-
"""
Created on Sun Sep 14 16:29:52 2014

@author: Eric

Tools for selecting data from the 'full' database

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



def select_fields(con, cur, field_s_l):
    """
    Selects fields specified in field_s_l from database; returns a dictionary where each field is a key.
    """
    
    # Grab data in the form of a tuple of dictionaries
    command_s = 'SELECT '
    for field_s in field_s_l:
        command_s += field_s + ', '
    command_s = command_s[:-2] + ' FROM full;'
    print(command_s)
    cur.execute(command_s)
    output_d_t = cur.fetchall()
    
    # Convert tuple of dictionaries to dictionary of lists
    field_d = {}
    for field_s in field_s_l:
        field_d[field_s] = [output_d[field_s] for output_d in output_d_t]
        
    return field_d