# -*- coding: utf-8 -*-
"""
Created on Sun May 18 20:36:32 2014

@author: Eric Smith

What trends exist within US-county-level demographic data?

Suffixes at the end of variable names:
A: numpy array
B: boolean
D: dictionary
L: list
S: string
T: tuple
Underscores indicate chaining: for instance, "fooT_T" is a tuple of tuples

2014-05-30: Figure out how to ignore certain columns in MySQL; then, figure out which columns you want in election2012 and load only those. Do this with all of your files. Also, probably use additional arguments in cursor.execute() because that's suppose to avoid SQL injections
"""



def main():