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

2014-06-03: Forget about csv2mysql.py. Just see this page: http://stackoverflow.com/questions/4202564/how-to-insert-selected-columns-from-csv-file-to-mysql-using-load-data-infile. First test it with foo.csv, and then try election2012. Write a script to auto-create a string '(@col001, @col002,...' so that you don't have to manually specify fields.
"""



def main():