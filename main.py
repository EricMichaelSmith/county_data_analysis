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

2014-05-31: Play around more with csv2mysql.py. Previous error: _mysql_exceptions.ProgrammingError: (1064, "You have an error in your SQL syntax; check the manual that corresponds to your MySQL server version for the right syntax to use near 'varchar(255),\ncounty_number integer,\nfips_code integer,\ncounty_name varchar(255)' at line 4"). Also, probably use additional arguments in cursor.execute() because that's suppose to avoid SQL injections.
"""



def main():