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

2014-08-28: You're currently testing fips.py. Then try merging it with election2012, and then probably figure out how to import election2008, yes. After that keep adding tables: make a list of all of your data sources, and for each source, write a script to read in all of the relevant data from that source. Then, write a function to join all of those tables together.
"""

import MySQLdb
import sys

import config
reload(config)
import utilities
reload(utilities)

import fips
reload(fips)
import election2012
reload(election2012)

sys.path.append(config.configLocalPathS)
import config_local
reload(config_local)


def main():
    
    con = MySQLdb.connect(user='root',
                          passwd=config_local.pwordS,
                          db='projects',
                          local_infile=1)
    cur = con.cursor()
    
    
    ## Import data
    fips(cur)
    election2012(cur)
    
    
    # Wrap up
    con.commit()
    con.close()