# -*- coding: utf-8 -*-
"""
@author: Eric Smith
Created 2014-06-26

Reads in 2012 US election results, from http://www.theguardian.com/news/datablog/2012/nov/07/us-2012-election-county-results-download#data

Suffixes at the end of variable names:
A: numpy array
B: boolean
D: dictionary
L: list
S: string
T: tuple
Underscores indicate chaining: for instance, "fooT_T" is a tuple of tuples
"""

import MySQLdb
import sys

import config
reload(config)
import utilities
reload(utilities)

sys.path.append(config.configLocalPathS)
import config_local
reload(config_local)



def main():

    con = MySQLdb.connect(user='root',
                          passwd=config_local.pwordS,
                          db='projects',
                          local_infile=1)
    cur = con.cursor()
    
    # Prepare for reading in 2008 election data
    filePathS = os.path.join(config.rawDataPathS, 'election_statistics', '2008',
                             'elpo08p020.dbf')
    cur.execute('DROP TABLE IF EXISTS election2008_raw;')
    
    
    # {{{}}}
    
    
    # Wrap up
    con.commit()
    con.close()