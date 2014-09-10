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

2014-08-30: Figure out how to import election2008. After that keep adding tables: make a list of all of your data sources, and for each source, write a script to read in all of the relevant data from that source. Then, write a function to join all of those tables together.
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


def main(con, cur):
        
    # Import data
    fips.main(cur)
    election2008.main(con, cur)
    election2012.main(cur)
    
    # Merge tables
    cur.execute('DROP TABLE IF EXISTS full;')
    cur.execute("""CREATE TABLE full
SELECT *
FROM fips LEFT JOIN election2012
ON fips.fips_fips = election2012.election2012_fips;""")

    # Print columns
    cur.execute('SELECT * FROM full;')
    for lRow in range(10):
        row = cur.fetchone()
        print(row)    
    
    

def wrapper():
    
    # Start connection
    con = MySQLdb.connect(user='root',
                          passwd=config_local.pwordS,
                          db='projects',
                          local_infile=1)
    cur = con.cursor()
    
    # Run code
    main(cur)
    
    # Wrap up
    con.commit()
    con.close()
    
    return (con, cur)