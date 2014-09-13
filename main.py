# -*- coding: utf-8 -*-
"""
Created on Sun May 18 20:36:32 2014

@author: Eric Smith

What trends exist within US-county-level demographic data?

Suffixes at the end of variable names:
a: numpy array
b: boolean
d: dictionary
df: pandas DataFrame
l: list
s: string
t: tuple
Underscores indicate chaining: for instance, "foo_t_t" is a tuple of tuples

2014-09-11: Keep adding tables: make a list of all of your data sources, and for each source, write a script to read in all of the relevant data from that source. Then, write a function to join all of those tables together.
"""

import MySQLdb
import sys

import config
reload(config)
import utilities
reload(utilities)

import fips
reload(fips)
import election2008
reload(election2008)
import election2012
reload(election2012)
import readin
reload(readin)

sys.path.append(config.config_local_path_s)
import config_local
reload(config_local)
    
    

def connect_to_sql():
    
    # Start connection
    con = MySQLdb.connect(user='root',
                          passwd=config_local.pword_s,
                          db='projects',
                          local_infile=1)
    cur = con.cursor()
    
    return (con, cur)
    
    
    
def create_database(con, cur):
        
    # Import data
    fips.main(con, cur)
    (shape_index_l, shape_l) = election2008.main(con, cur)
    election2012.main(con, cur)
    readin.main(con, cur)
    
    # Merge tables
    cur.execute('DROP TABLE IF EXISTS full;')
    cur.execute("""CREATE TABLE full
SELECT *
FROM fips
INNER JOIN election2008 ON fips.fips_fips = election2008.election2008_fips
INNER JOIN election2012 ON fips.fips_fips = election2012.election2012_fips;""")
    # {{{merge fields from readin}}}

    # Print columns
    cur.execute('SHOW COLUMNS FROM full;')
    for l_row in range(20):
        row = cur.fetchone()
        print(row)  
    cur.execute('SELECT * FROM full;')
    for l_row in range(10):
        row = cur.fetchone()
        print(row)
    
    

def sql_wrapper():
    
    (con, cur) = connect_to_sql()
    
    # Run code
    create_database(con, cur)
    
    # Wrap up
    con.commit()
    con.close()
    
    return (con, cur)