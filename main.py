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

2014-09-13: Troubleshoot import. Why is the TRIM(SUBSTR()) command failing? Why are floats defaulting to one decimal place? When all tables import correctly, go through and make sure that the values of the first few rows of each field are what they should be. Then, write a function to join all of those tables together and see which counties are left in the table and which are missing; that should tell you what rows are missing data. Also, when you're done with all of that, see the OneNote page task list for other stuff to do.
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
import read_many_files
reload(read_many_files)

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
    read_many_files.main(con, cur)
    
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