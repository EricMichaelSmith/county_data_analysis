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

2014-09-15: Add other derived features. Plot features on the shape plot from last time and see which counties are left in the table and which are missing; that should tell you what rows are missing data. Also, when you're done with all of that, see the OneNote page task list for other stuff to do.
"""

import MySQLdb
import sys

import config
reload(config)
import fips
reload(fips)
import election2008
reload(election2008)
import election2012
reload(election2012)
import read_many_files
reload(read_many_files)
import utilities
reload(utilities)

sys.path.append(config.config_local_path_s)
import config_local
reload(config_local)
    
    
    
def main():
    
    # Connect to SQL
    (con, cur) = connect_to_sql()
    
    # Create SQL database
    create_database(con, cur)
    
    # Add derived featueres
    add_derived_features(con, cur)
    
    # Wrap up
    con.commit()
    con.close()
    
    return (con, cur)
    
    

def add_derived_features(con, cur):
    
    # election2008_percent_dem
    command_s = 'ALTER TABLE full ADD election2008_fraction_dem FLOAT(6, 5);'
    cur.execute(command_s)
    command_s = """UPDATE full
SET election2008_fraction_dem = election2008_dem / election2008_total_votes;"""
    cur.execute(command_s)    

    # election2012_percent_dem
    command_s = 'ALTER TABLE full ADD election2012_fraction_dem FLOAT(6, 5);'
    cur.execute(command_s)
    command_s = """UPDATE full
SET election2012_fraction_dem = election2012_dem / election2012_total_votes;"""
    cur.execute(command_s)    

    # dem_shift
    command_s = 'ALTER TABLE full ADD fraction_dem_shift FLOAT(8, 5);'
    cur.execute(command_s)
    command_s = """UPDATE full
SET fraction_dem_shift = election2012_fraction_dem - election2008_fraction_dem;"""
    cur.execute(command_s)

    # unemployment_rate_shift
    command_s = 'ALTER TABLE full ADD unemployment_rate_shift FLOAT(8, 5);'
    cur.execute(command_s)
    command_s = """UPDATE full
SET unemployment_rate_shift = unemployment_rate_2012 - unemployment_rate_2008;"""
    cur.execute(command_s)   
    
    

def connect_to_sql():
    
    # Start connection
    con = MySQLdb.connect(user='root',
                          passwd=config_local.pword_s,
                          db=config.database_s,
                          local_infile=1)
    cur = con.cursor(MySQLdb.cursors.DictCursor)
    
    return (con, cur)
    
    
    
def create_database(con, cur):
        
    # Import data
    fips.main(con, cur)
    (shape_index_l, shape_l) = election2008.main(con, cur)
    election2012.main(con, cur)
    field_d = read_many_files.main(con, cur)
    
    # Merge tables
    cur.execute('DROP TABLE IF EXISTS full;')
    command_s = """CREATE TABLE full
SELECT *
FROM fips
INNER JOIN election2008 ON fips.fips_fips = election2008.election2008_fips
INNER JOIN election2012 ON fips.fips_fips = election2012.election2012_fips"""
    for folder_name in field_d:
        for file_name in field_d[folder_name]:
            table_name = file_name.replace('.', '')
            this_table_command_s = """
INNER JOIN {table_name} ON fips.fips_fips = {table_name}.{table_name}_fips"""
            this_table_command_s = this_table_command_s.format(table_name=table_name)
            command_s += this_table_command_s
    command_s += ';'
    cur.execute(command_s)

    # Print columns
#    cur.execute('SHOW COLUMNS FROM full;')
#    output_t = cur.fetchall()
#    for row in output_t:
#        print(row)  
#    cur.execute('SELECT * FROM full;')
#    output_t = cur.fetchall()
#    for l_row in range(10):
#        print(output_t)