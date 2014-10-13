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

2014-10-12:
- (3) Make a scatter plot of the top 8 or so correlators with Dem shift to give a sense of these groups' loyalty to Obama; but then, of course, you have to explain all of the features clearly and put units on them
- If it's easy, and AK and HI to the shape plots
- (1) Make some county shape plots plotting a few interesting features
- Maybe see, in OLS, which features have coeffs that are significantly away from 0?
- OLS and ridge give R^2 ~= 0.37 but lasso and elastic net give R^2 ~= 0.34: should you be worried?
- See: are the 9 features selected by lasso / elastic net generally the 9 highest features in the forward and backward selection models?
- Maybe do something to see if it makes sense that percent_non_senior_citizens_without_insurance should be the third most important feature?
- Are you satisfied now with your understanding of how the regression models work? Think about all of this some more...
- See how the ranking of the magnitude of the r-values lines up with the coefficients of the normalized features in the multiple linear regression: will this tell you something about the covariance of the features?
- In article, find something that backs up your finding that whites left Obama in droves (call it the Matt Damon effect? Or just be very careful about this...)
- In article, mention trying regularization in order to reduce the variance inherent in the fit (of course, make sure you know what you're talking about)
- Before posting, search the code for "{{{" and "[[[" and "((("
- When you're done with all of that, see the OneNote page task list for other stuff to do.
"""

import MySQLdb
import sys

import config
reload(config)
import confounding_factors
reload(confounding_factors)
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
    
    # Add derived features
    utilities.add_derived_features(con, cur)
        
    # Find confounding factors
    #confounding_factors.main(con, cur)
    
    # Wrap up
    con.commit()
    con.close()
    
    return (con, cur)
    
    

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
    
    print('Database created.')

    # Print columns
#    cur.execute('SHOW COLUMNS FROM full;')
#    output_t = cur.fetchall()
#    for row in output_t:
#        print(row)  
#    cur.execute('SELECT * FROM full;')
#    output_t = cur.fetchall()
#    for l_row in range(10):
#        print(output_t)