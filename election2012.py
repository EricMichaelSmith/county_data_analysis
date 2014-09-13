# -*- coding: utf-8 -*-
"""
@author: Eric Smith
Created 2014-03-03

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

import os
import sys

import config
reload(config)
import utilities
reload(utilities)

sys.path.append(config.config_local_path_s)
import config_local
reload(config_local)



def main(con, cur):
    
    # Prepare for reading in 2012 election data
    file_path_s = os.path.join(config.raw_data_path_s, 'election_statistics',
                             'US_elect_county__2012.csv')
    cur.execute('DROP TABLE IF EXISTS election2012_raw;')

    # Create table with necessary columns, including all 'Party' and 'Votes' columns
    i_first_party_column = 13
    num_party_columns = 16
    command_s = 'CREATE TABLE election2012_raw(election2012_fips CHAR(5), election2012_total_votes INT(10)'
    for l_column in xrange(num_party_columns):
        one_column_s = ', party%02d CHAR(3), votes%02d CHAR(10)' % (l_column, l_column)
        command_s += one_column_s
    command_s += ');'
    cur.execute(command_s)
    
    # Load all columns
    command_s = """LOAD DATA LOCAL INFILE '{file_path_s}'
INTO TABLE election2012_raw""".format(file_path_s=file_path_s).replace('\\', r'\\')
    command_s += r"""
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\r\n'
IGNORE 1 LINES"""
    command_s += utilities.construct_field_string(203)
    # Add a bracketed list of all columns
    command_s += """
SET election2012_fips=@col004, election2012_total_votes=@col011"""
    for l_column in xrange(num_party_columns):
        i_party_column = i_first_party_column + 12*l_column
        i_votes_column = i_party_column + 7
        one_column_s = (', party%02d=@col%03d, votes%02d=@col%03d'
                     % (l_column, i_party_column, l_column, i_votes_column))
        command_s += one_column_s
    command_s += ';'
    cur.execute(command_s)
    
    # Remove entries that correspond to the voting records of the entire state
    cur.execute("DELETE FROM election2012_raw WHERE election2012_fips='0';")
        
    # Extract Democratic and Republican votes
    extract_votes(cur, 'Dem')
    extract_votes(cur, 'GOP')
    
    # Create new table with only relevant columns
    cur.execute('DROP TABLE IF EXISTS election2012;')
    cur.execute("""CREATE TABLE election2012 AS
(SELECT election2012_fips, election2012_total_votes,
election2012_Dem, election2012_GOP FROM election2012_raw);""")

    # Edit column names
    cur.execute("""ALTER TABLE election2012
CHANGE election2012_Dem election2012_dem INT,
CHANGE election2012_GOP election2012_rep INT;""")

    # Print columns
#    cur.execute('SHOW COLUMNS FROM election2012;')
#    for l_row in range(10):
#        row = cur.fetchone()
#        print(row)
#    cur.execute('SELECT * FROM election2012;')
#    for l_row in range(10):
#        row = cur.fetchone()
#        print(row)
    
        
        
def extract_votes(cur, party_s):
    """
    For each row, find the column that corresponds to the party given in party_s
    and store the corresponding value in the 'party' column
    """
    
    # Add empty column to store vote total information in
    command_s = ("""ALTER TABLE election2012_raw
ADD election2012_%s CHAR(10) NULL;""" % party_s)
    cur.execute(command_s)    
    
    # Set votes from each 'party###' column
    i_party = 0
    num_null_values = None
    while num_null_values != 0:
        command_s = """
        UPDATE election2012_raw
        SET election2012_%s=votes%02d
        WHERE party%02d = '%s';
        """ % (party_s, i_party, i_party, party_s)
        cur.execute(command_s)
        
        # Count number of counties with unset values and iterate
        command_s = """
        SELECT COUNT(*) FROM election2012_raw
        WHERE election2012_%s IS NULL;
        """ % party_s
        cur.execute(command_s)
        row = cur.fetchone()
        num_null_values = row[0]
        i_party += 1