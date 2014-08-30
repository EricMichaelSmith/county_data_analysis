# -*- coding: utf-8 -*-
"""
@author: Eric Smith
Created 2014-08-24

Reads in FIPS codes from https://github.com/hadley/data-counties/blob/master/county-fips.csv (Hadley Wickham)

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

sys.path.append(config.configLocalPathS)
import config_local
reload(config_local)



def main(cur):

    # Prepare for reading in 2012 election data
    filePathS = os.path.join(config.rawDataPathS, 'fips_codes',
                             'county-fips.csv')
    cur.execute('DROP TABLE IF EXISTS fips_raw;')
    
    # Create analysis with necessary columns
    commandS = """
CREATE TABLE fips_raw(county_fips VARCHAR(3),
state_fips VARCHAR(2), fips_county VARCHAR(32),
fips_state VARCHAR(2));"""
#    commandS = """
#        CREATE TABLE fips_raw(foo1 CHAR(3), foo2 CHAR(3), foo3 CHAR(3));"""
    cur.execute(commandS)
#    commandS = """
#        CREATE TABLE fips_raw(county_fips LPAD(VARCHAR(3), 3, '0'),
#        state_fips VARCHAR(2), fips_county VARCHAR(32), fips_state VARCHAR(2));"""
#    cur.execute(commandS)
    
    # Load all columns
    commandS = r"""LOAD DATA LOCAL INFILE '{filePathS}'
INTO TABLE fips_raw
FIELDS TERMINATED BY ','
IGNORE 1 LINES
""".format(filePathS=filePathS).replace('\\', r'\\')
    commandS += utilities.construct_field_string(6)
    # Add a bracketed list of all columns
    commandS += """SET county_fips=@col005, state_fips=@col006,
fips_county=@col003, fips_state=@col004;"""
#    commandS = r"""LOAD DATA LOCAL INFILE '{filePathS}'
#INTO TABLE fips_raw
#FIELDS TERMINATED BY ','
#IGNORE 1 LINES
#(foo1, foo2, foo3);""".format(filePathS=filePathS).replace('\\', r'\\')
    print(commandS)
    cur.execute(commandS)
    
    # Concatenate the two fips fields
#    commandS = """
#        UPDATE fips_raw SET fips_fips =
#        CONCAT(state_fips, county_fips);"""
#    cur.execute(commandS)
    
    # Using the now-current FIPS code for Miami-Dade County, FL
#    commandS = """
#        UPDATE fips_raw
#        SET fips_fips = '12086', fips_county = 'Miami-Dade'
#        WHERE fips_fips = '12025';"""
#    cur.execute(commandS)
    
    # Create new table with only relevant columns
#    cur.execute('DROP TABLE IF EXISTS fips;')
#    commandS = """
#        CREATE TABLE fips AS
#        (SELECT fips_fips, fips_county, fips_state FROM fips_raw);"""
#    cur.execute(commandS)
    
    # Print all columns
#    cur.execute('SELECT * FROM fips;')
    cur.execute('SELECT * FROM fips_raw;')
    for lRow in range(10):
        row = cur.fetchone()
        print(row)