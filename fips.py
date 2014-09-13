# -*- coding: utf-8 -*-
"""
@author: Eric Smith
Created 2014-08-24

Reads in FIPS codes from https://github.com/hadley/data-counties/blob/master/county-fips.csv (Hadley Wickham)

Suffixes at the end of variable names:
a: numpy array
b: boolean
d: dictionary
df: pandas DataFrame
l: list
s: string
t: tuple
Underscores indicate chaining: for instance, "foo_t_t" is a tuple of tuples
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



def main(con, cur):

    # Prepare for reading in FIPS data
    file_path = os.path.join(config.rawDataPathS, 'fips_codes',
                             'county-fips.csv')
    cur.execute('DROP TABLE IF EXISTS fips_raw;')
    
    # Create table with necessary columns
    command_s = """CREATE TABLE fips_raw(fips_state_part VARCHAR(2),
fips_county_part CHAR(3), county_name VARCHAR(72),
state_name VARCHAR(22));"""
    cur.execute(command_s)
    
    # Load all columns
    command_s = """LOAD DATA LOCAL INFILE '{file_path}'
INTO TABLE fips_raw""".format(file_path=file_path).replace('\\', r'\\')
    command_s += r"""
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\r\n'
IGNORE 1 LINES"""
    command_s += utilities.construct_field_string(6)
    # Add a bracketed list of all columns
    command_s += """
SET fips_state_part=@col006, fips_county_part=@col005,
county_name=@col003, state_name=@col004;"""
    cur.execute(command_s)
    
    # Pad out fips_county_part
    cur.execute("""UPDATE fips_raw
SET fips_county_part = LPAD(fips_county_part, 3, '0');""")
    
    # Concatenate the two fips fields
    cur.execute('ALTER TABLE fips_raw ADD fips_fips VARCHAR(5);')
    cur.execute("""UPDATE fips_raw
SET fips_fips = CONCAT(fips_state_part, fips_county_part);""")
    
    # Remove quotation marks
    cur.execute("""UPDATE fips_raw
SET county_name = REPLACE(county_name, '"', ''),
state_name = REPLACE(state_name, '"', '');""")

    # Title DC correctly
    cur.execute("""UPDATE fips_raw
SET state_name = REPLACE(state_name, 'District of Columbia', 'DC');""")
    
    # Using the now-current FIPS code for Miami-Dade County, FL
    cur.execute("""UPDATE fips_raw
SET fips_fips = '12086', county_name = 'Miami-Dade'
WHERE fips_fips = '12025';""")
    
    # Create new table with only relevant columns
    cur.execute('DROP TABLE IF EXISTS fips;')
    cur.execute("""CREATE TABLE fips AS
(SELECT fips_fips, county_name, state_name FROM fips_raw);""")
    
    # Print columns
#    cur.execute('SELECT * FROM fips;')
#    for l_row in range(10):
#        row = cur.fetchone()
#        print(row)