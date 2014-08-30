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

sys.path.append(config.configLocalPathS)
import config_local
reload(config_local)



def main(cur):

    # Prepare for reading in 2012 election data
    filePathS = os.path.join([config.rawDataPathS, 'election_statistics',
                             'US_elect_county__2012.csv'])
    cur.execute('DROP TABLE IF EXISTS election2012_raw;')

    # Create table with necessary columns, including all 'Party' and 'Votes' columns
    iFirstPartyColumn = 13
    numPartyColumns = 16
    commandS = 'CREATE TABLE election2012_raw(election2012_fips CHAR(5), election2012_total_votes INT(10)'
    for lColumn in xrange(numPartyColumns):
        oneColumnS = ', party%02d CHAR(3), votes%02d INT(10)' % (lColumn, lColumn)
        commandS += oneColumnS
    commandS += ');'
    cur.execute(commandS)
    
    # Load all columns
    commandS = r"""LOAD DATA LOCAL INFILE '{filePathS}'
INTO TABLE election2012_raw
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\r\n'
IGNORE 1 LINES
""".format(filePathS=filePathS)
    commandS += utilities.construct_field_string(203)
    # Add a bracketed list of all columns
    commandS += '\nSET election2012_fips=@col004, election2012_total_votes=@col011'
    for lColumn in xrange(numPartyColumns):
        iPartyColumn = iFirstPartyColumn + 12*lColumn
        iVotesColumn = iPartyColumn + 7
        oneColumnS = (', party%02d=@col%03d, votes%02d=@col%03d'
                     % (lColumn, iPartyColumn, lColumn, iVotesColumn))
        commandS += oneColumnS
    commandS += ';'
    cur.execute(commandS)
    
    # Remove entries that correspond to the voting records of the entire state
    cur.execute('DELETE FROM election2012_raw WHERE election2012_fips=0;')
        
    # Test extract_votes
    extract_votes(cur, 'Dem')
    extract_votes(cur, 'GOP')
    
    # Create new table with only relevant columns
    cur.execute('DROP TABLE IF EXISTS election2012;')
    commandS = """
        CREATE TABLE election2012 AS
        (SELECT election2012_fips, election2012_total_votes,
        election2012_Dem, election2012_GOP FROM election2012_raw);"""
    cur.execute(commandS)
    
    # Print all columns
#    cur.execute('SELECT * FROM election2012;')
#    for lRow in range(1000):
#        row = cur.fetchone()
#        print(row)
    
        
        
def extract_votes(cur, partyS):
    """
    For each row, find the column that corresponds to the party given in partyS
    and store the corresponding value in the 'party' column
    """
    
    # Add empty column to store vote total information in
    commandS = ('ALTER TABLE election2012_raw ' +
                'ADD election2012_%s INT(10) NULL;' % partyS)
    cur.execute(commandS)    
    
    # Set votes from each 'party###' column
    iParty = 0
    numNullValues = None
    while numNullValues != 0:
        commandS = """
        UPDATE election2012_raw
        SET election2012_%s=votes%02d
        WHERE party%02d = '%s';
        """ % (partyS, iParty, iParty, partyS)
        cur.execute(commandS)
        
        # Count number of counties with unset values and iterate
        commandS = """
        SELECT COUNT(*) FROM election2012_raw
        WHERE election2012_%s IS NULL;
        """ % partyS
        cur.execute(commandS)
        row = cur.fetchone()
        numNullValues = row[0]
        iParty += 1