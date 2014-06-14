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

    # Prepare for reading in 2012 election data
    filePathS = '/'.join([config.rawDataPathS, 'election_statistics',
                             'US_elect_county__2012.csv'])
    cur.execute('DROP TABLE IF EXISTS election2012_raw;')

    # Read in necessary columns, including all 'Party' and 'Votes' columns
    iFirstPartyColumn = 13
    numPartyColumns = 16
    commandS = 'CREATE TABLE election2012_raw(fips_election2012 CHAR(5), total_votes INT(10)'
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
    commandS += '\nSET fips_election2012=@col004, total_votes=@col011'
    for lColumn in xrange(numPartyColumns):
        iPartyColumn = iFirstPartyColumn + 12*(lColumn-1)
        iVotesColumn = iPartyColumn + 7
        oneColumnS = (', party%02d=@col%03d, votes%02d=@col%03d'
                     % (lColumn, iPartyColumn, lColumn, iVotesColumn))
        commandS += oneColumnS
    commandS += ';'
    print(commandS)
    cur.execute(commandS)

    # Print all columns
    cur.execute('SELECT * FROM election2012_raw')
    for lRow in range(5):
        row = cur.fetchone()
        print row