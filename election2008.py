# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 08:25:48 2014

@author: Eric

Reads in 2008 county election data (2008 Presidential General Election, County Results, National Atlas of the United States, http://nationalatlas.gov/atlasftp.html?openChapters=chphist#chphist)

Suffixes at the end of variable names:
A: numpy array
B: boolean
D: dictionary
DF: pandas DataFrame
L: list
S: string
T: tuple
Underscores indicate chaining: for instance, "fooT_T" is a tuple of tuples
"""

import os
import sys

import config
reload(config)

sys.path.append(config.GeoDaSandboxPathS)
from pyGDsandbox.dataIO import dbf2df

def main(con, cur):

    # Prepare for reading in 2012 election data
    filePathS = os.path.join(config.rawDataPathS, 'election_statistics', '2008',
                             'elpo08p020.dbf')
    cur.execute('DROP TABLE IF EXISTS election2008_raw;')
    
    # Read fields into DataFrame
    fullDF = dbf2df(filePathS)
    fullDF = fullDF.convert_objects(convert_numeric=True)
    
    # Select relevant columns
    finalDF = fullDF.loc[:, ['FIPS', 'TOTAL_VOTE', 'VOTE_DEM', 'VOTE_REP']]
    finalDF.columns = ['election2008_fips', 'election2008_total_votes', 'election2008_dem', 'Election2008_rep']
    
    # Write dataframe to SQL
    # (((see http://stackoverflow.com/questions/16476413/how-to-insert-pandas-dataframe-via-mysqldb-into-database and http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.to_sql.html)))