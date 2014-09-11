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
import pandas as pd
import shapefile
import sys

import config
reload(config)

sys.path.append(config.GeoDaSandboxPathS)
from pyGDsandbox.dataIO import dbf2df

def main(con, cur):

    # Prepare for reading in 2012 election data
    filePathS = os.path.join(config.rawDataPathS, 'election_statistics', '2008',
                             'elpo08p020.dbf')
    
    # Read fields into DataFrame
    fullDF = dbf2df(filePathS)
    fullDF = fullDF.convert_objects(convert_numeric=True)
    
    # Select relevant columns
    finalDF = fullDF.loc[:, ['FIPS', 'TOTAL_VOTE', 'VOTE_DEM', 'VOTE_REP']]
    finalDF.columns = ['election2008_fips', 'election2008_total_votes', 'election2008_dem', 'election2008_rep']
    shapeIndexL = finalDF.election2008_fips.tolist()

    # Read in shapefile data
    fullShapeFile = shapefile.Reader(filePathS)
    shapeL = fullShapeFile.shapes()
    del fullShapeFile
    
        # Removing the second (incorrect) entry for Ottawa County, OH
    finalDF = finalDF.loc[~finalDF['election2008_fips'].isin([39123]) |
                          ~finalDF['election2008_dem'].isin([12064])]
    
    # Cleaning, sorting, and setting the index
    finalDF = finalDF.drop_duplicates()
    finalDF = finalDF.sort(columns='election2008_fips')
    finalDF = finalDF.set_index('election2008_fips', drop=False)
    
    # Converting the fips column to an int
    finalDF['election2008_fips'] = finalDF['election2008_fips'].astype(str)
    
    # This is a work-around for a NaN somewhere in Michigan.
    finalDF = finalDF[pd.notnull(finalDF['election2008_total_votes'])]
    
    # Correcting the total number of votes cast in Laclede County, MO: the number
    # in the source file is actually the number from the previous county in the
    # list, Knox County. If this error isn't corrected, the Obama vote swing in
    # Laclede County between 2008 and 2012 is -231%. The figure that I'm
    # replacing the vote total with was taken from Wikipedia
    # (http://en.wikipedia.org/wiki/Laclede_County,_Missouri, 2014-04-17).
    finalDF.loc[29105, 'election2008_total_votes'] = 16379

    # Correcting the number of votes in Washington County, OH, which is
    # erroneous; the correct number can be found at
    # http://en.wikipedia.org/wiki/United_States_presidential_election_in_Ohio,_2008,
    # 2014-04-17.
    finalDF.loc[39167, 'election2008_total_votes'] = 29932
    finalDF.loc[39167, 'election2008_dem'] = 12368

    # Correcting the number of votes in LaPorte County, IN, which is
    # erroneous; the correct number can be found at
    # http://en.wikipedia.org/wiki/United_States_presidential_election_in_Indiana,_2008,
    # 2014-04-17.
    finalDF.loc[18091, 'election2008_total_votes'] = 46919
            
    # Write to SQL database
    finalDF.to_sql(name='election2008', con=con, if_exists='replace', flavor='mysql',
                   index='False')
                   
    # Print columns
#    cur.execute('SELECT * FROM election2008;')
#    for lRow in range(10):
#        row = cur.fetchone()
#        print(row)
    
    return (shapeIndexL, shapeL)