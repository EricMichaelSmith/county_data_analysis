# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 08:25:48 2014

@author: Eric

Reads in 2008 county election data (2008 Presidential General Election, County Results, National Atlas of the United States, http://nationalatlas.gov/atlasftp.html?openChapters=chphist#chphist)

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
import pandas as pd
import shapefile
import sys

import config
reload(config)

sys.path.append(config.GeoDaSandbox_path_s)
from pyGDsandbox.dataIO import dbf2df

def main(con, cur):
    
    # Read in data
    final_df, shape_index_l, shape_l = read_data()
    
    # Removing the second (incorrect) entry for Ottawa County, OH
    final_df = final_df.loc[~final_df['election2008_fips'].isin([39123]) |
                          ~final_df['election2008_dem'].isin([12064])]
    
    # Cleaning, sorting, and setting the index
    final_df = final_df.drop_duplicates()
    final_df = final_df.sort(columns='election2008_fips')
    final_df = final_df.set_index('election2008_fips', drop=False)
    
    # Converting the fips column to an int
    final_df['election2008_fips'] = final_df['election2008_fips'].astype(str)
    
    # This is a work-around for a NaN somewhere in Michigan.
    final_df = final_df[pd.notnull(final_df['election2008_total_votes'])]
    
    # Correcting the total number of votes cast in Laclede County, MO: the number
    # in the source file is actually the number from the previous county in the
    # list, Knox County. If this error isn't corrected, the Obama vote swing in
    # Laclede County between 2008 and 2012 is -231%. The figure that I'm
    # replacing the vote total with was taken from Wikipedia
    # (http://en.wikipedia.org/wiki/Laclede_County,_Missouri, 2014-04-17).
    final_df.loc[29105, 'election2008_total_votes'] = 16379

    # Correcting the number of votes in Washington County, OH, which is
    # erroneous; the correct number can be found at
    # http://en.wikipedia.org/wiki/United_States_presidential_election_in_Ohio,_2008,
    # 2014-04-17.
    final_df.loc[39167, 'election2008_total_votes'] = 29932
    final_df.loc[39167, 'election2008_dem'] = 12368

    # Correcting the number of votes in LaPorte County, IN, which is
    # erroneous; the correct number can be found at
    # http://en.wikipedia.org/wiki/United_States_presidential_election_in_Indiana,_2008,
    # 2014-04-17.
    final_df.loc[18091, 'election2008_total_votes'] = 46919
            
    # Write to SQL database
    final_df.to_sql(name='election2008', con=con, if_exists='replace', flavor='mysql',
                   index='False')
                   
    # Print columns
#    cur.execute('SELECT * FROM election2008;')
#    for l_row in range(10):
#        row = cur.fetchone()
#        print(row)
    
    return (shape_index_l, shape_l)
    
    
    
def read_data():
    """ Read in data and return the uncleaned DataFrame as well as the shapes from the shapefile and an index of shapes """    
    
    # Prepare for reading in 2012 election data
    file_path_s = os.path.join(config.raw_data_path_s, 'election_statistics', '2008',
                             'elpo08p020.dbf')
    
    # Read fields into DataFrame
    full_df = dbf2df(file_path_s)
    full_df = full_df.convert_objects(convert_numeric=True)
    
    # Select relevant columns
    final_df = full_df.loc[:, ['FIPS', 'TOTAL_VOTE', 'VOTE_DEM', 'VOTE_REP']]
    final_df.columns = ['election2008_fips', 'election2008_total_votes', 'election2008_dem', 'election2008_rep']
    shape_index_l = final_df.election2008_fips.tolist()

    # Read in shapefile data
    fullShapeFile = shapefile.Reader(file_path_s)
    shape_l = fullShapeFile.shapes()
    del fullShapeFile
    
    return(final_df, shape_index_l, shape_l)