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

import config
reload(config)
import mysql_
reload(mysql_)



def main():
    
    # Connect to MySQL
    con = mysql_.connect_to_mysql()
    
    with con:
        
        # Read in 2012 election data
        filePathS = os.path.join(config.rawDataPathS, 'election_statistics',
                                 'US_elect_county__2012.csv')
        cur = con.cursor()
        cur.execute('DROP TABLE IF EXISTS election2012')
        mysqlS = """
                 LOAD DATA LOCAL INFILE {filePathS}
                 INTO TABLE election2012
                 FIELDS TERMINATED BY ','
                 LINES TERMINATED BY '\r\n'
                 IGNORE 2 LINES;
                 """
        cur.execute(mysqlS.format(filePathS=filePathS))
        cur.execute('SELECT * FROM election2012')
        for lRow in range(5):
            row = cur.fetchone()
            print row