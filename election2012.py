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
import os
import sys

import config
reload(config)

sys.path.append(config.configLocalPathS)
import config_local
reload(config_local)

sys.path.append(os.path.join(config.packagePathS, 'richardpenman-csv2mysql-5d059a4361fb'))



def main():

    con = MySQLdb.connect(user='root',
                          passwd=config_local.pwordS,
                          db='sudointellectual',
                          local_infile=1)
    cur = con.cursor()

    # Read in 2012 election data
    filePathS = os.path.join(config.rawDataPathS, 'election_statistics',
                             'US_elect_county__2012.csv')
    cur.execute('DROP TABLE IF EXISTS election2012')
    cur.execute('CREATE TABLE election2012(myindex INT PRIMARY KEY AUTO_INCREMENT, un VARCHAR(25), trois VARCHAR(25))')
#    mysqlS = """
#             LOAD DATA LOCAL INFILE {filePathS}
#             INTO TABLE election2012
#             FIELDS TERMINATED BY ','
#             LINES TERMINATED BY '\r\n'
#             IGNORE 1 LINES
#             (column1, column2);
#             """
#    cur.execute(mysqlS.format(filePathS=filePathS))
    cur.execute("LOAD DATA LOCAL INFILE 'foo.csv' INTO TABLE election2012 FIELDS TERMINATED BY ',' LINES TERMINATED BY '\r\n' (@col1, @col2, @col3) SET un=@col1, trois=@col3")
#    mysqlS = """
#             LOAD DATA LOCAL INFILE 'US_elect_county__2012.csv'
#             INTO TABLE election2012;
#             """
#    cur.execute(mysqlS)
    cur.execute('SELECT * FROM election2012')
    for lRow in range(5):
        row = cur.fetchone()
        print row