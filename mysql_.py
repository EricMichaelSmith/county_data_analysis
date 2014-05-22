# -*- coding: utf-8 -*-
"""
Created on Sun May 18 20:36:32 2014

@author: Eric Smith

What trends exist within US-county-level demographic data?

Suffixes at the end of variable names:
A: numpy array
B: boolean
D: dictionary
L: list
S: string
T: tuple
Underscores indicate chaining: for instance, "fooT_T" is a tuple of tuples
"""

import pymysql
import sys

import config
reload(config)

sys.path.append(config.configLocalPathS)
import config_local
reload(config_local)



def connect_to_mysql():
    """
    Connection code from http://zetcode.com/db/mysqlpython/, 2014-05-18
    """
    
    try:
        con = pymysql.connect(user='root', passwd=config_local.pwordS)
        return con
        
    except pymysql.Error, e:
        print 'Error %d: %s' % (e.args[0], e.args[1])
        
    finally:
        if con:
            con.close()