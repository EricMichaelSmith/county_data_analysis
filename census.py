# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 20:10:53 2014

@author: Eric

Reads in US Census data, collected from FactFinder

Suffixes at the end of variable names:
A: numpy array
B: boolean
D: dictionary
L: list
S: string
T: tuple
Underscores indicate chaining: for instance, "fooT_T" is a tuple of tuples
"""

import config
reload(config)