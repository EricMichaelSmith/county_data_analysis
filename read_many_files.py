# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 20:10:53 2014

@author: Eric

Reads in many csv and fixed-width files at once.

Sources:
- 2008_to_2012_age_and_sex
    - Searched the US Census' FactFinder with Geographies: {'County': ['All Counties within United States', 'All Counties within United States and Puerto Rico']}
    - Data from S0101: AGE AND SEX
    - 2012 ACS 5-year estimates
    - Downloaded 2014-05-04
- 2008_to_2012_social_characteristics
    - Searched the US Census' FactFinder with Geographies: {'County': ['All Counties within United States', 'All Counties within United States and Puerto Rico']}
    - Data from DP02: SELECTED SOCIAL CHARACTERISTICS IN THE UNITED STATES
    - 2012 ACS 5-year estimates
    - Downloaded 2014-05-04 
- 2010_to_2013_population
    - County-level population estimates
    - Data downloaded from http://www.census.gov/popest/data/counties/totals/2013/CO-EST2013-01.html, 2014-05-04
- 2012_income_and_poverty
    - County-level estimates of US poverty and income for 2012
    - Direct link: http://www.census.gov/did/www/saipe/downloads/estmod12/est12ALL.txt
    - Guide to fields can be found http://www.census.gov/did/www/saipe/data/statecounty/data/2012.html
- 2013_area
    - From http://www.census.gov/geo/maps-data/data/gazetteer2013.html; see link for column info
    - Downloaded 2014-05-04
- 2014_health_indicators
    - Available from http://www.countyhealthrankings.org/rankings/data, a program from Robert Wood Johnson
- unemployment_statistics
    - From http://www.bls.gov/lau/tables.htm

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



def main(con, cur):
    
    # {{{add all files to fieldsD: each file should contain three keys, one for all of the fields to import, one for the delimiter, and one for how many lines to ignore; anything else?}}}

    # Fields to extract: first level indicates folder, second level indicates
    # file name, third level indicates fields
    fieldsD = {'2008_to_2012_age_and_sex': {},
               '2008_to_2012_social_characteristics': {},
               '2010_to_2013_population': {},
               '2012_income_and_poverty': {},
               '2013_area': {},
               '2014_health_indicators': {},
               'unemployment_statistics': {}}