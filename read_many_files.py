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
- 2008_to_2012_race_and_ethnicity
    - Searched the US Census' FactFinder with Geographies: {'County': ['All Counties within United States', 'All Counties within United States and Puerto Rico']}
    - Data from B03002: HISPANIC OR LATINO ORIGIN BY RACE
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
    fieldsD = {'2008_to_2012_age_and_sex': 
               {'ACS_12_5YR_S0101_with_ann.csv':
                {'delimiter': ',', 'lines_to_ignore': 2,
                 'fields': {178: 'median_age',
                            184: 'sex_ratio'}}},
               '2008_to_2012_race_and_ethnicity':
               {'ACS_12_5YR_B03002_with_ann.csv':
                {'delimiter': ',', 'lines_to_ignore': 2,
                 'fields': {4: '2008_to_2012_race_and_ethnicity__total',
                            8: 'white_not_hispanic__number',
                            10: 'black_not_hispanic__number',
                            14: 'asian_not_hispanic__number',
                            26: 'hispanic_number'}}},
               '2008_to_2012_social_characteristics':
               {'ACS_12_5YR_DP02_with_ann.csv':
                {'delimiter': ',', 'lines_to_ignore': 2,
                 'fields': {54: 'households_with_children',
                            58: 'households_with_senior_citizens',
                            60: 'average_household_size',
                            102: 'never_married',
                            118: 'divorced',
                            156: 'fertility',
                            266: 'high_school_graduate',
                            278: 'veterans',
                            318: 'same_house_1_yr_ago',
                            370: 'foreign_born',
                            450: 'language_other_than_english_spoken_at_home'}}},
               '2010_to_2013_population': {},
               '2012_income_and_poverty': {},
               '2013_area': {},
               '2014_health_indicators': {},
               'unemployment_statistics': {}}