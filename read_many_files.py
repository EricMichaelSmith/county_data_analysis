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
    - laucntycur14.txt downloaded 2014-09-13, others downloaded 2014-02

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

import config
reload(config)



def main(con, cur):

    # Fields to extract: first level indicates folder, second level indicates
    # file name, third level indicates fields. In length-2 tuples, index 0
    # represents the field start column for fixed-width files and index 1
    # represents the length of the field.
    field_d = {'2008_to_2012_age_and_sex': 
               {'ACS_12_5YR_S0101_with_ann.csv':
                {'delimiter': ',',
                 'header_lines_to_ignore': 2,
                 'footer_lines_to_ignore': 1,
                 'fields': {2: 'fips_column',
                            178: 'median_age',
                            184: 'sex_ratio'}}},
               '2008_to_2012_race_and_ethnicity':
               {'ACS_12_5YR_B03002_with_ann.csv':
                {'delimiter': ',',
                 'header_lines_to_ignore': 2,
                 'footer_lines_to_ignore': 1,
                 'fields': {2: 'fips_column',
                            4: '2008_to_2012_race_and_ethnicity__total',
                            8: 'white_not_hispanic__number',
                            10: 'black_not_hispanic__number',
                            14: 'asian_not_hispanic__number',
                            26: 'hispanic_number'}}},
               '2008_to_2012_social_characteristics':
               {'ACS_12_5YR_DP02_with_ann.csv':
                {'delimiter': ',',
                 'header_lines_to_ignore': 2,
                 'footer_lines_to_ignore': 1,
                 'fields': {2: 'fips_column',
                            54: 'households_with_children',
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
               '2010_to_2013_population':
               {'PEP_2013_PEPANNRES_with_ann.csv':
                {'delimiter': ',',
                 'header_lines_to_ignore': 2,
                 'footer_lines_to_ignore': 1,
                 'fields': {2: 'fips_column',
                            4: 'population_2010_census',
                            9: 'population_2013_estimate'}}},
               '2012_income_and_poverty':
               {'est12ALL.txt':
                {'delimiter': None,
                 'header_lines_to_ignore': 0,
                 'footer_lines_to_ignore': 1,
                 'fields': {(1, 2): 'fips_state_column',
                            (4, 3): 'fips_county_column',
                            (35, 4): 'in_poverty',
                            (134, 6): 'median_household_income'}}},
               '2013_area':
               {'2013_Gaz_counties_national.txt':
                {'delimiter': '\t',
                 'header_lines_to_ignore': 2,
                 'footer_lines_to_ignore': 1,
                 'fields': {2: 'fips_column',
                            6: 'land_area'}}},
               '2014_health_indicators':
               {'2014 CHR analytic data.csv':
                {'delimiter': ',',
                 'header_lines_to_ignore': 2,
                 'footer_lines_to_ignore': 1,
                 'fields': {1: 'fips_state_column',
                            2: 'fips_county_column',
                            6: 'premature_death_rate',
                            31: 'smoking',
                            36: 'obese',
                            71: 'teen_birth_rate',
                            76: 'percent_non_senior_citizens_without_insurance',
                            144: 'violent_crime_rate'}}},
               'unemployment_statistics':
               {'laucnty08.txt':
                {'delimiter': None,
                 'header_lines_to_ignore': 6,
                 'footer_lines_to_ignore': 3,
                 'fields': {(19, 2): 'fips_state_column',
                            (26, 3): 'fips_county_column',
                            (129, 4): 'unemployment_rate_2008'}},
                'laucnty12.txt':
                {'delimiter': None,
                 'header_lines_to_ignore': 6,
                 'footer_lines_to_ignore': 3,
                 'fields': {(19, 2): 'fips_state_column',
                            (26, 3): 'fips_county_column',
                            (129, 4): 'unemployment_rate_2012'}},
                'laucntycur14.txt':
                {'delimiter': None,
                 'header_lines_to_ignore': 38646,
                 'footer_lines_to_ignore': 3226,
                 'fields': {(21, 2): 'fips_state_column',
                            (28, 3): 'fips_county_column',
                            (130, 4): 'unemployment_rate_2014_jun'}}}}
                 
                 
    ## Read in files
    for folder_name in field_d:
        for file_name in field_d[folder_name]:

            # Prepare for creating table
            table_name = file_name.replace('.', '')
            file_path = os.path.join(config.raw_data_path_s, folder_name,
                                     file_name)
            cur.execute('DROP TABLE IF EXISTS {table_name};'.format(table_name=table_name))
            
            # Create table
            command_s = 'CREATE TABLE {table_name}('
            command_s = command_s.format(table_name=table_name)
            this_table_field_d = field_d[folder_name][file_name]['fields']
            for field in this_table_field_d:
                field_s = '{field_name} FLOAT(10),'
                field_s = field_s.replace(field_name=this_table_field_d[field])
                command_s += field_s
            command_s = command_s[:-1] + ');'
            cur.execute(command_s)
            
            if field_d[folder_name][file_name]['delimiter']:
                # CSV or tab-delimited tables
            
                # {{{}}}
            
            else:
                # Fixed-width tables
            
                # {{{}}}
            
            # {{{make sure to convert either 'fips_column' or 'fips_state_column'+'fips_county_column' to table_name+'_fips'}}}